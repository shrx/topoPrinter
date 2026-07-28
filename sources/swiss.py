#!/usr/bin/env python3
"""
Generate Swiss SwissALTI3D or SwissSURFACE3D tile URLs for a given location and radius.

Usage:
    python -m sources.swiss --lat 46.1167 --lon 7.7167 --diameter 10
    python -m sources.swiss --lat 46.1167 --lon 7.7167 --bbox 10 --dataset surface3d
    python -m sources.swiss --lv95 2617000 1095000 --diameter 10
    python -m sources.swiss --rect-corners 46.02757,7.5648,49.12584,7.76655 --bearing 300

Coordinates are in WGS84 (lat/lon) or Swiss LV95 (easting/northing in meters).
Diameter and bbox are in kilometers.
Rectangle corners are specified as LAT1,LON1,LAT2,LON2.
Bearing is in degrees (0-360), where 0=North, 90=East, 180=South, 270=West.

Dataset options:
    alti3d (default)  - SwissALTI3D terrain elevation (buildings/trees removed)
    surface3d         - SwissSURFACE3D surface elevation (includes buildings/trees)
"""

import sys
import argparse
from typing import List, Tuple, Optional
from pyproj import Transformer
import requests

# Swiss tile URL patterns
SWISS_ALTI3D_URL_PATTERN = "https://data.geo.admin.ch/ch.swisstopo.swissalti3d/swissalti3d_{year}_{e}-{n}/swissalti3d_{year}_{e}-{n}_2_2056_5728.tif"
SWISS_SURFACE3D_URL_PATTERN = "https://data.geo.admin.ch/ch.swisstopo.swisssurface3d-raster/swisssurface3d-raster_{year}_{e}-{n}/swisssurface3d-raster_{year}_{e}-{n}_0.5_2056_5728.tif"

# Default years for tiles (can be overridden)
DEFAULT_YEAR_ALTI3D = 2019

# STAC API endpoints
STAC_API_BASE = "https://data.geo.admin.ch/api/stac/v0.9"
STAC_COLLECTION_SURFACE3D = "ch.swisstopo.swisssurface3d-raster"


def query_tile_years_from_stac(tiles: List[Tuple[int, int]]) -> dict:
    """
    Query STAC API once for the bbox covering all tiles and return per-tile years.

    A single paged bbox query replaces one HTTP request per tile.

    Args:
        tiles: List of (tile_e_km, tile_n_km) tuples (LV95)

    Returns:
        Dict mapping (tile_e_km, tile_n_km) -> year for every tile found.
    """
    transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)

    e_min = min(t[0] for t in tiles)
    e_max = max(t[0] for t in tiles) + 1
    n_min = min(t[1] for t in tiles)
    n_max = max(t[1] for t in tiles) + 1
    min_lon, min_lat = transformer.transform(e_min * 1000, n_min * 1000)
    max_lon, max_lat = transformer.transform(e_max * 1000, n_max * 1000)

    url = f"{STAC_API_BASE}/collections/{STAC_COLLECTION_SURFACE3D}/items"
    params = {"bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}", "limit": 100}

    years: dict = {}
    session = requests.Session()
    while url:
        response = session.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        for feature in data.get('features', []):
            item_id = feature.get('id', '')
            # Expected format: swisssurface3d-raster_YEAR_E-N
            parts = item_id.split('_')
            if len(parts) >= 3:
                try:
                    year = int(parts[1])
                    e_str, n_str = parts[2].split('-')
                    years[(int(e_str), int(n_str))] = year
                except ValueError:
                    continue

        # Follow pagination; the "next" href carries its own query params
        url = next((lnk.get('href') for lnk in data.get('links', [])
                    if lnk.get('rel') == 'next'), None)
        params = None

    return years


def latlon_to_lv95(lat: float, lon: float) -> Tuple[float, float]:
    """
    Convert WGS84 lat/lon to Swiss LV95 coordinates.

    Args:
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees

    Returns:
        Tuple of (easting, northing) in meters
    """
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
    easting, northing = transformer.transform(lon, lat)
    return easting, northing


def get_tiles_in_radius(center_e: float, center_n: float, radius_km: float) -> List[Tuple[int, int]]:
    """
    Get all 1km tiles that intersect with a circular radius of center point.

    Args:
        center_e: Center easting in meters
        center_n: Center northing in meters
        radius_km: Radius in kilometers

    Returns:
        List of (tile_e_km, tile_n_km) tuples
    """
    radius_m = radius_km * 1000.0
    # Add extra margin to ensure we get all intersecting tiles (half diagonal of 1km tile = ~707m)
    search_radius = radius_m + 707.0
    tiles = set()

    # Calculate bounding box with margin
    min_e = center_e - search_radius
    max_e = center_e + search_radius
    min_n = center_n - search_radius
    max_n = center_n + search_radius

    # Iterate through all tiles in bounding box
    tile_e_min = int(min_e / 1000)
    tile_e_max = int(max_e / 1000) + 1
    tile_n_min = int(min_n / 1000)
    tile_n_max = int(max_n / 1000) + 1

    for tile_e_km in range(tile_e_min, tile_e_max):
        for tile_n_km in range(tile_n_min, tile_n_max):
            # Check if tile intersects with circle
            # Find closest point in tile to center
            tile_min_e = tile_e_km * 1000
            tile_max_e = (tile_e_km + 1) * 1000
            tile_min_n = tile_n_km * 1000
            tile_max_n = (tile_n_km + 1) * 1000

            # Clamp center to tile bounds to find closest point
            closest_e = max(tile_min_e, min(center_e, tile_max_e))
            closest_n = max(tile_min_n, min(center_n, tile_max_n))

            # Distance from center to closest point in tile
            dist = ((closest_e - center_e)**2 + (closest_n - center_n)**2)**0.5

            if dist <= radius_m:
                tiles.add((tile_e_km, tile_n_km))

    return sorted(tiles)


def get_tiles_in_bbox(center_e: float, center_n: float, side_length_km: float) -> List[Tuple[int, int]]:
    """
    Get all 1km tiles within a square bounding box around center point.

    Args:
        center_e: Center easting in meters
        center_n: Center northing in meters
        side_length_km: Side length of square in kilometers

    Returns:
        List of (tile_e_km, tile_n_km) tuples
    """
    half_side_m = (side_length_km * 1000.0) / 2.0

    min_e = center_e - half_side_m
    max_e = center_e + half_side_m
    min_n = center_n - half_side_m
    max_n = center_n + half_side_m

    tile_e_min = int(min_e / 1000)
    tile_e_max = int(max_e / 1000)
    tile_n_min = int(min_n / 1000)
    tile_n_max = int(max_n / 1000)

    tiles = []
    for tile_e_km in range(tile_e_min, tile_e_max + 1):
        for tile_n_km in range(tile_n_min, tile_n_max + 1):
            tiles.append((tile_e_km, tile_n_km))

    return sorted(tiles)


def get_tiles_in_rotated_rect(corner1_e: float, corner1_n: float,
                                corner2_e: float, corner2_n: float,
                                bearing: float = 0.0) -> List[Tuple[int, int]]:
    """
    Get all 1km tiles that intersect with a rotated rectangle defined by opposite corners.

    Args:
        corner1_e: First corner easting in meters
        corner1_n: First corner northing in meters
        corner2_e: Second corner easting in meters
        corner2_n: Second corner northing in meters
        bearing: Bearing in degrees (0-360), where 0=North, 90=East

    Returns:
        List of (tile_e_km, tile_n_km) tuples
    """
    import numpy as np
    from shapely.geometry import Polygon, box
    from bearing_utils import rotate_to_bearing_frame, rotate_from_bearing_frame

    # Calculate center
    center_e = (corner1_e + corner2_e) / 2.0
    center_n = (corner1_n + corner2_n) / 2.0

    # If no rotation, use simple bounding box
    if bearing == 0.0:
        min_e = min(corner1_e, corner2_e)
        max_e = max(corner1_e, corner2_e)
        min_n = min(corner1_n, corner2_n)
        max_n = max(corner1_n, corner2_n)

        tile_e_min = int(min_e / 1000)
        tile_e_max = int(max_e / 1000)
        tile_n_min = int(min_n / 1000)
        tile_n_max = int(max_n / 1000)

        tiles = []
        for tile_e_km in range(tile_e_min, tile_e_max + 1):
            for tile_n_km in range(tile_n_min, tile_n_max + 1):
                tiles.append((tile_e_km, tile_n_km))
        return sorted(tiles)

    bearing_rad = np.radians(bearing)

    # Project opposite corners onto bearing-aligned local frame
    c1e_centered = corner1_e - center_e
    c1n_centered = corner1_n - center_n
    c2e_centered = corner2_e - center_e
    c2n_centered = corner2_n - center_n

    c1_perp, c1_along = rotate_to_bearing_frame(c1e_centered, c1n_centered, bearing_rad)
    c2_perp, c2_along = rotate_to_bearing_frame(c2e_centered, c2n_centered, bearing_rad)

    rect_min_perp = min(c1_perp, c2_perp)
    rect_max_perp = max(c1_perp, c2_perp)
    rect_min_along = min(c1_along, c2_along)
    rect_max_along = max(c1_along, c2_along)

    # Rotate the 4 rectangle corners back to CRS to build the exact polygon.
    # Ring order (min,min)->(max,min)->(max,max)->(min,max) so the polygon is
    # the true rotated rectangle, not a Z-ordered self-intersection.
    corners_local = [
        (rect_min_perp, rect_min_along),
        (rect_max_perp, rect_min_along),
        (rect_max_perp, rect_max_along),
        (rect_min_perp, rect_max_along),
    ]

    rect_corners_crs = []
    for perp, along in corners_local:
        de, dn = rotate_from_bearing_frame(perp, along, bearing_rad)
        rect_corners_crs.append((de + center_e, dn + center_n))
    rect = Polygon(rect_corners_crs)

    bbox_e = [e for e, _ in rect_corners_crs]
    bbox_n = [n for _, n in rect_corners_crs]

    # Get tile range from bounding box
    tile_e_min = int(min(bbox_e) / 1000) - 1
    tile_e_max = int(max(bbox_e) / 1000) + 1
    tile_n_min = int(min(bbox_n) / 1000) - 1
    tile_n_max = int(max(bbox_n) / 1000) + 1

    # Keep every tile whose 1km footprint actually overlaps the rectangle.
    # Exact polygon-box intersection (not a tile-center + margin approximation,
    # which misses tiles the rectangle only clips near its corners/edges).
    tiles = []
    for tile_e_km in range(tile_e_min, tile_e_max + 1):
        for tile_n_km in range(tile_n_min, tile_n_max + 1):
            tile_box = box(tile_e_km * 1000, tile_n_km * 1000,
                           (tile_e_km + 1) * 1000, (tile_n_km + 1) * 1000)
            if rect.intersects(tile_box):
                tiles.append((tile_e_km, tile_n_km))

    return sorted(tiles)


def generate_urls(tiles: List[Tuple[int, int]], dataset: str = "alti3d", year: int = None) -> List[str]:
    """
    Generate download URLs for tiles.

    Args:
        tiles: List of (easting_km, northing_km) tuples
        dataset: Dataset type: "alti3d" (terrain only) or "surface3d" (includes buildings/trees)
        year: Data year (only used for alti3d, defaults to 2019)

    Returns:
        List of download URLs
    """
    urls = []

    if dataset == "alti3d":
        url_pattern = SWISS_ALTI3D_URL_PATTERN
        if year is None:
            year = DEFAULT_YEAR_ALTI3D
        for e_km, n_km in tiles:
            url = url_pattern.format(year=year, e=e_km, n=n_km)
            urls.append(url)

    elif dataset == "surface3d":
        url_pattern = SWISS_SURFACE3D_URL_PATTERN
        print(f"[INFO] Querying STAC API for {len(tiles)} surface3d tiles (single bbox query)...",
              file=sys.stderr, flush=True)
        years = query_tile_years_from_stac(tiles)
        for e_km, n_km in tiles:
            tile_year = years.get((e_km, n_km))
            if tile_year is None:
                print(f"[WARN] Tile {e_km}-{n_km} not found in STAC API, skipping", file=sys.stderr, flush=True)
                continue
            url = url_pattern.format(year=tile_year, e=e_km, n=n_km)
            urls.append(url)

    else:
        raise ValueError(f"Invalid dataset: {dataset}. Must be 'alti3d' or 'surface3d'.")

    return urls


def main():
    parser = argparse.ArgumentParser(
        description="Generate Swiss SwissALTI3D or SwissSURFACE3D tile URLs for a given location and radius."
    )

    # Coordinate input (mutually exclusive)
    coord_group = parser.add_mutually_exclusive_group(required=True)
    coord_group.add_argument(
        "--lat",
        type=float,
        help="Center latitude in WGS84 decimal degrees (use with --lon)."
    )
    coord_group.add_argument(
        "--lv95",
        nargs=2,
        type=float,
        metavar=("EASTING", "NORTHING"),
        help="Center coordinates in Swiss LV95 (easting northing in meters)."
    )
    coord_group.add_argument(
        "--rect-corners",
        type=str,
        help="Rectangle corners as LAT1,LON1,LAT2,LON2 (e.g., '46.5,8.5,47.0,9.0')."
    )

    parser.add_argument(
        "--lon",
        type=float,
        help="Center longitude in WGS84 decimal degrees (required with --lat)."
    )

    # Region selection (mutually exclusive, only for center-based modes)
    region_group = parser.add_mutually_exclusive_group(required=False)
    region_group.add_argument(
        "--diameter",
        type=float,
        help="Circular diameter in kilometers (requires --lat/--lon or --lv95)."
    )
    region_group.add_argument(
        "--bbox",
        type=float,
        help="Square bounding box side length in kilometers (requires --lat/--lon or --lv95)."
    )

    parser.add_argument(
        "--bearing",
        type=float,
        default=0.0,
        help="Bearing in degrees (0-360) for cutout rotation. 0/360=North, 90=East, 180=South, 270=West. Default: 0."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["alti3d", "surface3d"],
        default="alti3d",
        help="Dataset to download: 'alti3d' (terrain elevation, default) or 'surface3d' (includes buildings/trees)."
    )

    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help=f"Data year for alti3d tiles (default: {DEFAULT_YEAR_ALTI3D}). For surface3d, year is determined via STAC API query."
    )

    args = parser.parse_args()

    # Validate lat/lon combination
    if args.lat is not None and args.lon is None:
        print("Error: --lat requires --lon", file=sys.stderr)
        sys.exit(1)

    if args.lon is not None and args.lat is None:
        print("Error: --lon requires --lat", file=sys.stderr)
        sys.exit(1)

    # Validate bearing
    if not (0.0 <= args.bearing <= 360.0):
        print("Error: --bearing must be between 0 and 360 degrees", file=sys.stderr)
        sys.exit(1)

    # Handle rect-corners mode
    if args.rect_corners is not None:
        try:
            parts = args.rect_corners.split(',')
            if len(parts) != 4:
                raise ValueError("must be LAT1,LON1,LAT2,LON2 format")
            rect_lat1, rect_lon1, rect_lat2, rect_lon2 = (
                float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
            )
            if not all(-90 <= lat <= 90 for lat in [rect_lat1, rect_lat2]):
                raise ValueError("latitude coordinates out of range")
            if not all(-180 <= lon <= 180 for lon in [rect_lon1, rect_lon2]):
                raise ValueError("longitude coordinates out of range")
            if rect_lat1 == rect_lat2 or rect_lon1 == rect_lon2:
                raise ValueError("corners must define a non-zero area rectangle")
        except ValueError as e:
            print(f"Error: Invalid --rect-corners: {e}", file=sys.stderr)
            sys.exit(1)

        # Convert corners to LV95
        corner1_e, corner1_n = latlon_to_lv95(rect_lat1, rect_lon1)
        corner2_e, corner2_n = latlon_to_lv95(rect_lat2, rect_lon2)
        print(f"[INFO] Rectangle corner 1: ({rect_lat1}, {rect_lon1}) -> LV95: ({corner1_e:.1f}, {corner1_n:.1f})", file=sys.stderr, flush=True)
        print(f"[INFO] Rectangle corner 2: ({rect_lat2}, {rect_lon2}) -> LV95: ({corner2_e:.1f}, {corner2_n:.1f})", file=sys.stderr, flush=True)

        bearing_info = f" with bearing {args.bearing}°" if args.bearing != 0.0 else ""
        tiles = get_tiles_in_rotated_rect(corner1_e, corner1_n, corner2_e, corner2_n, args.bearing)
        print(f"[INFO] Found {len(tiles)} tiles in rotated rectangle{bearing_info}", file=sys.stderr, flush=True)

    # Handle center-based modes
    else:
        # Require diameter or bbox for center-based modes
        if args.diameter is None and args.bbox is None:
            print("Error: --diameter or --bbox required with --lat/--lon or --lv95", file=sys.stderr)
            sys.exit(1)

        # Get center coordinates in LV95
        if args.lat is not None:
            center_e, center_n = latlon_to_lv95(args.lat, args.lon)
            print(f"[INFO] Converted ({args.lat}, {args.lon}) to Swiss LV95: ({center_e:.1f}, {center_n:.1f})", file=sys.stderr, flush=True)
        else:
            center_e, center_n = args.lv95[0], args.lv95[1]
            print(f"[INFO] Using Swiss LV95 coordinates: ({center_e:.1f}, {center_n:.1f})", file=sys.stderr, flush=True)

        # Get tiles
        if args.diameter is not None:
            radius_km = args.diameter / 2.0
            tiles = get_tiles_in_radius(center_e, center_n, radius_km)
            print(f"[INFO] Found {len(tiles)} tiles within {args.diameter}km diameter (radius {radius_km}km)", file=sys.stderr, flush=True)
        else:
            # For bbox with bearing, we need to handle rotation
            if args.bearing != 0.0:
                # Use the rotated rectangle logic with center-based dimensions
                half_side_m = (args.bbox * 1000.0) / 2.0
                corner1_e = center_e - half_side_m
                corner1_n = center_n - half_side_m
                corner2_e = center_e + half_side_m
                corner2_n = center_n + half_side_m
                tiles = get_tiles_in_rotated_rect(corner1_e, corner1_n, corner2_e, corner2_n, args.bearing)
                print(f"[INFO] Found {len(tiles)} tiles in {args.bbox}km × {args.bbox}km box with bearing {args.bearing}°", file=sys.stderr, flush=True)
            else:
                tiles = get_tiles_in_bbox(center_e, center_n, args.bbox)
                print(f"[INFO] Found {len(tiles)} tiles in {args.bbox}km × {args.bbox}km box", file=sys.stderr, flush=True)

    if not tiles:
        print("Error: No tiles found for specified region", file=sys.stderr, flush=True)
        sys.exit(1)

    # Generate URLs
    urls = generate_urls(tiles, dataset=args.dataset, year=args.year)

    # Output URLs to stdout (can be redirected to file)
    for url in urls:
        print(url, flush=True)

    dataset_name = "SwissALTI3D" if args.dataset == "alti3d" else "SwissSURFACE3D"
    print(f"[INFO] Generated {len(urls)} {dataset_name} tile URLs", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
