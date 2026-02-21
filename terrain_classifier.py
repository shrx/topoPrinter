"""
Terrain classification from OpenStreetMap data via Overpass API.

Queries OSM polygon features (glacier, water, foliage) and rasterizes
them onto the DEM grid. Unclassified pixels default to rock.
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from pyproj import Transformer

# Terrain class constants (priority: lower value = higher priority)
TERRAIN_ROCK = 0
TERRAIN_GLACIER = 1
TERRAIN_WATER = 2
TERRAIN_FOLIAGE = 3

TERRAIN_NAMES = {
    TERRAIN_ROCK: "rock",
    TERRAIN_GLACIER: "glacier",
    TERRAIN_WATER: "water",
    TERRAIN_FOLIAGE: "foliage",
}

# OSM tag → terrain class mapping
_GLACIER_NATURAL = {"glacier"}
_WATER_NATURAL = {"water"}
_WATER_LANDUSE = {"reservoir", "basin"}
_FOLIAGE_NATURAL = {"wood", "scrub", "heath", "grassland", "fell", "tundra", "moor", "wetland"}
_FOLIAGE_LANDUSE = {"forest", "meadow", "grass", "farmland", "orchard", "vineyard"}

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def _classify_element(tags: dict) -> int:
    """Classify an OSM element by its tags. Returns terrain class constant."""
    natural = tags.get("natural", "")
    landuse = tags.get("landuse", "")

    # Glacier (highest priority), excluding rock glaciers
    if natural in _GLACIER_NATURAL and tags.get("glacier:type") != "rock":
        return TERRAIN_GLACIER

    # Water
    if natural in _WATER_NATURAL:
        return TERRAIN_WATER
    if landuse in _WATER_LANDUSE:
        return TERRAIN_WATER

    # Foliage
    if natural in _FOLIAGE_NATURAL:
        return TERRAIN_FOLIAGE
    if landuse in _FOLIAGE_LANDUSE:
        return TERRAIN_FOLIAGE

    return TERRAIN_ROCK


def _build_overpass_query(bbox: Tuple[float, float, float, float]) -> str:
    """Build Overpass QL query for terrain polygons in bounding box.

    Args:
        bbox: (south, west, north, east) in WGS84 degrees.

    Returns:
        Overpass QL query string.
    """
    s, w, n, e = bbox
    bb = f"{s},{w},{n},{e}"

    # All tags we need, grouped by key
    tag_filters = [
        ("natural", "glacier"),
        ("natural", "water"),
        ("landuse", "reservoir"),
        ("landuse", "basin"),
        ("natural", "wood"),
        ("landuse", "forest"),
        ("natural", "scrub"),
        ("natural", "heath"),
        ("natural", "grassland"),
        ("natural", "fell"),
        ("natural", "tundra"),
        ("natural", "moor"),
        ("landuse", "meadow"),
        ("landuse", "grass"),
        ("landuse", "farmland"),
        ("landuse", "orchard"),
        ("landuse", "vineyard"),
        ("natural", "wetland"),
    ]

    lines = ["[out:json][timeout:120];", "("]
    for key, val in tag_filters:
        lines.append(f'  way["{key}"="{val}"]({bb});')
        lines.append(f'  relation["{key}"="{val}"]({bb});')
    lines.append(");")
    lines.append("out geom;")

    return "\n".join(lines)


def _way_to_geojson(element: dict) -> Optional[dict]:
    """Convert an Overpass way element (with geometry) to a GeoJSON Polygon."""
    geom = element.get("geometry", [])
    if len(geom) < 4:  # need at least 3 unique points + closing
        return None
    coords = [[pt["lon"], pt["lat"]] for pt in geom]
    # Ensure ring is closed
    if coords[0] != coords[-1]:
        coords.append(list(coords[0]))
    return {"type": "Polygon", "coordinates": [coords]}


def _join_ring_segments(segments: List[List[list]]) -> List[List[list]]:
    """Join way segments into closed rings by matching endpoints.

    Args:
        segments: list of coordinate sequences [[lon, lat], ...]

    Returns:
        List of closed rings (each a list of [lon, lat] coords).
    """
    if not segments:
        return []

    # Work with mutable copies
    remaining = [list(seg) for seg in segments]
    rings = []

    while remaining:
        current = remaining.pop(0)
        changed = True
        while changed:
            changed = False
            for i, seg in enumerate(remaining):
                # Try to append to current ring
                if _coords_close(current[-1], seg[0]):
                    current.extend(seg[1:])
                    remaining.pop(i)
                    changed = True
                    break
                elif _coords_close(current[-1], seg[-1]):
                    current.extend(reversed(seg[:-1]))
                    remaining.pop(i)
                    changed = True
                    break
                elif _coords_close(current[0], seg[-1]):
                    current = seg[:-1] + current
                    remaining.pop(i)
                    changed = True
                    break
                elif _coords_close(current[0], seg[0]):
                    current = list(reversed(seg[1:])) + current
                    remaining.pop(i)
                    changed = True
                    break

        # Close ring if endpoints match
        if _coords_close(current[0], current[-1]):
            if current[0] != current[-1]:
                current.append(list(current[0]))
            rings.append(current)
        # else: unclosed ring — discard with warning

    return rings


def _coords_close(a: list, b: list, tol: float = 1e-7) -> bool:
    """Check if two [lon, lat] coordinates are approximately equal."""
    return abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol


def _relation_to_geojson(element: dict) -> Optional[dict]:
    """Convert an Overpass relation element to a GeoJSON Polygon/MultiPolygon."""
    members = element.get("members", [])
    if not members:
        return None

    outer_segments = []
    inner_segments = []

    for member in members:
        role = member.get("role", "")
        geom = member.get("geometry", [])
        if not geom or len(geom) < 2:
            continue
        coords = [[pt["lon"], pt["lat"]] for pt in geom]
        if role == "outer":
            outer_segments.append(coords)
        elif role == "inner":
            inner_segments.append(coords)

    outer_rings = _join_ring_segments(outer_segments)
    inner_rings = _join_ring_segments(inner_segments)

    if not outer_rings:
        return None

    if len(outer_rings) == 1:
        # Single polygon, possibly with holes
        rings = [outer_rings[0]] + inner_rings
        return {"type": "Polygon", "coordinates": rings}
    else:
        # MultiPolygon — assign each inner ring to the outer ring that contains it
        from shapely.geometry import Polygon as _Poly, Point as _Point
        outer_polys = [_Poly(ring) for ring in outer_rings]
        inner_assignments: Dict[int, list] = {i: [] for i in range(len(outer_rings))}
        for inner in inner_rings:
            # Test a point on the inner ring against each outer
            pt = _Point(inner[0])
            for i, op in enumerate(outer_polys):
                if op.contains(pt):
                    inner_assignments[i].append(inner)
                    break
        polygons = []
        for i, outer in enumerate(outer_rings):
            polygons.append([outer] + inner_assignments[i])
        return {"type": "MultiPolygon", "coordinates": polygons}


def _parse_geometry(element: dict) -> Optional[dict]:
    """Convert an Overpass element to a GeoJSON geometry dict."""
    etype = element.get("type")
    if etype == "way":
        return _way_to_geojson(element)
    elif etype == "relation":
        return _relation_to_geojson(element)
    return None


def _transform_coords(coords_rings, transformer) -> list:
    """Transform polygon coordinate rings from WGS84 to target CRS."""
    transformed = []
    for ring in coords_rings:
        lons = [c[0] for c in ring]
        lats = [c[1] for c in ring]
        xs, ys = transformer.transform(lons, lats)
        transformed.append(list(zip(xs, ys)))
    return transformed


def _transform_geojson_to_crs(geom: dict, transformer) -> dict:
    """Reproject a GeoJSON geometry from WGS84 to a target CRS.

    Args:
        geom: GeoJSON geometry dict (Polygon or MultiPolygon).
        transformer: pyproj.Transformer (WGS84 → target CRS, always_xy=True).

    Returns:
        GeoJSON geometry dict with transformed coordinates.
    """
    gtype = geom["type"]
    if gtype == "Polygon":
        return {
            "type": "Polygon",
            "coordinates": _transform_coords(geom["coordinates"], transformer),
        }
    elif gtype == "MultiPolygon":
        return {
            "type": "MultiPolygon",
            "coordinates": [
                _transform_coords(polygon_rings, transformer)
                for polygon_rings in geom["coordinates"]
            ],
        }
    return geom


def classify_terrain(
    dem_shape: Tuple[int, int],
    ref_transform,
    ref_crs,
    overpass_url: str = OVERPASS_URL,
) -> Dict[int, list]:
    """Query OSM and collect terrain polygon geometries.

    Args:
        dem_shape: (rows, cols) of the DEM array.
        ref_transform: rasterio Affine transform for the DEM.
        ref_crs: CRS of the DEM (pyproj or rasterio CRS).
        overpass_url: Overpass API endpoint.
        terrain_types: optional list of terrain type names to include
            (e.g. ["glacier", "foliage"]). None means all.

    Returns:
        dict mapping terrain class → list of GeoJSON geometry dicts in CRS coords.
    """
    rows, cols = dem_shape

    # Compute DEM bounding box corners in CRS space
    corners_crs = [
        ref_transform * (0, 0),          # top-left
        ref_transform * (cols, 0),       # top-right
        ref_transform * (cols, rows),    # bottom-right
        ref_transform * (0, rows),       # bottom-left
    ]
    xs_crs = [c[0] for c in corners_crs]
    ys_crs = [c[1] for c in corners_crs]

    # Transform CRS corners to WGS84
    to_wgs84 = Transformer.from_crs(ref_crs, "EPSG:4326", always_xy=True)
    lons, lats = to_wgs84.transform(xs_crs, ys_crs)

    # Bounding box with small buffer
    buf = 0.001  # ~100m buffer in degrees
    bbox = (min(lats) - buf, min(lons) - buf, max(lats) + buf, max(lons) + buf)

    empty_features: Dict[int, list] = {
        TERRAIN_GLACIER: [],
        TERRAIN_WATER: [],
        TERRAIN_FOLIAGE: [],
    }

    # Query Overpass API
    query = _build_overpass_query(bbox)
    data = _query_overpass(query, overpass_url)
    if data is None:
        return empty_features

    elements = data.get("elements", [])
    if not elements:
        print("[terrain] No OSM features found in bounding box.")
        return empty_features

    # Parse, classify, and transform geometries
    from_wgs84 = Transformer.from_crs("EPSG:4326", ref_crs, always_xy=True)

    features_by_class: Dict[int, list] = {
        TERRAIN_GLACIER: [],
        TERRAIN_WATER: [],
        TERRAIN_FOLIAGE: [],
    }

    for element in elements:
        tags = element.get("tags", {})
        terrain_class = _classify_element(tags)
        if terrain_class == TERRAIN_ROCK:
            continue

        geom = _parse_geometry(element)
        if geom is None:
            continue

        geom_crs = _transform_geojson_to_crs(geom, from_wgs84)
        features_by_class[terrain_class].append(geom_crs)

    return features_by_class


def _query_overpass(
    query: str,
    url: str,
    max_retries: int = 3,
    initial_delay: float = 10.0,
) -> Optional[dict]:
    """Send Overpass query with retry on rate limiting.

    Returns parsed JSON dict, or None on failure.
    """
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            print(f"[terrain] Querying Overpass API (attempt {attempt + 1})...", flush=True)
            resp = requests.post(url, data={"data": query}, timeout=120)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code in (429, 504):
                print(f"[terrain] HTTP {resp.status_code}, retrying in {delay:.0f}s...", flush=True)
                time.sleep(delay)
                delay *= 2
            else:
                print(f"[terrain] Overpass API error: HTTP {resp.status_code}", flush=True)
                return None
        except Exception as e:
            print(f"[terrain] Overpass request failed: {e}", flush=True)
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2

    print("[terrain] Overpass API failed after retries, defaulting to all rock.", flush=True)
    return None
