#!/usr/bin/env python3
"""
Generate TanDEM-X 30m EDEM tile URLs for a given location and radius.

Covers Turkey (and the rest of the globe) — used for areas like Mount Ararat
where no open national LiDAR/DEM program exists. This is the best openly
accessible DEM there: TanDEM-X-derived, ~30 m, edited, no voids. (Turkey is
excluded from the open Copernicus GLO-30; the 12 m / EEA-10 data is gated
behind commercial purchase or EU-research eligibility.)

Usage:
    python -m sources.tandemx --lat 39.7025 --lon 44.2983 --diameter 20
    python -m sources.tandemx --lat 39.7025 --lon 44.2983 --bbox 15
    python -m sources.tandemx --rect-corners 39.6,44.1,39.8,44.5

Coordinates are in WGS84 (lat/lon). Diameter and bbox are in kilometers.
Rectangle corners are specified as LAT1,LON1,LAT2,LON2.

Tiles are 1°x1° geocells (wider in longitude above 60° latitude), ~30 m pixel
spacing, distributed as zip archives containing GeoTIFFs with ellipsoidal
(WGS84) and geoid (EGM2008) heights plus editing-mask aux layers.

Downloading requires a free DLR EOC account (self-registration):
    https://sso.eoc.dlr.de/tdm30-edited/selfservice/register
The download server uses preemptive HTTP basic auth, e.g.:
    wget -i urls.txt --auth-no-challenge --user=USER --ask-password
    aria2c -i urls.txt --http-user USER --http-passwd PASS

Credentials come from the environment variables DLR_EOC_USER and
DLR_EOC_PASSWORD, loaded from a gitignored env.sh at the repo root (see
env.sh.example) via python-dotenv, so you can just run:
    python -m sources.tandemx --lat 39.7025 --lon 44.2983 --diameter 20
Variables already set in the environment take precedence.

Without credentials this script emits candidate URLs following the documented
naming convention (two zip-name variants per tile — the downloader tries
alternatives, like the ARSO fetcher). With credentials, it lists the server
directories and emits only the exact, verified zip URLs.
"""

import os
import shutil
import sys
import math
import argparse
import re
import zipfile
from typing import List, Optional, Tuple

from dotenv import load_dotenv

BASE_URL = "https://download.geoservice.dlr.de/TDM30_EDEM/files"
REGISTER_URL = "https://sso.eoc.dlr.de/tdm30-edited/selfservice/register"

KM_PER_DEG_LAT = 111.32


def lon_tile_step(lat_deg: int) -> int:
    """Longitude extent (degrees) of a geocell at the given latitude band."""
    abs_lat = abs(lat_deg)
    if abs_lat >= 80:
        return 4
    if abs_lat >= 60:
        return 2
    return 1


def format_geocell(lat_deg: int, lon_deg: int) -> str:
    """Format a southwest corner as e.g. N39E044 / S05W071."""
    ns = "N" if lat_deg >= 0 else "S"
    ew = "E" if lon_deg >= 0 else "W"
    return f"{ns}{abs(lat_deg):02d}{ew}{abs(lon_deg):03d}"


def get_tiles_in_latlon_bbox(min_lat: float, min_lon: float,
                             max_lat: float, max_lon: float) -> List[Tuple[int, int]]:
    """
    Get all geocells intersecting a lat/lon bounding box.

    Returns:
        List of (lat_deg, lon_deg) southwest corners.
    """
    tiles = []
    for lat_deg in range(math.floor(min_lat), math.floor(max_lat) + 1):
        step = lon_tile_step(lat_deg)
        lon_start = math.floor(min_lon / step) * step
        lon_end = math.floor(max_lon / step) * step
        for lon_deg in range(lon_start, lon_end + 1, step):
            tiles.append((lat_deg, lon_deg))
    return sorted(tiles)


def bbox_from_center(lat: float, lon: float, half_side_km: float) -> Tuple[float, float, float, float]:
    """Lat/lon bounding box for a square of given half-side around a center point."""
    dlat = half_side_km / KM_PER_DEG_LAT
    dlon = half_side_km / (KM_PER_DEG_LAT * math.cos(math.radians(lat)))
    return lat - dlat, lon - dlon, lat + dlat, lon + dlon


def tile_dir_url(lat_deg: int, lon_deg: int) -> Tuple[str, str]:
    """
    Build the (longitude-bin directory URL, product directory name) for a geocell.

    Server layout (per DLR data guide):
        files/TDM1_EDEM_10_N49/TDM1_EDEM_10_N49E010/TDM1_EDEM__10_N49E010_V01_C/
    i.e. a latitude dir, a 10°-binned longitude dir named after the bin's first
    cell, then the product dir (note the double underscore in the product ID).
    """
    ns = "N" if lat_deg >= 0 else "S"
    lat_dir = f"TDM1_EDEM_10_{ns}{abs(lat_deg):02d}"
    lon_bin = math.floor(lon_deg / 10) * 10
    lon_dir = f"TDM1_EDEM_10_{format_geocell(lat_deg, lon_bin)}"
    prod_dir = f"TDM1_EDEM__10_{format_geocell(lat_deg, lon_deg)}_V01_C"
    return f"{BASE_URL}/{lat_dir}/{lon_dir}", prod_dir


def generate_candidate_urls(tiles: List[Tuple[int, int]]) -> List[str]:
    """
    Generate zip URLs from the documented naming convention (no server access).

    The data guide is ambiguous about the zip file name (product-ID style with
    double underscore vs. plain geocell name), so emit both variants per tile;
    the downloader tries alternatives like the ARSO fetcher.
    """
    urls = []
    for lat_deg, lon_deg in tiles:
        bin_url, prod_dir = tile_dir_url(lat_deg, lon_deg)
        geocell = format_geocell(lat_deg, lon_deg)
        urls.append(f"{bin_url}/{prod_dir}/{prod_dir}.zip")
        urls.append(f"{bin_url}/{prod_dir}/TDM1_EDEM_10_{geocell}.zip")
    return urls


def prepare(path: str) -> Optional[str]:
    """Claim TanDEM-X EDEM zip archives and extract the usable GeoTIFF."""
    if not zipfile.is_zipfile(path):
        return None
    return extract_tandemx_edem_tif(path)


def extract_tandemx_edem_tif(zip_path: str) -> str:
    """Extract the usable DEM GeoTIFF from a TanDEM-X EDEM zip into the cache.

    TanDEM-X EDEM tiles ship as a zip whose payload is a product directory with
    two elevation rasters — ellipsoidal WGS84 heights (``_W84.tif``) and
    geoid/orthometric heights (``_EGM.tif``) — plus auxiliary quality masks
    under ``EDEM_AUXFILES/``. We pick the EGM raster (metres above sea level)
    for a physically meaningful relief, falling back to W84, then any GeoTIFF.

    Returns the path to the extracted GeoTIFF (cached, so re-runs skip the work).
    """
    from downloader import CACHE_DIR

    with zipfile.ZipFile(zip_path) as zf:
        tifs = [n for n in zf.namelist() if n.lower().endswith(".tif")]
        # Prefer the main elevation layers over the auxiliary masks.
        elevation = [n for n in tifs if "auxfiles" not in n.lower()]
        pool = elevation or tifs
        egm = [n for n in pool if n.lower().endswith("_egm.tif")]
        w84 = [n for n in pool if n.lower().endswith("_w84.tif")]
        chosen = egm or w84 or pool
        if not chosen:
            raise RuntimeError(f"No GeoTIFF found inside archive: {zip_path}")
        member = chosen[0]

        out_path = os.path.join(CACHE_DIR, os.path.basename(member))
        if os.path.exists(out_path):
            return out_path

        # Extract to a temp file and rename on success, mirroring download_dem,
        # so an interrupted extraction never leaves a truncated cached tif.
        part_path = out_path + ".part"
        try:
            with zf.open(member) as src, open(part_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            os.replace(part_path, out_path)
        except Exception:
            if os.path.exists(part_path):
                os.remove(part_path)
            raise
    return out_path


def resolve_urls_from_server(tiles: List[Tuple[int, int]], user: str, password: str) -> List[str]:
    """
    Resolve exact zip URLs by listing the server directories (needs credentials).

    Returns one verified URL per available tile; tiles missing on the server
    (e.g. over open ocean) are reported on stderr and skipped.
    """
    import requests

    session = requests.Session()
    session.auth = (user, password)

    listing_cache = {}

    def list_dir(url: str) -> List[str]:
        if url not in listing_cache:
            response = session.get(url + "/", timeout=60)
            response.raise_for_status()
            if "text/html" not in response.headers.get("Content-Type", ""):
                raise RuntimeError(f"Unexpected response from {url} — check credentials")
            # Apache autoindex hrefs: skip parent (starts with /) and sort links (start with ?)
            listing_cache[url] = re.findall(r'href="([^"?/][^"?]*)"', response.text)
        return listing_cache[url]

    urls = []
    for lat_deg, lon_deg in tiles:
        bin_url, expected_prod_dir = tile_dir_url(lat_deg, lon_deg)
        geocell = format_geocell(lat_deg, lon_deg)

        prod_dirs = [entry.rstrip("/") for entry in list_dir(bin_url)
                     if geocell in entry]
        if not prod_dirs:
            print(f"[WARN] Tile {geocell} not found on server, skipping", file=sys.stderr, flush=True)
            continue
        prod_dir = prod_dirs[0]
        if prod_dir != expected_prod_dir:
            print(f"[INFO] Tile {geocell}: product dir is {prod_dir}", file=sys.stderr, flush=True)

        zips = [entry for entry in list_dir(f"{bin_url}/{prod_dir}")
                if entry.lower().endswith(".zip")]
        if not zips:
            print(f"[WARN] No zip found in {prod_dir}, skipping", file=sys.stderr, flush=True)
            continue
        urls.append(f"{bin_url}/{prod_dir}/{zips[0]}")

    return urls


def main():
    parser = argparse.ArgumentParser(
        description="Generate TanDEM-X 30m EDEM tile URLs for a given location and radius."
    )

    coord_group = parser.add_mutually_exclusive_group(required=True)
    coord_group.add_argument(
        "--lat",
        type=float,
        help="Center latitude in WGS84 decimal degrees (use with --lon)."
    )
    coord_group.add_argument(
        "--rect-corners",
        type=str,
        help="Rectangle corners as LAT1,LON1,LAT2,LON2 (e.g., '39.6,44.1,39.8,44.5')."
    )

    parser.add_argument(
        "--lon",
        type=float,
        help="Center longitude in WGS84 decimal degrees (required with --lat)."
    )

    region_group = parser.add_mutually_exclusive_group(required=False)
    region_group.add_argument(
        "--diameter",
        type=float,
        help="Bounding-circle diameter in kilometers (requires --lat/--lon)."
    )
    region_group.add_argument(
        "--bbox",
        type=float,
        help="Square bounding box side length in kilometers (requires --lat/--lon)."
    )

    args = parser.parse_args()

    if args.lat is not None and args.lon is None:
        print("Error: --lat requires --lon", file=sys.stderr)
        sys.exit(1)

    # Credentials come from the environment (source a gitignored env.sh at the
    # repo root), never from the command line — CLI args leak into shell
    # history and process lists.
    load_dotenv(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "env.sh"))
    user = os.environ.get("DLR_EOC_USER") or None
    password = os.environ.get("DLR_EOC_PASSWORD") or None
    if (user is None) != (password is None):
        print("Error: set both DLR_EOC_USER and DLR_EOC_PASSWORD (see env.sh.example), or neither",
              file=sys.stderr)
        sys.exit(1)

    if args.rect_corners is not None:
        try:
            parts = args.rect_corners.split(',')
            if len(parts) != 4:
                raise ValueError("must be LAT1,LON1,LAT2,LON2 format")
            lat1, lon1, lat2, lon2 = (float(p) for p in parts)
            if not all(-90 <= lat <= 90 for lat in [lat1, lat2]):
                raise ValueError("latitude coordinates out of range")
            if not all(-180 <= lon <= 180 for lon in [lon1, lon2]):
                raise ValueError("longitude coordinates out of range")
            if lat1 == lat2 or lon1 == lon2:
                raise ValueError("corners must define a non-zero area rectangle")
        except ValueError as e:
            print(f"Error: Invalid --rect-corners: {e}", file=sys.stderr)
            sys.exit(1)

        min_lat, max_lat = min(lat1, lat2), max(lat1, lat2)
        min_lon, max_lon = min(lon1, lon2), max(lon1, lon2)
        print(f"[INFO] Rectangle: ({min_lat}, {min_lon}) to ({max_lat}, {max_lon})",
              file=sys.stderr, flush=True)
    else:
        if args.diameter is None and args.bbox is None:
            print("Error: --diameter or --bbox required with --lat/--lon", file=sys.stderr)
            sys.exit(1)

        half_side_km = (args.diameter if args.diameter is not None else args.bbox) / 2.0
        min_lat, min_lon, max_lat, max_lon = bbox_from_center(args.lat, args.lon, half_side_km)
        region = (f"{args.diameter}km diameter" if args.diameter is not None
                  else f"{args.bbox}km x {args.bbox}km box")
        print(f"[INFO] Center ({args.lat}, {args.lon}), {region} -> "
              f"bbox ({min_lat:.4f}, {min_lon:.4f}) to ({max_lat:.4f}, {max_lon:.4f})",
              file=sys.stderr, flush=True)

    tiles = get_tiles_in_latlon_bbox(min_lat, min_lon, max_lat, max_lon)
    geocells = [format_geocell(lat, lon) for lat, lon in tiles]
    print(f"[INFO] Found {len(tiles)} geocell(s): {', '.join(geocells)}", file=sys.stderr, flush=True)

    if user is not None:
        print("[INFO] Resolving exact zip URLs via server directory listing...", file=sys.stderr, flush=True)
        urls = resolve_urls_from_server(tiles, user, password)
    else:
        urls = generate_candidate_urls(tiles)
        print("[INFO] Emitting candidate URLs from naming convention (2 variants per tile); "
              "set DLR_EOC_USER/DLR_EOC_PASSWORD to resolve exact URLs", file=sys.stderr, flush=True)

    if not urls:
        print("Error: No tile URLs resolved for specified region", file=sys.stderr, flush=True)
        sys.exit(1)

    for url in urls:
        print(url, flush=True)

    print(f"[INFO] Generated {len(urls)} TanDEM-X 30m EDEM URLs", file=sys.stderr, flush=True)
    print(f"[INFO] Download requires a free DLR EOC account ({REGISTER_URL}); "
          "use e.g.: wget -i urls.txt --auth-no-challenge --user=USER --ask-password",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
