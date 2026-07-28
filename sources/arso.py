#!/usr/bin/env python3
"""
Query ARSO DMR tile block numbers and generate download URLs.

Usage:
    python -m sources.arso 457 459 108 110
    (Downloads tiles from 457_108 to 459_110)

Downloaded tiles are XYZ point clouds (one ``E N Z`` line per metre); the
``prepare`` hook grids them into GeoTIFFs for the raster pipeline.
"""

import sys
import os
from typing import List, Optional, Tuple
from osgeo import gdal, ogr

# ARSO tile index shapefile (sits at the repo root, next to the cache)
TILE_INDEX_DBF_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "LIDAR_FISHNET_D48GK.dbf")

# ARSO DMR download URL pattern
ARSO_URL_PATTERN = "http://gis.arso.gov.si/lidar/dmr1/{block}/D48GK/GK1_{e}_{n}.asc"

def query_tiles(e_min: int, e_max: int, n_min: int, n_max: int) -> List[Tuple[int, int, str]]:
    """
    Query tile block numbers for a rectangular region.

    Args:
        e_min: Minimum easting (tile coordinate)
        e_max: Maximum easting (tile coordinate)
        n_min: Minimum northing (tile coordinate)
        n_max: Maximum northing (tile coordinate)

    Returns:
        List of tuples: (easting, northing, block)
    """

    ogr.UseExceptions()
    ds = ogr.Open(TILE_INDEX_DBF_FILE)
    if not ds:
        raise RuntimeError(f"Failed to open tile index: {TILE_INDEX_DBF_FILE}")

    layer = ds.GetLayer(0)
    tiles = []

    for feature in layer:
        name = feature.GetField('NAME')
        blok = feature.GetField('BLOK')

        if name and blok:
            parts = name.split('_')
            if len(parts) == 2:
                try:
                    e = int(parts[0])
                    n = int(parts[1])
                    if e_min <= e <= e_max and n_min <= n <= n_max:
                        tiles.append((e, n, blok))
                except ValueError:
                    pass

    ds = None

    # Sort by northing then easting
    tiles.sort(key=lambda t: (t[1], t[0]))

    return tiles


def generate_urls(tiles: List[Tuple[int, int, str]]) -> List[str]:
    """
    Generate download URLs for tiles.
    For tiles that exist in multiple blocks, output all possible URLs
    so the downloader can try alternatives if one fails.

    Args:
        tiles: List of (easting, northing, block) tuples

    Returns:
        List of download URLs (may contain multiple URLs for same tile)
    """
    # Group by tile coordinate to handle duplicates
    tile_dict = {}
    for e, n, block in tiles:
        key = (e, n)
        if key not in tile_dict:
            tile_dict[key] = []
        tile_dict[key].append(block)

    urls = []
    for (e, n), blocks in sorted(tile_dict.items()):
        # For tiles in multiple blocks, add all possible URLs
        # Sort blocks to be deterministic
        for block in sorted(set(blocks)):
            url = ARSO_URL_PATTERN.format(block=block, e=e, n=n)
            urls.append(url)

    return urls


def prepare(path: str) -> Optional[str]:
    """Claim ARSO XYZ point clouds and grid them; None for anything else."""
    try:
        with open(path, 'r') as f:
            parts = f.readline().strip().replace(';', ' ').replace(',', ' ').split()
        if len(parts) < 3:
            return None
        float(parts[0]), float(parts[1]), float(parts[2])
    except (FileNotFoundError, OSError):
        raise
    except Exception:
        return None
    print(f"[INFO] Detected XYZ point cloud: {path}, converting to grid...")
    return convert_xyz_to_grid(path)


def convert_xyz_to_grid(xyz_path: str) -> str:
    """Convert XYZ point cloud to gridded GeoTIFF (stored in cache)."""
    gdal.UseExceptions()

    from downloader import CACHE_DIR, ensure_dir
    ensure_dir(CACHE_DIR)

    base_name = os.path.splitext(os.path.basename(xyz_path))[0]
    output_path = os.path.join(CACHE_DIR, f'{base_name}_gridded.tif')

    if os.path.exists(output_path):
        return output_path

    vrt_content = f"""<OGRVRTDataSource>
  <OGRVRTLayer name="{base_name}">
    <SrcDataSource>CSV:{os.path.abspath(xyz_path)}</SrcDataSource>
    <SrcLayer>{base_name}</SrcLayer>
    <GeometryType>wkbPoint25D</GeometryType>
    <GeometryField encoding="PointFromColumns" x="field_1" y="field_2" z="field_3"/>
  </OGRVRTLayer>
</OGRVRTDataSource>"""

    vrt_path = f'/vsimem/{base_name}.vrt'
    gdal.FileFromMemBuffer(vrt_path, vrt_content)

    vrt_ds = ogr.Open(vrt_path)
    if vrt_ds is None:
        gdal.Unlink(vrt_path)
        raise ValueError(f"Failed to parse XYZ file: {xyz_path}")

    data_extent = vrt_ds.GetLayer(0).GetExtent()
    vrt_ds = None

    # Expand to 1km tile boundaries to eliminate gaps between adjacent tiles
    extent = (
        int(data_extent[0] / 1000) * 1000,
        (int(data_extent[1] / 1000) + 1) * 1000,
        int(data_extent[2] / 1000) * 1000,
        (int(data_extent[3] / 1000) + 1) * 1000,
    )

    cellsize = 1.0
    width = int((extent[1] - extent[0]) / cellsize)
    height = int((extent[3] - extent[2]) / cellsize)

    print(f"[INFO] Gridding {width}x{height} at {cellsize}m resolution...")

    gdal.Grid(
        output_path,
        vrt_path,
        algorithm='nearest:radius1=2.0:radius2=2.0:nodata=-9999',
        outputBounds=[extent[0], extent[2], extent[1], extent[3]],
        width=width,
        height=height,
        outputType=gdal.GDT_Float32,
        zfield='field_3'
    )

    gdal.Unlink(vrt_path)
    print(f"[INFO] Created gridded raster: {output_path}")
    return output_path


def main():
    if len(sys.argv) != 5:
        print("Usage: python arso_tile_lookup.py <e_min> <e_max> <n_min> <n_max>", file=sys.stderr)
        print("Example: python arso_tile_lookup.py 457 459 108 110", file=sys.stderr)
        sys.exit(1)

    try:
        e_min = int(sys.argv[1])
        e_max = int(sys.argv[2])
        n_min = int(sys.argv[3])
        n_max = int(sys.argv[4])
    except ValueError:
        print("Error: All arguments must be integers", file=sys.stderr)
        sys.exit(1)

    if e_min > e_max or n_min > n_max:
        print("Error: min values must be <= max values", file=sys.stderr)
        sys.exit(1)

    # Query tiles
    tiles = query_tiles(e_min, e_max, n_min, n_max)

    if not tiles:
        print(f"No tiles found in range {e_min}-{e_max}, {n_min}-{n_max}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Found {len(tiles)} tiles", file=sys.stderr)

    # Check for tiles in multiple blocks
    from collections import Counter
    tile_coords = [(e, n) for e, n, _ in tiles]
    duplicates = [coord for coord, count in Counter(tile_coords).items() if count > 1]

    if duplicates:
        print(f"[INFO] {len(duplicates)} tiles exist in multiple blocks, will try all alternatives", file=sys.stderr)

    # Generate URLs
    urls = generate_urls(tiles)

    # Output only URLs to stdout (can be redirected to a file)
    for url in urls:
        print(url)

    print(f"[INFO] Generated {len(urls)} URLs for {len(set((e, n) for e, n, _ in tiles))} unique tiles", file=sys.stderr)


if __name__ == "__main__":
    main()
