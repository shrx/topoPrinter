#!/usr/bin/env python3
"""
Batch convert swissALTI3D GeoTIFF DEMs into watertight relief STL models.
Uses: numpy, rasterio, numpy-stl, requests (plus Python stdlib).
"""

import argparse
import os
import sys
from typing import Iterable, List

from dem_processing import load_and_merge
from downloader import CACHE_DIR, download_dem, ensure_dir, read_url_list
from mesh_builder import dem_to_vertices_and_faces, save_stl


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert swissALTI3D GeoTIFF DEMs into watertight STL relief models."
    )
    parser.add_argument("--url-list", required=True, help="Path to text file with one GeoTIFF URL per line.")
    parser.add_argument("--output-dir", required=True, help="Directory to write STL files into.")
    parser.add_argument("--x-size-mm", type=float, default=200.0, help="Model size in X (mm).")
    parser.add_argument("--max-height-mm", type=float, default=30.0, help="Total model height including base (mm).")
    parser.add_argument("--z-exaggeration", type=float, default=1.0, help="Vertical exaggeration factor.")
    parser.add_argument("--downsample", type=int, default=1, help="Downsample factor to reduce mesh density.")
    parser.add_argument("--base-thickness-mm", type=float, default=2.0, help="Thickness of flat base (mm).")
    parser.add_argument(
        "--lake-range-percent",
        type=float,
        default=0.0,
        help="Treat cells within this percent above min elevation as lakes (0 disables lake lowering).",
    )
    parser.add_argument(
        "--lake-lowering-mm",
        type=float,
        default=0.0,
        help="Lower identified lake cells by this many millimeters (0 disables lake lowering).",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    urls = read_url_list(args.url_list)
    if not urls:
        print("No URLs found in url list.", file=sys.stderr)
        return 1

    print(f"[INFO] Found {len(urls)} URL(s) in list.")
    ensure_dir(args.output_dir)
    download_dir = os.path.join(args.output_dir, "tmp_dem")
    ensure_dir(download_dir)
    ensure_dir(CACHE_DIR)
    cache_dir_abs = os.path.abspath(CACHE_DIR)

    downloaded: List[str] = []
    for idx, url in enumerate(urls):
        print(f"[INFO] Downloading ({idx + 1}/{len(urls)}): {url}", flush=True)
        try:
            tif_path = download_dem(url, download_dir, idx + 1)
            downloaded.append(tif_path)
            loc = "cache" if os.path.abspath(tif_path).startswith(cache_dir_abs) else "download"
            print(f"[INFO]   -> {loc}: {tif_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] {exc}", file=sys.stderr)

    if not downloaded:
        print("No GeoTIFFs were downloaded successfully; nothing to process.", file=sys.stderr)
        return 1

    print(f"[INFO] Merging {len(downloaded)} DEM(s)...", flush=True)
    try:
        dem, px_size_x, px_size_y = load_and_merge(downloaded, args.downsample)
        print(
            f"[INFO] Merge complete. DEM shape: {dem.shape[0]} x {dem.shape[1]} "
            f"(downsample={args.downsample}), pixel size (m): {px_size_x:.3f} x {px_size_y:.3f}"
        )
        print("[INFO] Building mesh...", flush=True)
        vertices, faces, max_z, water_faces = dem_to_vertices_and_faces(
            dem,
            px_size_x,
            px_size_y,
            args.x_size_mm,
            args.max_height_mm,
            args.z_exaggeration,
            args.base_thickness_mm,
            args.lake_range_percent,
            args.lake_lowering_mm,
        )
        water_info = f", water faces: {water_faces.shape[0]}" if water_faces is not None else ""
        print(f"[INFO] Mesh built: {faces.shape[0]} faces, {vertices.shape[0]} vertices{water_info}.")
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Processing failed: {exc}", file=sys.stderr)
        return 1

    input_stub = os.path.splitext(os.path.basename(args.url_list))[0]
    first_tile_stub = os.path.splitext(os.path.basename(downloaded[0]))[0]
    base_name = f"{input_stub}_{first_tile_stub}"
    if len(downloaded) > 1:
        base_name = f"{base_name}_mosaic"
    stl_path = os.path.join(args.output_dir, f"{base_name}.stl")
    print(f"[INFO] Saving STL to {stl_path}...", flush=True)
    try:
        save_stl(vertices, faces, stl_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Failed to save STL: {exc}", file=sys.stderr)
        return 1

    if water_faces is not None and len(water_faces) > 0 and args.lake_range_percent > 0 and args.lake_lowering_mm > 0:
        water_path = os.path.join(args.output_dir, f"{base_name}_water.stl")
        print(f"[INFO] Saving water STL to {water_path}...", flush=True)
        try:
            save_stl(vertices, water_faces, water_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] Failed to save water STL: {exc}", file=sys.stderr)
            return 1

    rows, cols = dem.shape
    model_y = vertices[:, 1].max()
    print(
        f"[OK] Merged {len(downloaded)} DEM(s): {rows} x {cols} samples -> "
        f"model {args.x_size_mm:.2f} mm x {model_y:.2f} mm x {max_z:.2f} mm\n"
        f"     -> {stl_path}"
    )
    print(f"Temporary GeoTIFFs kept at {download_dir}.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
