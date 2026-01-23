#!/usr/bin/env python3
"""
Batch convert DEM tiles (GeoTIFF or ASCII Grid) into watertight relief STL models.
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
        description="Convert DEM tiles (GeoTIFF or ASC) into watertight STL relief models."
    )
    parser.add_argument(
        "--url-list",
        required=True,
        help="Path to file with DEM URLs (supports .txt, .csv, .xlsx).",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to write STL files into.")
    parser.add_argument("--x-size-mm", type=float, default=200.0, help="Model size in X (mm).")

    # Mutually exclusive scaling modes
    scale_group = parser.add_mutually_exclusive_group()
    scale_group.add_argument(
        "--max-height-mm",
        type=float,
        default=None,
        help="Total model height including base (mm). Uses normalized scale (fits elevation range into this height).",
    )
    scale_group.add_argument(
        "--z-exaggeration",
        type=float,
        default=None,
        help="Vertical exaggeration multiplier for true 1:1 scale. Default (no scale args) is true 1:1 scale with no exaggeration.",
    )

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
    ensure_dir(CACHE_DIR)

    downloaded: List[str] = []
    for idx, url in enumerate(urls):
        print(f"[INFO] Downloading ({idx + 1}/{len(urls)}): {url}", flush=True)
        try:
            dem_path = download_dem(url, idx + 1)
            downloaded.append(dem_path)
            print(f"[INFO]   -> cache: {dem_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] {exc}", file=sys.stderr)

    if not downloaded:
        print("No DEM files were downloaded successfully; nothing to process.", file=sys.stderr)
        return 1

    print(f"[INFO] Merging {len(downloaded)} DEM(s)...", flush=True)
    try:
        dem, px_size_x, px_size_y = load_and_merge(downloaded, args.downsample)
        print(
            f"[INFO] Merge complete. DEM shape: {dem.shape[0]} x {dem.shape[1]} "
            f"(downsample={args.downsample}), pixel size (m): {px_size_x:.3f} x {px_size_y:.3f}"
        )
        print("[INFO] Building mesh...", flush=True)

        if args.max_height_mm is not None:
            use_true_scale = False
            max_height_mm = args.max_height_mm
            z_exaggeration = 1.0
        else:
            use_true_scale = True
            max_height_mm = 30.0
            z_exaggeration = args.z_exaggeration if args.z_exaggeration is not None else 1.0

        vertices, faces, max_z, water_faces = dem_to_vertices_and_faces(
            dem,
            px_size_x,
            px_size_y,
            args.x_size_mm,
            max_height_mm,
            z_exaggeration,
            args.base_thickness_mm,
            args.lake_range_percent,
            args.lake_lowering_mm,
            use_true_scale=use_true_scale,
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
        f"     -> {stl_path}\n"
        f"Cached DEM files at: {os.path.abspath(CACHE_DIR)}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
