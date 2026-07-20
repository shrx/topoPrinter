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
import numpy as np
from mesh_builder import build_all_terrain_meshes, dem_to_vertices_and_faces, save_stl


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
    # Cutout region specification - mutually exclusive modes
    region_group = parser.add_mutually_exclusive_group()

    # Center-based cutout (existing)
    region_group.add_argument(
        "--center",
        type=str,
        default=None,
        help="Center point for cutout as LAT,LON (e.g., '46.9876,8.6543'). Use with --diameter or --side-length.",
    )

    # Rectangle corners (new)
    region_group.add_argument(
        "--rect-corners",
        type=str,
        default=None,
        help="Rectangle cutout specified by two opposite corners as LAT1,LON1,LAT2,LON2 (e.g., '46.5,8.5,47.0,9.0').",
    )

    # Size specification for center-based cutouts
    cutout_group = parser.add_mutually_exclusive_group()
    cutout_group.add_argument(
        "--diameter",
        type=float,
        default=None,
        help="Diameter in kilometers for circular cutout (requires --center).",
    )
    cutout_group.add_argument(
        "--side-length",
        type=float,
        default=None,
        help="Side length in kilometers for square cutout (requires --center).",
    )
    parser.add_argument(
        "--ngon-sides",
        type=int,
        default=64,
        help="Number of sides for circular cutout perimeter (default: 64, higher = smoother).",
    )
    parser.add_argument(
        "--bearing",
        type=float,
        default=0.0,
        help="Bearing in degrees (0-360) for cutout rotation. 0/360=North, 90=East, 180=South, 270=West. Default: 0 (North).",
    )

    # Terrain classification
    parser.add_argument(
        "--terrain",
        action="store_true",
        default=False,
        help="Enable terrain classification from OSM data (outputs separate STL per terrain type).",
    )
    parser.add_argument(
        "--terrain-thickness-mm",
        type=float,
        default=2.0,
        help="Thickness of terrain overlay shells in mm (default: 2.0).",
    )
    parser.add_argument(
        "--terrain-types",
        type=str,
        default=None,
        help="Comma-separated list of terrain types to include (default: all). "
             "Valid types: glacier, water, foliage.",
    )
    parser.add_argument(
        "--terrain-recess-mode",
        choices=["flat", "uniform"],
        default="flat",
        help="Recess algorithm: 'flat' (flat bottom at min elevation) or "
             "'uniform' (terrain-following uniform thickness). Default: flat.",
    )
    parser.add_argument(
        "--insert-xy-clearance-mm",
        type=float,
        default=0.0,
        help="Per-side horizontal gap between inserts and their rock pockets, "
             "for printing inserts separately on a single-nozzle printer. The "
             "insert walls are inset by this amount; the pocket stays full size. "
             "0 = touching fit for one-piece multimaterial printing (default). "
             "~0.1 mm gives a friction fit on a 0.4 mm nozzle.",
    )
    parser.add_argument(
        "--insert-z-clearance-mm",
        type=float,
        default=0.0,
        help="Vertical relief at the hidden pocket floor, so a separately-printed "
             "insert can seat fully flush on its walls instead of bottoming out. "
             "Only the pocket is deepened; the insert keeps its full height so its "
             "top stays flush. 0 = no relief (default). ~0.2 mm suits a 0.4 mm nozzle.",
    )

    return parser.parse_args(argv)


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)

    # Validate cutout arguments
    center_lat, center_lon = None, None
    rect_lat1, rect_lon1, rect_lat2, rect_lon2 = None, None, None, None

    # Handle rectangle corners
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
            print(f"[ERROR] Invalid --rect-corners: {e}", file=sys.stderr)
            return 1

    # Handle center-based cutouts
    elif args.center is not None:
        if args.diameter is None and args.side_length is None:
            print("[ERROR] --center requires either --diameter or --side-length.", file=sys.stderr)
            return 1
        try:
            parts = args.center.split(',')
            if len(parts) != 2:
                raise ValueError("must be LAT,LON format")
            center_lat, center_lon = float(parts[0]), float(parts[1])
            if not (-90 <= center_lat <= 90) or not (-180 <= center_lon <= 180):
                raise ValueError("coordinates out of range")
        except ValueError as e:
            print(f"[ERROR] Invalid --center: {e}", file=sys.stderr)
            return 1

    # Validate that diameter/side-length require center
    elif args.diameter is not None or args.side_length is not None:
        print("[ERROR] --diameter or --side-length require --center.", file=sys.stderr)
        return 1

    # Validate bearing
    if not (0.0 <= args.bearing <= 360.0):
        print("[ERROR] --bearing must be between 0 and 360 degrees.", file=sys.stderr)
        return 1

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
        dem, px_size_x, px_size_y, ref_crs, ref_transform = load_and_merge(
            downloaded,
            args.downsample,
        )
        print(
            f"[INFO] Merge complete. DEM shape: {dem.shape[0]} x {dem.shape[1]} "
            f"(downsample={args.downsample}), pixel size (m): {px_size_x:.3f} x {px_size_y:.3f}"
        )
        if args.rect_corners:
            bearing_info = f", bearing={args.bearing}°" if args.bearing != 0.0 else ""
            print(f"[INFO] Applied rectangular cutout with corners ({rect_lat1}, {rect_lon1}) to ({rect_lat2}, {rect_lon2}){bearing_info}")
        elif args.center:
            cutout_type = "circular" if args.diameter else "rectangular"
            cutout_size = f"{args.diameter}km diameter" if args.diameter else f"{args.side_length}km side"
            bearing_info = f", bearing={args.bearing}°" if args.bearing != 0.0 else ""
            print(f"[INFO] Applied {cutout_type} cutout at ({center_lat}, {center_lon}), {cutout_size}{bearing_info}")
        print("[INFO] Building mesh...", flush=True)

        if args.max_height_mm is not None:
            use_true_scale = False
            max_height_mm = args.max_height_mm
            z_exaggeration = 1.0
        else:
            use_true_scale = True
            max_height_mm = 30.0
            z_exaggeration = args.z_exaggeration if args.z_exaggeration is not None else 1.0

        # Prepare cutout parameters for mesh builder
        cutout_type_for_mesh = None
        cutout_radius_m = None
        if args.rect_corners:
            cutout_type_for_mesh = "rectangular"
        elif args.center:
            cutout_type_for_mesh = "circular" if args.diameter else "rectangular"
            if args.diameter:
                cutout_radius_m = (args.diameter / 2.0) * 1000.0  # Convert km to m

        # Terrain classification
        class_geometries = None
        if args.terrain:
            from terrain_classifier import classify_terrain, TERRAIN_NAMES
            if args.lake_range_percent > 0 and args.lake_lowering_mm > 0:
                print("[WARN] --terrain mode ignores --lake-range-percent and --lake-lowering-mm "
                      "(OSM water classification is used instead).", flush=True)
            print("[INFO] Classifying terrain from OSM data...", flush=True)
            terrain_type_list = args.terrain_types.split(",") if args.terrain_types else None
            class_geometries = classify_terrain(dem.shape, ref_transform, ref_crs)
            import shapely
            from shapely.geometry import box as shapely_box, shape as shapely_shape
            # Clip to the DEM extent before unioning: OSM polygons can extend far
            # beyond the model, which would both slow the union down and inflate
            # the reported percentages.
            bx0, by0 = ref_transform * (0, 0)
            bx1, by1 = ref_transform * (dem.shape[1], dem.shape[0])
            dem_bbox = shapely_box(min(bx0, bx1), min(by0, by1), max(bx0, bx1), max(by0, by1))
            class_areas = {}
            for cls_val, cls_name in sorted(TERRAIN_NAMES.items()):
                geoms = class_geometries.get(cls_val, [])
                if geoms:
                    polys = shapely.make_valid(
                        np.array([shapely_shape(g) for g in geoms], dtype=object))
                    clipped = shapely.intersection(polys, dem_bbox)
                    class_areas[cls_name] = float(shapely.union_all(clipped).area)
                else:
                    class_areas[cls_name] = 0.0
            total_area = sum(class_areas.values())
            if total_area > 0:
                rock_area = dem.shape[0] * abs(ref_transform.e) * dem.shape[1] * abs(ref_transform.a) - total_area
                class_areas["rock"] = max(rock_area, 0.0)
                total_with_rock = sum(class_areas.values())
                for name in ["rock", "glacier", "water", "foliage"]:
                    area = class_areas.get(name, 0.0)
                    if area > 0:
                        pct = 100.0 * area / total_with_rock
                        print(f"[INFO]   {name}: {pct:.1f}%")

        # Build meshes
        input_stub = os.path.splitext(os.path.basename(args.url_list))[0]
        first_tile_stub = os.path.splitext(os.path.basename(downloaded[0]))[0]
        base_name = f"{input_stub}_{first_tile_stub}"
        if len(downloaded) > 1:
            base_name = f"{base_name}_mosaic"

        if args.terrain and class_geometries is not None:
            terrain_meshes = build_all_terrain_meshes(
                dem, class_geometries,
                px_size_x, px_size_y,
                args.x_size_mm, max_height_mm, z_exaggeration,
                args.base_thickness_mm, args.terrain_thickness_mm,
                use_true_scale=use_true_scale,
                cutout_type=cutout_type_for_mesh,
                cutout_center_lat=center_lat,
                cutout_center_lon=center_lon,
                cutout_radius_m=cutout_radius_m,
                cutout_side_length_km=args.side_length,
                ref_transform=ref_transform,
                ref_crs=ref_crs,
                n_gon_sides=args.ngon_sides,
                bearing=args.bearing,
                rect_corner1_lat=rect_lat1,
                rect_corner1_lon=rect_lon1,
                rect_corner2_lat=rect_lat2,
                rect_corner2_lon=rect_lon2,
                recess_mode=args.terrain_recess_mode,
                terrain_types=terrain_type_list,
                insert_xy_clearance_mm=args.insert_xy_clearance_mm,
                insert_z_clearance_mm=args.insert_z_clearance_mm,
            )
            for terrain_name, mesh_data in terrain_meshes.items():
                if mesh_data is None:
                    continue
                verts, fcs, mz = mesh_data
                stl_path = os.path.join(args.output_dir, f"{base_name}_{terrain_name}.stl")
                print(f"[INFO] Saving {terrain_name} STL ({fcs.shape[0]} faces) to {stl_path}...", flush=True)
                save_stl(verts, fcs, stl_path)

            rows, cols = dem.shape
            rock_data = terrain_meshes["rock"]
            model_y = rock_data[0][:, 1].max()
            max_z = rock_data[2]
            print(
                f"[OK] Merged {len(downloaded)} DEM(s): {rows} x {cols} samples -> "
                f"model {args.x_size_mm:.2f} mm x {model_y:.2f} mm x {max_z:.2f} mm\n"
                f"     Terrain STLs saved to {args.output_dir}\n"
                f"Cached DEM files at: {os.path.abspath(CACHE_DIR)}"
            )
        else:
            if (args.lake_range_percent > 0 and args.lake_lowering_mm > 0
                    and cutout_type_for_mesh is not None):
                print("[WARN] Lake lowering is applied to the surface, but a separate "
                      "water STL is not produced when a cutout is used.", flush=True)
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
                cutout_type=cutout_type_for_mesh,
                cutout_center_lat=center_lat,
                cutout_center_lon=center_lon,
                cutout_radius_m=cutout_radius_m,
                cutout_side_length_km=args.side_length,
                ref_transform=ref_transform,
                ref_crs=ref_crs,
                n_gon_sides=args.ngon_sides,
                bearing=args.bearing,
                rect_corner1_lat=rect_lat1,
                rect_corner1_lon=rect_lon1,
                rect_corner2_lat=rect_lat2,
                rect_corner2_lon=rect_lon2,
            )
            water_info = f", water faces: {water_faces.shape[0]}" if water_faces is not None else ""
            print(f"[INFO] Mesh built: {faces.shape[0]} faces, {vertices.shape[0]} vertices{water_info}.")

            stl_path = os.path.join(args.output_dir, f"{base_name}.stl")
            print(f"[INFO] Saving STL to {stl_path}...", flush=True)
            save_stl(vertices, faces, stl_path)

            if water_faces is not None and len(water_faces) > 0 and args.lake_range_percent > 0 and args.lake_lowering_mm > 0:
                water_path = os.path.join(args.output_dir, f"{base_name}_water.stl")
                print(f"[INFO] Saving water STL to {water_path}...", flush=True)
                save_stl(vertices, water_faces, water_path)

            rows, cols = dem.shape
            model_y = vertices[:, 1].max()
            print(
                f"[OK] Merged {len(downloaded)} DEM(s): {rows} x {cols} samples -> "
                f"model {args.x_size_mm:.2f} mm x {model_y:.2f} mm x {max_z:.2f} mm\n"
                f"     -> {stl_path}\n"
                f"Cached DEM files at: {os.path.abspath(CACHE_DIR)}"
            )

    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Processing failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
