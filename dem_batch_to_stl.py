#!/usr/bin/env python3
"""
Batch convert DEM tiles (GeoTIFF or ASCII Grid) into watertight relief STL models.
Uses: numpy, rasterio, numpy-stl, requests (plus Python stdlib).
"""

import argparse
import os
import sys
from typing import Iterable, List

from downloader import CACHE_DIR, download_dem, ensure_dir, read_url_list
from mesh_builder import build_terrain_meshes, save_stl
from sources import load_dem, prepare_dem_files


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
        help="Read ground within this percent above the DEM minimum as water, printed "
             "as a water insert (0 disables the DEM lake mask).",
    )
    parser.add_argument(
        "--lake-lowering-mm",
        type=float,
        default=0.0,
        help="Sink the water class this many millimeters below the ground it seats in, "
             "flat-topped (0 leaves water flush, drawn at the DEM).",
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
        "--snow-geojson",
        type=str,
        default=None,
        help="Path to a satellite-derived snow polygon (e.g. from Sentinel-2 NDSI). "
             "Its polygons are cleaned for printability and added to the glacier "
             "terrain class (unioned with any OSM glacier).",
    )
    parser.add_argument(
        "--snow-iterations",
        type=int,
        default=220,
        help="Area-preserving curve-shortening (APCSF) iterations for snow cleanup: "
             "higher = more finger retraction / blobbier, thicker inserts (area is "
             "preserved throughout). Default 220 (sub-1 mm slivers < 1%%, 2 pieces, at "
             "RESAMPLE_M=15; tied to RESAMPLE_M, re-sweep if you change it).",
    )
    parser.add_argument(
        "--snow-min-feature-m2",
        type=float,
        default=None,
        help="Despeckle/hole-fill threshold (m^2): drop snow polygons and holes "
             "smaller than this. Default: derived from the true print scale as a "
             "2x2 mm square (diameter/x-size), not a fixed value.",
    )
    parser.add_argument(
        "--snow-dt",
        type=float,
        default=4.0,
        help="APCSF curve-shortening step size (m) per iteration. Default: 4.",
    )
    parser.add_argument(
        "--foliage-geojson",
        type=str,
        default=None,
        help="Path to a satellite-derived foliage/vegetation polygon (e.g. from "
             "Sentinel-2 NDVI). Required by --invert-base, where foliage becomes "
             "the base plate. Its polygons are cleaned for printability (APCSF).",
    )
    parser.add_argument(
        "--foliage-iterations",
        type=int,
        default=100,
        help="APCSF iterations for foliage cleanup (see --snow-iterations). "
             "Default 100 (sub-1 mm base-plate slivers < 1%% at RESAMPLE_M=15).",
    )
    parser.add_argument(
        "--invert-base",
        action="store_true",
        help="Invert the terrain print: use FOLIAGE as the base plate and seat "
             "rock + snow as inserts (rock = cutout - snow - foliage). For scenes "
             "where foliage dominates and rock would otherwise be a fragile web of "
             "river slivers (e.g. Ararat). Requires --snow-geojson, "
             "--foliage-geojson and a circular cutout (--diameter). Builds terrain "
             "from the satellite layers only (no OSM classification).",
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
    parser.add_argument(
        "--insert-corner-relief-mm",
        type=float,
        default=0.0,
        help="Extra clearance at sharp corners, on top of --insert-xy-clearance-mm, "
             "to defeat FDM inside-corner over-extrusion that otherwise locks the "
             "fit. Enlarges the rock pocket at convex corners and cuts the insert "
             "back at reflex corners. 0 = no corner relief (default). ~0.25 mm suits "
             "a 0.4 mm nozzle.",
    )
    parser.add_argument(
        "--insert-corner-min-angle-deg",
        type=float,
        default=45.0,
        help="Minimum boundary turn angle for a corner to get relief (default 45). "
             "Near-straight vertices are skipped so the flat clearance is preserved.",
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
        downloaded = prepare_dem_files(downloaded)
        product = load_dem(downloaded, args.downsample)
        dem = product.array
        px_size_x, px_size_y = product.px_size_x, product.px_size_y
        ref_crs, ref_transform = product.crs, product.transform
        print(
            f"[INFO] Merge complete. DEM shape: {dem.shape[0]} x {dem.shape[1]} "
            f"(downsample={args.downsample}), pixel size (m): {px_size_x:.3f} x {px_size_y:.3f}"
        )
        # Crop the raster to the cutout region so --x-size-mm is the cutout's output
        # size (not the whole tile) and only the cutout neighbourhood is meshed. The
        # final shape is still trimmed to the exact cutout by the 2D stage.
        if args.center or args.rect_corners:
            from dem_processing import crop_to_cutout
            _crop_radius_m = (args.diameter / 2.0 * 1000.0) if args.diameter else None
            dem, ref_transform = crop_to_cutout(
                dem, ref_transform, ref_crs,
                center_lat=center_lat, center_lon=center_lon,
                radius_m=_crop_radius_m, side_length_km=args.side_length,
                rect_lat1=rect_lat1, rect_lon1=rect_lon1,
                rect_lat2=rect_lat2, rect_lon2=rect_lon2,
            )
            print(f"[INFO] Cropped to cutout region: DEM shape {dem.shape[0]} x "
                  f"{dem.shape[1]}", flush=True)
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
        from terrain_layout import CutoutSpec, rect_extent_m

        cutout_type_for_mesh = None
        cutout_radius_m = None
        if args.rect_corners:
            cutout_type_for_mesh = "rectangular"
        elif args.center:
            cutout_type_for_mesh = "circular" if args.diameter else "rectangular"
            if args.diameter:
                cutout_radius_m = (args.diameter / 2.0) * 1000.0  # Convert km to m
        cutout = CutoutSpec(
            cutout_type=cutout_type_for_mesh,
            center_lat=center_lat,
            center_lon=center_lon,
            radius_m=cutout_radius_m,
            side_length_km=args.side_length,
            n_gon_sides=args.ngon_sides,
            bearing=args.bearing,
            rect_corner1_lat=rect_lat1,
            rect_corner1_lon=rect_lon1,
            rect_corner2_lat=rect_lat2,
            rect_corner2_lon=rect_lon2,
        )

        # Pin the model scale to the cutout so the printed cutout is exactly
        # --x-size-mm, independent of how crop_to_cutout rounded to whole pixels (and,
        # for a rotated rectangle, of how much wider its axis-aligned crop box is).
        # The mesh derives its scale from the raster width, so scale x_size up by
        # raster_width / cutout_width: printed diameter (circular) or printed AB edge
        # (rectangular) == x_size_mm. Without this the model is built at one scale and
        # would have to be rescaled after meshing, which is what used to happen for
        # rectangles -- and it rescaled xy only, leaving true-scale relief understated.
        x_size_model = args.x_size_mm
        terrain_w_m = (dem.shape[1] - 1) * px_size_x
        cutout_w_m = None
        if args.diameter:
            cutout_w_m = args.diameter * 1000.0
        elif cutout_type_for_mesh == "rectangular":
            cutout_w_m, _ = rect_extent_m(ref_crs, cutout)
        if cutout_w_m:
            x_size_model = args.x_size_mm * terrain_w_m / cutout_w_m

        # Terrain classification / composition. Every print goes through this
        # stage: with no mask provider at all it resolves to a single base layer
        # covering the whole cutout, which is the plain relief block.
        from masks import (TERRAIN_FOLIAGE, TERRAIN_NAMES, TERRAIN_ROCK,
                           merge_masks)
        from model_frame import ModelFrame
        from terrain_layout import frame_with_print_motion

        frame = ModelFrame.from_dem(dem.shape, px_size_x, px_size_y,
                                    x_size_model, ref_transform, ref_crs)
        # The grid -> print motion, so the 2D stage emits the coordinates the STL
        # will carry and its float32 snap is the last thing to touch them.
        frame = frame_with_print_motion(frame, cutout)

        terrain_type_list = args.terrain_types.split(",") if args.terrain_types else None

        # Masks come from whichever data sources are given, independent of the
        # bottom-layer mapping. Providers clean in CRS metres, applying mm
        # feature-size rules at the frame's true print scale.
        providers = []
        if args.terrain:
            from masks.osm import OsmMasks
            providers.append(OsmMasks())
        if args.lake_range_percent > 0:
            # DEM-thresholded lakes are just another water source; where OSM water
            # is queried too, the two merge into one class.
            from masks.lake import LakeMasks
            providers.append(LakeMasks(dem, args.lake_range_percent))
        if args.snow_geojson:
            from masks.sentinel2 import SnowMasks
            providers.append(SnowMasks(
                args.snow_geojson, iterations=args.snow_iterations,
                dt=args.snow_dt, min_feature_m2=args.snow_min_feature_m2))
        if args.foliage_geojson:
            from masks.sentinel2 import FoliageMasks
            providers.append(FoliageMasks(
                args.foliage_geojson, iterations=args.foliage_iterations))
        class_geometries = merge_masks(frame, providers)

        # The only thing --invert-base changes: which terrain type is the bottom.
        base_class = TERRAIN_FOLIAGE if args.invert_base else TERRAIN_ROCK
        if args.invert_base and not class_geometries.get(TERRAIN_FOLIAGE):
            raise SystemExit("--invert-base makes foliage the base plate; supply "
                             "a foliage mask with --foliage-geojson.")

        # Build meshes
        input_stub = os.path.splitext(os.path.basename(args.url_list))[0]
        first_tile_stub = os.path.splitext(os.path.basename(downloaded[0]))[0]
        base_name = f"{input_stub}_{first_tile_stub}"
        if len(downloaded) > 1:
            base_name = f"{base_name}_mosaic"

        from terrain_layout import (InsertFit, build_terrain_layout,
                                    cutout_footprint)

        # Stage 1: masks -> final 2D polygons (no elevations involved).
        layout = build_terrain_layout(
            frame, class_geometries,
            outline=cutout_footprint(frame, cutout),
            base_class=base_class,
            terrain_types=terrain_type_list,
            fit=InsertFit(
                xy_clearance_mm=args.insert_xy_clearance_mm,
                z_clearance_mm=args.insert_z_clearance_mm,
                corner_relief_mm=args.insert_corner_relief_mm,
                corner_min_angle_deg=args.insert_corner_min_angle_deg,
            ),
        )

        # Stage 2: extrude that layout over the DEM.
        terrain_meshes = build_terrain_meshes(
            layout, frame, dem,
            max_height_mm, z_exaggeration,
            args.base_thickness_mm, args.terrain_thickness_mm,
            use_true_scale=use_true_scale,
            recess_mode=args.terrain_recess_mode,
            insert_z_clearance_mm=args.insert_z_clearance_mm,
            water_lowering_mm=args.lake_lowering_mm,
        )

        # A print with no insert is one body, and it keeps the plain output name:
        # calling it "_rock" would name a terrain class the layout never resolved.
        base_name_out = TERRAIN_NAMES[base_class]
        built = {name: data for name, data in terrain_meshes.items()
                 if data is not None}
        if len(built) == 1 and base_name_out in built:
            paths = {base_name_out: os.path.join(args.output_dir, f"{base_name}.stl")}
        else:
            paths = {name: os.path.join(args.output_dir, f"{base_name}_{name}.stl")
                     for name in built}
        for name, data in built.items():
            verts, fcs, _mz = data
            print(f"[INFO] Saving {name} STL ({fcs.shape[0]} faces) to "
                  f"{paths[name]}...", flush=True)
            save_stl(verts, fcs, paths[name])

        rows, cols = dem.shape
        # Report the actual printed bounding box (the base plate = the cutout,
        # trimmed in 2D), not the raster grid extent.
        bv, _bf, max_z = built[base_name_out]
        model_x = float(bv[:, 0].max() - bv[:, 0].min())
        model_y = float(bv[:, 1].max() - bv[:, 1].min())
        print(
            f"[OK] Merged {len(downloaded)} DEM(s): {rows} x {cols} samples -> "
            f"model {model_x:.2f} mm x {model_y:.2f} mm x {max_z:.2f} mm\n"
            f"     -> {', '.join(paths.values())}\n"
            f"Cached DEM files at: {os.path.abspath(CACHE_DIR)}"
        )

    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Processing failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
