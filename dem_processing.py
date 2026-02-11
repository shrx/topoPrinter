"""
DEM loading, nodata handling, and merging.
"""

import os
import sys
from typing import Iterable, Tuple, List

import numpy as np
import rasterio
from rasterio.merge import merge
from osgeo import gdal, ogr
from pyproj import Transformer

from bearing_utils import rotate_to_bearing_frame


def preprocess_dem_files(paths: Iterable[str]) -> List[str]:
    """Preprocess DEM files, converting XYZ point clouds to gridded rasters if needed."""
    processed_paths = []

    for path in paths:
        # Try to open as raster
        try:
            with rasterio.open(path):
                processed_paths.append(path)
                continue
        except:
            pass

        # Check if it's XYZ point cloud (3 numeric values per line)
        try:
            with open(path, 'r') as f:
                parts = f.readline().strip().replace(';', ' ').replace(',', ' ').split()
                if len(parts) >= 3:
                    float(parts[0]), float(parts[1]), float(parts[2])
                    print(f"[INFO] Detected XYZ point cloud: {path}, converting to grid...")
                    processed_paths.append(_convert_xyz_to_grid(path))
                    continue
        except (FileNotFoundError, OSError):
            raise
        except:
            pass

        raise ValueError(f"Unable to read file as raster or XYZ point cloud: {path}")

    return processed_paths


def _convert_xyz_to_grid(xyz_path: str) -> str:
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


def _gather_metadata(paths: Iterable[str]) -> Tuple[float, float, float, object]:
    """Collect pixel sizes, nodata, and CRS consistency checks for merge."""
    paths = list(paths)
    if not paths:
        raise ValueError("No DEM datasets provided for merge.")
    ref_crs = None
    ref_px_x = None
    ref_px_y = None
    nodata_value = None
    for p in paths:
        with rasterio.open(p) as ds:
            if ref_crs is None:
                ref_crs = ds.crs
                ref_px_x = abs(ds.transform.a)
                ref_px_y = abs(ds.transform.e)
                nodata_value = ds.nodata
            else:
                if ds.crs != ref_crs:
                    raise ValueError("All DEMs must share the same CRS for merging.")
                if not np.isclose(abs(ds.transform.a), ref_px_x) or not np.isclose(abs(ds.transform.e), ref_px_y):
                    raise ValueError("All DEMs must have matching pixel sizes for merging.")
    return ref_px_x, ref_px_y, nodata_value, ref_crs


def load_and_merge(
    paths: Iterable[str],
    downsample: int,
) -> Tuple[np.ndarray, float, float, object, object]:
    """Merge DEM tiles, fill nodata, and optionally downsample the grid."""
    if downsample < 1:
        raise ValueError("downsample must be >= 1")

    path_list = list(paths)

    # Preprocess files (convert XYZ to grid if needed)
    path_list = preprocess_dem_files(path_list)

    px_size_x, px_size_y, nodata_value, ref_crs = _gather_metadata(path_list)
    merged, ref_transform = merge(
        path_list,
        nodata=nodata_value,
        method="first",
    )
    arr = merged[0]

    # Convert nodata values to NaN so missing areas appear as holes
    if nodata_value is not None:
        arr = np.where(arr == nodata_value, np.nan, arr)

    # Cutout cropping is handled by boolean intersection in mesh_builder

    if downsample > 1:
        arr = arr[::downsample, ::downsample]
        px_size_x *= downsample
        px_size_y *= downsample
        # Update transform to reflect downsampling
        ref_transform = ref_transform * ref_transform.scale(downsample, downsample)

    if arr.size == 0 or arr.shape[0] < 2 or arr.shape[1] < 2:
        raise ValueError("DEM too small after downsampling to form a mesh.")

    return arr, px_size_x, px_size_y, ref_crs, ref_transform


def apply_cutout_mask(
    arr: np.ndarray,
    transform: object,
    crs: object,
    center_lat: float,
    center_lon: float,
    radius_km: float = None,
    side_length_km: float = None,
    px_size_x: float = None,
    px_size_y: float = None,
    nodata_value: float = np.nan,
    rect_lat1: float = None,
    rect_lon1: float = None,
    rect_lat2: float = None,
    rect_lon2: float = None,
    bearing: float = 0.0,
) -> np.ndarray:
    """
    Apply circular or rectangular cutout mask to DEM array with optional rotation.

    For circular cutouts, includes a buffer to keep pixels partially within the circle.
    This buffer allows later interpolation to n-gon perimeter vertices.

    Args:
        arr: DEM array (rows x cols)
        transform: Affine transform from rasterio
        crs: CRS of the DEM
        center_lat: Center latitude (EPSG:4326), or None for rectangle corners mode
        center_lon: Center longitude (EPSG:4326), or None for rectangle corners mode
        radius_km: Radius for circular cutout (km), or None
        side_length_km: Side length for square cutout (km), or None
        px_size_x: Pixel size in X direction (meters), required for circular cutouts
        px_size_y: Pixel size in Y direction (meters), required for circular cutouts
        nodata_value: Value to set for masked areas
        rect_lat1: First corner latitude (EPSG:4326), or None
        rect_lon1: First corner longitude (EPSG:4326), or None
        rect_lat2: Second corner latitude (EPSG:4326), or None
        rect_lon2: Second corner longitude (EPSG:4326), or None
        bearing: Bearing in degrees (0-360) for cutout rotation. 0=North, 90=East, etc.

    Returns:
        Masked DEM array with areas outside cutout set to nodata
    """
    rows, cols = arr.shape
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    # Create coordinate grids for all pixels
    row_indices, col_indices = np.mgrid[0:rows, 0:cols]

    # Get x, y coordinates for each pixel using affine transform
    pixel_x = transform.c + transform.a * col_indices + transform.b * row_indices
    pixel_y = transform.f + transform.d * col_indices + transform.e * row_indices

    # Convert bearing to radians for rotation (bearing is clockwise from north)
    bearing_rad = np.radians(bearing)

    # Handle rectangle corners mode
    if rect_lat1 is not None and rect_lon1 is not None and rect_lat2 is not None and rect_lon2 is not None:
        # Transform both corners from EPSG:4326 to DEM's CRS
        corner1_x, corner1_y = transformer.transform(rect_lon1, rect_lat1)
        corner2_x, corner2_y = transformer.transform(rect_lon2, rect_lat2)

        # Calculate center of the rectangle
        center_x = (corner1_x + corner2_x) / 2.0
        center_y = (corner1_y + corner2_y) / 2.0

        if bearing != 0.0:
            # Project pixel offsets onto bearing-aligned local frame
            px_centered = pixel_x - center_x
            py_centered = pixel_y - center_y
            pixel_perp, pixel_along = rotate_to_bearing_frame(px_centered, py_centered, bearing_rad)

            # Project corner offsets onto bearing-aligned local frame
            c1_perp, c1_along = rotate_to_bearing_frame(corner1_x - center_x, corner1_y - center_y, bearing_rad)
            c2_perp, c2_along = rotate_to_bearing_frame(corner2_x - center_x, corner2_y - center_y, bearing_rad)

            # Determine min/max bounds in local frame
            min_perp = min(c1_perp, c2_perp)
            max_perp = max(c1_perp, c2_perp)
            min_along = min(c1_along, c2_along)
            max_along = max(c1_along, c2_along)

            # Create mask using local-frame pixel coordinates
            mask = (pixel_perp < min_perp) | (pixel_perp > max_perp) | (pixel_along < min_along) | (pixel_along > max_along)
        else:
            # No rotation - use original logic
            min_x = min(corner1_x, corner2_x)
            max_x = max(corner1_x, corner2_x)
            min_y = min(corner1_y, corner2_y)
            max_y = max(corner1_y, corner2_y)
            mask = (pixel_x < min_x) | (pixel_x > max_x) | (pixel_y < min_y) | (pixel_y > max_y)

    # Handle center-based cutouts
    else:
        # Transform center from EPSG:4326 to DEM's CRS
        center_x, center_y = transformer.transform(center_lon, center_lat)

        # Calculate distances from center
        dx = pixel_x - center_x
        dy = pixel_y - center_y

        # Apply rotation if bearing is non-zero
        if bearing != 0.0 and side_length_km is not None:
            # Project offset coordinates onto bearing-aligned local frame
            dx, dy = rotate_to_bearing_frame(dx, dy, bearing_rad)

        # Create mask based on cutout type
        if radius_km is not None:
            # Circular cutout - rotation doesn't affect circular shapes
            # Use exact radius for min/max calculation
            # For boolean intersection approach, we'll build a larger rectangular mesh
            # and let the boolean op cut it precisely
            radius_m = radius_km * 1000.0
            distances = np.sqrt(dx**2 + dy**2)
            mask = distances > radius_m  # True = outside = mask out
        else:
            # Rectangular (square) cutout
            half_side_m = (side_length_km * 1000.0) / 2.0
            mask = (np.abs(dx) > half_side_m) | (np.abs(dy) > half_side_m)

    # Apply mask
    arr_masked = arr.copy()
    arr_masked[mask] = nodata_value

    return arr_masked
