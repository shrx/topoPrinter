"""
DEM loading, nodata handling, and merging.
"""

import os
from typing import Iterable, Tuple, List

import numpy as np
import rasterio
from rasterio.merge import merge
from osgeo import gdal, ogr
from pyproj import Transformer


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
    center_lat: float = None,
    center_lon: float = None,
    radius_km: float = None,
    side_length_km: float = None,
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

    # Apply cutout mask if specified
    # For circular cutouts, skip masking - will use boolean intersection in mesh_builder
    # For rectangular cutouts, apply mask as before
    if center_lat is not None and side_length_km is not None:
        # Rectangular cutout - apply mask
        arr = apply_cutout_mask(
            arr,
            ref_transform,
            ref_crs,
            center_lat,
            center_lon,
            None,  # no radius
            side_length_km,
            px_size_x,
            px_size_y,
            np.nan
        )
        valid_count = np.sum(np.isfinite(arr))
        if valid_count == 0:
            raise ValueError(
                f"Cutout at ({center_lat}, {center_lon}) excluded all data. "
                "Check that cutout region intersects with DEM."
            )

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
) -> np.ndarray:
    """
    Apply circular or rectangular cutout mask to DEM array.

    For circular cutouts, includes a buffer to keep pixels partially within the circle.
    This buffer allows later interpolation to n-gon perimeter vertices.

    Args:
        arr: DEM array (rows x cols)
        transform: Affine transform from rasterio
        crs: CRS of the DEM
        center_lat: Center latitude (EPSG:4326)
        center_lon: Center longitude (EPSG:4326)
        radius_km: Radius for circular cutout (km), or None
        side_length_km: Side length for square cutout (km), or None
        px_size_x: Pixel size in X direction (meters), required for circular cutouts
        px_size_y: Pixel size in Y direction (meters), required for circular cutouts
        nodata_value: Value to set for masked areas

    Returns:
        Masked DEM array with areas outside cutout set to nodata
    """
    rows, cols = arr.shape

    # Transform center from EPSG:4326 to DEM's CRS
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    center_x, center_y = transformer.transform(center_lon, center_lat)

    # Create coordinate grids for all pixels
    row_indices, col_indices = np.mgrid[0:rows, 0:cols]

    # Get x, y coordinates for each pixel using affine transform
    pixel_x = transform.c + transform.a * col_indices + transform.b * row_indices
    pixel_y = transform.f + transform.d * col_indices + transform.e * row_indices

    # Calculate distances from center
    dx = pixel_x - center_x
    dy = pixel_y - center_y

    # Create mask based on cutout type
    if radius_km is not None:
        # Circular cutout - use exact radius for min/max calculation
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
