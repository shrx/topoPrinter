"""
DEM loading, nodata handling, and merging.

Inputs must already be rasterio-readable rasters; source-specific artifacts
(TanDEM-X zips, ARSO XYZ point clouds) are converted beforehand by
``sources.prepare_dem_files``.
"""

from typing import Iterable, Tuple

import numpy as np
import rasterio
from rasterio.merge import merge
from pyproj import Transformer, Geod

from bearing_utils import rotate_to_bearing_frame


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

    px_size_x, px_size_y, nodata_value, ref_crs = _gather_metadata(path_list)

    # Merge directly at the target resolution: rasterio then reads every source
    # tile decimated (nearest resampling), cutting peak memory by downsample^2
    # compared to building the full-resolution mosaic and slicing it, and the
    # returned transform describes the coarse grid correctly by construction.
    merge_kwargs = {}
    if downsample > 1:
        merge_kwargs["res"] = (px_size_x * downsample, px_size_y * downsample)
        px_size_x *= downsample
        px_size_y *= downsample
    merged, ref_transform = merge(
        path_list,
        nodata=nodata_value,
        method="first",
        **merge_kwargs,
    )
    arr = merged[0]

    # Convert nodata values to NaN so missing areas appear as holes
    if nodata_value is not None:
        arr = np.where(arr == nodata_value, np.nan, arr)

    # Cutout cropping is handled by boolean intersection in mesh_builder

    if arr.size == 0 or arr.shape[0] < 2 or arr.shape[1] < 2:
        raise ValueError("DEM too small after downsampling to form a mesh.")

    # Downstream (true-scale Z, aspect ratio, cutout radius) treats px_size as
    # METRES. Metric DEMs (Swiss, Slovenian, UTM) already satisfy that, but a
    # geographic DEM (e.g. TanDEM-X in EPSG:4979, degrees) reports px_size in
    # degrees -- so convert it to metres at the DEM's centre latitude using a
    # geodesic measure (no hard-coded m/deg constant). The transform stays in the
    # native CRS for georeferencing; only the physical pixel size is converted.
    if getattr(ref_crs, "is_geographic", False):
        rows, cols = arr.shape
        clon = ref_transform.c + ref_transform.a * (cols / 2.0)
        clat = ref_transform.f + ref_transform.e * (rows / 2.0)
        geod = Geod(ellps="WGS84")
        # metres spanned by one degree of lon / lat at the centre
        m_per_deg_lon = geod.inv(clon - 0.5, clat, clon + 0.5, clat)[2]
        m_per_deg_lat = geod.inv(clon, clat - 0.5, clon, clat + 0.5)[2]
        px_size_x *= m_per_deg_lon
        px_size_y *= m_per_deg_lat

    return arr, px_size_x, px_size_y, ref_crs, ref_transform


def crop_to_cutout(
    arr: np.ndarray,
    transform: object,
    crs: object,
    center_lat: float = None,
    center_lon: float = None,
    radius_m: float = None,
    side_length_km: float = None,
    rect_lat1: float = None,
    rect_lon1: float = None,
    rect_lat2: float = None,
    rect_lon2: float = None,
) -> Tuple[np.ndarray, object]:
    """Crop the DEM to the cutout region's bounding box.

    The output model scale is set from the cropped array, so ``--x-size-mm`` maps
    to the cutout, not the whole provided tile -- and only the cutout neighbourhood
    is meshed instead of a full 1-degree tile. The final (circular/rotated) shape is
    still trimmed later by the mesh boolean cutout; this only bounds the raster.

    Returns (cropped_arr, cropped_transform). A no-op (returns the inputs) when no
    cutout region is given or the window would be degenerate.
    """
    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    pts = []
    if rect_lat1 is not None and rect_lat2 is not None:
        pts = [(rect_lon1, rect_lat1), (rect_lon2, rect_lat2),
               (rect_lon1, rect_lat2), (rect_lon2, rect_lat1)]
    elif center_lat is not None:
        if radius_m is not None:
            reach = radius_m
        elif side_length_km is not None:
            reach = (side_length_km * 1000.0 / 2.0) * (2.0 ** 0.5)  # half-diagonal
        else:
            return arr, transform
        geod = Geod(ellps="WGS84")
        # Sample the boundary all the way round so a rotated/square region is fully
        # bounded regardless of bearing.
        for az in range(0, 360, 15):
            lon2, lat2, _ = geod.fwd(center_lon, center_lat, az, reach)
            pts.append((lon2, lat2))
    else:
        return arr, transform

    xs, ys = [], []
    for lon, lat in pts:
        x, y = tf.transform(lon, lat)
        xs.append(x)
        ys.append(y)
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)

    inv = ~transform
    cols, rows = [], []
    for x, y in [(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)]:
        c, r = inv * (x, y)
        cols.append(c)
        rows.append(r)
    # `inv` gives corner pixel coordinates, but the mesh vertices are pixel CENTRES
    # (index i sits at corner i+0.5). The kept pixels c0..c1-1 have their outer
    # centres at c0+0.5 and c1-0.5, so to make those centres bracket the cutout
    # bbox [min, max] we shift the rounding by that half-pixel: c0 = floor(min-0.5),
    # c1 = ceil(max+0.5). Otherwise the outermost vertex lands inside the bbox and
    # the boolean cutout clips the shape (the ~0.1% shortfall seen at the poles).
    c0 = max(int(np.floor(min(cols) - 0.5)), 0)
    c1 = min(int(np.ceil(max(cols) + 0.5)), arr.shape[1])
    r0 = max(int(np.floor(min(rows) - 0.5)), 0)
    r1 = min(int(np.ceil(max(rows) + 0.5)), arr.shape[0])
    if c1 - c0 < 2 or r1 - r0 < 2:
        return arr, transform
    return arr[r0:r1, c0:c1], transform * rasterio.Affine.translation(c0, r0)


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
