"""
DEM loading, nodata handling, and merging.
"""

from typing import Iterable, Tuple

import numpy as np
import rasterio
from rasterio.merge import merge


def fill_nodata(arr: np.ndarray, nodata_value) -> np.ndarray:
    data = arr.astype(np.float64, copy=False)
    mask = ~np.isfinite(data)
    if nodata_value is not None:
        mask |= data == nodata_value
    if mask.any():
        valid = data[~mask]
        if valid.size == 0:
            raise ValueError("DEM contains only nodata values.")
        # Use minimum valid value so masked water surfaces do not get lifted above surroundings.
        fill_value = float(valid.min())
        data = data.copy()
        data[mask] = fill_value
    return data


def _gather_metadata(paths: Iterable[str]) -> Tuple[float, float, float, object]:
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


def load_and_merge(paths: Iterable[str], downsample: int) -> Tuple[np.ndarray, float, float]:
    if downsample < 1:
        raise ValueError("downsample must be >= 1")

    path_list = list(paths)
    px_size_x, px_size_y, nodata_value, _ = _gather_metadata(path_list)
    merged, _transform = merge(
        path_list,
        nodata=nodata_value,
        method="first",
    )
    arr = merged[0]
    arr = fill_nodata(arr, nodata_value)

    if downsample > 1:
        arr = arr[::downsample, ::downsample]
        px_size_x *= downsample
        px_size_y *= downsample

    if arr.size == 0 or arr.shape[0] < 2 or arr.shape[1] < 2:
        raise ValueError("DEM too small after downsampling to form a mesh.")

    return arr, px_size_x, px_size_y
