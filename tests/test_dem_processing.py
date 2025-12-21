import numpy as np

from dem_processing import fill_nodata


def test_fill_nodata_replaces_nan_with_min() -> None:
    arr = np.array([[1.0, np.nan], [3.0, 2.0]], dtype=float)
    result = fill_nodata(arr, nodata_value=None)
    assert np.isfinite(result).all()
    assert result[0, 1] == 1.0


def test_fill_nodata_replaces_nodata_value() -> None:
    arr = np.array([[5.0, -9999.0], [7.0, 6.0]], dtype=float)
    result = fill_nodata(arr, nodata_value=-9999.0)
    assert result[0, 1] == 5.0
