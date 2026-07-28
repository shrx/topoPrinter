"""Tests for DEM loading, merging, cropping and masking.

This module replaces a suite that tested ``fill_nodata``, a function that no longer
exists -- nodata is now turned into NaN inside ``load_and_merge`` and left as holes.
Because the stale import raised at COLLECTION time, it aborted the whole run
(``pytest`` exits 2 having run nothing), so the tests below also serve to get the
rest of the suite executing again.

Everything here works on small synthetic rasters written to ``tmp_path``; the real
fixture tiles are exercised by ``test_integration.py``.
"""

import numpy as np
import pytest
import rasterio
from pyproj import Geod, Transformer
from rasterio.transform import from_origin

from dem_processing import (_gather_metadata, apply_cutout_mask, crop_to_cutout,
                            load_and_merge)


CRS = "EPSG:32633"             # UTM 33N -- metric, so px_size is already metres
ORIGIN_X, ORIGIN_Y = 500000.0, 5000000.0
PX = 10.0


def _write_tif(path, arr, *, crs=CRS, origin=(ORIGIN_X, ORIGIN_Y), px=PX, nodata=None):
    arr = np.asarray(arr, dtype="float32")
    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
        dtype="float32", crs=crs, nodata=nodata,
        transform=from_origin(origin[0], origin[1], px, px),
    ) as ds:
        ds.write(arr, 1)
    return str(path)


def _ramp(rows, cols):
    return np.arange(rows * cols, dtype="float32").reshape(rows, cols)


def _to_lonlat(x, y, crs=CRS):
    return Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(x, y)


# --------------------------------------------------------------------------
# _gather_metadata
# --------------------------------------------------------------------------

class TestGatherMetadata:
    def test_rejects_an_empty_list(self):
        with pytest.raises(ValueError, match="No DEM datasets"):
            _gather_metadata([])

    def test_reports_absolute_pixel_sizes_and_nodata(self):
        """``transform.e`` is negative for a north-up raster; callers want the size."""
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            p = _write_tif(os.path.join(d, "a.tif"), _ramp(8, 8), nodata=-9999.0)
            px_x, px_y, nodata, crs = _gather_metadata([p])
        assert px_x == pytest.approx(PX)
        assert px_y == pytest.approx(PX), "must be abs(e), not the negative value"
        assert nodata == -9999.0
        assert crs is not None

    def test_rejects_mixed_crs(self, tmp_path):
        a = _write_tif(tmp_path / "a.tif", _ramp(8, 8))
        b = _write_tif(tmp_path / "b.tif", _ramp(8, 8), crs="EPSG:32632")
        with pytest.raises(ValueError, match="same CRS"):
            _gather_metadata([a, b])

    def test_rejects_mixed_pixel_sizes(self, tmp_path):
        a = _write_tif(tmp_path / "a.tif", _ramp(8, 8))
        b = _write_tif(tmp_path / "b.tif", _ramp(8, 8), px=20.0)
        with pytest.raises(ValueError, match="matching pixel sizes"):
            _gather_metadata([a, b])

    def test_identical_pixel_sizes_from_separate_files_pass(self, tmp_path):
        """The check is np.isclose, not ==, so float round-trips must not trip it."""
        a = _write_tif(tmp_path / "a.tif", _ramp(8, 8))
        b = _write_tif(tmp_path / "b.tif", _ramp(8, 8),
                       origin=(ORIGIN_X + 8 * PX, ORIGIN_Y))
        assert _gather_metadata([a, b])[0] == pytest.approx(PX)


# --------------------------------------------------------------------------
# load_and_merge
# --------------------------------------------------------------------------

class TestLoadAndMerge:
    def test_rejects_downsample_below_one(self, tmp_path):
        p = _write_tif(tmp_path / "a.tif", _ramp(8, 8))
        with pytest.raises(ValueError, match="downsample must be >= 1"):
            load_and_merge([p], downsample=0)

    def test_merges_two_adjacent_tiles_into_one_grid(self, tmp_path):
        a = _write_tif(tmp_path / "a.tif", np.ones((8, 8)))
        b = _write_tif(tmp_path / "b.tif", np.full((8, 8), 2.0),
                       origin=(ORIGIN_X + 8 * PX, ORIGIN_Y))
        arr, px_x, px_y, _crs, transform = load_and_merge([a, b], downsample=1)
        assert arr.shape == (8, 16), "tiles should sit side by side"
        assert (px_x, px_y) == pytest.approx((PX, PX))
        assert transform.c == pytest.approx(ORIGIN_X)
        assert np.all(arr[:, :8] == 1.0) and np.all(arr[:, 8:] == 2.0)

    def test_nodata_becomes_nan(self, tmp_path):
        arr_in = np.ones((8, 8), dtype="float32")
        arr_in[2, 3] = -9999.0
        p = _write_tif(tmp_path / "a.tif", arr_in, nodata=-9999.0)
        arr, *_ = load_and_merge([p], downsample=1)
        assert np.isnan(arr[2, 3]), "missing data must be a hole, not -9999"
        assert np.isfinite(arr).sum() == arr.size - 1

    def test_no_nodata_declared_leaves_values_alone(self, tmp_path):
        """With nodata=None nothing may be reinterpreted -- not even a -9999."""
        arr_in = np.ones((8, 8), dtype="float32")
        arr_in[1, 1] = -9999.0
        p = _write_tif(tmp_path / "a.tif", arr_in, nodata=None)
        arr, *_ = load_and_merge([p], downsample=1)
        assert np.isfinite(arr).all()
        assert arr[1, 1] == -9999.0

    def test_downsample_scales_the_reported_pixel_size(self, tmp_path):
        """Merging happens AT the coarse resolution, so px_size must follow."""
        p = _write_tif(tmp_path / "a.tif", _ramp(20, 20))
        arr1, px1, _, _, _ = load_and_merge([p], downsample=1)
        arr2, px2, py2, _, tf2 = load_and_merge([p], downsample=2)
        assert px2 == pytest.approx(px1 * 2)
        assert py2 == pytest.approx(px1 * 2)
        assert arr2.shape == (10, 10)
        assert tf2.a == pytest.approx(PX * 2), "the transform must describe the coarse grid"

    def test_rejects_a_grid_too_small_to_mesh(self, tmp_path):
        """Fewer than 2 rows/cols cannot form a single quad."""
        p = _write_tif(tmp_path / "a.tif", _ramp(2, 2))
        with pytest.raises(ValueError, match="DEM too small"):
            load_and_merge([p], downsample=2)

    def test_geographic_pixel_size_is_converted_to_metres(self, tmp_path):
        """A degrees-CRS DEM reports px_size in degrees; downstream needs metres.

        Latitude degrees are longer than longitude degrees away from the equator, so
        the two must come out different -- a single hard-coded m/deg constant would
        make them equal.
        """
        p = _write_tif(tmp_path / "geo.tif", _ramp(16, 16), crs="EPSG:4326",
                       origin=(8.0, 46.0), px=0.001)
        _arr, px_x, px_y, _crs, _tf = load_and_merge([p], downsample=1)
        assert 60.0 < px_x < 90.0, "0.001 deg lon at 46N is ~77 m"
        assert 100.0 < px_y < 120.0, "0.001 deg lat is ~111 m"
        assert px_x < px_y

    def test_metric_pixel_size_is_left_alone(self, tmp_path):
        p = _write_tif(tmp_path / "a.tif", _ramp(8, 8))
        _arr, px_x, px_y, *_ = load_and_merge([p], downsample=1)
        assert (px_x, px_y) == pytest.approx((PX, PX))


# --------------------------------------------------------------------------
# crop_to_cutout
# --------------------------------------------------------------------------

def _rect_lonlat(x0, y0, x1, y1):
    lon1, lat1 = _to_lonlat(x0, y0)
    lon2, lat2 = _to_lonlat(x1, y1)
    return dict(rect_lat1=lat1, rect_lon1=lon1, rect_lat2=lat2, rect_lon2=lon2)


def _centres(arr, transform):
    """(first_x, last_x, first_y, last_y) pixel-CENTRE coordinates of a window."""
    h, w = arr.shape
    return (transform.c + transform.a * 0.5,
            transform.c + transform.a * (w - 0.5),
            transform.f + transform.e * 0.5,
            transform.f + transform.e * (h - 0.5))


class TestCropToCutout:
    """The crop window must BRACKET the cutout with pixel centres, not corners."""

    def _arr_tf(self, rows=40, cols=40):
        return np.zeros((rows, cols), "float32"), from_origin(ORIGIN_X, ORIGIN_Y, PX, PX)

    def test_no_cutout_is_a_no_op(self):
        arr, tf = self._arr_tf()
        out, out_tf = crop_to_cutout(arr, tf, CRS)
        assert out is arr and out_tf is tf

    def test_centre_without_a_size_is_a_no_op(self):
        """A bare centre says nothing about extent, so nothing can be cropped."""
        arr, tf = self._arr_tf()
        lon, lat = _to_lonlat(ORIGIN_X + 200, ORIGIN_Y - 200)
        out, out_tf = crop_to_cutout(arr, tf, CRS, center_lat=lat, center_lon=lon)
        assert out is arr and out_tf is tf

    def test_outer_pixel_centres_bracket_the_requested_box(self):
        """The documented half-pixel rule.

        Plain floor/ceil on the corner columns puts the outermost VERTEX inside the
        requested box, and the later boolean cutout then clips the shape (the ~0.1%
        shortfall seen at the poles). Bracketing is what stops that, so it is
        asserted as an inequality on the centres rather than on exact indices.
        """
        arr, tf = self._arr_tf()
        x0, y0, x1, y1 = 500100.0, 4999750.0, 500250.0, 4999900.0
        out, out_tf = crop_to_cutout(arr, tf, CRS, **_rect_lonlat(x0, y0, x1, y1))
        fx, lx, fy, ly = _centres(out, out_tf)
        assert fx <= x0 and lx >= x1, "x centres must bracket the box"
        assert fy >= y1 and ly <= y0, "y centres must bracket the box (y descends)"

    def test_crop_is_tight_to_within_one_pixel(self):
        """Bracketing must not degrade into 'keep everything'."""
        arr, tf = self._arr_tf()
        x0, y0, x1, y1 = 500100.0, 4999750.0, 500250.0, 4999900.0
        out, out_tf = crop_to_cutout(arr, tf, CRS, **_rect_lonlat(x0, y0, x1, y1))
        fx, lx, fy, ly = _centres(out, out_tf)
        assert x0 - fx <= PX and lx - x1 <= PX
        assert fy - y1 <= PX and y0 - ly <= PX
        assert out.shape[0] < arr.shape[0] and out.shape[1] < arr.shape[1]

    def test_cropped_transform_matches_the_returned_pixels(self):
        """The transform must describe the SAME pixels that were sliced out."""
        rows, cols = 40, 40
        arr = _ramp(rows, cols)
        tf = from_origin(ORIGIN_X, ORIGIN_Y, PX, PX)
        out, out_tf = crop_to_cutout(arr, tf, CRS,
                                     **_rect_lonlat(500100.0, 4999750.0,
                                                    500250.0, 4999900.0))
        # Recover the offset the transform claims, and read the original there.
        c0 = int(round((out_tf.c - tf.c) / tf.a))
        r0 = int(round((out_tf.f - tf.f) / tf.e))
        assert out[0, 0] == arr[r0, c0]
        assert out[-1, -1] == arr[r0 + out.shape[0] - 1, c0 + out.shape[1] - 1]
        assert out_tf.a == tf.a and out_tf.e == tf.e, "resolution must not change"

    def test_corner_order_does_not_matter(self):
        arr, tf = self._arr_tf()
        a = crop_to_cutout(arr, tf, CRS, **_rect_lonlat(500100.0, 4999750.0,
                                                        500250.0, 4999900.0))
        b = crop_to_cutout(arr, tf, CRS, **_rect_lonlat(500250.0, 4999900.0,
                                                        500100.0, 4999750.0))
        assert a[0].shape == b[0].shape
        assert a[1] == b[1]

    def test_window_clamps_to_the_raster(self):
        """A cutout hanging off the edge must not produce negative indices."""
        arr, tf = self._arr_tf()
        out, out_tf = crop_to_cutout(arr, tf, CRS,
                                     **_rect_lonlat(ORIGIN_X - 500.0, 4999900.0,
                                                    500100.0, ORIGIN_Y + 500.0))
        assert out.shape[0] >= 2 and out.shape[1] >= 2
        assert out_tf.c >= tf.c, "cannot start left of the raster"
        assert out_tf.f <= tf.f, "cannot start above the raster"

    def test_cutout_entirely_outside_the_raster_is_a_no_op(self):
        """A degenerate window returns the inputs rather than an empty array."""
        arr, tf = self._arr_tf()
        out, out_tf = crop_to_cutout(arr, tf, CRS,
                                     **_rect_lonlat(ORIGIN_X - 2000.0, ORIGIN_Y - 100.0,
                                                    ORIGIN_X - 1000.0, ORIGIN_Y - 50.0))
        assert out is arr and out_tf is tf

    def test_radius_window_contains_the_whole_circle(self):
        """The boundary is sampled all the way round, so every azimuth is bounded."""
        arr, tf = self._arr_tf(60, 60)
        cx, cy = ORIGIN_X + 300.0, ORIGIN_Y - 300.0
        lon, lat = _to_lonlat(cx, cy)
        radius_m = 150.0
        out, out_tf = crop_to_cutout(arr, tf, CRS, center_lat=lat, center_lon=lon,
                                     radius_m=radius_m)
        fx, lx, fy, ly = _centres(out, out_tf)
        geod = Geod(ellps="WGS84")
        for az in range(0, 360, 5):
            lon2, lat2, _ = geod.fwd(lon, lat, az, radius_m)
            x, y = Transformer.from_crs("EPSG:4326", CRS,
                                        always_xy=True).transform(lon2, lat2)
            assert fx <= x <= lx and ly <= y <= fy
        assert out.shape[1] < arr.shape[1], "and it still crops"

    def test_square_window_allows_any_bearing(self):
        """side_length uses the half-DIAGONAL, so a rotated square always fits."""
        arr, tf = self._arr_tf(80, 80)
        lon, lat = _to_lonlat(ORIGIN_X + 400.0, ORIGIN_Y - 400.0)
        side_km = 0.2
        out, out_tf = crop_to_cutout(arr, tf, CRS, center_lat=lat, center_lon=lon,
                                     side_length_km=side_km)
        width_m = out.shape[1] * abs(out_tf.a)
        assert width_m >= side_km * 1000.0 * (2 ** 0.5) - 2 * PX

    def test_radius_wins_over_side_length(self):
        """radius_m is checked first; the two must not silently combine."""
        arr, tf = self._arr_tf(80, 80)
        lon, lat = _to_lonlat(ORIGIN_X + 400.0, ORIGIN_Y - 400.0)
        both = crop_to_cutout(arr, tf, CRS, center_lat=lat, center_lon=lon,
                              radius_m=100.0, side_length_km=0.5)
        only = crop_to_cutout(arr, tf, CRS, center_lat=lat, center_lon=lon,
                              radius_m=100.0)
        assert both[0].shape == only[0].shape


# --------------------------------------------------------------------------
# apply_cutout_mask
# --------------------------------------------------------------------------

class TestApplyCutoutMask:
    ROWS = COLS = 61

    def _setup(self):
        arr = np.ones((self.ROWS, self.COLS), dtype=float)
        tf = from_origin(ORIGIN_X, ORIGIN_Y, PX, PX)
        cx = ORIGIN_X + self.COLS // 2 * PX
        cy = ORIGIN_Y - self.ROWS // 2 * PX
        lon, lat = _to_lonlat(cx, cy)
        return arr, tf, lat, lon

    def _kept(self, masked):
        return np.isfinite(masked)

    def test_input_array_is_not_modified(self):
        arr, tf, lat, lon = self._setup()
        before = arr.copy()
        apply_cutout_mask(arr, tf, CRS, lat, lon, radius_km=0.1)
        assert np.array_equal(arr, before), "must work on a copy"

    def test_circle_keeps_the_centre_and_drops_the_far_corners(self):
        arr, tf, lat, lon = self._setup()
        out = apply_cutout_mask(arr, tf, CRS, lat, lon, radius_km=0.1)
        kept = self._kept(out)
        assert kept[self.ROWS // 2, self.COLS // 2]
        assert not kept[0, 0] and not kept[0, -1]
        assert not kept[-1, 0] and not kept[-1, -1]

    def test_circle_area_is_about_pi_r_squared(self):
        arr, tf, lat, lon = self._setup()
        radius_m = 200.0
        out = apply_cutout_mask(arr, tf, CRS, lat, lon, radius_km=radius_m / 1000.0)
        expected_px = np.pi * radius_m ** 2 / (PX * PX)
        assert self._kept(out).sum() == pytest.approx(expected_px, rel=0.05)

    def test_circle_is_symmetric_about_its_centre(self):
        """A disc is centrally symmetric; an off-by-one centre would break this.

        The radius is deliberately not a whole number of pixels: at exactly 15 px the
        four axial boundary pixels sit at distance == radius, where the lat/lon
        round-trip decides ``>`` differently on opposite sides and costs a pixel of
        symmetry. That is harmless (the final shape is cut by the mesh boolean, not
        by this mask) but it would make this test measure the round-trip, not the mask.
        """
        arr, tf, lat, lon = self._setup()
        kept = self._kept(apply_cutout_mask(arr, tf, CRS, lat, lon, radius_km=0.155))
        assert np.array_equal(kept, kept[::-1, ::-1])

    def test_bearing_does_not_rotate_a_circle(self):
        """Documented invariance -- rotation is meaningless for a disc."""
        arr, tf, lat, lon = self._setup()
        a = apply_cutout_mask(arr, tf, CRS, lat, lon, radius_km=0.15, bearing=0.0)
        b = apply_cutout_mask(arr, tf, CRS, lat, lon, radius_km=0.15, bearing=37.0)
        assert np.array_equal(self._kept(a), self._kept(b))

    def test_square_keeps_an_axis_aligned_block(self):
        arr, tf, lat, lon = self._setup()
        side_km = 0.2
        kept = self._kept(apply_cutout_mask(arr, tf, CRS, lat, lon,
                                            side_length_km=side_km))
        rows = np.flatnonzero(kept.any(axis=1))
        cols = np.flatnonzero(kept.any(axis=0))
        # Every row in the band must be identical -- that is what "axis aligned" means.
        block = kept[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
        assert block.all()
        expected_px = (side_km * 1000.0 / PX) ** 2
        assert kept.sum() == pytest.approx(expected_px, rel=0.1)

    def test_bearing_rotates_a_square(self):
        arr, tf, lat, lon = self._setup()
        plain = self._kept(apply_cutout_mask(arr, tf, CRS, lat, lon,
                                             side_length_km=0.2, bearing=0.0))
        turned = self._kept(apply_cutout_mask(arr, tf, CRS, lat, lon,
                                              side_length_km=0.2, bearing=45.0))
        assert not np.array_equal(plain, turned), "45 deg must change the footprint"
        # A rotation preserves area and keeps the centre.
        assert turned.sum() == pytest.approx(plain.sum(), rel=0.1)
        assert turned[self.ROWS // 2, self.COLS // 2]

    def test_square_rotated_by_90_returns_to_itself(self):
        """A square has 4-fold symmetry, so 90 deg is the identity on its footprint."""
        arr, tf, lat, lon = self._setup()
        a = self._kept(apply_cutout_mask(arr, tf, CRS, lat, lon,
                                         side_length_km=0.2, bearing=0.0))
        b = self._kept(apply_cutout_mask(arr, tf, CRS, lat, lon,
                                         side_length_km=0.2, bearing=90.0))
        assert np.array_equal(a, b)

    def test_rect_corners_ignore_corner_order(self):
        arr, tf, _lat, _lon = self._setup()
        x0, y0 = ORIGIN_X + 100.0, ORIGIN_Y - 400.0
        x1, y1 = ORIGIN_X + 400.0, ORIGIN_Y - 100.0
        a = apply_cutout_mask(arr, tf, CRS, None, None, **_rect_lonlat(x0, y0, x1, y1))
        b = apply_cutout_mask(arr, tf, CRS, None, None, **_rect_lonlat(x1, y1, x0, y0))
        assert np.array_equal(self._kept(a), self._kept(b))

    def test_rect_corners_keep_the_named_box(self):
        arr, tf, _lat, _lon = self._setup()
        x0, y0 = ORIGIN_X + 100.0, ORIGIN_Y - 400.0
        x1, y1 = ORIGIN_X + 400.0, ORIGIN_Y - 100.0
        kept = self._kept(apply_cutout_mask(arr, tf, CRS, None, None,
                                            **_rect_lonlat(x0, y0, x1, y1)))
        expected_px = ((x1 - x0) / PX) * ((y1 - y0) / PX)
        assert kept.sum() == pytest.approx(expected_px, rel=0.1)

    def test_custom_nodata_value_is_written(self):
        arr, tf, lat, lon = self._setup()
        out = apply_cutout_mask(arr, tf, CRS, lat, lon, radius_km=0.1,
                                nodata_value=-9999.0)
        assert np.isfinite(out).all(), "nothing should be NaN with an explicit nodata"
        assert (out == -9999.0).any()
        assert out[self.ROWS // 2, self.COLS // 2] == 1.0
