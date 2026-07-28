"""Tests for ModelFrame's CRS -> model-mm conversions.

The frame's derived scalars and its bit-identity with the mesh grid are covered by
``test_terrain_layout.TestModelFrame``; this module takes the conversion methods and
the geometry repair path, which nothing exercised. ``geojsons_to_mm`` is where raw
OSM rings enter the pipeline, and OSM rings are routinely self-intersecting, so its
repair branch runs on real data and needs to be pinned.
"""

import numpy as np
import pytest
import shapely
from rasterio.transform import from_origin
from shapely.geometry import (LineString, MultiPolygon, Point,
                              Polygon as ShapelyPolygon, box, mapping)

from model_frame import ModelFrame


ROWS, COLS = 21, 41
PX = 10.0
ORIGIN_X, ORIGIN_Y = 500000.0, 5000000.0
X_SIZE_MM = 80.0


def _frame(rows=ROWS, cols=COLS, x_size_mm=X_SIZE_MM, px=PX):
    return ModelFrame.from_dem((rows, cols), px, px, x_size_mm,
                               from_origin(ORIGIN_X, ORIGIN_Y, px, px), "EPSG:32633")


def _crs_box(col0, row0, col1, row1):
    """A box given in DEM pixel-CENTRE indices, as a GeoJSON dict."""
    return mapping(box(ORIGIN_X + (col0 + 0.5) * PX, ORIGIN_Y - (row1 + 0.5) * PX,
                       ORIGIN_X + (col1 + 0.5) * PX, ORIGIN_Y - (row0 + 0.5) * PX))


class TestDerivedScalars:
    def test_model_y_follows_the_pixel_centre_aspect(self):
        """cols-1 spacings span x_size_mm, so the aspect uses rows-1 / cols-1."""
        frame = _frame()
        assert frame.model_y_mm == pytest.approx(X_SIZE_MM * (ROWS - 1) / (COLS - 1))

    def test_non_square_pixels_are_accounted_for(self):
        frame = ModelFrame.from_dem((ROWS, COLS), 10.0, 20.0, X_SIZE_MM,
                                    from_origin(ORIGIN_X, ORIGIN_Y, 10.0, 20.0),
                                    "EPSG:32633")
        assert frame.model_y_mm == pytest.approx(
            X_SIZE_MM * ((ROWS - 1) * 20.0) / ((COLS - 1) * 10.0))

    def test_bounds_start_at_the_origin(self):
        frame = _frame()
        assert frame.bounds_mm == (0.0, 0.0, X_SIZE_MM, frame.model_y_mm)

    def test_grid_pitch_is_guarded_against_a_single_column(self):
        """max(cols - 1, 1) keeps a degenerate frame from dividing by zero."""
        assert _frame(rows=1, cols=1).grid_pitch_mm == X_SIZE_MM

    def test_output_resolution_is_taken_at_the_largest_coordinate(self):
        """float32's absolute step is coarsest there, so that bound is the safe one."""
        wide = _frame(rows=5, cols=101)          # model_y_mm < x_size_mm
        tall = _frame(rows=101, cols=5)          # model_y_mm > x_size_mm
        assert wide.output_resolution == np.spacing(np.float32(wide.x_size_mm))
        assert tall.output_resolution == np.spacing(np.float32(tall.model_y_mm))

    def test_scale_is_terrain_metres_per_printed_mm(self):
        frame = _frame()
        assert frame.scale_m_per_mm == pytest.approx((COLS - 1) * PX / X_SIZE_MM)
        # Halving the print size doubles the scale denominator.
        assert _frame(x_size_mm=X_SIZE_MM / 2).scale_m_per_mm \
            == pytest.approx(frame.scale_m_per_mm * 2)

    def test_grid_arrays_span_the_model_and_ascend(self):
        frame = _frame()
        assert len(frame.grid_xs) == COLS and len(frame.grid_ys) == ROWS
        assert frame.grid_xs[0] == 0.0 and frame.grid_xs[-1] == X_SIZE_MM
        assert frame.grid_ys[0] == 0.0 and frame.grid_ys[-1] == frame.model_y_mm
        assert np.all(np.diff(frame.grid_xs) > 0)
        assert np.all(np.diff(frame.grid_ys) > 0)

    def test_frame_is_immutable(self):
        """A frozen dataclass -- the 2D and mesh stages share one instance."""
        with pytest.raises(Exception):
            _frame().x_size_mm = 1.0


class TestPointConversion:
    def test_coords_to_mm_matches_point_to_mm_exactly(self):
        """The vectorized path is what real rings go through; it must not drift."""
        frame = _frame()
        rng = np.random.default_rng(5)
        pts = np.column_stack((
            rng.uniform(ORIGIN_X, ORIGIN_X + COLS * PX, 300),
            rng.uniform(ORIGIN_Y - ROWS * PX, ORIGIN_Y, 300),
        ))
        got = frame.coords_to_mm(pts)
        for (x, y), row in zip(pts, got):
            assert tuple(row) == frame.point_to_mm(x, y)

    def test_pixel_centres_map_to_the_model_corners(self):
        frame = _frame()
        assert frame.point_to_mm(ORIGIN_X + 0.5 * PX, ORIGIN_Y - 0.5 * PX) \
            == pytest.approx((0.0, frame.model_y_mm))
        assert frame.point_to_mm(ORIGIN_X + (COLS - 0.5) * PX,
                                 ORIGIN_Y - (ROWS - 0.5) * PX) \
            == pytest.approx((X_SIZE_MM, 0.0))

    def test_conversion_is_affine(self):
        """No projection happens here -- midpoints must map to midpoints."""
        frame = _frame()
        a = (ORIGIN_X + 30.0, ORIGIN_Y - 40.0)
        b = (ORIGIN_X + 230.0, ORIGIN_Y - 140.0)
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        ma, mb, mm = (frame.point_to_mm(*p) for p in (a, b, mid))
        assert mm == pytest.approx(((ma[0] + mb[0]) / 2, (ma[1] + mb[1]) / 2))

    def test_points_outside_the_dem_are_not_clamped(self):
        """Clipping is the layout stage's job; the mapping stays linear."""
        frame = _frame()
        x, y = frame.point_to_mm(ORIGIN_X - 100.0, ORIGIN_Y + 100.0)
        assert x < 0.0 and y > frame.model_y_mm

    def test_coords_to_mm_accepts_a_list_of_tuples(self):
        frame = _frame()
        got = frame.coords_to_mm([(ORIGIN_X + 5.0, ORIGIN_Y - 5.0),
                                  (ORIGIN_X + 15.0, ORIGIN_Y - 15.0)])
        assert got.shape == (2, 2)


class TestGeojsonToMm:
    def test_polygon_lands_on_the_model_grid(self):
        frame = _frame()
        got = frame.geojson_to_mm(_crs_box(0, 0, COLS - 1, ROWS - 1))
        assert got.bounds == pytest.approx((0.0, 0.0, X_SIZE_MM, frame.model_y_mm))

    def test_holes_are_preserved(self):
        frame = _frame()
        outer = box(ORIGIN_X + 50, ORIGIN_Y - 150, ORIGIN_X + 250, ORIGIN_Y - 50)
        inner = box(ORIGIN_X + 100, ORIGIN_Y - 120, ORIGIN_X + 180, ORIGIN_Y - 80)
        got = frame.geojson_to_mm(mapping(outer.difference(inner)))
        assert len(got.interiors) == 1
        assert got.area == pytest.approx(
            (outer.area - inner.area) / frame.scale_m_per_mm ** 2, rel=1e-9)

    def test_multipolygon_keeps_every_part(self):
        frame = _frame()
        multi = MultiPolygon([box(ORIGIN_X + 20, ORIGIN_Y - 60, ORIGIN_X + 60,
                                 ORIGIN_Y - 20),
                              box(ORIGIN_X + 120, ORIGIN_Y - 160, ORIGIN_X + 180,
                                  ORIGIN_Y - 100)])
        got = frame.geojson_to_mm(mapping(multi))
        assert got.geom_type == "MultiPolygon" and len(got.geoms) == 2

    def test_non_polygonal_input_is_returned_unconverted(self):
        """A trap worth pinning: only polygons are mapped.

        A LineString or Point falls through the two branches and is returned as it
        arrived -- still in CRS metres, not model mm. Nothing in the pipeline feeds
        non-polygonal geometry in today, so this documents the limitation rather than
        endorsing it: a caller that starts passing lines would get coordinates six
        orders of magnitude off with no error.
        """
        frame = _frame()
        line = LineString([(ORIGIN_X, ORIGIN_Y), (ORIGIN_X + 100, ORIGIN_Y - 100)])
        got = frame.geojson_to_mm(mapping(line))
        assert got.equals(line), "returned in CRS coordinates, unmapped"
        assert got.bounds[0] > 1e5, "still an Easting, not a millimetre"


class TestGeojsonsToMm:
    def test_no_geometries_gives_none(self):
        frame = _frame()
        assert frame.geojsons_to_mm([]) is None
        assert frame.geojsons_to_mm(None) is None

    def test_overlapping_geometries_are_unioned(self):
        frame = _frame()
        got = frame.geojsons_to_mm([_crs_box(2, 2, 12, 12), _crs_box(8, 8, 20, 20)])
        assert got.geom_type == "Polygon", "the two boxes overlap, so they merge"
        parts = [frame.geojson_to_mm(g) for g in
                 (_crs_box(2, 2, 12, 12), _crs_box(8, 8, 20, 20))]
        assert got.area == pytest.approx(shapely.union_all(parts).area)

    def test_disjoint_geometries_stay_separate(self):
        frame = _frame()
        got = frame.geojsons_to_mm([_crs_box(1, 1, 5, 5), _crs_box(20, 10, 30, 18)])
        assert got.geom_type == "MultiPolygon" and len(got.geoms) == 2

    def test_a_self_intersecting_ring_is_repaired_not_dropped(self):
        """OSM rings cross themselves constantly; the feature must survive."""
        frame = _frame()
        bowtie = {"type": "Polygon", "coordinates": [[
            (ORIGIN_X + 50, ORIGIN_Y - 50), (ORIGIN_X + 150, ORIGIN_Y - 150),
            (ORIGIN_X + 150, ORIGIN_Y - 50), (ORIGIN_X + 50, ORIGIN_Y - 150),
            (ORIGIN_X + 50, ORIGIN_Y - 50)]]}
        assert not frame.geojson_to_mm(bowtie).is_valid, "fixture must be invalid"

        got = frame.geojsons_to_mm([bowtie])
        assert got is not None and got.is_valid
        assert got.area > 0.0
        assert got.geom_type in ("Polygon", "MultiPolygon")

    def test_an_invalid_ring_alongside_a_valid_one_does_not_lose_either(self):
        frame = _frame()
        bowtie = {"type": "Polygon", "coordinates": [[
            (ORIGIN_X + 50, ORIGIN_Y - 50), (ORIGIN_X + 150, ORIGIN_Y - 150),
            (ORIGIN_X + 150, ORIGIN_Y - 50), (ORIGIN_X + 50, ORIGIN_Y - 150),
            (ORIGIN_X + 50, ORIGIN_Y - 50)]]}
        good = _crs_box(25, 2, 35, 8)
        got = frame.geojsons_to_mm([bowtie, good])
        assert got.is_valid
        assert got.area > frame.geojson_to_mm(good).area

    def test_a_degenerate_zero_area_ring_is_skipped(self):
        """make_valid turns a collapsed ring into a LineString, which is not kept."""
        frame = _frame()
        flat = {"type": "Polygon", "coordinates": [[
            (ORIGIN_X + 50, ORIGIN_Y - 50), (ORIGIN_X + 150, ORIGIN_Y - 50),
            (ORIGIN_X + 50, ORIGIN_Y - 50)]]}
        assert frame.geojsons_to_mm([flat]) is None

    def test_a_degenerate_ring_does_not_take_a_valid_one_with_it(self):
        frame = _frame()
        flat = {"type": "Polygon", "coordinates": [[
            (ORIGIN_X + 50, ORIGIN_Y - 50), (ORIGIN_X + 150, ORIGIN_Y - 50),
            (ORIGIN_X + 50, ORIGIN_Y - 50)]]}
        good = _crs_box(25, 2, 35, 8)
        got = frame.geojsons_to_mm([flat, good])
        assert got is not None
        assert got.area == pytest.approx(frame.geojson_to_mm(good).area)

    def test_result_is_in_model_millimetres(self):
        frame = _frame()
        got = frame.geojsons_to_mm([_crs_box(0, 0, COLS - 1, ROWS - 1)])
        assert got.bounds == pytest.approx((0.0, 0.0, X_SIZE_MM, frame.model_y_mm))
