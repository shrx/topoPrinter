"""Tests for the pure-geometry helpers underneath the mesh stage.

These are the small functions the prism builders lean on: the CRS->model mapping,
the DEM sampler, and the three grid/segment predicates that make the base plate
fast. They carry no trimesh state, so they can be pinned exactly -- and several of
them have a stated completeness requirement (``_cells_crossed_by`` may over-mark
but must never under-mark) that is only meaningful if something checks it against
an exact answer. That is what the brute-force comparisons below are.
"""

import numpy as np
import pytest
import shapely
from rasterio.transform import from_origin
from shapely.geometry import (LineString, MultiLineString, Point,
                              Polygon as ShapelyPolygon, box)

from mesh_builder import (_cells_crossed_by, _constraint_segments,
                          _crs_point_to_model_xy, _dem_sampler,
                          _interior_grid_points, _nodes_near_segments)
from model_frame import ModelFrame
from terrain_layout import densify_on_grid


ROWS, COLS = 31, 41
PX = 10.0
ORIGIN_X, ORIGIN_Y = 500000.0, 5000000.0
X_SIZE_MM = 120.0


def _frame():
    return ModelFrame.from_dem((ROWS, COLS), PX, PX, X_SIZE_MM,
                               from_origin(ORIGIN_X, ORIGIN_Y, PX, PX), "EPSG:32633")


def _grid_frame(cols, rows, x_size_mm):
    """A frame whose lattice is exactly ``linspace(0, x_size_mm, cols)`` by
    ``linspace(0, x_size_mm * (rows - 1) / (cols - 1), rows)``.

    Square pixels, so the frame's aspect reproduces the plain grids these tests are
    written against. Asserted in TestGridFrameFixture rather than assumed -- if the
    frame ever stopped agreeing, every expectation below would silently shift.
    """
    return ModelFrame.from_dem((rows, cols), 1.0, 1.0, x_size_mm,
                               from_origin(0.0, float(rows), 1.0, 1.0), "EPSG:3857")


class TestGridFrameFixture:
    def test_the_fixture_frame_reproduces_the_plain_linspace_grids(self):
        frame = _grid_frame(11, 9, 10.0)
        assert np.array_equal(frame.grid_xs, np.linspace(0.0, 10.0, 11))
        assert np.array_equal(frame.grid_ys, np.linspace(0.0, 8.0, 9))


class TestCrsPointToModelXy:
    """Byte-for-byte the same mapping as ModelFrame.point_to_mm.

    The two are independent copies of the same four lines, in the module that meshes
    and the module that lays out polygons. If they ever drift, inserts stop matching
    the pockets they seat into -- and the drift would be a fraction of a millimetre,
    far too small to notice in a render. So equality is asserted to the last bit.
    """

    def _pairs(self):
        rng = np.random.default_rng(11)
        return np.column_stack((
            rng.uniform(ORIGIN_X, ORIGIN_X + COLS * PX, 200),
            rng.uniform(ORIGIN_Y - ROWS * PX, ORIGIN_Y, 200),
        ))

    def test_matches_model_frame_exactly(self):
        frame = _frame()
        for x, y in self._pairs():
            mine = _crs_point_to_model_xy(x, y, frame.ref_transform, ROWS, COLS,
                                          X_SIZE_MM, frame.model_y_mm)
            assert mine == frame.point_to_mm(x, y)

    def test_pixel_centres_land_on_the_model_corners(self):
        """Vertex (i, j) is the CENTRE of pixel (i, j), so the half-pixel comes off."""
        frame = _frame()
        first = _crs_point_to_model_xy(ORIGIN_X + 0.5 * PX, ORIGIN_Y - 0.5 * PX,
                                       frame.ref_transform, ROWS, COLS, X_SIZE_MM,
                                       frame.model_y_mm)
        last = _crs_point_to_model_xy(ORIGIN_X + (COLS - 0.5) * PX,
                                      ORIGIN_Y - (ROWS - 0.5) * PX,
                                      frame.ref_transform, ROWS, COLS, X_SIZE_MM,
                                      frame.model_y_mm)
        assert first == pytest.approx((0.0, frame.model_y_mm))
        assert last == pytest.approx((X_SIZE_MM, 0.0))

    def test_northing_is_flipped_but_easting_is_not(self):
        """Row 0 is the TOP of the DEM, so Y descends while X ascends."""
        frame = _frame()
        args = (frame.ref_transform, ROWS, COLS, X_SIZE_MM, frame.model_y_mm)
        west = _crs_point_to_model_xy(ORIGIN_X + 100.0, ORIGIN_Y - 100.0, *args)
        east = _crs_point_to_model_xy(ORIGIN_X + 200.0, ORIGIN_Y - 100.0, *args)
        south = _crs_point_to_model_xy(ORIGIN_X + 100.0, ORIGIN_Y - 200.0, *args)
        assert east[0] > west[0], "more Easting -> more model X"
        assert south[1] < west[1], "less Northing -> less model Y"


class TestDemSampler:
    """z_grid_asc[i, j] is the surface at (xs[j], ys[i])."""

    def _grid(self):
        frame = _grid_frame(5, 4, 40.0)         # pitch 10
        xs, ys = frame.grid_xs, frame.grid_ys
        i, j = np.mgrid[0:4, 0:5]
        z = (10.0 * i + j).astype(float)
        return _dem_sampler(z, frame), xs, ys, z

    def test_is_exact_at_the_grid_nodes(self):
        sample, xs, ys, z = self._grid()
        pts = np.array([(xs[j], ys[i]) for i in range(len(ys)) for j in range(len(xs))])
        want = np.array([z[i, j] for i in range(len(ys)) for j in range(len(xs))])
        assert np.allclose(sample(pts), want)
        assert sample(np.array([[xs[2], ys[1]]]))[0] == pytest.approx(z[1, 2]), \
            "index order must be [row, col], not [col, row]"

    def test_cell_centre_is_the_mean_of_four_corners(self):
        sample, xs, ys, z = self._grid()
        mid = np.array([[(xs[1] + xs[2]) / 2, (ys[1] + ys[2]) / 2]])
        expected = (z[1, 1] + z[1, 2] + z[2, 1] + z[2, 2]) / 4.0
        assert sample(mid)[0] == pytest.approx(expected)

    def test_edge_midpoint_is_the_mean_of_two_corners(self):
        sample, xs, ys, z = self._grid()
        mid = np.array([[(xs[1] + xs[2]) / 2, ys[1]]])
        assert sample(mid)[0] == pytest.approx((z[1, 1] + z[1, 2]) / 2.0)

    def test_is_linear_along_a_cell_edge(self):
        sample, xs, ys, z = self._grid()
        ts = np.linspace(0.0, 1.0, 7)
        pts = np.column_stack((xs[1] + ts * (xs[2] - xs[1]), np.full(len(ts), ys[1])))
        got = sample(pts)
        assert np.allclose(got, z[1, 1] + ts * (z[1, 2] - z[1, 1]))

    def test_outside_the_grid_clamps_instead_of_extrapolating(self):
        """Boundary vertices can sit a hair outside; they must not fly off."""
        sample, xs, ys, z = self._grid()
        far = np.array([[-1000.0, -1000.0], [1e4, 1e4], [xs[0] - 5.0, ys[2]]])
        got = sample(far)
        assert np.all(got >= z.min()) and np.all(got <= z.max())
        assert got[0] == pytest.approx(z[0, 0])
        assert got[1] == pytest.approx(z[-1, -1])
        assert got[2] == pytest.approx(z[2, 0]), "clamped in x, exact in y"

    def test_accepts_an_empty_query(self):
        sample, *_ = self._grid()
        assert sample(np.empty((0, 2))).shape == (0,)

    def test_descending_grid_would_be_wrong_so_ascending_is_required(self):
        """The builders reverse the DEM before calling this; document the contract."""
        sample, xs, ys, _z = self._grid()
        assert xs[0] < xs[-1] and ys[0] < ys[-1]
        got = sample(np.array([[xs[0], ys[0]]]))
        assert got[0] == pytest.approx(0.0), "ys[0] must be the FIRST row of z"


class TestConstraintSegments:
    def test_line_string_becomes_consecutive_pairs(self):
        A, B = _constraint_segments(LineString([(0, 0), (1, 0), (1, 1)]))
        assert len(A) == 2
        assert np.array_equal(A, [[0, 0], [1, 0]])
        assert np.array_equal(B, [[1, 0], [1, 1]])

    def test_polygon_boundary_closes_the_ring(self):
        poly = box(0, 0, 2, 3)
        A, B = _constraint_segments(poly.boundary)
        assert len(A) == 4, "a rectangle has four closing segments"
        assert np.allclose(np.hypot(*(B - A).T).sum(), poly.exterior.length)

    def test_holes_are_included(self):
        poly = ShapelyPolygon(box(0, 0, 10, 10).exterior.coords,
                              [box(3, 3, 6, 6).exterior.coords])
        A, _B = _constraint_segments(poly.boundary)
        assert len(A) == 8, "exterior plus the hole ring"

    def test_multi_line_string_parts_are_concatenated(self):
        mls = MultiLineString([[(0, 0), (1, 1)], [(5, 5), (6, 6), (7, 7)]])
        A, B = _constraint_segments(mls)
        assert len(A) == len(B) == 3

    def test_non_linear_parts_are_skipped(self):
        empty_A, empty_B = _constraint_segments(shapely.geometrycollections(
            [shapely.points([[0.0, 0.0]])[0]]))
        assert empty_A.shape == (0, 2) and empty_B.shape == (0, 2)

    def test_degenerate_single_point_line_yields_nothing(self):
        A, B = _constraint_segments(LineString())
        assert len(A) == 0 and len(B) == 0


class TestCellsCrossedBy:
    """Over-marking is harmless; UNDER-marking silently mis-assigns triangles."""

    FRAME = _grid_frame(11, 9, 10.0)
    XS, YS = FRAME.grid_xs, FRAME.grid_ys

    def _exact(self, geom):
        """Every cell whose box actually meets the boundary."""
        boundary = geom.boundary
        ni, nj = len(self.YS) - 1, len(self.XS) - 1
        out = np.zeros((ni, nj), bool)
        for i in range(ni):
            for j in range(nj):
                cell = box(self.XS[j], self.YS[i], self.XS[j + 1], self.YS[i + 1])
                out[i, j] = boundary.intersects(cell)
        return out

    def _dense(self, poly):
        # The documented precondition: boundaries arrive densified on the grid lines.
        return densify_on_grid(poly, self.FRAME)

    @pytest.mark.parametrize("poly", [
        box(2.0, 2.0, 7.0, 6.0),                                    # grid aligned
        box(2.3, 1.7, 6.8, 5.4),                                    # offset
        ShapelyPolygon([(1.5, 1.2), (8.4, 2.9), (6.1, 6.7), (2.2, 5.3)]),   # slanted
        ShapelyPolygon([(0.5, 0.5), (9.5, 0.5), (9.5, 7.5), (0.5, 7.5)],
                       [[(3.0, 3.0), (6.0, 3.0), (6.0, 5.0), (3.0, 5.0)]]),  # hole
    ])
    def test_never_under_marks(self, poly):
        dense = self._dense(poly)
        got = _cells_crossed_by(dense, self.FRAME)
        exact = self._exact(dense)
        missed = exact & ~got
        assert not missed.any(), f"under-marked {int(missed.sum())} cell(s)"

    def test_marks_are_not_wildly_generous(self):
        """A useful shortcut: the mask must stay far from 'every cell'."""
        dense = self._dense(box(2.3, 1.7, 6.8, 5.4))
        got = _cells_crossed_by(dense, self.FRAME)
        assert got.sum() < got.size * 0.6

    def test_multipolygon_covers_every_part(self):
        parts = shapely.multipolygons([box(1.2, 1.2, 3.4, 3.4).exterior.coords,
                                       box(6.1, 4.2, 8.9, 6.8).exterior.coords])
        dense = shapely.union_all([self._dense(g) for g in parts.geoms])
        got = _cells_crossed_by(dense, self.FRAME)
        missed = self._exact(dense) & ~got
        assert not missed.any()

    def test_shape_is_cells_not_nodes(self):
        got = _cells_crossed_by(box(2.0, 2.0, 7.0, 6.0), self.FRAME)
        assert got.shape == (len(self.YS) - 1, len(self.XS) - 1)


class TestNodesNearSegments:
    """The proximity filter that keeps sub-float32 slivers out of the CDT."""

    def _brute(self, A, B, wx, wy, resolution):
        mask = np.zeros((len(wy), len(wx)), bool)
        for i, y in enumerate(wy):
            for j, x in enumerate(wx):
                p = Point(x, y)
                for a, b in zip(A, B):
                    if p.distance(LineString([a, b])) < resolution:
                        mask[i, j] = True
                        break
        return mask

    def test_matches_a_brute_force_distance_test(self):
        wx = np.linspace(0.0, 5.0, 11)
        wy = np.linspace(0.0, 4.0, 9)
        A, B = _constraint_segments(
            ShapelyPolygon([(0.6, 0.55), (4.4, 1.05), (3.05, 3.45)]).boundary)
        for resolution in (0.05, 0.2, 0.6):
            got = _nodes_near_segments(A, B, wx, wy, resolution)
            assert np.array_equal(got, self._brute(A, B, wx, wy, resolution))

    def test_a_node_exactly_on_a_segment_is_flagged(self):
        wx = np.array([0.0, 1.0, 2.0])
        wy = np.array([0.0, 1.0, 2.0])
        A, B = np.array([[0.0, 1.0]]), np.array([[2.0, 1.0]])   # the y=1 grid line
        got = _nodes_near_segments(A, B, wx, wy, 1e-9)
        assert got[1, :].all(), "the whole middle row lies on the segment"
        assert not got[0, :].any() and not got[2, :].any()

    def test_a_distant_segment_flags_nothing(self):
        wx = np.linspace(0.0, 2.0, 5)
        wy = np.linspace(0.0, 2.0, 5)
        A, B = np.array([[100.0, 100.0]]), np.array([[101.0, 100.0]])
        assert not _nodes_near_segments(A, B, wx, wy, 0.1).any()

    def test_zero_length_segment_is_treated_as_a_point(self):
        """dd == 0 must not divide by zero."""
        wx = np.array([0.0, 1.0, 2.0])
        wy = np.array([0.0, 1.0, 2.0])
        A = B = np.array([[1.0, 1.0]])
        got = _nodes_near_segments(A, B, wx, wy, 0.5)
        assert got[1, 1] and got.sum() == 1

    def test_empty_segment_list_returns_an_all_false_mask(self):
        wx, wy = np.linspace(0, 1, 3), np.linspace(0, 1, 4)
        got = _nodes_near_segments(np.empty((0, 2)), np.empty((0, 2)), wx, wy, 0.1)
        assert got.shape == (4, 3) and not got.any()


class TestInteriorGridPoints:
    FRAME = _grid_frame(21, 13, 10.0)    # pitch 0.5
    XS, YS = FRAME.grid_xs, FRAME.grid_ys

    def test_returns_only_points_inside_the_region(self):
        region = box(2.0, 1.0, 7.0, 4.0)
        pts = _interior_grid_points(region, region.boundary, self.FRAME, 1e-9)
        assert len(pts)
        assert shapely.contains_xy(region, pts[:, 0], pts[:, 1]).all()

    def test_returns_every_strictly_interior_grid_node(self):
        region = box(2.0, 1.0, 7.0, 4.0)
        pts = _interior_grid_points(region, region.boundary, self.FRAME, 1e-9)
        GX, GY = np.meshgrid(self.XS, self.YS)
        expected = int(shapely.contains_xy(region, GX, GY).sum())
        assert len(pts) == expected

    def test_nodes_hugging_the_boundary_are_dropped(self):
        """Within one float32 step of an edge, a node collapses onto it on export."""
        resolution = 1e-3
        # Left edge a hair right of the x=2.0 grid line, so x=2.0 is inside but hugging.
        region = ShapelyPolygon([(2.0 - 1e-4, 1.2), (7.2, 1.2), (7.2, 3.8),
                                 (2.0 - 1e-4, 3.8)])
        loose = _interior_grid_points(region, region.boundary, self.FRAME, 1e-12)
        assert (loose[:, 0] == 2.0).sum() > 0, "fixture must contain the column"
        tight = _interior_grid_points(region, region.boundary, self.FRAME,
                                      resolution)
        assert (tight[:, 0] == 2.0).sum() == 0, "the hugging column must go"
        assert len(tight) < len(loose)

    def test_a_pocket_seam_also_excludes_points(self):
        """Constraints, not just the outer rim -- the base plate hugs pocket seams."""
        region = box(1.0, 1.0, 9.0, 5.0)
        seam = LineString([(5.0, 1.0), (5.0, 5.0)])     # straight down a grid column
        with_seam = _interior_grid_points(
            region, shapely.union_all([region.boundary, seam]),
            self.FRAME, 1e-3)
        assert (with_seam[:, 0] == 5.0).sum() == 0, "the seam column must be excluded"
        without = _interior_grid_points(region, region.boundary, self.FRAME, 1e-3)
        assert (without[:, 0] == 5.0).sum() > 0

    def test_a_region_holding_no_grid_node_returns_empty(self):
        tiny = box(0.6, 0.6, 0.9, 0.9)      # fits between grid lines at pitch 0.5
        pts = _interior_grid_points(tiny, tiny.boundary, self.FRAME, 1e-9)
        assert pts.shape == (0, 2)

    def test_holes_are_excluded(self):
        region = ShapelyPolygon(box(1.0, 1.0, 9.0, 5.0).exterior.coords,
                                [box(3.0, 2.0, 6.0, 4.0).exterior.coords])
        pts = _interior_grid_points(region, region.boundary, self.FRAME, 1e-9)
        assert shapely.contains_xy(region, pts[:, 0], pts[:, 1]).all()
        hole = box(3.0, 2.0, 6.0, 4.0)
        assert not shapely.contains_xy(hole, pts[:, 0], pts[:, 1]).any()
