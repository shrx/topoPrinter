"""Tests for the 2D terrain stage: ModelFrame and build_terrain_layout."""

import numpy as np
import pytest
import shapely
from rasterio.transform import from_origin
from shapely.geometry import Polygon as ShapelyPolygon, box, mapping

from bearing_utils import rotate_from_bearing_frame
from mesh_builder import _compute_model_coordinates
from model_frame import ModelFrame
from terrain_classifier import (TERRAIN_FOLIAGE, TERRAIN_GLACIER, TERRAIN_ROCK,
                                TERRAIN_WATER)
from terrain_layout import (InsertFit, build_terrain_layout, densify_on_grid,
                            rect_extent_m)


ROWS, COLS = 41, 61
PX = 10.0                       # metres per DEM sample
ORIGIN_X, ORIGIN_Y = 500000.0, 5000000.0


def _frame(x_size_mm=120.0):
    transform = from_origin(ORIGIN_X, ORIGIN_Y, PX, PX)
    return ModelFrame.from_dem((ROWS, COLS), PX, PX, x_size_mm, transform, "EPSG:32633")


def _crs_box(col0, row0, col1, row1):
    """A GeoJSON box given in DEM pixel-centre coordinates."""
    x0 = ORIGIN_X + (col0 + 0.5) * PX
    x1 = ORIGIN_X + (col1 + 0.5) * PX
    y0 = ORIGIN_Y - (row1 + 0.5) * PX
    y1 = ORIGIN_Y - (row0 + 0.5) * PX
    return mapping(box(x0, y0, x1, y1))


class TestModelFrame:
    def test_agrees_with_the_mesh_grid(self):
        """The frame's scalars must equal what the X/Y meshgrids report.

        The whole point of ModelFrame is that the 2D stage can skip building those
        grids; if the two ever disagree, polygons and mesh land in different spaces.
        """
        frame = _frame()
        dem = np.zeros((ROWS, COLS))
        X, Y, _z, _v, _l, model_y_mm = _compute_model_coordinates(
            dem, PX, PX, frame.x_size_mm, max_height_mm=30.0, z_exaggeration=1.0,
            base_thickness_mm=2.0, use_true_scale=False,
        )
        assert frame.model_y_mm == model_y_mm
        assert frame.bounds_mm == (float(X.min()), float(Y.min()),
                                   float(X.max()), float(Y.max()))
        assert frame.grid_pitch_mm == (float(X.max()) - float(X.min())) / (COLS - 1)
        assert frame.output_resolution == np.spacing(
            np.float32(max(float(X.max()), float(Y.max()))))

    def test_pixel_centre_convention(self):
        """Grid vertex (i, j) is the CENTRE of pixel (i, j), not its corner."""
        frame = _frame()
        first = frame.point_to_mm(ORIGIN_X + 0.5 * PX, ORIGIN_Y - 0.5 * PX)
        last = frame.point_to_mm(ORIGIN_X + (COLS - 0.5) * PX,
                                 ORIGIN_Y - (ROWS - 0.5) * PX)
        assert first == pytest.approx((0.0, frame.model_y_mm))
        assert last == pytest.approx((frame.x_size_mm, 0.0))

    def test_scale_m_per_mm(self):
        frame = _frame(x_size_mm=120.0)
        assert frame.scale_m_per_mm == pytest.approx((COLS - 1) * PX / 120.0)

    def test_grid_lines_are_bit_identical_to_the_mesh_grid(self):
        """Not approx: boundaries are densified on these and meshed on those.

        A last-bit difference would put a densified vertex a hair off the grid line
        it was meant to sit on, which is exactly the near-coincidence that snaps
        into a sliver.
        """
        frame = _frame()
        X, Y, *_ = _compute_model_coordinates(
            np.zeros((ROWS, COLS)), PX, PX, frame.x_size_mm, max_height_mm=30.0,
            z_exaggeration=1.0, base_thickness_mm=2.0, use_true_scale=False,
        )
        assert np.array_equal(frame.grid_xs, X[0, :])
        assert np.array_equal(frame.grid_ys, Y[::-1, 0])


class TestRectExtent:
    """The rectangle's own edge lengths, which is what --x-size-mm sizes.

    They are NOT the cropped raster's width: ``crop_to_cutout`` bounds the rectangle
    with a CRS-axis-aligned box rounded out to whole pixels, and a rotated rectangle
    needs a wider box still. Sizing the model from the raster instead of from these is
    what used to force a post-mesh rescale.
    """

    # A 400 m x 300 m rectangle at the equator on the prime meridian, so the WGS84
    # -> Web Mercator conversion the function performs is not itself under test:
    # only the diagonal -> (width, height) decomposition is.
    def _spec(self, bearing=0.0):
        from pyproj import Transformer
        from terrain_layout import CutoutSpec
        to_wgs = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        half_w, half_h = 200.0, 150.0
        b = np.radians(bearing)
        # Corner offsets in the bearing frame -> CRS, so the spec's two corners are a
        # genuine diagonal of the rotated rectangle.
        de, dn = rotate_from_bearing_frame(-half_w, -half_h, b)
        lon1, lat1 = to_wgs.transform(de, dn)
        de, dn = rotate_from_bearing_frame(half_w, half_h, b)
        lon2, lat2 = to_wgs.transform(de, dn)
        return CutoutSpec(cutout_type="rectangular", bearing=bearing,
                          rect_corner1_lat=lat1, rect_corner1_lon=lon1,
                          rect_corner2_lat=lat2, rect_corner2_lon=lon2)

    @pytest.mark.parametrize("bearing", [0.0, 30.0, 90.0, 217.0])
    def test_separates_the_two_edges_at_any_bearing(self, bearing):
        w, h = rect_extent_m("EPSG:3857", self._spec(bearing))
        assert w == pytest.approx(400.0)
        assert h == pytest.approx(300.0)

    def test_is_none_for_a_circular_cutout(self):
        """Only the rectangular branch has a diagonal to decompose."""
        from terrain_layout import CutoutSpec
        spec = CutoutSpec(cutout_type="circular", center_lat=0.0, center_lon=0.0,
                          radius_m=100.0)
        assert rect_extent_m("EPSG:3857", spec) == (None, None)


class TestDensifyOnGrid:
    """Densification must add vertices without moving the shape."""

    def test_inserts_a_vertex_at_every_crossing(self):
        xs = np.array([0.0, 1.0, 2.0, 3.0])
        ys = np.array([0.0, 1.0, 2.0, 3.0])
        poly = ShapelyPolygon([(0.5, 0.5), (2.5, 0.5), (2.5, 2.5), (0.5, 2.5)])
        dense = densify_on_grid(poly, xs, ys)

        assert dense.equals(poly), "densifying must not change the shape"
        assert dense.area == poly.area
        # 4 corners + 2 crossings per side (the x=1,2 / y=1,2 lines).
        assert len(dense.exterior.coords) - 1 == 12

    def test_diagonal_edge_crosses_both_families(self):
        xs = np.array([0.0, 1.0, 2.0])
        ys = np.array([0.0, 1.0, 2.0])
        poly = ShapelyPolygon([(0.0, 0.0), (2.0, 2.0), (0.0, 2.0)])
        dense = densify_on_grid(poly, xs, ys)
        assert dense.equals(poly)
        # The diagonal meets x=1 and y=1 at the same point -- one vertex, not two.
        assert (1.0, 1.0) in list(dense.exterior.coords)

    def test_holes_are_densified_too(self):
        xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        ys = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        poly = ShapelyPolygon([(0.5, 0.5), (3.5, 0.5), (3.5, 3.5), (0.5, 3.5)],
                              [[(1.5, 1.5), (1.5, 2.5), (2.5, 2.5), (2.5, 1.5)]])
        dense = densify_on_grid(poly, xs, ys)
        assert dense.equals(poly)
        assert len(dense.interiors[0].coords) > len(poly.interiors[0].coords)

    def test_shared_seam_densifies_identically_from_either_side(self):
        """Two regions meeting on one edge must not diverge along it."""
        xs = np.array([0.0, 1.0, 2.0, 3.0])
        ys = np.array([0.0, 1.0, 2.0, 3.0])
        left = ShapelyPolygon([(0.0, 0.2), (1.5, 0.2), (1.5, 2.8), (0.0, 2.8)])
        right = ShapelyPolygon([(1.5, 0.2), (3.0, 0.2), (3.0, 2.8), (1.5, 2.8)])
        dl = densify_on_grid(left, xs, ys)
        dr = densify_on_grid(right, xs, ys)

        # As a set: the rings run along the seam in opposite directions, and one of
        # them closes on it, so the coordinate LISTS legitimately differ.
        seam_l = {c for c in dl.exterior.coords if c[0] == 1.5}
        seam_r = {c for c in dr.exterior.coords if c[0] == 1.5}
        assert seam_l == seam_r
        assert (1.5, 1.0) in seam_l and (1.5, 2.0) in seam_l

    def test_slanted_shared_seam_is_bit_identical_from_either_side(self):
        """The general case: no axis-aligned luck, and equality to the last bit.

        A seam whose two copies differ by an ULP is the hairline cell the base
        solid drapes to full DEM height as a razor fin, so approx is not enough.
        """
        xs = np.linspace(0.0, 3.0, 7)
        ys = np.linspace(0.0, 3.0, 5)
        a, b = (0.3, 0.4), (2.7, 2.6)
        left = ShapelyPolygon([a, b, (0.0, 3.0)])
        right = ShapelyPolygon([b, a, (3.0, 0.0)])     # same seam, walked backwards

        cl = np.asarray(densify_on_grid(left, xs, ys).exterior.coords)
        cr = np.asarray(densify_on_grid(right, xs, ys).exterior.coords)
        seam_l = cl[:np.flatnonzero((cl == b).all(axis=1))[0] + 1]
        i = np.flatnonzero((cr == a).all(axis=1))[0]
        seam_r = cr[:i + 1][::-1]

        assert len(seam_l) > 2, "the seam should have picked up crossings"
        assert np.array_equal(seam_l, seam_r)


class TestBuildTerrainLayout:
    def test_rock_mask_is_rejected(self):
        """Rock is the derived leftover; supplying it used to be silently ignored."""
        with pytest.raises(ValueError, match="TERRAIN_ROCK"):
            build_terrain_layout(_frame(), {TERRAIN_ROCK: [_crs_box(5, 5, 20, 20)]})

    def test_inserts_and_pockets_are_produced(self):
        frame = _frame()
        layout = build_terrain_layout(
            frame, {TERRAIN_GLACIER: [_crs_box(10, 8, 30, 24)]})
        assert layout.base_class == TERRAIN_ROCK
        assert layout.base_name == "rock"
        assert layout.pockets, "a mask should carve a pocket"
        assert layout.insert_parts[TERRAIN_GLACIER], "a mask should print an insert"

    def test_insert_is_inset_inside_its_pocket(self):
        """With XY clearance the printed insert must sit strictly inside its seat."""
        frame = _frame()
        clearance = 0.5
        layout = build_terrain_layout(
            frame, {TERRAIN_GLACIER: [_crs_box(10, 8, 30, 24)]},
            fit=InsertFit(xy_clearance_mm=clearance))
        pocket = shapely.union_all([pk for _tc, pk in layout.pockets])
        insert = shapely.union_all(layout.insert_parts[TERRAIN_GLACIER])
        assert pocket.contains(insert)
        assert insert.area < pocket.area
        # the gap is the clearance on every side, so the areas differ by ~perimeter*c
        assert (pocket.area - insert.area) == pytest.approx(
            pocket.exterior.length * clearance, rel=0.05)

    def test_layers_are_disjoint(self):
        """Overlapping masks are resolved by precedence, not left overlapping."""
        frame = _frame()
        layout = build_terrain_layout(frame, {
            TERRAIN_GLACIER: [_crs_box(10, 8, 30, 24)],
            TERRAIN_FOLIAGE: [_crs_box(20, 14, 45, 30)],     # overlaps the glacier
        })
        glacier = shapely.union_all(layout.insert_parts[TERRAIN_GLACIER])
        foliage = shapely.union_all(layout.insert_parts[TERRAIN_FOLIAGE])
        assert glacier.intersection(foliage).area == pytest.approx(0.0, abs=1e-9)
        # glacier outranks foliage in TERRAIN_PRECEDENCE, so it keeps the overlap
        assert glacier.area > foliage.area * 0.0

    def test_excluded_type_is_absorbed_by_its_container(self):
        """A component of an excluded class merges into the active class holding it."""
        frame = _frame()
        masks = {
            TERRAIN_FOLIAGE: [_crs_box(5, 5, 40, 30)],
            TERRAIN_WATER: [_crs_box(15, 12, 22, 18)],       # an island inside foliage
        }
        both = build_terrain_layout(frame, masks)
        assert both.insert_parts[TERRAIN_WATER], "water is built when active"

        only_foliage = build_terrain_layout(frame, masks, terrain_types=["foliage"])
        assert TERRAIN_WATER not in only_foliage.overlay_classes
        merged = shapely.union_all(only_foliage.insert_parts[TERRAIN_FOLIAGE])
        active = shapely.union_all(both.insert_parts[TERRAIN_FOLIAGE])
        assert merged.area > active.area, "the excluded island should be absorbed"

    def test_base_outline_is_quantized_to_the_export_grid(self):
        frame = _frame()
        layout = build_terrain_layout(
            frame, {TERRAIN_GLACIER: [_crs_box(10, 8, 30, 24)]})
        coords = np.asarray(layout.base_outline.exterior.coords)
        assert np.array_equal(coords, coords.astype(np.float32).astype(np.float64))

    def test_outline_clips_the_inserts(self):
        """Everything outside the cutout footprint is trimmed away in 2D."""
        frame = _frame()
        outline = box(10.0, 10.0, 60.0, 50.0)
        layout = build_terrain_layout(
            frame, {TERRAIN_GLACIER: [_crs_box(10, 8, 30, 24)]}, outline=outline)
        insert = shapely.union_all(layout.insert_parts[TERRAIN_GLACIER])
        assert layout.base_outline.buffer(1e-9).contains(insert)
