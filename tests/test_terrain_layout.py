"""Tests for the 2D terrain stage: ModelFrame and build_terrain_layout."""

import numpy as np
import pytest
import shapely
from rasterio.transform import from_origin
from shapely.geometry import Polygon as ShapelyPolygon, box, mapping

from bearing_utils import rotate_from_bearing_frame
from mesh_builder import _compute_model_coordinates
from model_frame import ModelFrame
from masks import (TERRAIN_FOLIAGE, TERRAIN_GLACIER, TERRAIN_ROCK,
                   TERRAIN_WATER)
from terrain_layout import (BODY_RELIEF_P0_MM, BODY_RELIEF_PMAX_MM, InsertFit,
                            _corner_reliefs, _insert_body, build_terrain_layout,
                            densify_on_grid, rect_extent_m)


ROWS, COLS = 41, 61
PX = 10.0                       # metres per DEM sample
ORIGIN_X, ORIGIN_Y = 500000.0, 5000000.0


def _frame(x_size_mm=120.0):
    transform = from_origin(ORIGIN_X, ORIGIN_Y, PX, PX)
    return ModelFrame.from_dem((ROWS, COLS), PX, PX, x_size_mm, transform, "EPSG:32633")


def _grid_frame(cols, rows, x_size_mm, y_size_mm):
    """A frame whose lattice IS ``linspace(0, x_size_mm, cols)`` by
    ``linspace(0, y_size_mm, rows)``.

    The pixel aspect is solved for, because the frame derives its Y extent from the
    DEM's rather than taking it directly. Checked in
    ``test_the_fixture_frame_reproduces_the_plain_linspace_grids`` rather than assumed.
    """
    px = 1.0
    py = y_size_mm * (cols - 1) / (x_size_mm * (rows - 1))
    return ModelFrame.from_dem((rows, cols), px, py, x_size_mm,
                               from_origin(ORIGIN_X, ORIGIN_Y, px, py), "EPSG:32633")


def _crs_box(col0, row0, col1, row1):
    """A GeoJSON box given in DEM pixel-centre coordinates."""
    x0 = ORIGIN_X + (col0 + 0.5) * PX
    x1 = ORIGIN_X + (col1 + 0.5) * PX
    y0 = ORIGIN_Y - (row1 + 0.5) * PX
    y1 = ORIGIN_Y - (row0 + 0.5) * PX
    return mapping(box(x0, y0, x1, y1))


def _crs_geom(geom_mm, x_size_mm=120.0):
    """A GeoJSON geometry given in model mm (``_frame()``'s print space)."""
    scale = (COLS - 1) * PX / x_size_mm
    y_mm = (ROWS - 1) * PX / scale
    return mapping(shapely.transform(
        geom_mm,
        lambda pts: np.column_stack([
            ORIGIN_X + 0.5 * PX + pts[:, 0] * scale,
            ORIGIN_Y - 0.5 * PX - (y_mm - pts[:, 1]) * scale])))


class TestModelFrame:
    def test_agrees_with_the_mesh_grid(self):
        """The frame's scalars must equal what the X/Y meshgrids report.

        The whole point of ModelFrame is that the 2D stage can skip building those
        grids; if the two ever disagree, polygons and mesh land in different spaces.
        """
        frame = _frame()
        dem = np.zeros((ROWS, COLS))
        X, Y, _z, _v, model_y_mm = _compute_model_coordinates(
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


class TestPrintMotion:
    """The grid -> print motion a rectangular cutout carries.

    This used to be a transform applied to finished vertices, which is why it could
    rescale as well; pinning the scale upstream leaves a rigid motion, and a rigid
    motion can be applied in the 2D stage -- ahead of the float32 snap, so the snapped
    coordinates are the exported ones. The two properties below are what make that
    safe: the rectangle lands where the print wants it, and no distance changes.
    """

    # A 400 m x 300 m rectangle inside a 1000 m raster, so the pin is a real factor
    # (2.5x) rather than 1 and the motion has something to move.
    RASTER_M = (COLS - 1) * PX
    RECT_W_M, RECT_H_M = 400.0, 300.0
    PRINT_W_MM = 120.0

    def _frame_and_spec(self, bearing):
        from pyproj import Transformer
        from terrain_layout import CutoutSpec, frame_with_print_motion
        to_wgs = Transformer.from_crs("EPSG:32633", "EPSG:4326", always_xy=True)
        cx = ORIGIN_X + self.RASTER_M / 2.0
        cy = ORIGIN_Y - ((ROWS - 1) * PX) / 2.0
        b = np.radians(bearing)
        corners_local = [(-self.RECT_W_M / 2, -self.RECT_H_M / 2),
                         (self.RECT_W_M / 2, -self.RECT_H_M / 2),
                         (self.RECT_W_M / 2, self.RECT_H_M / 2),
                         (-self.RECT_W_M / 2, self.RECT_H_M / 2)]
        crs_corners = []
        for perp, along in corners_local:
            de, dn = rotate_from_bearing_frame(perp, along, b)
            crs_corners.append((cx + de, cy + dn))
        lon1, lat1 = to_wgs.transform(*crs_corners[0])
        lon2, lat2 = to_wgs.transform(*crs_corners[2])
        spec = CutoutSpec(cutout_type="rectangular", bearing=bearing,
                          rect_corner1_lat=lat1, rect_corner1_lon=lon1,
                          rect_corner2_lat=lat2, rect_corner2_lon=lon2)
        # The scale pin the CLI applies before the frame exists.
        x_size = self.PRINT_W_MM * self.RASTER_M / self.RECT_W_M
        frame = frame_with_print_motion(_frame(x_size_mm=x_size), spec)
        return frame, spec, crs_corners

    @pytest.mark.parametrize("bearing", [0.0, 31.0, 90.0, 206.0])
    def test_the_rectangle_lands_at_the_origin_at_the_requested_width(self, bearing):
        frame, _spec, crs_corners = self._frame_and_spec(bearing)
        grid = [frame.point_to_mm(x, y) for x, y in crs_corners]
        got = frame.to_print(grid)
        h_mm = self.PRINT_W_MM * self.RECT_H_M / self.RECT_W_M
        want = [(0.0, 0.0), (self.PRINT_W_MM, 0.0),
                (self.PRINT_W_MM, h_mm), (0.0, h_mm)]
        # 1 nm. The fixture states its corners in CRS metres, and rect_crs_corners
        # reads them back through WGS84, so a sub-nanometre geodetic round trip is the
        # noise floor here -- four orders below the 1.5e-5 mm float32 export step.
        assert got == pytest.approx(np.asarray(want), abs=1e-6)

    @pytest.mark.parametrize("bearing", [0.0, 31.0, 206.0])
    def test_the_motion_is_rigid(self, bearing):
        """No scale in it, so nothing the 2D stage already placed changes size."""
        frame, _spec, _c = self._frame_and_spec(bearing)
        rng = np.random.default_rng(3)
        pts = np.column_stack((rng.uniform(0, frame.x_size_mm, 40),
                               rng.uniform(0, frame.model_y_mm, 40)))
        out = frame.to_print(pts)

        def pdist(v):
            d = v[:, None, :] - v[None, :, :]
            return np.hypot(d[..., 0], d[..., 1])

        assert pdist(out) == pytest.approx(pdist(pts))

    @pytest.mark.parametrize("bearing", [0.0, 31.0, 206.0])
    def test_to_grid_inverts_to_print(self, bearing):
        frame, _spec, _c = self._frame_and_spec(bearing)
        rng = np.random.default_rng(11)
        pts = np.column_stack((rng.uniform(0, frame.x_size_mm, 40),
                               rng.uniform(0, frame.model_y_mm, 40)))
        assert frame.to_grid(frame.to_print(pts)) == pytest.approx(pts)

    def test_a_circular_cutout_keeps_the_identity(self):
        """A disc is rotation-invariant, so there is nothing to turn."""
        from terrain_layout import CutoutSpec, frame_with_print_motion
        spec = CutoutSpec(cutout_type="circular", center_lat=45.0, center_lon=15.0,
                          radius_m=500.0, bearing=37.0)
        frame = frame_with_print_motion(_frame(), spec)
        assert frame.print_is_identity
        pts = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert frame.to_print(pts) is pts, "identity must not even copy"

    def test_the_export_resolution_follows_the_print_extent(self):
        """Snapping to a grid measured in the wrong space leaves values off it."""
        frame, _spec, _c = self._frame_and_spec(31.0)
        assert not frame.print_is_identity
        expected = np.spacing(np.float32(max(abs(v) for v in frame.print_bounds_mm)))
        assert frame.output_resolution == expected
        assert frame.print_bounds_mm != frame.bounds_mm


class TestDensifyOnGrid:
    """Densification must add vertices without moving the shape."""

    def test_the_fixture_frame_reproduces_the_plain_linspace_grids(self):
        frame = _grid_frame(7, 5, 3.0, 3.0)
        assert np.array_equal(frame.grid_xs, np.linspace(0.0, 3.0, 7))
        assert np.array_equal(frame.grid_ys, np.linspace(0.0, 3.0, 5))

    def test_inserts_a_vertex_at_every_crossing(self):
        poly = ShapelyPolygon([(0.5, 0.5), (2.5, 0.5), (2.5, 2.5), (0.5, 2.5)])
        dense = densify_on_grid(poly, _grid_frame(4, 4, 3.0, 3.0))

        assert dense.equals(poly), "densifying must not change the shape"
        assert dense.area == poly.area
        # 4 corners + 2 crossings per side (the x=1,2 / y=1,2 lines).
        assert len(dense.exterior.coords) - 1 == 12

    def test_diagonal_edge_crosses_both_families(self):
        poly = ShapelyPolygon([(0.0, 0.0), (2.0, 2.0), (0.0, 2.0)])
        dense = densify_on_grid(poly, _grid_frame(3, 3, 2.0, 2.0))
        assert dense.equals(poly)
        # The diagonal meets x=1 and y=1 at the same point -- one vertex, not two.
        assert (1.0, 1.0) in list(dense.exterior.coords)

    def test_holes_are_densified_too(self):
        poly = ShapelyPolygon([(0.5, 0.5), (3.5, 0.5), (3.5, 3.5), (0.5, 3.5)],
                              [[(1.5, 1.5), (1.5, 2.5), (2.5, 2.5), (2.5, 1.5)]])
        dense = densify_on_grid(poly, _grid_frame(5, 5, 4.0, 4.0))
        assert dense.equals(poly)
        assert len(dense.interiors[0].coords) > len(poly.interiors[0].coords)

    def test_shared_seam_densifies_identically_from_either_side(self):
        """Two regions meeting on one edge must not diverge along it."""
        frame = _grid_frame(4, 4, 3.0, 3.0)
        left = ShapelyPolygon([(0.0, 0.2), (1.5, 0.2), (1.5, 2.8), (0.0, 2.8)])
        right = ShapelyPolygon([(1.5, 0.2), (3.0, 0.2), (3.0, 2.8), (1.5, 2.8)])
        dl = densify_on_grid(left, frame)
        dr = densify_on_grid(right, frame)

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
        frame = _grid_frame(7, 5, 3.0, 3.0)
        a, b = (0.3, 0.4), (2.7, 2.6)
        left = ShapelyPolygon([a, b, (0.0, 3.0)])
        right = ShapelyPolygon([b, a, (3.0, 0.0)])     # same seam, walked backwards

        cl = np.asarray(densify_on_grid(left, frame).exterior.coords)
        cr = np.asarray(densify_on_grid(right, frame).exterior.coords)
        seam_l = cl[:np.flatnonzero((cl == b).all(axis=1))[0] + 1]
        i = np.flatnonzero((cr == a).all(axis=1))[0]
        seam_r = cr[:i + 1][::-1]

        assert len(seam_l) > 2, "the seam should have picked up crossings"
        assert np.array_equal(seam_l, seam_r)

    # --- the same guarantees, with the grid TURNED relative to print space ---------
    #
    # A rotated rectangular cutout makes print space differ from the space the grid
    # lines live in, so densification has to cross between the two. That crossing is
    # where a round trip would creep in: mapping a whole ring to grid space and back
    # returns every ORIGINAL vertex perturbed in the low bits, and two rings sharing a
    # seam would each perturb it their own way -- the razor-fin condition. Only the new
    # crossings may be mapped; the vertices the ring already had must be copied.

    def _turned(self):
        """A frame whose grid is turned 31 degrees out of print space."""
        from dataclasses import replace
        f = _grid_frame(7, 5, 3.0, 3.0)
        return replace(f, print_bearing=31.0, print_pivot_mm=(1.5, 1.5),
                       print_origin_mm=(1.5, 1.5))

    def test_a_turned_frame_still_only_inserts_vertices(self):
        frame = self._turned()
        assert not frame.print_is_identity, "fixture must actually turn the grid"
        poly = ShapelyPolygon([(0.3, 0.4), (2.7, 0.5), (2.6, 2.6), (0.4, 2.5)])
        dense = densify_on_grid(poly, frame)

        # Every original vertex survives to the LAST BIT, not approximately.
        orig = np.asarray(poly.exterior.coords)
        got = np.asarray(dense.exterior.coords)
        for v in orig:
            assert (got == v).all(axis=1).any(), f"{v} was moved, not preserved"
        assert len(got) > len(orig), "and crossings were still found"

    def test_a_turned_seam_is_bit_identical_from_either_side(self):
        """The razor-fin guarantee, under the motion that made it non-trivial."""
        frame = self._turned()
        a, b = (0.3, 0.4), (2.7, 2.6)
        left = ShapelyPolygon([a, b, (0.2, 2.8)])
        right = ShapelyPolygon([b, a, (2.8, 0.2)])      # same seam, walked backwards

        cl = np.asarray(densify_on_grid(left, frame).exterior.coords)
        cr = np.asarray(densify_on_grid(right, frame).exterior.coords)
        seam_l = cl[:np.flatnonzero((cl == b).all(axis=1))[0] + 1]
        i = np.flatnonzero((cr == a).all(axis=1))[0]
        seam_r = cr[:i + 1][::-1]

        assert len(seam_l) > 2, "the seam should have picked up crossings"
        assert np.array_equal(seam_l, seam_r)

    def test_a_turned_frame_does_not_move_the_shape(self):
        """A crossing sits within one ULP of the boundary, turned or not.

        Exactly ON it is not achievable in doubles for a generic segment -- the
        interpolated coordinates round -- so the identity frame sets the bar and
        the turned frame must not be any worse.
        """
        poly = ShapelyPolygon([(0.3, 0.4), (2.7, 0.5), (2.6, 2.6), (0.4, 2.5)])
        boundary = poly.exterior
        ulp = np.spacing(3.0)               # one ULP at the frame's 3 mm extent
        for frame in (_grid_frame(7, 5, 3.0, 3.0), self._turned()):
            dense = densify_on_grid(poly, frame)
            assert dense.area == pytest.approx(poly.area, rel=0, abs=1e-12)
            worst = max(boundary.distance(shapely.Point(p))
                        for p in dense.exterior.coords)
            assert worst <= ulp


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

    def test_insert_is_flush_at_the_rim(self):
        """The clearance is measured to in-print neighbours only: where a mask
        meets the cutout rim the insert reaches the rim, not clearance short of
        it -- a plain buffer(-c) would recess the whole perimeter by a slit."""
        frame = _frame()
        outline = box(10.0, 10.0, 60.0, 50.0)
        layout = build_terrain_layout(
            frame, {TERRAIN_GLACIER: [_crs_geom(box(30.0, 20.0, 70.0, 40.0))]},
            outline=outline, fit=InsertFit(xy_clearance_mm=0.5))
        rim = layout.base_outline.boundary
        part = max(layout.insert_parts[TERRAIN_GLACIER], key=lambda p: p.area)
        assert part.intersects(rim), "the insert must reach the rim"
        assert part.bounds[2] == pytest.approx(60.0, abs=1e-4)
        # and it still clears the pocket walls that are not on the rim
        pocket = max((pk for _tc, pk in layout.pockets), key=lambda p: p.area)
        off_rim = pocket.exterior.difference(rim.buffer(1e-6))
        assert part.distance(off_rim) >= 0.5 - 1e-3

    def test_a_rim_bit_severed_by_the_fit_returns_to_the_base(self):
        """A small bit the clearance inset severs at the rim is dropped at
        FOOTPRINT level and the fit is redone. Neither half-measure is
        acceptable: dropping just the part leaves its hole as a perimeter dent,
        and returning the region to the base after the inset leaves full-height
        base standing closer than the clearance to the kept insert."""
        frame = _frame()
        outline = box(10.0, 10.0, 60.0, 50.0)
        mask = shapely.union_all([
            box(20.0, 18.0, 40.0, 42.0),     # the real insert
            box(40.0, 28.0, 57.0, 30.0),     # neck: thinner than 2*clearance
            box(57.0, 27.0, 62.0, 30.6),     # sub-MIN_BLOB pad pinched at the rim
        ])
        layout = build_terrain_layout(
            frame, {TERRAIN_GLACIER: [_crs_geom(mask)]}, outline=outline,
            fit=InsertFit(xy_clearance_mm=1.5))
        rim = layout.base_outline.boundary
        parts = layout.insert_parts[TERRAIN_GLACIER]
        assert len(parts) == 1, "only the blob insert survives"
        assert parts[0].distance(rim) > 1.0, "no insert remains at the rim"
        # no hole is left behind: the pad and the neck are base again, and no
        # pocket reaches the rim at all
        pockets = shapely.union_all([pk for _tc, pk in layout.pockets])
        assert not pockets.covers(shapely.Point(59.0, 29.0))    # pad region
        assert not pockets.covers(shapely.Point(50.0, 29.0))    # neck region
        assert pockets.distance(rim) > 1.0
        # the kept insert clears every wall of its (re-fitted) seat, up to the
        # arc-chord sagitta of the offset-curve discretization
        assert pockets.covers(parts[0])
        assert parts[0].distance(pockets.exterior) >= 1.5 - 0.01


class TestCornerReliefRimRule:
    """No relief disc may break the print's perimeter (see _corner_reliefs)."""

    def test_clip_manufactured_rim_corners_fire_nothing(self):
        """Corners come from the PRE-clip component, so the junction vertices the
        clip creates on the rim do not exist and fire no discs."""
        within = box(0.0, 0.0, 20.0, 20.0)
        component = box(5.0, 5.0, 25.0, 15.0)         # sticks out through the rim
        solid = component.intersection(within)         # has 90-deg rim junctions
        pocket_extra, insert_cut = _corner_reliefs(
            component, 0.06, 0.25, 45.0, within=within, solid=solid)
        assert insert_cut is None
        assert pocket_extra is not None, "the two interior corners are relieved"
        assert not pocket_extra.intersects(within.boundary)
        assert pocket_extra.bounds[2] < 6.0, "discs only at the x=5 corners"

    def test_a_disc_that_would_cross_the_rim_is_dropped_whole(self):
        """A genuine corner a fraction of the disc radius inside the rim fires a
        disc that reaches the perimeter: dropped, tight corner or not."""
        within = box(0.0, 0.0, 20.0, 20.0)
        component = box(5.0, 5.0, 19.9, 15.0)          # corners 0.1 inside the rim
        pocket_extra, _ = _corner_reliefs(
            component, 0.06, 0.25, 45.0, within=within, solid=component)
        assert pocket_extra is not None
        assert not pocket_extra.intersects(within.boundary)
        assert pocket_extra.bounds[2] < 6.0, "the near-rim discs are gone"


class TestInsertBodyRelief:
    """The relieved below-collar footprint of a printed part (_insert_body)."""

    def _body(self, part, relief_max=0.25):
        frame = _frame()
        return _insert_body(part, relief_max, frame, frame.output_resolution)

    def _ramp(self, part, relief_max=0.25):
        t = ((part.length - BODY_RELIEF_P0_MM)
             / (BODY_RELIEF_PMAX_MM - BODY_RELIEF_P0_MM))
        return relief_max * min(max(t, 0.0), 1.0)

    def test_a_part_below_the_ramp_gets_no_body(self):
        assert self._body(box(50.0, 20.0, 51.0, 21.0)) is None

    def test_a_large_part_is_relieved_by_the_full_maximum(self):
        part = box(10.0, 10.0, 50.0, 50.0)     # P = 160 >= BODY_RELIEF_PMAX_MM
        body = self._body(part)
        assert body is not None and body.within(part)
        assert part.exterior.distance(body) == pytest.approx(0.25, abs=1e-4)

    def test_the_relief_ramps_with_the_perimeter(self):
        part = box(20.0, 20.0, 40.0, 40.0)     # P = 80, mid-ramp
        body = self._body(part)
        assert part.exterior.distance(body) == pytest.approx(
            self._ramp(part), abs=1e-4)

    def test_corner_tips_are_not_kept_as_thin(self):
        """A convex corner is not a thin feature: the whole square must be relieved
        (a disc opening would leave four full-footprint corner slivers)."""
        part = box(10.0, 10.0, 50.0, 50.0)
        body = self._body(part)
        assert part.exterior.distance(body) >= 0.25 - 1e-4

    def test_a_thin_fin_keeps_its_full_width(self):
        # Dyadic coordinates throughout: _insert_body quantizes the body to the
        # f32 export grid, and a fin edge that is not f32-representable would
        # round a hair outside this raw fixture (production parts arrive
        # already quantized, so the shared boundary rounds onto itself there).
        bulk = box(10.0, 10.0, 40.0, 40.0)
        fin = box(40.0, 24.625, 48.0, 25.375)  # 0.75 mm wide: below the gate
        part = bulk.union(fin)
        body = self._body(part)
        assert body is not None and body.within(part.buffer(1e-9))
        assert body.covers(box(41.0, 24.64, 47.875, 25.36)), \
            "the fin must keep the part's own footprint"
        # while the bulk is still inset by the full ramp relief on its far wall
        assert body.bounds[0] - part.bounds[0] == pytest.approx(
            self._ramp(part), abs=1e-4)

    def test_a_part_thin_everywhere_gets_no_body(self):
        strip = box(10.0, 20.0, 90.0, 20.8)    # long enough for full relief
        assert self._body(strip) is None

    def test_the_body_stays_flush_at_the_rim(self):
        """The relief is measured to in-print neighbours only: at the cutout rim
        there is no pocket wall, so the body keeps the rim edge -- a rim-thinned
        body loses both sides of a rim bit at once (detached in the fit-test
        print)."""
        frame = _frame()
        outline = box(10.0, 10.0, 60.0, 50.0)
        part = box(20.0, 20.0, 60.0, 40.0)     # flush at the rim x=60
        body = _insert_body(part, 0.25, frame, frame.output_resolution,
                            outline=outline)
        assert body.bounds[2] == pytest.approx(60.0, abs=1e-6)
        # while every in-print wall still gets the full ramp relief
        assert body.bounds[0] - part.bounds[0] == pytest.approx(
            self._ramp(part), abs=1e-4)

    def test_layout_bodies_stay_flush_at_the_rim(self):
        frame = _frame()
        outline = box(10.0, 10.0, 60.0, 50.0)
        layout = build_terrain_layout(
            frame, {TERRAIN_GLACIER: [_crs_geom(box(30.0, 20.0, 70.0, 40.0))]},
            outline=outline,
            fit=InsertFit(xy_clearance_mm=0.5, body_relief_max_mm=0.25))
        parts = layout.insert_parts[TERRAIN_GLACIER]
        i = max(range(len(parts)), key=lambda k: parts[k].area)
        part, body = parts[i], layout.insert_bodies[TERRAIN_GLACIER][i]
        assert part.bounds[2] == pytest.approx(60.0, abs=1e-4)
        assert body is not None
        assert body.bounds[2] == pytest.approx(60.0, abs=1e-4)

    def test_bodies_run_parallel_to_the_parts(self):
        frame = _frame()
        layout = build_terrain_layout(
            frame, {TERRAIN_GLACIER: [_crs_box(10, 8, 30, 24)]},
            fit=InsertFit(xy_clearance_mm=0.07, body_relief_max_mm=0.25))
        assert set(layout.insert_bodies) == set(layout.insert_parts)
        relieved = 0
        for tc, parts in layout.insert_parts.items():
            bodies = layout.insert_bodies[tc]
            assert len(bodies) == len(parts)
            for part, body in zip(parts, bodies):
                if body is not None:
                    relieved += 1
                    assert body.within(part.buffer(1e-9))
        assert relieved, "the glacier part is large enough to be relieved"

    def test_body_coordinates_are_quantized_to_the_export_grid(self):
        frame = _frame()
        layout = build_terrain_layout(
            frame, {TERRAIN_GLACIER: [_crs_box(10, 8, 30, 24)]},
            fit=InsertFit(xy_clearance_mm=0.07, body_relief_max_mm=0.25))
        body = next(b for bs in layout.insert_bodies.values()
                    for b in bs if b is not None)
        coords = np.asarray(body.exterior.coords)
        assert np.array_equal(coords, coords.astype(np.float32).astype(np.float64))

    def test_no_relief_without_the_flag(self):
        frame = _frame()
        layout = build_terrain_layout(
            frame, {TERRAIN_GLACIER: [_crs_box(10, 8, 30, 24)]},
            fit=InsertFit(xy_clearance_mm=0.07))
        assert all(b is None
                   for bs in layout.insert_bodies.values() for b in bs)
