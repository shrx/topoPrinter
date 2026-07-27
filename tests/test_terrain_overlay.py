"""Tests for the live insert-prism path in mesh_builder.

These replace an older suite that exercised ``_build_overlay_component`` /
``_compute_component_flat_z`` -- a solid-boolean path the 2D-first rewrite retired
and which no longer existed by the time the terrain pipeline was split into
``terrain_layout`` (2D) + ``mesh_builder`` (extrusion).
"""

import numpy as np
import pytest
import shapely
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

from mesh_builder import (
    _assign_pockets,
    _compute_model_coordinates,
    _dem_sampler,
    _dem_min_over,
    _region_top_surface,
    build_region_prism_fast,
)
from terrain_layout import densify_on_grid


def _make_grid(rows=20, cols=20):
    """A simple DEM plus the ascending model grid the prism builder wants."""
    dem = np.random.default_rng(42).uniform(100, 200, (rows, cols))
    X, Y, z_surface_mm, _valid, _lake, model_y_mm = _compute_model_coordinates(
        dem, px_size_x=1.0, px_size_y=1.0, x_size_mm=100.0,
        max_height_mm=50.0, z_exaggeration=1.0, base_thickness_mm=2.0,
        use_true_scale=False,
    )
    xs = np.asarray(X[0, :], dtype=float)
    ys = np.asarray(Y[::-1, 0], dtype=float)
    sample = _dem_sampler(np.asarray(z_surface_mm[::-1, :], dtype=float), xs, ys)
    return sample, xs, ys, model_y_mm


# The float32 export step at this fixture's scale; grid points nearer than this to
# a boundary are not distinguishable from it once written.
RESOLUTION = float(np.spacing(np.float32(100.0)))


def _flat(z):
    return lambda xy: np.full(len(xy), float(z))


def _build(poly, thickness=2.0):
    sample, xs, ys, _ = _make_grid()
    zmin = _dem_min_over(poly, sample, xs, ys, RESOLUTION)
    if zmin is None:
        return None
    return build_region_prism_fast(poly, sample, _flat(max(zmin - thickness, 0.01)),
                                   xs, ys, RESOLUTION)


class TestBuildRegionPrism:
    def test_region_smaller_than_a_grid_cell_still_meshes(self):
        """A region containing no grid point must still be built.

        The cell-corner classifier this path used to have dropped it, and dropped
        any sub-cell excursion of a large region too, while the pocket it seats in
        was cut exactly -- so insert and seat disagreed by up to a cell.
        """
        sample, xs, ys, _ = _make_grid()
        pitch = float(xs[1] - xs[0])
        centre = (float(xs[5]) + 0.5 * pitch, float(ys[5]) + 0.5 * pitch)
        tiny = ShapelyPolygon([
            (centre[0] - 0.1 * pitch, centre[1] - 0.1 * pitch),
            (centre[0] + 0.1 * pitch, centre[1] - 0.1 * pitch),
            (centre[0] + 0.1 * pitch, centre[1] + 0.1 * pitch),
            (centre[0] - 0.1 * pitch, centre[1] + 0.1 * pitch),
        ])
        assert _dem_min_over(tiny, sample, xs, ys, RESOLUTION) is not None
        result = _build(tiny, thickness=1.0)
        assert result is not None and result.is_watertight

    def test_grid_point_within_export_resolution_of_the_edge_is_dropped(self):
        """Such a point flattens a sliver into a zero-area face on export.

        Its coordinate on one axis rounds onto the boundary's while the other does
        not, so the triangle it anchors collapses. Dropping it is safe: boundaries
        are densified on these very grid lines, so it adds no sample the boundary
        vertices do not already carry.
        """
        _, xs, ys, _ = _make_grid()
        # A rectangle whose left edge is a hair to the right of grid column 4, and
        # whose other edges sit well clear of any grid line.
        eps = 0.25 * RESOLUTION
        poly = ShapelyPolygon([(xs[4] - eps, ys[3] + 0.5), (xs[8] + 0.5, ys[3] + 0.5),
                               (xs[8] + 0.5, ys[8] - 0.5), (xs[4] - eps, ys[8] - 0.5)])
        # A grid point sits at exactly xs[4]; the polygon's own corners sit at
        # xs[4] - eps, so an exact test tells them apart.
        loose = _region_top_surface(poly, xs, ys, 0.1 * eps)[0]
        assert (loose[:, 0] == xs[4]).sum() > 0, "fixture must include the column"

        xy, faces = _region_top_surface(poly, xs, ys, RESOLUTION)
        assert (xy[:, 0] == xs[4]).sum() == 0, "the hugging column must be dropped"
        # The area is still exact: dropping a sample does not move the boundary.
        a, b, c = xy[faces[:, 0]], xy[faces[:, 1]], xy[faces[:, 2]]
        area = float(np.abs(0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                                   - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))).sum())
        assert area == pytest.approx(poly.area)

    def test_sub_cell_excursion_of_a_large_region_is_kept(self):
        """Area is preserved exactly, not sampled at the grid corners."""
        _, xs, ys, _ = _make_grid()
        pitch = float(xs[1] - xs[0])
        big = ShapelyPolygon([(xs[3], ys[3]), (xs[9], ys[3]),
                              (xs[9], ys[9]), (xs[3], ys[9])])
        bump = ShapelyPolygon([
            (xs[9], ys[5] + 0.2 * pitch), (xs[9] + 0.4 * pitch, ys[5] + 0.2 * pitch),
            (xs[9] + 0.4 * pitch, ys[5] + 0.6 * pitch), (xs[9], ys[5] + 0.6 * pitch),
        ])
        poly = unary_union([big, bump])
        xy, faces = _region_top_surface(poly, xs, ys, RESOLUTION)
        a, b, c = xy[faces[:, 0]], xy[faces[:, 1]], xy[faces[:, 2]]
        area = float(np.abs(0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                                   - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))).sum())
        assert area == pytest.approx(poly.area)

    def test_produces_mesh_for_polygon_inside_dem(self):
        _, _, _, model_y_mm = _make_grid()
        poly = ShapelyPolygon([(30, model_y_mm * 0.3), (70, model_y_mm * 0.3),
                               (70, model_y_mm * 0.7), (30, model_y_mm * 0.7)])
        result = _build(poly)
        assert result is not None
        assert len(result.faces) > 0

    def test_prism_is_watertight(self):
        _, _, _, model_y_mm = _make_grid()
        poly = ShapelyPolygon([(20, model_y_mm * 0.2), (80, model_y_mm * 0.2),
                               (80, model_y_mm * 0.8), (20, model_y_mm * 0.8)])
        result = _build(poly)
        assert result is not None
        assert result.is_watertight, "insert prism should be watertight"

    def test_flat_bottom_sits_one_thickness_below_the_surface_minimum(self):
        """Flat mode floors the insert at (surface min - thickness), above zero."""
        sample, xs, ys, model_y_mm = _make_grid()
        thickness = 1.0
        poly = ShapelyPolygon([(20, model_y_mm * 0.2), (80, model_y_mm * 0.2),
                               (80, model_y_mm * 0.8), (20, model_y_mm * 0.8)])
        result = _build(poly, thickness)
        assert result is not None
        zmin = _dem_min_over(poly, sample, xs, ys, RESOLUTION)
        assert zmin - thickness > 0.01, "fixture must not hit the floor clamp"
        assert float(np.min(result.vertices[:, 2])) == pytest.approx(zmin - thickness)

    def test_referenced_vertices_stay_within_the_polygon_bounds(self):
        """Only vertices the faces reference are on the printed body.

        _region_top_surface also emits the OUTSIDE grid corners of boundary cells --
        needed so the walls close -- and those sit beyond the polygon by up to one
        grid pitch. No face references them, so they are not part of the surface.
        """
        _, _, _, model_y_mm = _make_grid()
        poly = ShapelyPolygon([(30, model_y_mm * 0.3), (70, model_y_mm * 0.3),
                               (70, model_y_mm * 0.7), (30, model_y_mm * 0.7)])
        result = _build(poly)
        assert result is not None
        verts = result.vertices[np.unique(result.faces)]
        assert float(np.min(verts[:, 0])) >= 30.0 - 0.1
        assert float(np.max(verts[:, 0])) <= 70.0 + 0.1
        assert float(np.min(verts[:, 1])) >= model_y_mm * 0.3 - 0.1
        assert float(np.max(verts[:, 1])) <= model_y_mm * 0.7 + 0.1


def _exact_assign(cen, pockets):
    """The per-triangle query the raster path replaces."""
    out = np.full(len(cen), -1, np.int64)
    hits = shapely.STRtree(pockets).query(shapely.points(cen), predicate="within")
    if hits.size:
        o = np.lexsort((hits[1], hits[0]))
        pc, pp = hits[0][o], hits[1][o]
        fst = np.ones(len(pc), bool)
        fst[1:] = pc[1:] != pc[:-1]
        out[pc[fst]] = pp[fst]
    return out


class TestAssignPockets:
    """The raster shortcut must agree with the exact query, triangle for triangle."""

    def _pockets(self, xs, ys):
        # Deliberately awkward: a rotated square, a ring with a hole, and a small
        # blob that overlaps the first so precedence is exercised. None of their
        # edges are axis-aligned or grid-aligned.
        cx, cy = xs[10], ys[10]
        rot = ShapelyPolygon([(cx - 3.1, cy), (cx, cy - 2.7),
                              (cx + 3.3, cy), (cx, cy + 2.9)])
        ring = ShapelyPolygon(
            [(xs[3] + 0.3, ys[3] + 0.7), (xs[8] - 0.4, ys[3] + 0.2),
             (xs[8] + 0.1, ys[8] - 0.6), (xs[3] - 0.2, ys[8] + 0.3)],
            [[(xs[5], ys[5]), (xs[6] + 0.2, ys[5] - 0.1),
              (xs[6] - 0.1, ys[6]), (xs[5] + 0.3, ys[6] + 0.2)]])
        blob = ShapelyPolygon([(cx + 1.0, cy + 1.0), (cx + 4.2, cy + 0.6),
                               (cx + 3.8, cy + 3.4), (cx + 0.7, cy + 3.1)])
        # The precondition: the layout densifies every boundary on the grid lines.
        return [densify_on_grid(p, xs, ys) for p in (rot, ring, blob)]

    def test_matches_the_exact_query(self):
        _, xs, ys, _ = _make_grid()
        pockets = self._pockets(xs, ys)
        rng = np.random.default_rng(7)
        cen = np.column_stack((rng.uniform(xs[0], xs[-1], 20000),
                               rng.uniform(ys[0], ys[-1], 20000)))
        assert np.array_equal(_assign_pockets(cen, pockets, xs, ys),
                              _exact_assign(cen, pockets))

    def test_overlap_goes_to_the_first_pocket(self):
        """TerrainLayout.pockets is ordered and the base solid honours that order."""
        _, xs, ys, _ = _make_grid()
        pockets = self._pockets(xs, ys)
        overlap = pockets[0].intersection(pockets[2])
        assert not overlap.is_empty, "fixture must actually overlap"
        pt = np.asarray(overlap.representative_point().coords)
        assert _assign_pockets(pt, pockets, xs, ys)[0] == 0

    def test_a_point_in_a_hole_belongs_to_no_pocket(self):
        _, xs, ys, _ = _make_grid()
        pockets = self._pockets(xs, ys)
        hole = ShapelyPolygon(pockets[1].interiors[0])
        pt = np.asarray(hole.representative_point().coords)
        assert _assign_pockets(pt, pockets, xs, ys)[0] == -1
