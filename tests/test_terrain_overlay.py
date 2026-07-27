"""Tests for the live insert-prism path in mesh_builder.

These replace an older suite that exercised ``_build_overlay_component`` /
``_compute_component_flat_z`` -- a solid-boolean path the 2D-first rewrite retired
and which no longer existed by the time the terrain pipeline was split into
``terrain_layout`` (2D) + ``mesh_builder`` (extrusion).
"""

import numpy as np
import pytest
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

from mesh_builder import (
    _compute_model_coordinates,
    _dem_sampler,
    _dem_min_over,
    _region_top_surface,
    build_region_prism_fast,
)


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


def _flat(z):
    return lambda xy: np.full(len(xy), float(z))


def _build(poly, thickness=2.0):
    sample, xs, ys, _ = _make_grid()
    zmin = _dem_min_over(poly, sample, xs, ys)
    if zmin is None:
        return None
    return build_region_prism_fast(poly, sample, _flat(max(zmin - thickness, 0.01)),
                                   xs, ys)


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
        assert _dem_min_over(tiny, sample, xs, ys) is not None
        result = _build(tiny, thickness=1.0)
        assert result is not None and result.is_watertight

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
        xy, faces = _region_top_surface(poly, xs, ys)
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
        zmin = _dem_min_over(poly, sample, xs, ys)
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
