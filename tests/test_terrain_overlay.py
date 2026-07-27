"""Tests for the live insert-prism path in mesh_builder.

These replace an older suite that exercised ``_build_overlay_component`` /
``_compute_component_flat_z`` -- a solid-boolean path the 2D-first rewrite retired
and which no longer existed by the time the terrain pipeline was split into
``terrain_layout`` (2D) + ``mesh_builder`` (extrusion).
"""

import numpy as np
import pytest
from shapely.geometry import Polygon as ShapelyPolygon

from mesh_builder import (
    _compute_model_coordinates,
    _dem_sampler,
    _dem_min_over,
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
    def test_returns_none_for_polygon_outside_dem(self):
        """A polygon entirely off the grid covers no cell, so there is no prism."""
        poly = ShapelyPolygon([(-50, -50), (-40, -50), (-40, -40), (-50, -40)])
        assert _build(poly) is None

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
