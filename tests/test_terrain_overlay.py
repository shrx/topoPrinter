"""Tests for vector-boundary terrain overlay building in mesh_builder."""

import numpy as np
import pytest
import trimesh
from shapely.geometry import Polygon as ShapelyPolygon

from mesh_builder import (
    _build_overlay_component,
    _compute_component_flat_z,
    _compute_model_coordinates,
)


def _make_dem_and_coords(rows=20, cols=20):
    """Create a simple DEM and compute model coordinates."""
    dem = np.random.default_rng(42).uniform(100, 200, (rows, cols))
    X, Y, z_surface_mm, valid_mask, _, model_y_mm = _compute_model_coordinates(
        dem, px_size_x=1.0, px_size_y=1.0, x_size_mm=100.0,
        max_height_mm=50.0, z_exaggeration=1.0, base_thickness_mm=2.0,
        use_true_scale=False,
    )
    return dem, X, Y, z_surface_mm, valid_mask, model_y_mm


def _build_with_auto_flat_z(poly, z_surface_mm, X, Y, valid_mask, thickness):
    """Helper: compute flat_z then build overlay component."""
    flat_z = _compute_component_flat_z(poly, z_surface_mm, X, Y, valid_mask, thickness)
    if flat_z is None:
        return None
    return _build_overlay_component(poly, flat_z, z_surface_mm, X, Y, valid_mask, thickness)


class TestBuildOverlayComponent:
    def test_returns_none_for_polygon_outside_dem(self):
        """Polygon entirely outside the DEM extent returns None."""
        _, X, Y, z_surface_mm, valid_mask, _ = _make_dem_and_coords()
        poly = ShapelyPolygon([(-50, -50), (-40, -50), (-40, -40), (-50, -40)])
        result = _build_with_auto_flat_z(poly, z_surface_mm, X, Y, valid_mask, 2.0)
        assert result is None

    def test_produces_mesh_for_polygon_inside_dem(self):
        """A polygon covering part of the DEM should produce a mesh."""
        _, X, Y, z_surface_mm, valid_mask, model_y_mm = _make_dem_and_coords()
        poly = ShapelyPolygon([
            (30, model_y_mm * 0.3),
            (70, model_y_mm * 0.3),
            (70, model_y_mm * 0.7),
            (30, model_y_mm * 0.7),
        ])
        result = _build_with_auto_flat_z(poly, z_surface_mm, X, Y, valid_mask, 2.0)
        assert result is not None
        assert len(result.faces) > 0

    def test_overlay_is_watertight(self):
        """The overlay mesh should be watertight."""
        _, X, Y, z_surface_mm, valid_mask, model_y_mm = _make_dem_and_coords()
        poly = ShapelyPolygon([
            (20, model_y_mm * 0.2),
            (80, model_y_mm * 0.2),
            (80, model_y_mm * 0.8),
            (20, model_y_mm * 0.8),
        ])
        result = _build_with_auto_flat_z(poly, z_surface_mm, X, Y, valid_mask, 2.0)
        assert result is not None
        assert result.is_watertight, "Overlay mesh should be watertight"

    def test_flat_bottom(self):
        """The bottom of the overlay should be a flat plane."""
        _, X, Y, z_surface_mm, valid_mask, model_y_mm = _make_dem_and_coords()
        thickness = 3.0
        poly = ShapelyPolygon([
            (20, model_y_mm * 0.2),
            (80, model_y_mm * 0.2),
            (80, model_y_mm * 0.8),
            (20, model_y_mm * 0.8),
        ])
        result = _build_with_auto_flat_z(poly, z_surface_mm, X, Y, valid_mask, thickness)
        assert result is not None
        min_z = float(np.min(result.vertices[:, 2]))
        assert min_z > 0

    def test_vector_boundary(self):
        """Vertices should not extend beyond the polygon boundary."""
        _, X, Y, z_surface_mm, valid_mask, model_y_mm = _make_dem_and_coords()
        poly = ShapelyPolygon([
            (30, model_y_mm * 0.3),
            (70, model_y_mm * 0.3),
            (70, model_y_mm * 0.7),
            (30, model_y_mm * 0.7),
        ])
        result = _build_with_auto_flat_z(poly, z_surface_mm, X, Y, valid_mask, 2.0)
        assert result is not None
        verts = result.vertices
        assert float(np.min(verts[:, 0])) >= 30.0 - 0.1
        assert float(np.max(verts[:, 0])) <= 70.0 + 0.1
        assert float(np.min(verts[:, 1])) >= model_y_mm * 0.3 - 0.1
        assert float(np.max(verts[:, 1])) <= model_y_mm * 0.7 + 0.1
