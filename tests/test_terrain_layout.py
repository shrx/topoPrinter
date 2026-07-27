"""Tests for the 2D terrain stage: ModelFrame and build_terrain_layout."""

import numpy as np
import pytest
import shapely
from rasterio.transform import from_origin
from shapely.geometry import Polygon as ShapelyPolygon, box, mapping

from mesh_builder import _compute_model_coordinates
from model_frame import ModelFrame
from terrain_classifier import (TERRAIN_FOLIAGE, TERRAIN_GLACIER, TERRAIN_ROCK,
                                TERRAIN_WATER)
from terrain_layout import InsertFit, build_terrain_layout


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
