"""The two things the mesh stage owes a print that has no terrain masks at all.

There is one path through the pipeline now: masks -> 2D layout -> extrusion. A
plain relief block is that path with no mask provider, so it is worth pinning that
it really does come out as ONE watertight body spanning the whole grid, and that
the water class -- the one feature the old raster path carried on its own -- still
prints as a lowered, flat-topped pool.
"""
import numpy as np
import pytest
import trimesh
from rasterio.transform import from_origin

from masks import TERRAIN_WATER
from model_frame import ModelFrame
from mesh_builder import build_terrain_meshes
from terrain_layout import InsertFit, build_terrain_layout

ROWS, COLS, PX = 30, 40, 10.0
X_SIZE_MM = 100.0
BASE_MM = 6.0
THICK_MM = 2.0
Z_CLEAR_MM = 0.2


def _frame():
    return ModelFrame.from_dem((ROWS, COLS), PX, PX, X_SIZE_MM,
                               from_origin(500000.0, 5000000.0, PX, PX), "EPSG:32633")


def _bowl():
    """A DEM whose lowest ground is a basin in the middle -- i.e. a lake."""
    yy, xx = np.mgrid[0:ROWS, 0:COLS]
    return 100.0 + 40.0 * np.hypot(xx - COLS / 2, yy - ROWS / 2) / (COLS / 2)


def _build(class_geometries, **mesh_kwargs):
    frame = _frame()
    layout = build_terrain_layout(
        frame, class_geometries,
        fit=InsertFit(xy_clearance_mm=0.07, corner_relief_mm=0.25,
                      corner_min_angle_deg=90.0))
    meshes = build_terrain_meshes(layout, frame, _bowl(), 30.0, 1.0, BASE_MM,
                                  THICK_MM, insert_z_clearance_mm=Z_CLEAR_MM,
                                  **mesh_kwargs)
    return layout, meshes


def _mesh(data):
    verts, faces, _max_z = data
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


class TestPlainReliefBlock:
    """No mask provider -> one base layer covering the whole grid."""

    def test_single_watertight_body_over_the_whole_grid(self):
        layout, meshes = _build({})
        assert layout.pockets == []
        assert [m for m in meshes.values() if m is not None] == [meshes["rock"]]

        block = _mesh(meshes["rock"])
        assert block.is_watertight
        assert block.volume > 0
        lo, hi = block.bounds
        assert (lo[0], lo[1], lo[2]) == pytest.approx((0.0, 0.0, 0.0))
        assert (hi[0], hi[1]) == pytest.approx((X_SIZE_MM, _frame().model_y_mm))

    def test_surface_carries_every_dem_sample(self):
        """The plate is the grid: one top and one bottom vertex per DEM sample."""
        _layout, meshes = _build({})
        assert len(meshes["rock"][0]) == 2 * ROWS * COLS

    def test_the_lowest_ground_sits_at_the_base_thickness(self):
        _layout, meshes = _build({})
        verts = meshes["rock"][0]
        top = verts[verts[:, 2] > 0.0]
        assert top[:, 2].min() == pytest.approx(BASE_MM, abs=1e-4)


def _water_geoms(radius_m=80.0):
    """A round water mask centred on the basin, as GeoJSON in the DEM's CRS."""
    cx = 500000.0 + COLS / 2 * PX
    cy = 5000000.0 - ROWS / 2 * PX
    ang = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    ring = [(cx + radius_m * np.cos(a), cy + radius_m * np.sin(a)) for a in ang]
    return {TERRAIN_WATER: [{"type": "Polygon", "coordinates": [ring + [ring[0]]]}]}


class TestWaterLowering:
    """Water is an ordinary insert; the lowering sinks the whole column."""

    def test_flush_water_is_draped_like_any_other_insert(self):
        _layout, meshes = _build(_water_geoms(), water_lowering_mm=0.0)
        verts = meshes["water"][0]
        top = verts[verts[:, 2] > verts[:, 2].min()]
        assert len(np.unique(top[:, 2].round(4))) > 1, "should follow the DEM"

    def test_lowered_water_is_one_flat_plane(self):
        _layout, sunk = _build(_water_geoms(), water_lowering_mm=1.0)
        verts = sunk["water"][0]
        assert len(np.unique(verts[:, 2].round(4))) == 2, "flat top, flat bottom"

    def test_lowering_moves_the_body_without_thinning_it(self):
        _layout, flush = _build(_water_geoms(), water_lowering_mm=0.0)
        _layout2, sunk = _build(_water_geoms(), water_lowering_mm=1.0)
        underside = sunk["water"][0][:, 2].min()
        surface = sunk["water"][0][:, 2].max()
        assert surface - underside == pytest.approx(THICK_MM - Z_CLEAR_MM, abs=1e-3)
        # A flush pool's underside is the same flat face, one drop higher.
        assert underside == pytest.approx(flush["water"][0][:, 2].min() - 1.0,
                                          abs=1e-3)

    def test_the_surface_sits_the_drop_below_the_ground_it_covers(self):
        _layout, flush = _build(_water_geoms(), water_lowering_mm=0.0)
        _layout2, sunk = _build(_water_geoms(), water_lowering_mm=1.0)
        ground = flush["water"][0][:, 2].min() + (THICK_MM - Z_CLEAR_MM)
        assert sunk["water"][0][:, 2].max() == pytest.approx(ground - 1.0, abs=1e-3)

    def test_pocket_floor_follows_the_water_down(self):
        """The insert keeps its designed z clearance over the sunk floor."""
        _layout, sunk = _build(_water_geoms(), water_lowering_mm=1.0)
        base = _mesh(sunk["rock"])
        assert base.is_watertight
        floor = sunk["water"][0][:, 2].min() - Z_CLEAR_MM
        # The recess floor is a plateau of the base plate: vertices sit on it.
        assert np.isclose(base.vertices[:, 2], floor, atol=1e-3).any()

    def test_water_and_base_are_both_watertight(self):
        _layout, sunk = _build(_water_geoms(), water_lowering_mm=1.0)
        assert _mesh(sunk["water"]).is_watertight
        assert _mesh(sunk["rock"]).is_watertight

    def test_a_drop_below_the_model_floor_is_refused(self):
        with pytest.raises(ValueError, match="lake-lowering"):
            _build(_water_geoms(), water_lowering_mm=BASE_MM)
