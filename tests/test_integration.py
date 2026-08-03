"""Integration tests using real DEM files."""

import os
import tempfile
import shutil
from dem_processing import load_and_merge
from downloader import download_dem
from mesh_builder import build_terrain_meshes, save_stl
from model_frame import ModelFrame
from sources import prepare_dem_files
from terrain_layout import build_terrain_layout

# Path to test fixtures
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
SWISS_TEST_FILE = os.path.join(FIXTURES_DIR, "swissalti3d_2019_2742-1234_2_2056_5728.tif")
SLOVENIA_TEST_FILE = os.path.join(FIXTURES_DIR, "GK1_399_45.asc")


def _plain_block(dem, px_size_x, px_size_y, crs, transform, x_size_mm=100.0):
    """The whole pipeline below the CLI, with no mask provider: one relief block.

    Same two calls dem_batch_to_stl makes -- layout, then extrusion -- so these
    tests exercise the real path a plain print takes, not a shortcut.
    """
    frame = ModelFrame.from_dem(dem.shape, px_size_x, px_size_y, x_size_mm,
                                transform, crs)
    layout = build_terrain_layout(frame, {})
    return build_terrain_meshes(layout, frame, dem, max_height_mm=30.0,
                                z_exaggeration=1.0, base_thickness_mm=2.0,
                                overlay_thickness_mm=2.0)[layout.base_name]


def test_swiss_geotiff_loads() -> None:
    """Test that Swiss GeoTIFF file loads and processes."""
    assert os.path.exists(SWISS_TEST_FILE), "Swiss test fixture missing"

    dem, px_size_x, px_size_y, _, _ = load_and_merge([SWISS_TEST_FILE], downsample=1)

    assert dem.shape[0] > 0
    assert dem.shape[1] > 0
    assert px_size_x > 0
    assert px_size_y > 0


def test_slovenian_asc_loads() -> None:
    """Test that Slovenian ASC file loads and processes."""
    assert os.path.exists(SLOVENIA_TEST_FILE), "Slovenian test fixture missing"

    prepared = prepare_dem_files([SLOVENIA_TEST_FILE])
    dem, px_size_x, px_size_y, _, _ = load_and_merge(prepared, downsample=1)

    assert dem.shape[0] > 0
    assert dem.shape[1] > 0
    assert px_size_x > 0
    assert px_size_y > 0


def test_swiss_geotiff_to_stl() -> None:
    """Test complete Swiss GeoTIFF to STL pipeline."""
    assert os.path.exists(SWISS_TEST_FILE), "Swiss test fixture missing"

    dem, px_size_x, px_size_y, crs, transform = load_and_merge(
        [SWISS_TEST_FILE], downsample=4)

    vertices, faces, max_z = _plain_block(dem, px_size_x, px_size_y, crs, transform)

    assert vertices.shape[0] > 0
    assert faces.shape[0] > 0
    assert max_z > 0

    # Test STL export
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        save_stl(vertices, faces, tmp_path)
        assert os.path.exists(tmp_path)
        assert os.path.getsize(tmp_path) > 0
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_slovenian_asc_to_stl() -> None:
    """Test complete Slovenian ASC to STL pipeline."""
    assert os.path.exists(SLOVENIA_TEST_FILE), "Slovenian test fixture missing"

    # Use higher downsample for the larger Slovenian file
    prepared = prepare_dem_files([SLOVENIA_TEST_FILE])
    dem, px_size_x, px_size_y, crs, transform = load_and_merge(prepared, downsample=8)

    vertices, faces, max_z = _plain_block(dem, px_size_x, px_size_y, crs, transform)

    assert vertices.shape[0] > 0
    assert faces.shape[0] > 0
    assert max_z > 0

    # Test STL export
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        save_stl(vertices, faces, tmp_path)
        assert os.path.exists(tmp_path)
        assert os.path.getsize(tmp_path) > 0
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
