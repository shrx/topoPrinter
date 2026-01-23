"""Integration tests using real DEM files."""

import os
import tempfile
import shutil
from dem_processing import load_and_merge
from downloader import download_dem
from mesh_builder import dem_to_vertices_and_faces, save_stl

# Path to test fixtures
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
SWISS_TEST_FILE = os.path.join(FIXTURES_DIR, "swissalti3d_2019_2742-1234_2_2056_5728.tif")
SLOVENIA_TEST_FILE = os.path.join(FIXTURES_DIR, "GK1_399_45.asc")


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

    dem, px_size_x, px_size_y, _, _ = load_and_merge([SLOVENIA_TEST_FILE], downsample=1)

    assert dem.shape[0] > 0
    assert dem.shape[1] > 0
    assert px_size_x > 0
    assert px_size_y > 0


def test_swiss_geotiff_to_stl() -> None:
    """Test complete Swiss GeoTIFF to STL pipeline."""
    assert os.path.exists(SWISS_TEST_FILE), "Swiss test fixture missing"

    dem, px_size_x, px_size_y, _, _ = load_and_merge([SWISS_TEST_FILE], downsample=4)

    vertices, faces, max_z, water_faces = dem_to_vertices_and_faces(
        dem, px_size_x, px_size_y,
        x_size_mm=100.0,
        max_height_mm=30.0,
        z_exaggeration=1.0,
        base_thickness_mm=2.0,
        lake_range_percent=0.0,
        lake_lowering_mm=0.0,
    )

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
    dem, px_size_x, px_size_y, _, _ = load_and_merge([SLOVENIA_TEST_FILE], downsample=8)

    vertices, faces, max_z, water_faces = dem_to_vertices_and_faces(
        dem, px_size_x, px_size_y,
        x_size_mm=100.0,
        max_height_mm=30.0,
        z_exaggeration=1.0,
        base_thickness_mm=2.0,
        lake_range_percent=0.0,
        lake_lowering_mm=0.0,
    )

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
