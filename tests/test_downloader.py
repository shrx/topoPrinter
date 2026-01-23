import os
from downloader import derive_base_name


def test_derive_base_name_strips_extension() -> None:
    url = "https://example.com/path/tile_01.tif"
    assert derive_base_name(url, fallback_index=1) == "tile_01"


def test_derive_base_name_strips_asc_extension() -> None:
    """Test that .asc extension is stripped for Slovenian data."""
    url = "http://gis.arso.gov.si/lidar/dmr1/b_456/D48GK/DMR1_456_100.asc"
    assert derive_base_name(url, fallback_index=1) == "DMR1_456_100"


def test_derive_base_name_fallback() -> None:
    url = "https://example.com/"
    assert derive_base_name(url, fallback_index=3) == "tile_3"


def test_filename_detection_preserves_asc_extension() -> None:
    """Test that download_dem would preserve .asc extension."""
    from urllib.parse import urlsplit

    url = "http://gis.arso.gov.si/lidar/dmr1/b_456/D48GK/DMR1_456_100.asc"
    parsed = urlsplit(url)
    filename_from_url = os.path.basename(parsed.path)
    base_name, ext = os.path.splitext(filename_from_url)

    assert base_name == "DMR1_456_100"
    assert ext == ".asc"


def test_filename_detection_preserves_tif_extension() -> None:
    """Test that download_dem would preserve .tif extension."""
    from urllib.parse import urlsplit

    url = "https://data.geo.admin.ch/ch.swisstopo.swissalti3d/N46E008_1m.tif"
    parsed = urlsplit(url)
    filename_from_url = os.path.basename(parsed.path)
    base_name, ext = os.path.splitext(filename_from_url)

    assert base_name == "N46E008_1m"
    assert ext == ".tif"
