from downloader import derive_base_name


def test_derive_base_name_strips_extension() -> None:
    url = "https://example.com/path/tile_01.tif"
    assert derive_base_name(url, fallback_index=1) == "tile_01"


def test_derive_base_name_fallback() -> None:
    url = "https://example.com/"
    assert derive_base_name(url, fallback_index=3) == "tile_3"
