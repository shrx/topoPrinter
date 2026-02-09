"""Tests for cutout argument parsing and rotation functionality."""
import numpy as np
import pytest
from unittest.mock import Mock

from dem_batch_to_stl import parse_args
from dem_processing import apply_cutout_mask


def test_parse_args_rect_corners():
    """Test parsing rectangle corners argument."""
    args = parse_args([
        '--url-list', 'test.txt',
        '--output-dir', 'out',
        '--rect-corners', '46.5,8.5,47.0,9.0'
    ])
    assert args.rect_corners == '46.5,8.5,47.0,9.0'
    assert args.center is None
    assert args.diameter is None
    assert args.side_length is None


def test_parse_args_rect_corners_with_bearing():
    """Test parsing rectangle corners with bearing."""
    args = parse_args([
        '--url-list', 'test.txt',
        '--output-dir', 'out',
        '--rect-corners', '46.5,8.5,47.0,9.0',
        '--bearing', '45'
    ])
    assert args.rect_corners == '46.5,8.5,47.0,9.0'
    assert args.bearing == 45.0


def test_parse_args_center_side_length_with_bearing():
    """Test parsing center + side-length with bearing."""
    args = parse_args([
        '--url-list', 'test.txt',
        '--output-dir', 'out',
        '--center', '46.75,8.75',
        '--side-length', '10',
        '--bearing', '90'
    ])
    assert args.center == '46.75,8.75'
    assert args.side_length == 10.0
    assert args.bearing == 90.0


def test_parse_args_default_bearing():
    """Test that bearing defaults to 0."""
    args = parse_args([
        '--url-list', 'test.txt',
        '--output-dir', 'out',
        '--center', '46.75,8.75',
        '--diameter', '5'
    ])
    assert args.bearing == 0.0


def test_parse_args_center_and_rect_corners_mutually_exclusive():
    """Test that --center and --rect-corners are mutually exclusive."""
    with pytest.raises(SystemExit):
        parse_args([
            '--url-list', 'test.txt',
            '--output-dir', 'out',
            '--center', '46.75,8.75',
            '--rect-corners', '46.5,8.5,47.0,9.0'
        ])


def test_apply_cutout_mask_rect_corners_no_rotation():
    """Test rectangular cutout with corners and no rotation."""
    # Create a simple test array
    arr = np.ones((100, 100), dtype=float)

    # Mock transform and CRS (we'll use a simple identity-like transform)
    # Assuming pixel coordinates map to projected coordinates 1:1
    # with origin at (0, 0)
    transform = Mock()
    transform.a = 1.0  # pixel width
    transform.b = 0.0
    transform.c = 0.0  # x origin
    transform.d = 0.0
    transform.e = -1.0  # pixel height (negative for north-up)
    transform.f = 100.0  # y origin

    crs = Mock()
    crs.__str__ = lambda self: "EPSG:3857"

    # Mock transformer that just returns the input coordinates
    # (assuming input is already in projected coordinates for simplicity)
    from unittest.mock import patch
    with patch('dem_processing.Transformer') as mock_transformer:
        mock_transformer.from_crs.return_value.transform = lambda lon, lat: (lon, lat)

        # Apply cutout with corners at (10, 10) and (50, 50)
        # This should mask out everything except the 40x40 box
        result = apply_cutout_mask(
            arr, transform, crs,
            None, None,  # no center
            rect_lat1=10, rect_lon1=10,
            rect_lat2=50, rect_lon2=50,
            bearing=0.0
        )

    # Check that some pixels are masked (NaN) and some are not
    assert np.isnan(result).any(), "Some pixels should be masked"
    assert np.isfinite(result).any(), "Some pixels should remain unmasked"


def test_apply_cutout_mask_center_square_no_rotation():
    """Test square cutout with center and no rotation."""
    arr = np.ones((100, 100), dtype=float)

    transform = Mock()
    transform.a = 1.0
    transform.b = 0.0
    transform.c = 0.0
    transform.d = 0.0
    transform.e = -1.0
    transform.f = 100.0

    crs = Mock()

    from unittest.mock import patch
    with patch('dem_processing.Transformer') as mock_transformer:
        mock_transformer.from_crs.return_value.transform = lambda lon, lat: (lon, lat)

        # Apply square cutout centered at (50, 50) with 20km side (20000m)
        result = apply_cutout_mask(
            arr, transform, crs,
            50.0, 50.0,  # center
            side_length_km=0.02,  # 20m in km (since our pixels are 1m)
            bearing=0.0
        )

    assert np.isnan(result).any(), "Some pixels should be masked"
    assert np.isfinite(result).any(), "Some pixels should remain unmasked"


def test_apply_cutout_mask_with_rotation():
    """Test that rotation parameter is accepted and doesn't crash."""
    arr = np.ones((100, 100), dtype=float)

    transform = Mock()
    transform.a = 1.0
    transform.b = 0.0
    transform.c = 0.0
    transform.d = 0.0
    transform.e = -1.0
    transform.f = 100.0

    crs = Mock()

    from unittest.mock import patch
    with patch('dem_processing.Transformer') as mock_transformer:
        mock_transformer.from_crs.return_value.transform = lambda lon, lat: (lon, lat)

        # Apply cutout with 45 degree rotation
        result = apply_cutout_mask(
            arr, transform, crs,
            50.0, 50.0,
            side_length_km=0.02,
            bearing=45.0
        )

    # Just verify it runs without error and returns valid data
    assert result.shape == arr.shape
    assert np.isnan(result).any() or np.isfinite(result).any()


def test_circular_cutout_ignores_rotation():
    """Test that circular cutouts are unaffected by rotation (as they should be)."""
    arr = np.ones((100, 100), dtype=float)

    transform = Mock()
    transform.a = 1.0
    transform.b = 0.0
    transform.c = 0.0
    transform.d = 0.0
    transform.e = -1.0
    transform.f = 100.0

    crs = Mock()

    from unittest.mock import patch
    with patch('dem_processing.Transformer') as mock_transformer:
        mock_transformer.from_crs.return_value.transform = lambda lon, lat: (lon, lat)

        # Apply circular cutout with rotation
        result1 = apply_cutout_mask(
            arr, transform, crs,
            50.0, 50.0,
            radius_km=0.01,  # 10m radius
            bearing=0.0
        )

        result2 = apply_cutout_mask(
            arr, transform, crs,
            50.0, 50.0,
            radius_km=0.01,
            bearing=45.0
        )

    # Results should be identical for circular cutout regardless of rotation
    assert np.array_equal(np.isnan(result1), np.isnan(result2)), \
        "Circular cutout should be unaffected by rotation"
