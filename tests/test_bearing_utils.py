"""Tests for the bearing-frame rotation used by every rotated cutout.

Bearing is degrees CLOCKWISE from North, and the DEM axes are (Easting, Northing).
That combination is a left-handed-looking setup that is easy to get subtly wrong --
a sign slip still round-trips, still preserves lengths, and only shows up as a
cutout rotated the wrong way. So the tests below pin the two BASIS DIRECTIONS
explicitly rather than only checking that the pair of functions are inverses.
"""

import numpy as np
import pytest

from bearing_utils import rotate_from_bearing_frame, rotate_to_bearing_frame


BEARINGS = [0.0, 30.0, 45.0, 90.0, 135.0, 180.0, 270.0, 359.0]


class TestBasisDirections:
    """The two local axes, stated as directions in (Easting, Northing)."""

    @pytest.mark.parametrize("bearing_deg", BEARINGS)
    def test_bearing_direction_is_pure_along(self, bearing_deg):
        """A step of 1 in the bearing direction (sin, cos) is along=1, perp=0."""
        b = np.radians(bearing_deg)
        de, dn = np.sin(b), np.cos(b)
        perp, along = rotate_to_bearing_frame(de, dn, b)
        assert perp == pytest.approx(0.0, abs=1e-12)
        assert along == pytest.approx(1.0)

    @pytest.mark.parametrize("bearing_deg", BEARINGS)
    def test_perpendicular_direction_is_pure_perp(self, bearing_deg):
        """The bearing+90 direction (cos, -sin) is perp=1, along=0."""
        b = np.radians(bearing_deg)
        de, dn = np.cos(b), -np.sin(b)
        perp, along = rotate_to_bearing_frame(de, dn, b)
        assert perp == pytest.approx(1.0)
        assert along == pytest.approx(0.0, abs=1e-12)

    def test_north_bearing_is_the_identity(self):
        """Bearing 0: the local frame IS (Easting, Northing)."""
        perp, along = rotate_to_bearing_frame(3.0, 7.0, 0.0)
        assert (perp, along) == pytest.approx((3.0, 7.0))

    def test_east_bearing_maps_easting_to_along(self):
        """Bearing 90 points along +Easting, so its perp axis is -Northing."""
        perp, along = rotate_to_bearing_frame(3.0, 7.0, np.radians(90.0))
        assert along == pytest.approx(3.0)
        assert perp == pytest.approx(-7.0)

    def test_south_bearing_negates_both(self):
        perp, along = rotate_to_bearing_frame(3.0, 7.0, np.radians(180.0))
        assert (perp, along) == pytest.approx((-3.0, -7.0))


class TestRotationProperties:
    @pytest.mark.parametrize("bearing_deg", BEARINGS)
    def test_round_trip_restores_the_offsets(self, bearing_deg):
        b = np.radians(bearing_deg)
        de, dn = 12.5, -4.25
        perp, along = rotate_to_bearing_frame(de, dn, b)
        assert rotate_from_bearing_frame(perp, along, b) == pytest.approx((de, dn))

    @pytest.mark.parametrize("bearing_deg", BEARINGS)
    def test_length_is_preserved(self, bearing_deg):
        """A rotation, not a shear or a scale -- cutout radii depend on this."""
        b = np.radians(bearing_deg)
        de, dn = 12.5, -4.25
        perp, along = rotate_to_bearing_frame(de, dn, b)
        assert np.hypot(perp, along) == pytest.approx(np.hypot(de, dn))

    @pytest.mark.parametrize("bearing_deg", BEARINGS)
    def test_matrix_is_a_proper_rotation(self, bearing_deg):
        """det = +1, not -1.

        A reflection would also preserve lengths and still round-trip through the
        inverse, so neither of those tests can catch a mirrored frame; the
        determinant's sign is what distinguishes them.
        """
        b = np.radians(bearing_deg)
        e_perp, e_along = rotate_to_bearing_frame(1.0, 0.0, b)   # image of Easting
        n_perp, n_along = rotate_to_bearing_frame(0.0, 1.0, b)   # image of Northing
        assert e_perp * n_along - e_along * n_perp == pytest.approx(1.0)

    @pytest.mark.parametrize("bearing_deg", BEARINGS)
    def test_inverse_is_the_transpose(self, bearing_deg):
        """rotate_from is the adjoint of rotate_to, so dot products survive."""
        b = np.radians(bearing_deg)
        de, dn = 2.0, 5.0
        perp, along = 1.5, -3.0
        fwd = rotate_to_bearing_frame(de, dn, b)
        back = rotate_from_bearing_frame(perp, along, b)
        assert fwd[0] * perp + fwd[1] * along == pytest.approx(back[0] * de + back[1] * dn)

    def test_angles_wrap(self):
        """bearing and bearing+360 are the same frame."""
        de, dn = 3.0, -8.0
        a = rotate_to_bearing_frame(de, dn, np.radians(37.0))
        b = rotate_to_bearing_frame(de, dn, np.radians(37.0 + 360.0))
        assert a == pytest.approx(b)


class TestVectorized:
    """apply_cutout_mask rotates a whole pixel meshgrid at once."""

    def test_arrays_rotate_elementwise(self):
        b = np.radians(30.0)
        de = np.array([[1.0, 2.0], [3.0, 4.0]])
        dn = np.array([[5.0, 6.0], [7.0, 8.0]])
        perp, along = rotate_to_bearing_frame(de, dn, b)
        assert perp.shape == de.shape and along.shape == dn.shape
        for idx in np.ndindex(de.shape):
            one = rotate_to_bearing_frame(float(de[idx]), float(dn[idx]), b)
            assert (perp[idx], along[idx]) == pytest.approx(one)

    def test_array_round_trip(self):
        b = np.radians(115.0)
        rng = np.random.default_rng(3)
        de, dn = rng.normal(size=64), rng.normal(size=64)
        back = rotate_from_bearing_frame(*rotate_to_bearing_frame(de, dn, b), b)
        assert np.allclose(back[0], de)
        assert np.allclose(back[1], dn)

    def test_origin_is_fixed(self):
        """No translation component -- the frame is centred on the cutout centre."""
        perp, along = rotate_to_bearing_frame(0.0, 0.0, np.radians(77.0))
        assert (perp, along) == pytest.approx((0.0, 0.0))
