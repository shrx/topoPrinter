"""
Bearing rotation utilities for coordinate transformations.

Bearing is measured in degrees clockwise from North. In (Easting, Northing)
coordinates, the bearing direction is (sin θ, cos θ) and the perpendicular
direction (bearing + 90°) is (cos θ, -sin θ).
"""

import numpy as np


def rotate_to_bearing_frame(de, dn, bearing_rad):
    """Transform (dEasting, dNorthing) offsets to bearing-aligned local frame.

    Projects offsets onto the rectangle's local axes:
    - perp: component along the perpendicular direction (bearing + 90°)
    - along: component along the bearing direction

    Args:
        de: Offset in easting direction (scalar or array)
        dn: Offset in northing direction (scalar or array)
        bearing_rad: Bearing angle in radians (clockwise from North)

    Returns:
        (perp, along): Components perpendicular to and along the bearing direction
    """
    cos_b = np.cos(bearing_rad)
    sin_b = np.sin(bearing_rad)
    perp = de * cos_b - dn * sin_b
    along = de * sin_b + dn * cos_b
    return perp, along


def rotate_from_bearing_frame(perp, along, bearing_rad):
    """Transform bearing-aligned local coordinates to (dEasting, dNorthing) offsets.

    Inverse of rotate_to_bearing_frame.

    Args:
        perp: Component perpendicular to bearing (scalar or array)
        along: Component along bearing direction (scalar or array)
        bearing_rad: Bearing angle in radians (clockwise from North)

    Returns:
        (de, dn): Offsets in easting and northing directions
    """
    cos_b = np.cos(bearing_rad)
    sin_b = np.sin(bearing_rad)
    de = perp * cos_b + along * sin_b
    dn = -perp * sin_b + along * cos_b
    return de, dn
