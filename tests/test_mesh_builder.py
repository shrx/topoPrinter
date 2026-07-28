import numpy as np
import pytest
from rasterio.transform import from_origin

from bearing_utils import rotate_from_bearing_frame
from mesh_builder import _apply_rect_cutout_transform, dem_to_vertices_and_faces
from model_frame import ModelFrame


def test_dem_to_vertices_and_faces_basic_grid() -> None:
    dem = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=float)
    vertices, faces, max_z, water_faces = dem_to_vertices_and_faces(
        dem,
        px_size_x=1.0,
        px_size_y=1.0,
        x_size_mm=10.0,
        max_height_mm=5.0,
        z_exaggeration=1.0,
        base_thickness_mm=1.0,
        lake_range_percent=0.0,
        lake_lowering_mm=0.0,
    )

    assert vertices.shape == (8, 3)
    assert faces.shape == (12, 3)
    assert max_z > 1.0
    assert water_faces is None


# --- rectangular cutout: the print-frame placement -------------------------
#
# A 1000 m square raster at 10 m pixels, holding a 400 m x 300 m rectangle. The
# caller pins the model scale to the RECTANGLE, so the raster is built wider than
# the requested print width by raster_width / rect_width:
ROWS = COLS = 101
PX = 10.0
RASTER_M = (COLS - 1) * PX          # 1000 m
RECT_W_M, RECT_H_M = 400.0, 300.0
PRINT_W_MM = 150.0                  # what --x-size-mm asks for
X_SIZE_MM = PRINT_W_MM * RASTER_M / RECT_W_M        # the pinned frame width, 375 mm
SCALE_MM_PER_M = X_SIZE_MM / RASTER_M               # 0.375
PRINT_H_MM = RECT_H_M * SCALE_MM_PER_M              # 112.5
CENTRE_X_CRS, CENTRE_Y_CRS = 1500.0, 4500.0


def _rect_setup(bearing):
    """(frame, X, Y, corners_crs) for the rectangle at ``bearing``."""
    transform = from_origin(1000.0, 5000.0, PX, PX)
    frame = ModelFrame.from_dem((ROWS, COLS), PX, PX, X_SIZE_MM, transform,
                                "EPSG:3857")
    b = np.radians(bearing)
    corners = []
    for perp, along in [(-RECT_W_M / 2, -RECT_H_M / 2), (RECT_W_M / 2, -RECT_H_M / 2),
                        (RECT_W_M / 2, RECT_H_M / 2), (-RECT_W_M / 2, RECT_H_M / 2)]:
        de, dn = rotate_from_bearing_frame(perp, along, b)
        corners.append((CENTRE_X_CRS + de, CENTRE_Y_CRS + dn))
    X, Y = np.meshgrid(frame.grid_xs, frame.grid_ys[::-1])
    return frame, X, Y, corners


def _transform(frame, X, Y, corners, verts, bearing):
    # The spec carries one diagonal: corner A and the opposite corner C.
    (ax, ay), _, (cx, cy), _ = corners
    return _apply_rect_cutout_transform(
        verts, (ROWS, COLS), PX, frame.x_size_mm, frame.ref_transform, X, Y,
        bearing, ax, ay, cx, cy)


class TestRectCutoutTransform:
    """What is left of the post-mesh transform once the scale is pinned upstream.

    It used to rescale as well, which is why it had to run after the mesh was built --
    and it could only reach xy, so under true scale the relief came out understated by
    rect_width / raster_width. Pinning the scale before the frame exists leaves a rigid
    motion here: turn the rectangle onto the print axes, put its corner at the origin.
    """

    @pytest.mark.parametrize("bearing", [0.0, 37.0, 90.0, 214.0])
    def test_rectangle_lands_at_the_origin_at_the_requested_width(self, bearing):
        frame, X, Y, corners = _rect_setup(bearing)
        verts = np.array([[*frame.point_to_mm(x, y), 0.0] for x, y in corners])
        out = _transform(frame, X, Y, corners, verts, bearing)

        got = sorted((round(x, 9), round(y, 9)) for x, y, _ in out)
        want = sorted([(0.0, 0.0), (PRINT_W_MM, 0.0),
                       (PRINT_W_MM, PRINT_H_MM), (0.0, PRINT_H_MM)])
        assert got == pytest.approx(want)

    @pytest.mark.parametrize("bearing", [0.0, 37.0, 214.0])
    def test_is_a_rigid_motion(self, bearing):
        """No scale left in it: every distance survives unchanged.

        This is the property that lets the transform move to the 2D stage, ahead of
        the float32 quantization -- a rescale would invalidate the snap, a rigid
        motion applied with identical constants to every body does not.
        """
        frame, X, Y, corners = _rect_setup(bearing)
        rng = np.random.default_rng(3)
        verts = np.column_stack((rng.uniform(0, X_SIZE_MM, 50),
                                 rng.uniform(0, frame.model_y_mm, 50),
                                 rng.uniform(0, 10, 50)))
        out = _transform(frame, X, Y, corners, verts, bearing)

        def pdist(v):
            d = v[:, None, :2] - v[None, :, :2]
            return np.hypot(d[..., 0], d[..., 1])

        assert pdist(out) == pytest.approx(pdist(verts))
        assert out[:, 2] == pytest.approx(verts[:, 2]), "z must not be touched"
