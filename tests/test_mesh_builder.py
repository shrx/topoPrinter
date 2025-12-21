import numpy as np

from mesh_builder import dem_to_vertices_and_faces


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
