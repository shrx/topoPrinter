"""
Mesh generation and STL export helpers.
"""

from typing import List, Tuple, Optional

import numpy as np
from stl import Mode, mesh


def dem_to_vertices_and_faces(
    dem: np.ndarray,
    px_size_x: float,
    px_size_y: float,
    x_size_mm: float,
    max_height_mm: float,
    z_exaggeration: float,
    base_thickness_mm: float,
    lake_range_percent: float = 0.0,
    lake_lowering_mm: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, float, Optional[np.ndarray]]:
    """Convert a DEM grid into watertight mesh vertices/faces and optional lake faces."""
    rows, cols = dem.shape
    aspect_ratio = (rows * px_size_y) / (cols * px_size_x)
    model_y_mm = x_size_mm * aspect_ratio

    min_elev = float(np.min(dem))
    max_elev = float(np.max(dem))
    height_range = max_elev - min_elev
    relief_mm = max(max_height_mm - base_thickness_mm, 0.0)

    if height_range == 0:
        normalized = np.zeros_like(dem, dtype=np.float64)
    else:
        normalized = (dem - min_elev) / height_range

    z_relief_mm = normalized * relief_mm * z_exaggeration
    z_surface_mm = base_thickness_mm + z_relief_mm

    lake_mask = None
    if lake_lowering_mm > 0 and lake_range_percent > 0:
        threshold = min_elev + height_range * (lake_range_percent / 100.0)
        lake_mask = dem <= threshold
        if lake_mask.any():
            lake_min_mm = float(np.min(z_surface_mm[lake_mask]))
            target_lake_mm = max(lake_min_mm - lake_lowering_mm, 0.0)
            z_surface_mm = np.where(lake_mask, target_lake_mm, z_surface_mm)

    # Flip X so the output matches the expected orientation in common viewers (e.g., Blender without extra flips).
    xs = np.linspace(x_size_mm, 0, cols)
    ys = np.linspace(0, model_y_mm, rows)
    X, Y = np.meshgrid(xs, ys)

    top_vertices = np.column_stack((X.ravel(), Y.ravel(), z_surface_mm.ravel()))
    base_vertices = np.column_stack((X.ravel(), Y.ravel(), np.zeros(rows * cols)))
    vertices = np.vstack((top_vertices, base_vertices))

    def idx(i: int, j: int) -> int:
        return i * cols + j

    faces: List[Tuple[int, int, int]] = []

    # Top surface (upward normals).
    for i in range(rows - 1):
        for j in range(cols - 1):
            v00 = idx(i, j)
            v10 = idx(i + 1, j)
            v11 = idx(i + 1, j + 1)
            v01 = idx(i, j + 1)
            faces.append((v00, v10, v11))
            faces.append((v00, v11, v01))

    base_offset = rows * cols

    # Base surface (downward normals).
    for i in range(rows - 1):
        for j in range(cols - 1):
            b00 = base_offset + idx(i, j)
            b10 = base_offset + idx(i + 1, j)
            b11 = base_offset + idx(i + 1, j + 1)
            b01 = base_offset + idx(i, j + 1)
            faces.append((b00, b11, b10))
            faces.append((b00, b01, b11))

    # Left wall (x = 0, outward normal ~ -X).
    for i in range(rows - 1):
        t0 = idx(i, 0)
        t1 = idx(i + 1, 0)
        b0 = base_offset + idx(i, 0)
        b1 = base_offset + idx(i + 1, 0)
        faces.append((t0, t1, b0))
        faces.append((t1, b1, b0))

    # Right wall (x = max, outward normal ~ +X).
    for i in range(rows - 1):
        t0 = idx(i, cols - 1)
        t1 = idx(i + 1, cols - 1)
        b0 = base_offset + idx(i, cols - 1)
        b1 = base_offset + idx(i + 1, cols - 1)
        faces.append((t0, b0, t1))
        faces.append((t1, b0, b1))

    # Top edge (y = 0, outward normal ~ -Y).
    for j in range(cols - 1):
        t0 = idx(0, j)
        t1 = idx(0, j + 1)
        b0 = base_offset + idx(0, j)
        b1 = base_offset + idx(0, j + 1)
        faces.append((t0, t1, b0))
        faces.append((t1, b0, b1))

    # Bottom edge (y = max, outward normal ~ +Y).
    for j in range(cols - 1):
        t0 = idx(rows - 1, j)
        t1 = idx(rows - 1, j + 1)
        b0 = base_offset + idx(rows - 1, j)
        b1 = base_offset + idx(rows - 1, j + 1)
        faces.append((t0, t1, b1))
        faces.append((t0, b1, b0))

    faces_array = np.array(faces, dtype=np.int64)

    water_faces_array: Optional[np.ndarray] = None
    if lake_mask is not None and lake_mask.any():
        cell_mask = lake_mask[:-1, :-1] & lake_mask[1:, :-1] & lake_mask[:-1, 1:] & lake_mask[1:, 1:]

        def add_water_wall(side: str, i: int, j: int, acc: List[Tuple[int, int, int]]) -> None:
            if side == "north":
                t0 = idx(i, j)
                t1 = idx(i, j + 1)
                b0 = base_offset + idx(i, j)
                b1 = base_offset + idx(i, j + 1)
                acc.append((t0, t1, b0))
                acc.append((t1, b1, b0))
            elif side == "south":
                t0 = idx(i + 1, j)
                t1 = idx(i + 1, j + 1)
                b0 = base_offset + idx(i + 1, j)
                b1 = base_offset + idx(i + 1, j + 1)
                acc.append((t0, t1, b1))
                acc.append((t0, b1, b0))
            elif side == "west":
                t0 = idx(i, j)
                t1 = idx(i + 1, j)
                b0 = base_offset + idx(i, j)
                b1 = base_offset + idx(i + 1, j)
                acc.append((t0, t1, b0))
                acc.append((t1, b1, b0))
            elif side == "east":
                t0 = idx(i, j + 1)
                t1 = idx(i + 1, j + 1)
                b0 = base_offset + idx(i, j + 1)
                b1 = base_offset + idx(i + 1, j + 1)
                acc.append((t0, b0, t1))
                acc.append((t1, b0, b1))

        water_faces: List[Tuple[int, int, int]] = []
        for i in range(rows - 1):
            for j in range(cols - 1):
                if not cell_mask[i, j]:
                    continue
                v00 = idx(i, j)
                v10 = idx(i + 1, j)
                v11 = idx(i + 1, j + 1)
                v01 = idx(i, j + 1)
                water_faces.append((v00, v10, v11))
                water_faces.append((v00, v11, v01))

                b00 = base_offset + idx(i, j)
                b10 = base_offset + idx(i + 1, j)
                b11 = base_offset + idx(i + 1, j + 1)
                b01 = base_offset + idx(i, j + 1)
                water_faces.append((b00, b11, b10))
                water_faces.append((b00, b01, b11))

                if i == 0 or not cell_mask[i - 1, j]:
                    add_water_wall("north", i, j, water_faces)
                if i == rows - 2 or not cell_mask[i + 1, j]:
                    add_water_wall("south", i, j, water_faces)
                if j == 0 or not cell_mask[i, j - 1]:
                    add_water_wall("west", i, j, water_faces)
                if j == cols - 2 or not cell_mask[i, j + 1]:
                    add_water_wall("east", i, j, water_faces)

        if water_faces:
            water_faces_array = np.array(water_faces, dtype=np.int64)

    max_z = float(np.max(z_surface_mm))
    return vertices.astype(np.float32), faces_array, max_z, water_faces_array


def save_stl(vertices: np.ndarray, faces: np.ndarray, output_path: str) -> None:
    """Write vertices/faces to a binary STL file."""
    vectors = np.zeros((faces.shape[0], 3, 3), dtype=np.float32)
    for i, face in enumerate(faces):
        vectors[i] = vertices[face]
    stl_mesh = mesh.Mesh(np.zeros(vectors.shape[0], dtype=mesh.Mesh.dtype))
    stl_mesh.vectors[:] = vectors
    stl_mesh.save(output_path, mode=Mode.BINARY)
