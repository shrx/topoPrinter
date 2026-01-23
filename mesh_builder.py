"""
Mesh generation and STL export helpers.
"""

from typing import List, Tuple, Optional

import numpy as np
from pyproj import Transformer
from stl import Mode, mesh


def bilinear_interpolate(x: float, y: float, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, valid_mask: np.ndarray) -> float:
    """
    Bilinearly interpolate Z value at (x, y) from grid defined by X, Y, Z.

    Args:
        x, y: Target coordinates in model space (mm)
        X, Y: 2D meshgrid arrays of coordinates (rows x cols)
        Z: 2D array of values to interpolate (rows x cols)
        valid_mask: Boolean mask of valid cells

    Returns:
        Interpolated Z value, or nearest valid value if outside grid bounds
    """
    rows, cols = Z.shape

    # Find the grid cell containing (x, y)
    # Note: X decreases with j (cols), Y increases with i (rows)
    # xs = np.linspace(x_size_mm, 0, cols) - decreasing
    # ys = np.linspace(0, model_y_mm, rows) - increasing

    # Find j such that X[0, j+1] <= x <= X[0, j] (X decreases)
    # Find i such that Y[i, 0] <= y <= Y[i+1, 0] (Y increases)

    # Simple approach: find nearest indices and their neighbors
    # Calculate distances to all grid points
    dists = (X - x)**2 + (Y - y)**2

    # Find 4 nearest valid points
    valid_dists = np.where(valid_mask, dists, np.inf)

    if not np.any(np.isfinite(valid_dists)):
        # No valid points, return NaN
        return np.nan

    # Get the 4 closest valid points
    flat_indices = np.argsort(valid_dists.ravel())[:4]
    indices = np.unravel_index(flat_indices, Z.shape)

    # Use inverse distance weighting for simplicity
    weights = []
    values = []
    for idx in range(len(flat_indices)):
        i, j = indices[0][idx], indices[1][idx]
        if valid_mask[i, j]:
            dist = np.sqrt((X[i, j] - x)**2 + (Y[i, j] - y)**2)
            if dist < 1e-6:  # Very close, just use this value
                return Z[i, j]
            weights.append(1.0 / dist)
            values.append(Z[i, j])

    if not weights:
        return np.nan

    weights = np.array(weights)
    values = np.array(values)
    return np.sum(weights * values) / np.sum(weights)


def point_in_polygon(x: float, y: float, poly_x: np.ndarray, poly_y: np.ndarray) -> bool:
    """
    Test if point (x, y) is inside polygon defined by vertices (poly_x, poly_y).

    Uses ray casting algorithm.

    Args:
        x, y: Point coordinates
        poly_x, poly_y: Polygon vertex coordinates (arrays of same length)

    Returns:
        True if point is inside polygon
    """
    n = len(poly_x)
    inside = False

    p1x, p1y = poly_x[0], poly_y[0]
    for i in range(1, n + 1):
        p2x, p2y = poly_x[i % n], poly_y[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def _build_circular_cutout_mesh(
    dem: np.ndarray,
    px_size_x: float,
    px_size_y: float,
    x_size_mm: float,
    model_y_mm: float,
    z_surface_mm: np.ndarray,
    valid_mask: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    center_lat: float,
    center_lon: float,
    radius_m: float,
    ref_transform: object,
    ref_crs: object,
    n_gon_sides: int,
    base_thickness_mm: float,
) -> Tuple[np.ndarray, np.ndarray, float, Optional[np.ndarray]]:
    """
    Build mesh with smooth n-gon perimeter for circular cutout.

    Args:
        dem: DEM array (with buffer pixels around exact radius)
        px_size_x, px_size_y: Pixel sizes in meters
        x_size_mm, model_y_mm: Model dimensions in mm
        z_surface_mm: Surface elevations in mm (rows x cols)
        valid_mask: Valid data mask (rows x cols)
        X, Y: Meshgrid of model coordinates (rows x cols)
        center_lat, center_lon: Center coordinates (WGS84)
        radius_m: Exact radius in meters
        ref_transform: Rasterio affine transform
        ref_crs: DEM's CRS
        n_gon_sides: Number of polygon sides
        base_thickness_mm: Base thickness

    Returns:
        Tuple of (vertices, faces, max_z, None)
    """
    rows, cols = dem.shape

    # Step 1: Convert center from WGS84 to DEM's CRS
    transformer = Transformer.from_crs("EPSG:4326", ref_crs, always_xy=True)
    center_x_crs, center_y_crs = transformer.transform(center_lon, center_lat)

    # Step 2: Convert center from DEM CRS to model mm coordinates
    # We need to find which pixel contains the center, then map to model space
    # Inverse transform: CRS coords → pixel indices
    from rasterio.transform import rowcol
    center_row, center_col = rowcol(ref_transform, center_x_crs, center_y_crs)

    # Map pixel indices to model coordinates using X, Y meshgrid
    # Handle case where center might be outside grid bounds
    center_row = max(0, min(rows - 1, center_row))
    center_col = max(0, min(cols - 1, center_col))
    center_x_mm = X[center_row, center_col]
    center_y_mm = Y[center_row, center_col]

    # Step 3: Convert radius from meters to model mm
    # terrain_width_m corresponds to x_size_mm in model
    terrain_width_m = cols * px_size_x
    scale = x_size_mm / terrain_width_m  # mm per meter
    radius_mm = radius_m * scale

    # Step 4: Generate n-gon vertices at exact radius
    angles = np.linspace(0, 2 * np.pi, n_gon_sides, endpoint=False)
    ngon_x = center_x_mm + radius_mm * np.cos(angles)
    ngon_y = center_y_mm + radius_mm * np.sin(angles)

    # Step 5: Interpolate Z for each n-gon vertex from DEM
    ngon_z = np.array([
        bilinear_interpolate(ngon_x[i], ngon_y[i], X, Y, z_surface_mm, valid_mask)
        for i in range(n_gon_sides)
    ])

    # Step 6: Determine which DEM cells are fully interior to n-gon
    # A cell (quad) is interior if all 4 corners are inside the polygon
    cell_is_interior = np.zeros((rows - 1, cols - 1), dtype=bool)
    for i in range(rows - 1):
        for j in range(cols - 1):
            # Check all 4 corners of this cell
            corners_inside = (
                point_in_polygon(X[i, j], Y[i, j], ngon_x, ngon_y) and
                point_in_polygon(X[i + 1, j], Y[i + 1, j], ngon_x, ngon_y) and
                point_in_polygon(X[i + 1, j + 1], Y[i + 1, j + 1], ngon_x, ngon_y) and
                point_in_polygon(X[i, j + 1], Y[i, j + 1], ngon_x, ngon_y)
            )
            # Also check that all corners are valid data
            corners_valid = (
                valid_mask[i, j] and valid_mask[i + 1, j] and
                valid_mask[i + 1, j + 1] and valid_mask[i, j + 1]
            )
            cell_is_interior[i, j] = corners_inside and corners_valid

    # Step 7: Generate vertices for interior DEM cells
    vertex_map = np.full((rows, cols), -1, dtype=np.int32)
    vertex_list = []
    vertex_idx = 0

    # First, add DEM vertices that are part of interior cells
    for i in range(rows):
        for j in range(cols):
            # Check if this vertex is used by any interior cell
            used = False
            if i > 0 and j > 0 and cell_is_interior[i - 1, j - 1]:
                used = True
            elif i > 0 and j < cols - 1 and cell_is_interior[i - 1, j]:
                used = True
            elif i < rows - 1 and j > 0 and cell_is_interior[i, j - 1]:
                used = True
            elif i < rows - 1 and j < cols - 1 and cell_is_interior[i, j]:
                used = True

            if used and valid_mask[i, j]:
                vertex_list.append([X[i, j], Y[i, j], z_surface_mm[i, j]])
                vertex_map[i, j] = vertex_idx
                vertex_idx += 1

    # Add base vertices for DEM cells
    base_offset = len(vertex_list)
    for i in range(rows):
        for j in range(cols):
            if vertex_map[i, j] >= 0:  # This vertex was added
                vertex_list.append([X[i, j], Y[i, j], 0.0])

    # Add n-gon top vertices
    ngon_offset = len(vertex_list)
    for i in range(n_gon_sides):
        if np.isfinite(ngon_z[i]):
            vertex_list.append([ngon_x[i], ngon_y[i], ngon_z[i]])
        else:
            # Fallback: use base_thickness_mm if interpolation failed
            vertex_list.append([ngon_x[i], ngon_y[i], base_thickness_mm])

    # Add n-gon base vertices
    ngon_base_offset = len(vertex_list)
    for i in range(n_gon_sides):
        vertex_list.append([ngon_x[i], ngon_y[i], 0.0])

    vertices = np.array(vertex_list, dtype=np.float32)

    def idx(i: int, j: int) -> int:
        """Get vertex index for DEM cell at (i,j). Returns -1 if not used."""
        return vertex_map[i, j]

    faces: List[Tuple[int, int, int]] = []

    # Step 8: Generate top surface faces for interior cells
    for i in range(rows - 1):
        for j in range(cols - 1):
            if not cell_is_interior[i, j]:
                continue

            v00 = idx(i, j)
            v10 = idx(i + 1, j)
            v11 = idx(i + 1, j + 1)
            v01 = idx(i, j + 1)

            if v00 >= 0 and v10 >= 0 and v11 >= 0 and v01 >= 0:
                faces.append((v00, v10, v11))
                faces.append((v00, v11, v01))

    # Step 9: Generate base faces for interior cells
    for i in range(rows - 1):
        for j in range(cols - 1):
            if not cell_is_interior[i, j]:
                continue

            v00 = idx(i, j)
            v10 = idx(i + 1, j)
            v11 = idx(i + 1, j + 1)
            v01 = idx(i, j + 1)

            if v00 >= 0 and v10 >= 0 and v11 >= 0 and v01 >= 0:
                b00 = base_offset + v00
                b10 = base_offset + v10
                b11 = base_offset + v11
                b01 = base_offset + v01
                faces.append((b00, b11, b10))
                faces.append((b00, b01, b11))

    # Step 10: Boundary triangulation
    # Find DEM vertices on the boundary (interior cells but adjacent to non-interior)
    boundary_vertices = set()
    for i in range(rows - 1):
        for j in range(cols - 1):
            if cell_is_interior[i, j]:
                # Check if any neighbor cell is not interior
                neighbors = [
                    (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)
                ]
                for ni, nj in neighbors:
                    if ni < 0 or ni >= rows - 1 or nj < 0 or nj >= cols - 1:
                        # Edge of array
                        for vi, vj in [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]:
                            if vertex_map[vi, vj] >= 0:
                                boundary_vertices.add((vi, vj))
                        break
                    elif not cell_is_interior[ni, nj]:
                        # Non-interior neighbor
                        for vi, vj in [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]:
                            if vertex_map[vi, vj] >= 0:
                                boundary_vertices.add((vi, vj))
                        break

    # Simple triangulation: connect each boundary DEM vertex to nearest n-gon vertices
    for vi, vj in boundary_vertices:
        dem_v_idx = vertex_map[vi, vj]
        if dem_v_idx < 0:
            continue

        dem_x, dem_y, dem_z = X[vi, vj], Y[vi, vj], z_surface_mm[vi, vj]

        # Find 2 nearest n-gon vertices
        dists = np.sqrt((ngon_x - dem_x)**2 + (ngon_y - dem_y)**2)
        nearest_indices = np.argsort(dists)[:2]

        for k in range(len(nearest_indices) - 1):
            ng1 = ngon_offset + nearest_indices[k]
            ng2 = ngon_offset + nearest_indices[k + 1]

            # Top face
            faces.append((dem_v_idx, ng1, ng2))

            # Base face
            faces.append((base_offset + dem_v_idx, ng2 + (ngon_base_offset - ngon_offset), ng1 + (ngon_base_offset - ngon_offset)))

    # Step 11: Generate smooth n-gon walls
    for i in range(n_gon_sides):
        next_i = (i + 1) % n_gon_sides
        t0 = ngon_offset + i
        t1 = ngon_offset + next_i
        b0 = ngon_base_offset + i
        b1 = ngon_base_offset + next_i

        faces.append((t0, b0, t1))
        faces.append((t1, b0, b1))

    faces_array = np.array(faces, dtype=np.int64)
    max_z = float(np.max(vertices[:, 2]))

    return vertices, faces_array, max_z, None


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
    use_true_scale: bool = False,
    cutout_type: Optional[str] = None,
    cutout_center_lat: Optional[float] = None,
    cutout_center_lon: Optional[float] = None,
    cutout_radius_m: Optional[float] = None,
    ref_transform: Optional[object] = None,
    ref_crs: Optional[object] = None,
    n_gon_sides: int = 64,
) -> Tuple[np.ndarray, np.ndarray, float, Optional[np.ndarray]]:
    """
    Convert DEM grid into watertight mesh vertices/faces.

    For circular cutouts, generates smooth n-gon perimeter walls instead of jagged pixel-based walls.
    """
    rows, cols = dem.shape
    aspect_ratio = (rows * px_size_y) / (cols * px_size_x)
    model_y_mm = x_size_mm * aspect_ratio

    # Create mask for valid data (not NaN or infinite)
    valid_mask = np.isfinite(dem)
    if not valid_mask.any():
        raise ValueError("DEM contains no valid data (all NaN/infinite)")

    # Calculate min/max only from valid data
    valid_data = dem[valid_mask]
    min_elev = float(np.min(valid_data))
    max_elev = float(np.max(valid_data))
    height_range = max_elev - min_elev

    if use_true_scale:
        terrain_width_m = cols * px_size_x
        horizontal_scale = (terrain_width_m * 1000.0) / x_size_mm
        z_relief_mm = (dem - min_elev) * 1000.0 / horizontal_scale
        z_relief_mm = z_relief_mm * z_exaggeration
        z_surface_mm = base_thickness_mm + z_relief_mm
    else:
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

    # Handle circular cutout with smooth n-gon perimeter
    if cutout_type == "circular" and cutout_center_lat is not None and cutout_radius_m is not None:
        return _build_circular_cutout_mesh(
            dem, px_size_x, px_size_y, x_size_mm, model_y_mm,
            z_surface_mm, valid_mask, X, Y,
            cutout_center_lat, cutout_center_lon, cutout_radius_m,
            ref_transform, ref_crs, n_gon_sides, base_thickness_mm
        )

    # Create vertex index mapping - only for valid cells
    vertex_map = np.full((rows, cols), -1, dtype=np.int32)
    vertex_list = []
    vertex_idx = 0

    for i in range(rows):
        for j in range(cols):
            if valid_mask[i, j]:
                # Add top vertex
                x_pos = X[i, j]
                y_pos = Y[i, j]
                z_pos = z_surface_mm[i, j]
                vertex_list.append([x_pos, y_pos, z_pos])
                vertex_map[i, j] = vertex_idx
                vertex_idx += 1

    # Add base vertices for valid cells
    base_offset = len(vertex_list)
    for i in range(rows):
        for j in range(cols):
            if valid_mask[i, j]:
                x_pos = X[i, j]
                y_pos = Y[i, j]
                vertex_list.append([x_pos, y_pos, 0.0])

    vertices = np.array(vertex_list, dtype=np.float32)

    def idx(i: int, j: int) -> int:
        """Get vertex index for valid cell at (i,j). Returns -1 if invalid."""
        return vertex_map[i, j]

    faces: List[Tuple[int, int, int]] = []

    # Top surface (upward normals) - only create faces where all 4 corners are valid.
    for i in range(rows - 1):
        for j in range(cols - 1):
            # Check if all 4 corners of this quad are valid
            if not (valid_mask[i, j] and valid_mask[i+1, j] and
                    valid_mask[i+1, j+1] and valid_mask[i, j+1]):
                continue

            v00 = idx(i, j)
            v10 = idx(i + 1, j)
            v11 = idx(i + 1, j + 1)
            v01 = idx(i, j + 1)
            faces.append((v00, v10, v11))
            faces.append((v00, v11, v01))

    # Base surface (downward normals) - only create faces where all 4 corners are valid.
    for i in range(rows - 1):
        for j in range(cols - 1):
            # Check if all 4 corners of this quad are valid
            if not (valid_mask[i, j] and valid_mask[i+1, j] and
                    valid_mask[i+1, j+1] and valid_mask[i, j+1]):
                continue

            b00 = base_offset + idx(i, j)
            b10 = base_offset + idx(i + 1, j)
            b11 = base_offset + idx(i + 1, j + 1)
            b01 = base_offset + idx(i, j + 1)
            faces.append((b00, b11, b10))
            faces.append((b00, b01, b11))

    # Perimeter walls - create walls at boundaries between valid and invalid cells.
    # Check vertical edges (between columns).
    for i in range(rows):
        for j in range(cols - 1):
            current_valid = valid_mask[i, j]
            next_valid = valid_mask[i, j + 1]

            # Create wall if one side is valid and the other is not
            if current_valid and not next_valid:
                # Wall on right side of current cell (facing outward)
                if i < rows - 1 and valid_mask[i + 1, j]:
                    t0 = idx(i, j)
                    t1 = idx(i + 1, j)
                    b0 = base_offset + idx(i, j)
                    b1 = base_offset + idx(i + 1, j)
                    faces.append((t0, b0, t1))
                    faces.append((t1, b0, b1))
            elif not current_valid and next_valid:
                # Wall on left side of next cell (facing outward)
                if i < rows - 1 and valid_mask[i + 1, j + 1]:
                    t0 = idx(i, j + 1)
                    t1 = idx(i + 1, j + 1)
                    b0 = base_offset + idx(i, j + 1)
                    b1 = base_offset + idx(i + 1, j + 1)
                    faces.append((t0, t1, b0))
                    faces.append((t1, b1, b0))

    # Check horizontal edges (between rows).
    for i in range(rows - 1):
        for j in range(cols):
            current_valid = valid_mask[i, j]
            next_valid = valid_mask[i + 1, j]

            # Create wall if one side is valid and the other is not
            if current_valid and not next_valid:
                # Wall on bottom side of current cell (facing outward)
                if j < cols - 1 and valid_mask[i, j + 1]:
                    t0 = idx(i, j)
                    t1 = idx(i, j + 1)
                    b0 = base_offset + idx(i, j)
                    b1 = base_offset + idx(i, j + 1)
                    faces.append((t0, t1, b0))
                    faces.append((t1, b1, b0))
            elif not current_valid and next_valid:
                # Wall on top side of next cell (facing outward)
                if j < cols - 1 and valid_mask[i + 1, j + 1]:
                    t0 = idx(i + 1, j)
                    t1 = idx(i + 1, j + 1)
                    b0 = base_offset + idx(i + 1, j)
                    b1 = base_offset + idx(i + 1, j + 1)
                    faces.append((t0, b0, t1))
                    faces.append((t1, b0, b1))

    # Note: No walls at array boundaries - all walls should be internal
    # (at valid/invalid cell boundaries) for circular/rectangular cutouts.

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
