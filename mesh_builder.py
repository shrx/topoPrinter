"""
Mesh generation and STL export helpers.
"""

from typing import List, Tuple, Optional

import numpy as np
from pyproj import Transformer
import trimesh

from bearing_utils import rotate_to_bearing_frame, rotate_from_bearing_frame


def _build_rectangular_mesh(
    rows: int,
    cols: int,
    X: np.ndarray,
    Y: np.ndarray,
    z_surface_mm: np.ndarray,
    valid_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build rectangular watertight mesh from DEM grid.

    Returns:
        Tuple of (vertices, faces, vertex_map)
    """
    # A cell is valid if all 4 corners have valid data
    cell_is_valid = (
        valid_mask[:-1, :-1] &
        valid_mask[1:, :-1] &
        valid_mask[1:, 1:] &
        valid_mask[:-1, 1:]
    )

    # Generate vertices for all valid DEM cells
    vertex_map = np.full((rows, cols), -1, dtype=np.int32)
    vertex_list = []
    vertex_idx = 0

    # Add DEM vertices that are part of valid cells
    for i in range(rows):
        for j in range(cols):
            # Check if this vertex is used by any valid cell
            used = False
            if i > 0 and j > 0 and cell_is_valid[i - 1, j - 1]:
                used = True
            elif i > 0 and j < cols - 1 and cell_is_valid[i - 1, j]:
                used = True
            elif i < rows - 1 and j > 0 and cell_is_valid[i, j - 1]:
                used = True
            elif i < rows - 1 and j < cols - 1 and cell_is_valid[i, j]:
                used = True

            if used and valid_mask[i, j]:
                vertex_list.append([X[i, j], Y[i, j], z_surface_mm[i, j]])
                vertex_map[i, j] = vertex_idx
                vertex_idx += 1

    # Add base vertices for DEM cells
    base_offset = len(vertex_list)
    for i in range(rows):
        for j in range(cols):
            if vertex_map[i, j] >= 0:
                vertex_list.append([X[i, j], Y[i, j], 0.0])

    vertices = np.array(vertex_list, dtype=np.float32)

    faces: List[Tuple[int, int, int]] = []

    # Top surface faces
    for i in range(rows - 1):
        for j in range(cols - 1):
            if not cell_is_valid[i, j]:
                continue

            v00, v10, v11, v01 = vertex_map[i, j], vertex_map[i + 1, j], vertex_map[i + 1, j + 1], vertex_map[i, j + 1]
            if v00 >= 0 and v10 >= 0 and v11 >= 0 and v01 >= 0:
                faces.append((v00, v10, v11))
                faces.append((v00, v11, v01))

    # Base surface faces
    for i in range(rows - 1):
        for j in range(cols - 1):
            if not cell_is_valid[i, j]:
                continue

            v00, v10, v11, v01 = vertex_map[i, j], vertex_map[i + 1, j], vertex_map[i + 1, j + 1], vertex_map[i, j + 1]
            if v00 >= 0 and v10 >= 0 and v11 >= 0 and v01 >= 0:
                b00 = base_offset + v00
                b10 = base_offset + v10
                b11 = base_offset + v11
                b01 = base_offset + v01
                faces.append((b00, b11, b10))
                faces.append((b00, b01, b11))

    # Perimeter walls
    for i in range(rows - 1):
        for j in range(cols - 1):
            if not (valid_mask[i, j] and valid_mask[i+1, j] and
                    valid_mask[i+1, j+1] and valid_mask[i, j+1]):
                continue

            v00, v10, v11, v01 = vertex_map[i, j], vertex_map[i + 1, j], vertex_map[i + 1, j + 1], vertex_map[i, j + 1]
            if v00 < 0 or v10 < 0 or v11 < 0 or v01 < 0:
                continue

            # Left edge
            if not (j > 0 and valid_mask[i, j-1] and valid_mask[i+1, j-1]):
                b00, b10 = base_offset + v00, base_offset + v10
                faces.append((v00, b00, v10))
                faces.append((v10, b00, b10))

            # Right edge
            if not (j < cols - 2 and valid_mask[i, j+2] and valid_mask[i+1, j+2]):
                b01, b11 = base_offset + v01, base_offset + v11
                faces.append((v01, v11, b01))
                faces.append((v11, b11, b01))

            # Top edge
            if not (i > 0 and valid_mask[i-1, j] and valid_mask[i-1, j+1]):
                b00, b01 = base_offset + v00, base_offset + v01
                faces.append((v00, v01, b00))
                faces.append((v01, b01, b00))

            # Bottom edge
            if not (i < rows - 2 and valid_mask[i+2, j] and valid_mask[i+2, j+1]):
                b10, b11 = base_offset + v10, base_offset + v11
                faces.append((v10, b10, v11))
                faces.append((v11, b10, b11))

    return vertices, np.array(faces, dtype=np.int64), vertex_map


def _build_rect_cutout_mesh(
    dem: np.ndarray,
    px_size_x: float,
    px_size_y: float,
    x_size_mm: float,
    model_y_mm: float,
    z_surface_mm: np.ndarray,
    valid_mask: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    c1_x_crs: float,
    c1_y_crs: float,
    c2_x_crs: float,
    c2_y_crs: float,
    bearing: float,
    ref_transform: object,
    base_thickness_mm: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Build mesh with exact rectangular bounds using boolean intersection.

    Args:
        dem: DEM array
        px_size_x, px_size_y: Pixel sizes in meters
        x_size_mm, model_y_mm: Model dimensions in mm
        z_surface_mm: Surface elevations in mm (rows x cols)
        valid_mask: Valid data mask (rows x cols)
        X, Y: Meshgrid of model coordinates (rows x cols)
        c1_x_crs, c1_y_crs: First corner in CRS coordinates (corner A)
        c2_x_crs, c2_y_crs: Second corner in CRS coordinates (corner C, opposite to A)
        bearing: Bearing in degrees (direction of AD edge)
        ref_transform: Rasterio affine transform
        base_thickness_mm: Base thickness

    Returns:
        Tuple of (vertices, faces, max_z)
    """
    rows, cols = dem.shape

    # Build rectangular DEM mesh for boolean intersection
    vertices_dem, faces_dem, _ = _build_rectangular_mesh(rows, cols, X, Y, z_surface_mm, valid_mask)
    dem_mesh = trimesh.Trimesh(vertices=vertices_dem, faces=faces_dem)

    # Decompose diagonal into width (perpendicular to bearing) and height (along bearing)
    dx_crs = c2_x_crs - c1_x_crs
    dy_crs = c2_y_crs - c1_y_crs
    bearing_rad = np.radians(bearing)
    AB_length_m, AD_length_m = rotate_to_bearing_frame(dx_crs, dy_crs, bearing_rad)
    AB_length_m = abs(AB_length_m)
    AD_length_m = abs(AD_length_m)

    # DEM mesh scale: mm per CRS meter
    terrain_width_m = abs(ref_transform.a) * cols
    dem_scale = x_size_mm / terrain_width_m

    # Rectangle dimensions in DEM mesh coordinate space
    rect_width_mm_dem = AB_length_m * dem_scale
    rect_height_mm_dem = AD_length_m * dem_scale

    # Final model scale: rectangle width → x_size_mm
    final_scale = x_size_mm / AB_length_m
    rect_width_mm_final = x_size_mm
    rect_height_mm_final = AD_length_m * final_scale

    # Find center in model mm via pixel lookup
    center_x_crs = (c1_x_crs + c2_x_crs) / 2.0
    center_y_crs = (c1_y_crs + c2_y_crs) / 2.0

    from rasterio.transform import rowcol
    center_row, center_col = rowcol(ref_transform, center_x_crs, center_y_crs)
    center_row = max(0, min(rows - 1, center_row))
    center_col = max(0, min(cols - 1, center_col))
    center_x_mm = X[center_row, center_col]
    center_y_mm = Y[center_row, center_col]

    # Create box for intersection
    max_terrain_z = float(np.max(z_surface_mm[valid_mask]))
    box_height = max(max_terrain_z * 2, base_thickness_mm * 3)

    half_w = rect_width_mm_dem / 2.0
    half_h = rect_height_mm_dem / 2.0

    box_verts = [
        [-half_w, -half_h, 0], [half_w, -half_h, 0], [half_w, half_h, 0], [-half_w, half_h, 0],
        [-half_w, -half_h, box_height], [half_w, -half_h, box_height],
        [half_w, half_h, box_height], [-half_w, half_h, box_height],
    ]

    # Rotate box from bearing-local frame to CRS-aligned model space and translate to center
    box_verts_rot = []
    for vx, vy, vz in box_verts:
        de, dn = rotate_from_bearing_frame(vx, vy, bearing_rad)
        box_verts_rot.append([de + center_x_mm, dn + center_y_mm, vz])

    box_faces = [
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 6, 5], [4, 7, 6],  # top
        [0, 4, 1], [1, 4, 5],  # sides
        [1, 5, 2], [2, 5, 6],
        [2, 6, 3], [3, 6, 7],
        [3, 7, 0], [0, 7, 4],
    ]

    box_mesh = trimesh.Trimesh(vertices=box_verts_rot, faces=box_faces)
    box_mesh.fix_normals()

    # Boolean intersection
    if not dem_mesh.is_volume or not box_mesh.is_volume:
        raise ValueError("Meshes are not volumes for boolean intersection")

    result_mesh = dem_mesh.intersection(box_mesh)

    # Undo bearing rotation: project model offsets onto bearing-local frame
    verts = result_mesh.vertices.copy()
    dx = verts[:, 0] - center_x_mm
    dy = verts[:, 1] - center_y_mm
    local_perp, local_along = rotate_to_bearing_frame(dx, dy, bearing_rad)

    # Rescale from DEM mesh scale to final model scale
    scale_factor = final_scale / dem_scale
    local_perp *= scale_factor
    local_along *= scale_factor

    # Translate to origin (center at half-width, half-height)
    verts[:, 0] = local_perp + rect_width_mm_final / 2.0
    verts[:, 1] = local_along + rect_height_mm_final / 2.0

    vertices = verts.astype(np.float32)
    faces = result_mesh.faces.astype(np.int64)
    max_z = float(np.max(vertices[:, 2]))

    return vertices, faces, max_z


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
    Build mesh with smooth n-gon perimeter using boolean intersection.

    Builds a watertight rectangular DEM mesh, creates an n-gon cylinder at
    the exact radius, then uses boolean intersection to cut the DEM precisely.

    Args:
        dem: DEM array
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

    # Convert center to model mm coordinates
    transformer = Transformer.from_crs("EPSG:4326", ref_crs, always_xy=True)
    center_x_crs, center_y_crs = transformer.transform(center_lon, center_lat)

    from rasterio.transform import rowcol
    center_row, center_col = rowcol(ref_transform, center_x_crs, center_y_crs)
    center_row = max(0, min(rows - 1, center_row))
    center_col = max(0, min(cols - 1, center_col))
    center_x_mm = X[center_row, center_col]
    center_y_mm = Y[center_row, center_col]

    # Convert radius to model mm
    terrain_width_m = cols * px_size_x
    scale = x_size_mm / terrain_width_m  # mm per meter
    radius_mm = radius_m * scale

    # Generate n-gon vertices at exact radius
    angles = np.linspace(0, 2 * np.pi, n_gon_sides, endpoint=False)
    ngon_x = center_x_mm + radius_mm * np.cos(angles)
    ngon_y = center_y_mm + radius_mm * np.sin(angles)

    # Build rectangular DEM mesh for boolean intersection
    vertices_dem, faces_dem, _ = _build_rectangular_mesh(rows, cols, X, Y, z_surface_mm, valid_mask)

    # Boolean intersection with n-gon cylinder for smooth walls
    dem_mesh = trimesh.Trimesh(vertices=vertices_dem, faces=faces_dem)

    # Create n-gon cylinder (from base to well above terrain)
    max_terrain_z = float(np.max(z_surface_mm[valid_mask]))
    cylinder_height = max(max_terrain_z * 2, base_thickness_mm * 3)

    # Create n-gon prism vertices
    cylinder_verts = []
    # Bottom ring
    for i in range(n_gon_sides):
        cylinder_verts.append([ngon_x[i], ngon_y[i], 0.0])
    # Top ring
    for i in range(n_gon_sides):
        cylinder_verts.append([ngon_x[i], ngon_y[i], cylinder_height])

    # Create n-gon prism faces with consistent outward-facing normals
    cylinder_faces = []
    # Side walls (outward-facing)
    for i in range(n_gon_sides):
        next_i = (i + 1) % n_gon_sides
        # Two triangles per side, winding for outward normals
        cylinder_faces.append([i, n_gon_sides + i, next_i])
        cylinder_faces.append([next_i, n_gon_sides + i, n_gon_sides + next_i])

    # Bottom cap (downward-facing: clockwise when viewed from below)
    for i in range(1, n_gon_sides - 1):
        cylinder_faces.append([0, i + 1, i])

    # Top cap (upward-facing: counter-clockwise when viewed from above)
    for i in range(1, n_gon_sides - 1):
        cylinder_faces.append([n_gon_sides, n_gon_sides + i, n_gon_sides + i + 1])

    cylinder_mesh = trimesh.Trimesh(vertices=cylinder_verts, faces=cylinder_faces)

    # Fix normals to ensure consistent orientation
    cylinder_mesh.fix_normals()

    # Boolean intersection
    if not dem_mesh.is_volume or not cylinder_mesh.is_volume:
        raise ValueError("Not all meshes are volumes!")

    result_mesh = dem_mesh.intersection(cylinder_mesh)

    vertices = result_mesh.vertices.astype(np.float32)
    faces_array = result_mesh.faces.astype(np.int64)
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
    cutout_side_length_km: Optional[float] = None,
    ref_transform: Optional[object] = None,
    ref_crs: Optional[object] = None,
    n_gon_sides: int = 64,
    bearing: float = 0.0,
    rect_corner1_lat: Optional[float] = None,
    rect_corner1_lon: Optional[float] = None,
    rect_corner2_lat: Optional[float] = None,
    rect_corner2_lon: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float, Optional[np.ndarray]]:
    """
    Convert DEM grid into watertight mesh vertices/faces.

    Cutout cropping is handled by boolean intersection for all cutout types.
    """
    rows, cols = dem.shape
    aspect_ratio = (rows * px_size_y) / (cols * px_size_x)
    model_y_mm = x_size_mm * aspect_ratio

    # Create mask for valid data (not NaN or infinite)
    valid_mask = np.isfinite(dem)
    if not valid_mask.any():
        raise ValueError("DEM contains no valid data (all NaN/infinite)")

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

    # Model axes aligned with CRS: X ∝ Easting, Y ∝ Northing
    xs = np.linspace(0, x_size_mm, cols)
    ys = np.linspace(model_y_mm, 0, rows)
    X, Y = np.meshgrid(xs, ys)

    # Handle circular cutout with smooth n-gon perimeter
    if cutout_type == "circular" and cutout_center_lat is not None and cutout_radius_m is not None:
        return _build_circular_cutout_mesh(
            dem, px_size_x, px_size_y, x_size_mm, model_y_mm,
            z_surface_mm, valid_mask, X, Y,
            cutout_center_lat, cutout_center_lon, cutout_radius_m,
            ref_transform, ref_crs, n_gon_sides, base_thickness_mm
        )

    # Handle all rectangular cutouts via boolean intersection
    if cutout_type == "rectangular":
        transformer = Transformer.from_crs("EPSG:4326", ref_crs, always_xy=True)
        bearing_rad = np.radians(bearing)

        if rect_corner1_lat is not None:
            # rect-corners mode: convert lat/lon to CRS
            c1_x, c1_y = transformer.transform(rect_corner1_lon, rect_corner1_lat)
            c2_x, c2_y = transformer.transform(rect_corner2_lon, rect_corner2_lat)
        else:
            # center + side-length mode: compute CRS corners
            cx, cy = transformer.transform(cutout_center_lon, cutout_center_lat)
            half = cutout_side_length_km * 1000.0 / 2.0
            de1, dn1 = rotate_from_bearing_frame(-half, -half, bearing_rad)
            c1_x, c1_y = cx + de1, cy + dn1
            de2, dn2 = rotate_from_bearing_frame(half, half, bearing_rad)
            c2_x, c2_y = cx + de2, cy + dn2

        vertices, faces_array, max_z = _build_rect_cutout_mesh(
            dem, px_size_x, px_size_y, x_size_mm, model_y_mm,
            z_surface_mm, valid_mask, X, Y,
            c1_x, c1_y, c2_x, c2_y,
            bearing, ref_transform, base_thickness_mm
        )
        return vertices, faces_array, max_z, None

    # Build rectangular mesh (no cutout)
    vertices, faces_array, vertex_map = _build_rectangular_mesh(rows, cols, X, Y, z_surface_mm, valid_mask)
    base_offset = len(vertices) // 2

    def idx(i: int, j: int) -> int:
        """Get vertex index for valid cell at (i,j). Returns -1 if invalid."""
        return vertex_map[i, j]

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
    tm = trimesh.Trimesh(vertices=vertices, faces=faces)
    tm.export(output_path)
