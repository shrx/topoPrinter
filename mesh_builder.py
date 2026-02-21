"""
Mesh generation and STL export helpers.
"""

from typing import List, Tuple, Optional

import numpy as np
from pyproj import Transformer
import trimesh

from shapely.geometry import shape as shapely_shape, Polygon as ShapelyPolygon, MultiPolygon
from shapely.ops import unary_union
import shapely

from bearing_utils import rotate_to_bearing_frame, rotate_from_bearing_frame


def _build_rectangular_mesh(
    rows: int,
    cols: int,
    X: np.ndarray,
    Y: np.ndarray,
    z_surface_mm: np.ndarray,
    valid_mask: np.ndarray,
    z_base: Optional[np.ndarray] = None,
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
                z_b = z_base[i, j] if z_base is not None else 0.0
                vertex_list.append([X[i, j], Y[i, j], z_b])

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


def _crs_to_model_xy(
    crs_coords: List[Tuple[float, float]],
    ref_transform,
    rows: int,
    cols: int,
    x_size_mm: float,
    model_y_mm: float,
) -> List[Tuple[float, float]]:
    """Convert CRS coordinates to model mm coordinates.

    Uses the same linear mapping as _compute_model_coordinates:
    model_x = col_frac / (cols-1) * x_size_mm
    model_y = model_y_mm * (1 - row_frac / (rows-1))
    """
    result = []
    for x_crs, y_crs in crs_coords:
        col_frac = (x_crs - ref_transform.c) / ref_transform.a
        row_frac = (y_crs - ref_transform.f) / ref_transform.e
        model_x = col_frac / (cols - 1) * x_size_mm
        model_y = model_y_mm * (1 - row_frac / (rows - 1))
        result.append((model_x, model_y))
    return result


def _geojson_to_shapely_mm(
    geojson_geom: dict,
    ref_transform,
    rows: int,
    cols: int,
    x_size_mm: float,
    model_y_mm: float,
) -> shapely.Geometry:
    """Convert a GeoJSON geometry (CRS coords) to a shapely geometry in model mm."""
    geom = shapely_shape(geojson_geom)
    # Transform all coordinates from CRS to model mm
    def _transform_ring(ring):
        crs_coords = list(ring.coords)
        return _crs_to_model_xy(crs_coords, ref_transform, rows, cols, x_size_mm, model_y_mm)

    if geom.geom_type == "Polygon":
        exterior = _transform_ring(geom.exterior)
        holes = [_transform_ring(h) for h in geom.interiors]
        return ShapelyPolygon(exterior, holes)
    elif geom.geom_type == "MultiPolygon":
        polys = []
        for poly in geom.geoms:
            exterior = _transform_ring(poly.exterior)
            holes = [_transform_ring(h) for h in poly.interiors]
            polys.append(ShapelyPolygon(exterior, holes))
        return MultiPolygon(polys)
    return shapely_shape(geojson_geom)


def _polygon_bbox_to_grid(
    polygon_mm: ShapelyPolygon,
    X: np.ndarray,
    Y: np.ndarray,
) -> Optional[Tuple[int, int, int, int]]:
    """Convert polygon bounding box to grid index range (i_min, i_max, j_min, j_max).

    Returns None if the crop region is too small.
    """
    rows, cols = X.shape
    x_size_mm = float(X[0, -1])
    model_y_mm = float(Y[0, 0])
    minx, miny, maxx, maxy = polygon_mm.bounds

    j_min = max(int(np.floor(minx * (cols - 1) / x_size_mm)) - 1, 0)
    j_max = min(int(np.ceil(maxx * (cols - 1) / x_size_mm)) + 1, cols - 1)
    i_min = max(int(np.floor((1 - maxy / model_y_mm) * (rows - 1))) - 1, 0)
    i_max = min(int(np.ceil((1 - miny / model_y_mm) * (rows - 1))) + 1, rows - 1)

    if (i_max - i_min + 1) < 2 or (j_max - j_min + 1) < 2:
        return None
    return i_min, i_max, j_min, j_max


def _compute_component_flat_z(
    polygon_mm: ShapelyPolygon,
    z_surface_mm: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    valid_mask: np.ndarray,
    thickness_mm: float,
) -> Optional[float]:
    """Compute the flat Z bottom for an overlay component via point-in-polygon."""
    bbox = _polygon_bbox_to_grid(polygon_mm, X, Y)
    if bbox is None:
        return None
    i_min, i_max, j_min, j_max = bbox

    X_crop = X[i_min:i_max + 1, j_min:j_max + 1]
    Y_crop = Y[i_min:i_max + 1, j_min:j_max + 1]
    z_crop = z_surface_mm[i_min:i_max + 1, j_min:j_max + 1]
    valid_crop = valid_mask[i_min:i_max + 1, j_min:j_max + 1]

    pts = shapely.points(X_crop.ravel(), Y_crop.ravel())
    inside = shapely.contains(polygon_mm, pts).reshape(X_crop.shape)
    inside &= valid_crop

    if not inside.any():
        return None
    return max(float(np.min(z_crop[inside])) - thickness_mm, 0.01)


def _build_polygon_prism(
    polygon_mm: ShapelyPolygon,
    z_bottom: float,
    z_top: float,
) -> Optional[trimesh.Trimesh]:
    """Build a watertight extruded prism from a shapely Polygon.

    Uses trimesh.creation.extrude_polygon for correct triangulation of
    non-convex polygons and polygons with holes.  Applies buffer(0) to fix
    self-touching boundaries from shapely boolean ops (may yield MultiPolygon).

    Returns trimesh volume or None if degenerate.
    """
    polygon_mm = polygon_mm.buffer(0)
    if polygon_mm.is_empty:
        return None
    height = z_top - z_bottom
    if height <= 0:
        return None

    if polygon_mm.geom_type == "Polygon":
        polys = [polygon_mm]
    elif polygon_mm.geom_type == "MultiPolygon":
        polys = list(polygon_mm.geoms)
    else:
        return None

    meshes = []
    for poly in polys:
        if poly.is_empty or not poly.is_valid:
            continue
        mesh = trimesh.creation.extrude_polygon(poly, height)
        mesh.apply_translation([0, 0, z_bottom])
        if mesh.is_empty or len(mesh.faces) == 0:
            continue
        mesh.fix_normals()
        meshes.append(mesh)

    if not meshes:
        return None
    if len(meshes) == 1:
        return meshes[0]
    return trimesh.boolean.union(meshes, check_volume=False)


def _build_overlay_component(
    polygon_mm: ShapelyPolygon,
    flat_z: Optional[float],
    z_surface_mm: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    valid_mask: np.ndarray,
    thickness_mm: float,
    recess_mode: str = "flat",
) -> Optional[trimesh.Trimesh]:
    """Build one overlay component mesh via boolean intersection.

    Args:
        polygon_mm: shapely Polygon in model mm coordinates.
        flat_z: pre-computed flat Z bottom (used only in "flat" mode).
        z_surface_mm: terrain surface Z array (rows x cols).
        X, Y: model coordinate grids (rows x cols).
        valid_mask: valid DEM data mask (rows x cols).
        thickness_mm: overlay shell thickness.
        recess_mode: "flat" for flat-bottomed recess, "uniform" for
            terrain-following uniform-thickness shell.

    Returns:
        trimesh.Trimesh or None if no DEM data inside polygon.
    """
    bbox = _polygon_bbox_to_grid(polygon_mm, X, Y)
    if bbox is None:
        return None
    i_min, i_max, j_min, j_max = bbox

    X_crop = X[i_min:i_max + 1, j_min:j_max + 1]
    Y_crop = Y[i_min:i_max + 1, j_min:j_max + 1]
    z_crop = z_surface_mm[i_min:i_max + 1, j_min:j_max + 1]
    valid_crop = valid_mask[i_min:i_max + 1, j_min:j_max + 1]
    crop_rows = i_max - i_min + 1
    crop_cols = j_max - j_min + 1

    if recess_mode == "uniform":
        z_base_crop = z_crop - thickness_mm
        verts_dem, faces_dem, _ = _build_rectangular_mesh(
            crop_rows, crop_cols, X_crop, Y_crop, z_crop, valid_crop,
            z_base=z_base_crop,
        )
    else:
        verts_dem, faces_dem, _ = _build_rectangular_mesh(
            crop_rows, crop_cols, X_crop, Y_crop, z_crop, valid_crop,
        )
    if len(faces_dem) == 0:
        return None

    dem_mesh = trimesh.Trimesh(vertices=verts_dem, faces=faces_dem)

    max_terrain_z = float(np.max(z_crop[valid_crop]))
    if recess_mode == "uniform":
        min_base_z = float(np.min(z_crop[valid_crop])) - thickness_mm
        prism_bottom = min_base_z - thickness_mm
        prism_top = max_terrain_z + thickness_mm * 2
    else:
        prism_bottom = flat_z
        prism_top = max(max_terrain_z + thickness_mm * 2, flat_z + thickness_mm * 10)

    prism_mesh = _build_polygon_prism(polygon_mm, prism_bottom, prism_top)
    if prism_mesh is None:
        return None

    result = trimesh.boolean.intersection([dem_mesh, prism_mesh], check_volume=False)
    if result.is_empty or len(result.faces) == 0:
        return None

    return result



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


def _compute_model_coordinates(
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], float]:
    """Compute model-space coordinates from DEM data.

    Returns:
        (X, Y, z_surface_mm, valid_mask, lake_mask, model_y_mm)
    """
    rows, cols = dem.shape
    aspect_ratio = (rows * px_size_y) / (cols * px_size_x)
    model_y_mm = x_size_mm * aspect_ratio

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

    xs = np.linspace(0, x_size_mm, cols)
    ys = np.linspace(model_y_mm, 0, rows)
    X, Y = np.meshgrid(xs, ys)

    return X, Y, z_surface_mm, valid_mask, lake_mask, model_y_mm


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
    X, Y, z_surface_mm, valid_mask, lake_mask, model_y_mm = _compute_model_coordinates(
        dem, px_size_x, px_size_y, x_size_mm, max_height_mm,
        z_exaggeration, base_thickness_mm, lake_range_percent,
        lake_lowering_mm, use_true_scale,
    )
    rows, cols = dem.shape

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


def _build_cutout_shape(
    cutout_type: str,
    dem_shape: Tuple[int, int],
    px_size_x: float,
    px_size_y: float,
    x_size_mm: float,
    z_surface_mm: np.ndarray,
    valid_mask: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    base_thickness_mm: float,
    ref_transform: object,
    ref_crs: object,
    cutout_center_lat: Optional[float] = None,
    cutout_center_lon: Optional[float] = None,
    cutout_radius_m: Optional[float] = None,
    cutout_side_length_km: Optional[float] = None,
    n_gon_sides: int = 64,
    bearing: float = 0.0,
    rect_corner1_lat: Optional[float] = None,
    rect_corner1_lon: Optional[float] = None,
    rect_corner2_lat: Optional[float] = None,
    rect_corner2_lon: Optional[float] = None,
) -> Optional[trimesh.Trimesh]:
    """Build the cutout shape (cylinder or rotated box) as a trimesh solid.

    Returns the cutout shape mesh, or None if cutout_type is not recognized.
    """
    rows, cols = dem_shape
    max_terrain_z = float(np.max(z_surface_mm[valid_mask]))
    box_height = max(max_terrain_z * 2, base_thickness_mm * 3)

    transformer = Transformer.from_crs("EPSG:4326", ref_crs, always_xy=True)

    if cutout_type == "circular":
        center_x_crs, center_y_crs = transformer.transform(cutout_center_lon, cutout_center_lat)
        from rasterio.transform import rowcol
        cr, cc = rowcol(ref_transform, center_x_crs, center_y_crs)
        cr = max(0, min(rows - 1, cr))
        cc = max(0, min(cols - 1, cc))
        center_x_mm = X[cr, cc]
        center_y_mm = Y[cr, cc]

        terrain_width_m = cols * px_size_x
        scale = x_size_mm / terrain_width_m
        radius_mm = cutout_radius_m * scale

        angles = np.linspace(0, 2 * np.pi, n_gon_sides, endpoint=False)
        ngon_x = center_x_mm + radius_mm * np.cos(angles)
        ngon_y = center_y_mm + radius_mm * np.sin(angles)

        verts = []
        for i in range(n_gon_sides):
            verts.append([ngon_x[i], ngon_y[i], 0.0])
        for i in range(n_gon_sides):
            verts.append([ngon_x[i], ngon_y[i], box_height])

        faces = []
        for i in range(n_gon_sides):
            ni = (i + 1) % n_gon_sides
            faces.append([i, n_gon_sides + i, ni])
            faces.append([ni, n_gon_sides + i, n_gon_sides + ni])
        for i in range(1, n_gon_sides - 1):
            faces.append([0, i + 1, i])
        for i in range(1, n_gon_sides - 1):
            faces.append([n_gon_sides, n_gon_sides + i, n_gon_sides + i + 1])

        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        mesh.fix_normals()
        return mesh

    elif cutout_type == "rectangular":
        bearing_rad = np.radians(bearing)

        if rect_corner1_lat is not None:
            c1_x, c1_y = transformer.transform(rect_corner1_lon, rect_corner1_lat)
            c2_x, c2_y = transformer.transform(rect_corner2_lon, rect_corner2_lat)
        else:
            cx, cy = transformer.transform(cutout_center_lon, cutout_center_lat)
            half = cutout_side_length_km * 1000.0 / 2.0
            de1, dn1 = rotate_from_bearing_frame(-half, -half, bearing_rad)
            c1_x, c1_y = cx + de1, cy + dn1
            de2, dn2 = rotate_from_bearing_frame(half, half, bearing_rad)
            c2_x, c2_y = cx + de2, cy + dn2

        dx_crs = c2_x - c1_x
        dy_crs = c2_y - c1_y
        AB_length_m, AD_length_m = rotate_to_bearing_frame(dx_crs, dy_crs, bearing_rad)
        AB_length_m = abs(AB_length_m)
        AD_length_m = abs(AD_length_m)

        terrain_width_m = abs(ref_transform.a) * cols
        dem_scale = x_size_mm / terrain_width_m
        half_w = AB_length_m * dem_scale / 2.0
        half_h = AD_length_m * dem_scale / 2.0

        center_x_crs = (c1_x + c2_x) / 2.0
        center_y_crs = (c1_y + c2_y) / 2.0
        from rasterio.transform import rowcol
        cr, cc = rowcol(ref_transform, center_x_crs, center_y_crs)
        cr = max(0, min(rows - 1, cr))
        cc = max(0, min(cols - 1, cc))
        center_x_mm = X[cr, cc]
        center_y_mm = Y[cr, cc]

        box_verts_local = [
            [-half_w, -half_h, 0], [half_w, -half_h, 0],
            [half_w, half_h, 0], [-half_w, half_h, 0],
            [-half_w, -half_h, box_height], [half_w, -half_h, box_height],
            [half_w, half_h, box_height], [-half_w, half_h, box_height],
        ]
        box_verts_rot = []
        for vx, vy, vz in box_verts_local:
            de, dn = rotate_from_bearing_frame(vx, vy, bearing_rad)
            box_verts_rot.append([de + center_x_mm, dn + center_y_mm, vz])

        box_faces = [
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 4, 1], [1, 4, 5],
            [1, 5, 2], [2, 5, 6],
            [2, 6, 3], [3, 6, 7],
            [3, 7, 0], [0, 7, 4],
        ]
        mesh = trimesh.Trimesh(vertices=box_verts_rot, faces=box_faces)
        mesh.fix_normals()
        return mesh

    return None


def _rasterize_cutout_mask(
    dem_shape: Tuple[int, int],
    ref_transform,
    ref_crs,
    cutout_type: str,
    cutout_center_lat: Optional[float] = None,
    cutout_center_lon: Optional[float] = None,
    cutout_radius_m: Optional[float] = None,
    cutout_side_length_km: Optional[float] = None,
    bearing: float = 0.0,
    rect_corner1_lat: Optional[float] = None,
    rect_corner1_lon: Optional[float] = None,
    rect_corner2_lat: Optional[float] = None,
    rect_corner2_lon: Optional[float] = None,
) -> np.ndarray:
    """Compute a boolean pixel mask for the cutout region on the DEM grid.

    Returns:
        Boolean array of shape dem_shape, True inside cutout.
    """
    rows, cols = dem_shape
    transformer = Transformer.from_crs("EPSG:4326", ref_crs, always_xy=True)

    # Build CRS coordinates for each pixel center
    col_idx, row_idx = np.meshgrid(np.arange(cols), np.arange(rows))
    # Pixel center: (col + 0.5, row + 0.5) in pixel space -> CRS via transform
    px_x_crs = ref_transform.a * (col_idx + 0.5) + ref_transform.c
    px_y_crs = ref_transform.e * (row_idx + 0.5) + ref_transform.f

    if cutout_type == "circular":
        center_x_crs, center_y_crs = transformer.transform(cutout_center_lon, cutout_center_lat)
        dist = np.sqrt((px_x_crs - center_x_crs)**2 + (px_y_crs - center_y_crs)**2)
        return dist <= cutout_radius_m

    elif cutout_type == "rectangular":
        bearing_rad = np.radians(bearing)

        if rect_corner1_lat is not None:
            c1_x, c1_y = transformer.transform(rect_corner1_lon, rect_corner1_lat)
            c2_x, c2_y = transformer.transform(rect_corner2_lon, rect_corner2_lat)
        else:
            cx, cy = transformer.transform(cutout_center_lon, cutout_center_lat)
            half = cutout_side_length_km * 1000.0 / 2.0
            de1, dn1 = rotate_from_bearing_frame(-half, -half, bearing_rad)
            c1_x, c1_y = cx + de1, cy + dn1
            de2, dn2 = rotate_from_bearing_frame(half, half, bearing_rad)
            c2_x, c2_y = cx + de2, cy + dn2

        center_x_crs = (c1_x + c2_x) / 2.0
        center_y_crs = (c1_y + c2_y) / 2.0
        dx_crs = c2_x - c1_x
        dy_crs = c2_y - c1_y
        half_w, half_h = rotate_to_bearing_frame(dx_crs, dy_crs, bearing_rad)
        half_w = abs(half_w) / 2.0
        half_h = abs(half_h) / 2.0

        # Project pixel offsets from center onto bearing-aligned frame
        de = px_x_crs - center_x_crs
        dn = px_y_crs - center_y_crs
        perp, along = rotate_to_bearing_frame(de, dn, bearing_rad)

        return (np.abs(perp) <= half_w) & (np.abs(along) <= half_h)

    # No cutout
    return np.ones(dem_shape, dtype=bool)


def _apply_rect_cutout_transform(
    verts: np.ndarray,
    dem_shape: Tuple[int, int],
    px_size_x: float,
    x_size_mm: float,
    ref_transform: object,
    X: np.ndarray,
    Y: np.ndarray,
    bearing: float,
    c1_x_crs: float,
    c1_y_crs: float,
    c2_x_crs: float,
    c2_y_crs: float,
) -> np.ndarray:
    """Apply the rectangular cutout rescale and bearing un-rotation to vertices.

    Transforms vertices from DEM mesh space to final model space
    (same transform as _build_rect_cutout_mesh post-processing).
    """
    rows, cols = dem_shape
    bearing_rad = np.radians(bearing)

    dx_crs = c2_x_crs - c1_x_crs
    dy_crs = c2_y_crs - c1_y_crs
    AB_length_m, AD_length_m = rotate_to_bearing_frame(dx_crs, dy_crs, bearing_rad)
    AB_length_m = abs(AB_length_m)
    AD_length_m = abs(AD_length_m)

    terrain_width_m = abs(ref_transform.a) * cols
    dem_scale = x_size_mm / terrain_width_m
    final_scale = x_size_mm / AB_length_m

    rect_width_mm_final = x_size_mm
    rect_height_mm_final = AD_length_m * final_scale

    center_x_crs = (c1_x_crs + c2_x_crs) / 2.0
    center_y_crs = (c1_y_crs + c2_y_crs) / 2.0
    from rasterio.transform import rowcol
    cr, cc = rowcol(ref_transform, center_x_crs, center_y_crs)
    cr = max(0, min(rows - 1, cr))
    cc = max(0, min(cols - 1, cc))
    center_x_mm = X[cr, cc]
    center_y_mm = Y[cr, cc]

    result = verts.copy()
    dx = result[:, 0] - center_x_mm
    dy = result[:, 1] - center_y_mm
    local_perp, local_along = rotate_to_bearing_frame(dx, dy, bearing_rad)

    scale_factor = final_scale / dem_scale
    local_perp *= scale_factor
    local_along *= scale_factor

    result[:, 0] = local_perp + rect_width_mm_final / 2.0
    result[:, 1] = local_along + rect_height_mm_final / 2.0
    return result


def _polygons_to_model_mm(
    geojson_geoms: List[dict],
    ref_transform,
    rows: int,
    cols: int,
    x_size_mm: float,
    model_y_mm: float,
) -> Optional[shapely.Geometry]:
    """Convert a list of GeoJSON geometries (CRS) to a single unioned shapely geometry in model mm."""
    if not geojson_geoms:
        return None
    polys_mm = []
    for g in geojson_geoms:
        p = _geojson_to_shapely_mm(g, ref_transform, rows, cols, x_size_mm, model_y_mm)
        if p.is_valid and not p.is_empty:
            polys_mm.append(p)
    if not polys_mm:
        return None
    union = unary_union(polys_mm)
    if union.is_empty:
        return None
    return union


def _iter_polygon_components(geom: shapely.Geometry):
    """Yield individual Polygon components from a Polygon or MultiPolygon."""
    if geom.geom_type == "Polygon":
        yield geom
    elif geom.geom_type == "MultiPolygon":
        yield from geom.geoms


def build_all_terrain_meshes(
    dem: np.ndarray,
    classification: np.ndarray,
    class_geometries: dict,
    px_size_x: float,
    px_size_y: float,
    x_size_mm: float,
    max_height_mm: float,
    z_exaggeration: float,
    base_thickness_mm: float,
    overlay_thickness_mm: float,
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
    recess_mode: str = "flat",
) -> dict:
    """Build rock base mesh and terrain overlay meshes.

    Args:
        class_geometries: dict mapping terrain class int → list of GeoJSON
            geometry dicts in CRS coordinates (from classify_terrain).

    Returns:
        dict mapping terrain name to (vertices, faces, max_z) or None.
    """
    from terrain_classifier import TERRAIN_ROCK, TERRAIN_GLACIER, TERRAIN_WATER, TERRAIN_FOLIAGE, TERRAIN_NAMES

    rows, cols = dem.shape

    # Compute model coordinates once (shared by rock base and overlays)
    X, Y, z_surface_mm, valid_mask, _, model_y_mm = _compute_model_coordinates(
        dem, px_size_x, px_size_y, x_size_mm, max_height_mm,
        z_exaggeration, base_thickness_mm,
        lake_range_percent=0.0, lake_lowering_mm=0.0,
        use_true_scale=use_true_scale,
    )

    # Convert all class geometries to shapely in model mm, union per class
    class_polys_mm = {}
    for tc in [TERRAIN_GLACIER, TERRAIN_WATER, TERRAIN_FOLIAGE]:
        geoms = class_geometries.get(tc, [])
        poly = _polygons_to_model_mm(
            geoms, ref_transform, rows, cols, x_size_mm, model_y_mm,
        )
        class_polys_mm[tc] = poly

    # Subtract higher-priority classes from lower-priority ones to make
    # mutually exclusive (matches the pixel rasterization priority order:
    # glacier > water > foliage)
    if class_polys_mm[TERRAIN_WATER] is not None and class_polys_mm[TERRAIN_GLACIER] is not None:
        diff = class_polys_mm[TERRAIN_WATER].difference(class_polys_mm[TERRAIN_GLACIER])
        class_polys_mm[TERRAIN_WATER] = diff if not diff.is_empty else None
    higher = [p for p in [class_polys_mm[TERRAIN_GLACIER], class_polys_mm[TERRAIN_WATER]] if p is not None]
    if class_polys_mm[TERRAIN_FOLIAGE] is not None and higher:
        combined_higher = unary_union(higher)
        diff = class_polys_mm[TERRAIN_FOLIAGE].difference(combined_higher)
        class_polys_mm[TERRAIN_FOLIAGE] = diff if not diff.is_empty else None

    # Pre-compute flat_z per overlay component (shared by recess and overlay)
    # List of (terrain_class, component_polygon, flat_z)
    overlay_components = []
    for tc in [TERRAIN_GLACIER, TERRAIN_WATER, TERRAIN_FOLIAGE]:
        union_poly = class_polys_mm[tc]
        if union_poly is None:
            continue
        for component in _iter_polygon_components(union_poly):
            if recess_mode == "uniform":
                overlay_components.append((tc, component, None))
            else:
                fz = _compute_component_flat_z(
                    component, z_surface_mm, X, Y, valid_mask, overlay_thickness_mm,
                )
                if fz is not None:
                    overlay_components.append((tc, component, fz))

    # Compute CRS corners for rectangular cutout (needed for rock mesh and transforms)
    c1_x_crs = c1_y_crs = c2_x_crs = c2_y_crs = None
    if cutout_type == "rectangular" and ref_transform and ref_crs:
        transformer = Transformer.from_crs("EPSG:4326", ref_crs, always_xy=True)
        bearing_rad = np.radians(bearing)
        if rect_corner1_lat is not None:
            c1_x_crs, c1_y_crs = transformer.transform(rect_corner1_lon, rect_corner1_lat)
            c2_x_crs, c2_y_crs = transformer.transform(rect_corner2_lon, rect_corner2_lat)
        else:
            cx, cy = transformer.transform(cutout_center_lon, cutout_center_lat)
            half = cutout_side_length_km * 1000.0 / 2.0
            de1, dn1 = rotate_from_bearing_frame(-half, -half, bearing_rad)
            c1_x_crs, c1_y_crs = cx + de1, cy + dn1
            de2, dn2 = rotate_from_bearing_frame(half, half, bearing_rad)
            c2_x_crs, c2_y_crs = cx + de2, cy + dn2

    # Build cutout shape once (shared by rock mesh and all overlays)
    cutout_shape = _build_cutout_shape(
        cutout_type, dem.shape, px_size_x, px_size_y, x_size_mm,
        z_surface_mm, valid_mask, X, Y, base_thickness_mm,
        ref_transform, ref_crs,
        cutout_center_lat=cutout_center_lat,
        cutout_center_lon=cutout_center_lon,
        cutout_radius_m=cutout_radius_m,
        cutout_side_length_km=cutout_side_length_km,
        n_gon_sides=n_gon_sides,
        bearing=bearing,
        rect_corner1_lat=rect_corner1_lat,
        rect_corner1_lon=rect_corner1_lon,
        rect_corner2_lat=rect_corner2_lat,
        rect_corner2_lon=rect_corner2_lon,
    )

    # Build rock base mesh: recess first, then cutout.
    # Overlay polygons are NOT clipped to the cutout boundary, so recess
    # prisms extend beyond the DEM grid — no coplanar faces with the DEM
    # perimeter wall.  The cutout shape then cleanly trims the result.
    rock_verts, rock_faces, _ = _build_rectangular_mesh(
        rows, cols, X, Y, z_surface_mm, valid_mask,
    )
    rock_mesh = trimesh.Trimesh(vertices=rock_verts, faces=rock_faces)

    # Step 1: Subtract recess volumes from the full DEM mesh
    max_terrain_z = float(np.max(z_surface_mm[valid_mask]))
    recess_volumes = []
    if recess_mode == "uniform":
        for tc, component, _ in overlay_components:
            bbox = _polygon_bbox_to_grid(component, X, Y)
            if bbox is None:
                continue
            i_min, i_max, j_min, j_max = bbox
            X_crop = X[i_min:i_max + 1, j_min:j_max + 1]
            Y_crop = Y[i_min:i_max + 1, j_min:j_max + 1]
            z_crop = z_surface_mm[i_min:i_max + 1, j_min:j_max + 1]
            valid_crop = valid_mask[i_min:i_max + 1, j_min:j_max + 1]
            crop_rows = i_max - i_min + 1
            crop_cols = j_max - j_min + 1
            z_base_crop = z_crop - overlay_thickness_mm
            # Use a high flat top (well above terrain) to avoid coplanarity
            # with the rock mesh's terrain surface during boolean difference.
            z_top_high = np.full_like(z_crop, max_terrain_z + overlay_thickness_mm * 2)
            verts, faces, _ = _build_rectangular_mesh(
                crop_rows, crop_cols, X_crop, Y_crop, z_top_high, valid_crop,
                z_base=z_base_crop,
            )
            if len(faces) == 0:
                continue
            recess_vol = trimesh.Trimesh(vertices=verts, faces=faces)
            # Intersect with tall polygon prism for XY clipping
            min_base_z = float(np.min(z_crop[valid_crop])) - overlay_thickness_mm
            prism = _build_polygon_prism(component, min_base_z - overlay_thickness_mm,
                                         max_terrain_z + overlay_thickness_mm * 2)
            if prism is not None:
                clipped = trimesh.boolean.intersection([recess_vol, prism], check_volume=False)
                if not clipped.is_empty and len(clipped.faces) > 0:
                    recess_volumes.append(clipped)
    else:
        for tc, component, flat_z in overlay_components:
            prism_top = max(max_terrain_z * 2, flat_z + overlay_thickness_mm * 10)
            prism = _build_polygon_prism(component, flat_z, prism_top)
            if prism is not None:
                recess_volumes.append(prism)

    if recess_volumes and rock_mesh.is_volume:
        if len(recess_volumes) == 1:
            combined = recess_volumes[0]
        else:
            combined = trimesh.boolean.union(recess_volumes, check_volume=False)
        subtracted = trimesh.boolean.difference([rock_mesh, combined], check_volume=False)
        if not subtracted.is_empty and len(subtracted.faces) > 0:
            rock_mesh = subtracted

    # Step 2: Apply cutout shape
    if cutout_shape is not None:
        result_mesh = trimesh.boolean.intersection([rock_mesh, cutout_shape], check_volume=False)
        if not result_mesh.is_empty and len(result_mesh.faces) > 0:
            rock_mesh = result_mesh

    # Step 3: For rectangular cutout, rescale and un-rotate to final model space
    rock_verts = rock_mesh.vertices
    if cutout_type == "rectangular" and c1_x_crs is not None:
        rock_verts = _apply_rect_cutout_transform(
            rock_verts, dem.shape, px_size_x, x_size_mm,
            ref_transform, X, Y, bearing,
            c1_x_crs, c1_y_crs, c2_x_crs, c2_y_crs,
        )
    rock_verts = rock_verts.astype(np.float32)
    rock_faces = rock_mesh.faces.astype(np.int64)
    rock_max_z = float(np.max(rock_verts[:, 2]))

    result = {"rock": (rock_verts, rock_faces, rock_max_z)}

    # Build overlays via boolean intersection with vector polygon prisms
    # Group components by terrain class
    from collections import defaultdict
    class_components = defaultdict(list)
    for tc, component, flat_z in overlay_components:
        class_components[tc].append((component, flat_z))

    for terrain_class in [TERRAIN_GLACIER, TERRAIN_WATER, TERRAIN_FOLIAGE]:
        name = TERRAIN_NAMES[terrain_class]
        components = class_components.get(terrain_class, [])
        if not components:
            result[name] = None
            continue

        component_meshes = []
        for component, flat_z in components:
            mesh = _build_overlay_component(
                component, flat_z, z_surface_mm, X, Y, valid_mask, overlay_thickness_mm,
                recess_mode=recess_mode,
            )
            if mesh is not None:
                component_meshes.append(mesh)

        if not component_meshes:
            result[name] = None
            continue

        # Boolean-union components into a proper volume, then apply cutout
        if len(component_meshes) == 1:
            combined = component_meshes[0]
        else:
            combined = trimesh.boolean.union(component_meshes, check_volume=False)

        if cutout_shape is not None:
            trimmed = trimesh.boolean.intersection([combined, cutout_shape], check_volume=False)
            if not trimmed.is_empty and len(trimmed.faces) > 0:
                combined = trimmed

        overlay_verts = combined.vertices
        overlay_faces = combined.faces

        # For rectangular cutout, apply rescale + rotation undo
        if cutout_type == "rectangular" and c1_x_crs is not None:
            overlay_verts = _apply_rect_cutout_transform(
                overlay_verts, dem.shape, px_size_x, x_size_mm,
                ref_transform, X, Y, bearing,
                c1_x_crs, c1_y_crs, c2_x_crs, c2_y_crs,
            )

        max_z = float(np.max(overlay_verts[:, 2]))
        result[name] = (
            overlay_verts.astype(np.float32),
            overlay_faces.astype(np.int64),
            max_z,
        )

    return result


def save_stl(vertices: np.ndarray, faces: np.ndarray, output_path: str) -> None:
    """Write vertices/faces to a binary STL file."""
    tm = trimesh.Trimesh(vertices=vertices, faces=faces)
    tm.export(output_path)
