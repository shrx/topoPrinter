"""
Mesh generation and STL export helpers.
"""

from typing import List, Tuple, Optional, Union

import numpy as np
from pyproj import Transformer
import trimesh

from shapely.geometry import shape as shapely_shape, Polygon as ShapelyPolygon, MultiPolygon, Point
from shapely.geometry.polygon import orient
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

    Fully vectorized (NumPy) construction of the top surface, flat/draped base,
    and perimeter walls. Produces the same indexed mesh as the previous
    per-cell Python loops, but at C speed and with much lower peak memory.

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

    # A vertex is "used" if any of the (up to 4) cells incident to it is valid.
    # Scatter each valid cell onto its four corner vertices via shifted ORs.
    vertex_used = np.zeros((rows, cols), dtype=bool)
    vertex_used[:-1, :-1] |= cell_is_valid   # corner (i,   j)
    vertex_used[1:, :-1] |= cell_is_valid    # corner (i+1, j)
    vertex_used[1:, 1:] |= cell_is_valid     # corner (i+1, j+1)
    vertex_used[:-1, 1:] |= cell_is_valid    # corner (i,   j+1)
    vertex_used &= valid_mask

    # Assign sequential vertex indices in row-major order (matches np.where).
    n_used = int(vertex_used.sum())
    vertex_map = np.full((rows, cols), -1, dtype=np.int32)
    vertex_map[vertex_used] = np.arange(n_used, dtype=np.int32)

    if n_used == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.int64),
            vertex_map,
        )

    # Vertices: top surface block followed by base block (row-major order).
    ii, jj = np.where(vertex_used)
    xv, yv = X[ii, jj], Y[ii, jj]
    z_b = z_base[ii, jj] if z_base is not None else np.zeros(n_used, dtype=X.dtype)
    vertices = np.empty((2 * n_used, 3), dtype=np.float32)
    vertices[:n_used, 0] = xv
    vertices[:n_used, 1] = yv
    vertices[:n_used, 2] = z_surface_mm[ii, jj]
    vertices[n_used:, 0] = xv
    vertices[n_used:, 1] = yv
    vertices[n_used:, 2] = z_b
    base_offset = n_used

    # Corner vertex indices for every valid cell (all guaranteed >= 0).
    ci, cj = np.where(cell_is_valid)
    v00 = vertex_map[ci, cj]
    v10 = vertex_map[ci + 1, cj]
    v11 = vertex_map[ci + 1, cj + 1]
    v01 = vertex_map[ci, cj + 1]
    b00, b10, b11, b01 = (
        v00 + base_offset, v10 + base_offset,
        v11 + base_offset, v01 + base_offset,
    )

    def _interleave(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Interleave two (N,3) arrays into (2N,3): a[0], b[0], a[1], b[1], ..."""
        out = np.empty((a.shape[0] * 2, 3), dtype=np.int64)
        out[0::2] = a
        out[1::2] = b
        return out

    # Top surface: two triangles per valid cell.
    top_faces = _interleave(
        np.column_stack((v00, v10, v11)),
        np.column_stack((v00, v11, v01)),
    )
    # Base surface (reversed winding).
    base_faces = _interleave(
        np.column_stack((b00, b11, b10)),
        np.column_stack((b00, b01, b11)),
    )

    # Perimeter walls: an edge gets walls unless the adjacent cell is also
    # valid. Neighbor lookups use a 2-cell zero-padded mask so out-of-bounds
    # neighbors read as invalid (reproducing the original boundary behavior).
    P = np.zeros((rows + 4, cols + 4), dtype=bool)
    P[2:2 + rows, 2:2 + cols] = valid_mask

    def _V(di: int, dj: int) -> np.ndarray:
        return P[ci + 2 + di, cj + 2 + dj]

    add_left = ~(_V(0, -1) & _V(1, -1))
    add_right = ~(_V(0, 2) & _V(1, 2))
    add_top = ~(_V(-1, 0) & _V(-1, 1))
    add_bottom = ~(_V(2, 0) & _V(2, 1))

    wall_blocks = []
    if add_left.any():
        m = add_left
        wall_blocks.append(_interleave(
            np.column_stack((v00[m], b00[m], v10[m])),
            np.column_stack((v10[m], b00[m], b10[m])),
        ))
    if add_right.any():
        m = add_right
        wall_blocks.append(_interleave(
            np.column_stack((v01[m], v11[m], b01[m])),
            np.column_stack((v11[m], b11[m], b01[m])),
        ))
    if add_top.any():
        m = add_top
        wall_blocks.append(_interleave(
            np.column_stack((v00[m], v01[m], b00[m])),
            np.column_stack((v01[m], b01[m], b00[m])),
        ))
    if add_bottom.any():
        m = add_bottom
        wall_blocks.append(_interleave(
            np.column_stack((v10[m], b10[m], v11[m])),
            np.column_stack((v11[m], b10[m], b11[m])),
        ))

    faces = np.concatenate([top_faces, base_faces, *wall_blocks], axis=0)
    return vertices, faces, vertex_map


def _crs_point_to_model_xy(
    x_crs: float,
    y_crs: float,
    ref_transform,
    rows: int,
    cols: int,
    x_size_mm: float,
    model_y_mm: float,
) -> Tuple[float, float]:
    """Map one CRS point to model mm exactly (pixel-center convention).

    Grid vertex (i, j) carries the DEM sample of pixel (i, j), whose CRS
    location is the pixel CENTER (col + 0.5, row + 0.5 in pixel space), so the
    CRS->model mapping must subtract that half-pixel before scaling:
    model_x = ((x - c)/a - 0.5) / (cols-1) * x_size_mm
    model_y = model_y_mm * (1 - ((y - f)/e - 0.5) / (rows-1))
    """
    col_frac = (x_crs - ref_transform.c) / ref_transform.a - 0.5
    row_frac = (y_crs - ref_transform.f) / ref_transform.e - 0.5
    model_x = col_frac / (cols - 1) * x_size_mm
    model_y = model_y_mm * (1 - row_frac / (rows - 1))
    return model_x, model_y


def _crs_to_model_xy(
    crs_coords,
    ref_transform,
    rows: int,
    cols: int,
    x_size_mm: float,
    model_y_mm: float,
) -> np.ndarray:
    """Convert an (N, 2) array of CRS coordinates to model mm (vectorized).

    Same mapping as _crs_point_to_model_xy, applied to whole rings at once —
    OSM polygon rings can carry tens of thousands of points.
    """
    arr = np.asarray(crs_coords, dtype=np.float64)
    col_frac = (arr[:, 0] - ref_transform.c) / ref_transform.a - 0.5
    row_frac = (arr[:, 1] - ref_transform.f) / ref_transform.e - 0.5
    model_x = col_frac / (cols - 1) * x_size_mm
    model_y = model_y_mm * (1 - row_frac / (rows - 1))
    return np.column_stack((model_x, model_y))


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
        crs_coords = np.asarray(ring.coords)
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


def _build_overlay_solid(
    polygon_mm: ShapelyPolygon,
    z_surface_mm: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    valid_mask: np.ndarray,
    thickness_mm: float,
) -> Optional[trimesh.Trimesh]:
    """Build the DEM surface clipped to ``polygon_mm`` as a watertight solid.

    The top follows the terrain (including the vertices created where the outline
    crosses the grid); the bottom is a provisional flat plane placed well below
    the lowest terrain.  Callers slice that bottom to the desired floor per part,
    or read the solid's true surface minimum.  Returns None if the polygon covers
    no valid DEM data.
    """
    bbox = _polygon_bbox_to_grid(polygon_mm, X, Y)
    if bbox is None:
        return None
    i_min, i_max, j_min, j_max = bbox

    X_crop = X[i_min:i_max + 1, j_min:j_max + 1]
    Y_crop = Y[i_min:i_max + 1, j_min:j_max + 1]
    z_crop = z_surface_mm[i_min:i_max + 1, j_min:j_max + 1]
    valid_crop = valid_mask[i_min:i_max + 1, j_min:j_max + 1]
    if not valid_crop.any():
        return None
    crop_rows = i_max - i_min + 1
    crop_cols = j_max - j_min + 1

    deep = float(np.min(z_crop[valid_crop])) - thickness_mm - 1.0
    verts, faces, _ = _build_rectangular_mesh(
        crop_rows, crop_cols, X_crop, Y_crop, z_crop, valid_crop,
        z_base=np.full_like(z_crop, deep),
    )
    if len(faces) == 0:
        return None
    dem_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    max_terrain_z = float(np.max(z_crop[valid_crop]))
    prism = _build_polygon_prism(polygon_mm, deep - 1.0, max_terrain_z + thickness_mm * 2)
    if prism is None:
        return None
    clipped = trimesh.boolean.intersection([dem_mesh, prism], check_volume=False)
    if clipped.is_empty or len(clipped.faces) == 0:
        return None
    return clipped


def _surface_top_min(solid: trimesh.Trimesh) -> Optional[float]:
    """Minimum Z of the terrain-top vertices of a solid from _build_overlay_solid.

    Excludes the provisional flat bottom (the single lowest plane), so the result
    is the true minimum of the clipped surface — outline vertices included.
    """
    z = solid.vertices[:, 2]
    deep = float(z.min())
    # The flat bottom is the single lowest plane (all vertices exactly at deep);
    # everything strictly above it is terrain.
    top = z[z > deep]
    if top.size == 0:
        return None
    return float(top.min())


def _compute_component_flat_z(
    polygon_mm: ShapelyPolygon,
    z_surface_mm: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    valid_mask: np.ndarray,
    thickness_mm: float,
    cutout_shape: Optional[trimesh.Trimesh] = None,
) -> Optional[float]:
    """Flat Z bottom for an overlay component: true surface minimum - thickness.

    The minimum is taken over the actual clipped terrain surface — the DEM grid
    points inside the (full) outline *and* the outline points themselves —
    restricted to the cutout, so terrain outside the model never deepens the
    recess.  The full outline is used, so the depth is independent of XY
    clearance.  Computed on the triangulated mesh (via clip), so it matches the
    printed geometry exactly.
    """
    solid = _build_overlay_solid(polygon_mm, z_surface_mm, X, Y, valid_mask, thickness_mm)
    if solid is None:
        return None
    if cutout_shape is not None:
        solid = trimesh.boolean.intersection([solid, cutout_shape], check_volume=False)
        if solid.is_empty or len(solid.faces) == 0:
            return None
    top_min = _surface_top_min(solid)
    if top_min is None:
        return None
    return max(top_min - thickness_mm, 0.01)


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


def _inset_polygon(
    polygon_mm: ShapelyPolygon,
    distance_mm: float,
) -> Optional[ShapelyPolygon]:
    """Shrink a polygon inward by ``distance_mm`` on every side.

    Used to give a separately-printed insert horizontal clearance from its rock
    pocket.  ``buffer(-d)`` may return a (Multi)Polygon or empty geometry when a
    feature is narrower than 2*distance; both are handled downstream.  Returns
    the original polygon when distance is non-positive, or None if the inset
    erases the feature entirely.
    """
    if distance_mm <= 0:
        return polygon_mm
    inset = polygon_mm.buffer(-distance_mm)
    if inset.is_empty:
        return None
    return inset


def _corner_reliefs(
    component: ShapelyPolygon,
    clearance_mm: float,
    relief_mm: float,
    min_turn_deg: float,
) -> Tuple[Optional[Union[ShapelyPolygon, MultiPolygon]],
           Optional[Union[ShapelyPolygon, MultiPolygon]]]:
    """Corner-relief geometry for a separately-printed insert.

    FDM over-extrudes reentrant (inside) corners, so a sharp corner seats tighter
    than the flat clearance and can lock.  This adds ``relief_mm`` of extra
    clearance at sharp corners, on whichever body owns the inside corner:

      * convex footprint corner -> the rock pocket has the inside corner, so
        enlarge the pocket: a disc of radius (clearance+relief) centred on the
        insert's inset corner point (concentric with the insert corner, making the
        corner gap clearance+relief).
      * reflex footprint corner -> the insert has the inside corner, so shrink the
        insert: the same disc, centred on the nominal vertex.

    Only corners turning by at least ``min_turn_deg`` are relieved; a disc at a
    near-straight vertex would push the flat wall out by ``relief_mm`` and wreck
    the flat clearance.  The signed turn angle classifies convex vs reflex and
    catches needle-sharp corners (turn near 180 deg) that a sine test would miss.

    Returns (pocket_extra, insert_cut): the union of convex discs to add to the
    pocket (or None) and the union of reflex discs to subtract from the insert
    (or None).
    """
    if relief_mm <= 0:
        return None, None
    r = clearance_mm + relief_mm
    convex, reflex = [], []
    polys = component.geoms if component.geom_type == "MultiPolygon" else [component]
    rings = []
    for poly in polys:
        oriented = orient(poly, 1.0)    # exterior CCW, holes CW: solid is on the left
        rings.extend([oriented.exterior, *oriented.interiors])
    for ring in rings:
        pts = list(ring.coords)[:-1]
        n = len(pts)
        if n < 3:
            continue
        for i in range(n):
            A = np.asarray(pts[i - 1]); V = np.asarray(pts[i]); B = np.asarray(pts[(i + 1) % n])
            u = V - A; w = B - V
            lu = float(np.hypot(*u)); lw = float(np.hypot(*w))
            if lu == 0.0 or lw == 0.0:
                continue
            u /= lu; w /= lw
            cross = float(u[0] * w[1] - u[1] * w[0])
            turn = np.degrees(np.arctan2(cross, float(u @ w)))   # signed, -180..180
            if abs(turn) < min_turn_deg:
                continue
            if turn > 0.0:                                       # convex -> relieve pocket
                nu = np.array([-u[1], u[0]]); nw = np.array([-w[1], w[0]])  # inward normals
                denom = max(1.0 + float(nu @ nw), 0.1)           # cap miter length at needles
                P = V + clearance_mm * (nu + nw) / denom
                convex.append(Point(P).buffer(r))
            else:                                                # reflex -> relieve insert
                reflex.append(Point(V).buffer(r))
    pocket_extra = unary_union(convex) if convex else None
    insert_cut = unary_union(reflex) if reflex else None
    return pocket_extra, insert_cut


def _clip_to_footprint(
    polygon_mm: Union[ShapelyPolygon, MultiPolygon],
    footprint: Optional[ShapelyPolygon],
) -> Optional[Union[ShapelyPolygon, MultiPolygon]]:
    """Clip a polygon to the cutout footprint in 2D.

    Done before any 3D mesh is built so the cutout boundary becomes ordinary
    prism walls (a single vertical segment), instead of being applied as a
    solid-vs-solid boolean that subdivides the existing walls with intermediate
    vertices.  Returns the original polygon when there is no cutout, a cleaned
    (Multi)Polygon when the clip is non-empty, or None when nothing remains.
    """
    if footprint is None:
        return polygon_mm
    clipped = polygon_mm.intersection(footprint).buffer(0)
    if clipped.is_empty:
        return None
    if clipped.geom_type in ("Polygon", "MultiPolygon"):
        return clipped
    if clipped.geom_type == "GeometryCollection":
        polys = [g for g in clipped.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        if polys:
            merged = unary_union(polys)
            if not merged.is_empty:
                return merged
    return None


def _drop_small_rim_bits(
    polygon_mm: Optional[Union[ShapelyPolygon, MultiPolygon]],
    rim,
    min_area_mm2: float,
) -> Optional[Union[ShapelyPolygon, MultiPolygon]]:
    """Drop connected components smaller than ``min_area_mm2`` that touch the rim.

    Clipping an insert footprint to the cutout can shear a sub-printable lobe off at
    the rim.  Removed here -- BEFORE the inset clearance step -- so the insert and
    its pocket are built from one cleaned footprint: otherwise the inset erases the
    lobe from the insert (``buffer(-clearance)``) while the pocket keeps it, leaving
    an empty recessed notch in the perimeter.  The dropped area sits in no pocket, so
    it stays ordinary base terrain (base = cutout - pockets).  Interior components
    are untouched (resolve_layers already enforced min_blob on whole components);
    only rim-touching runts go.  Returns the cleaned geometry, or None if nothing
    printable remains.
    """
    if polygon_mm is None or polygon_mm.is_empty:
        return None
    keep = [c for c in _iter_polygon_components(polygon_mm)
            if not (c.area < min_area_mm2 and rim.intersects(c))]
    if not keep:
        return None
    return keep[0] if len(keep) == 1 else unary_union(keep)


def _quantize_to_f32(
    polygon_mm: Union[ShapelyPolygon, MultiPolygon],
    output_resolution: np.float32,
) -> Optional[Union[ShapelyPolygon, MultiPolygon]]:
    """Quantize polygon coordinates to float32 precision, before 2D->3D.

    Binary STL stores vertices as 32-bit floats, so any polygon feature finer
    than float32 resolution (a near-pinch where two boundary points are a few nm
    apart) collapses to coincident vertices on export and becomes a zero-area
    triangle that breaks slicing.  ``set_precision`` rounds coordinates to
    ``output_resolution`` *and* removes the resulting duplicate/collapsed vertices
    (the default "valid_output" mode), so no sub-resolution sliver is ever built.
    float32 precision is relative, so quantizing in the polygon's own space carries
    correctly through the later affine rescale to final model space.  Returns the
    quantized (Multi)Polygon, or None if it collapses to nothing.
    """
    quantized = shapely.set_precision(polygon_mm, output_resolution)
    if quantized.is_empty or quantized.geom_type not in ("Polygon", "MultiPolygon"):
        return None
    # set_precision leaves a *persistent* fixed precision model on the geometry.  Any
    # later GEOS overlay (the per-cell poly.intersection in _region_top_surface) would
    # then run OverlayNG under that model and snap-round every output vertex to the
    # grid -- moving exact grid corners off-grid and producing duplicate vertices,
    # spurious interior walls, and non-manifold edges.  The coordinates are already
    # quantized; clear the model back to floating so downstream intersections preserve
    # pass-through vertices exactly (grid_size=0 leaves coordinates bit-identical).
    return shapely.set_precision(quantized, 0.0)


def _slice_solid_to_parts(
    solid: trimesh.Trimesh,
    floor_offset_mm: float,
) -> List[trimesh.Trimesh]:
    """Split a clipped surface solid into connected parts with flat bottoms.

    Each part's bottom is cut to its own surface minimum minus ``floor_offset_mm``.
    Because the minimum is read from the part's real surface (outline vertices
    included), every part's thinnest point equals ``floor_offset_mm`` exactly,
    with no interpolation artifacts.
    """
    parts = []
    for part in solid.split(only_watertight=False):
        top_min = _surface_top_min(part)
        if top_min is None:
            continue
        floor = max(top_min - floor_offset_mm, 0.01)
        sliced = part.slice_plane([0, 0, floor], [0, 0, 1], cap=True)
        if sliced is None or sliced.is_empty or len(sliced.faces) == 0:
            continue
        parts.append(sliced)
    return parts


def _build_overlay_component(
    polygon_mm: ShapelyPolygon,
    flat_z: Optional[float],
    z_surface_mm: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    valid_mask: np.ndarray,
    thickness_mm: float,
    recess_mode: str = "flat",
    z_clearance_mm: float = 0.0,
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
        z_clearance_mm: vertical clearance taken off the insert so it seats
            below the pocket floor; the shell becomes (thickness - clearance)
            thick.  Used by "uniform" mode (flat mode uses
            _build_overlay_parts_flat).

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
        z_base_crop = z_crop - (thickness_mm - z_clearance_mm)
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

    dem_mesh = trimesh.Trimesh(vertices=verts_dem, faces=faces_dem, process=False)

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
    dem_mesh = trimesh.Trimesh(vertices=vertices_dem, faces=faces_dem, process=False)

    # Decompose diagonal into width (perpendicular to bearing) and height (along bearing)
    dx_crs = c2_x_crs - c1_x_crs
    dy_crs = c2_y_crs - c1_y_crs
    bearing_rad = np.radians(bearing)
    AB_length_m, AD_length_m = rotate_to_bearing_frame(dx_crs, dy_crs, bearing_rad)
    AB_length_m = abs(AB_length_m)
    AD_length_m = abs(AD_length_m)

    # DEM mesh scale: mm per CRS meter (grid spans cols-1 pixel spacings,
    # first to last pixel center)
    terrain_width_m = abs(ref_transform.a) * (cols - 1)
    dem_scale = x_size_mm / terrain_width_m

    # Rectangle dimensions in DEM mesh coordinate space
    rect_width_mm_dem = AB_length_m * dem_scale
    rect_height_mm_dem = AD_length_m * dem_scale

    # Final model scale: rectangle width → x_size_mm
    final_scale = x_size_mm / AB_length_m
    rect_width_mm_final = x_size_mm
    rect_height_mm_final = AD_length_m * final_scale

    # Find center in model mm (exact, no pixel snapping)
    center_x_crs = (c1_x_crs + c2_x_crs) / 2.0
    center_y_crs = (c1_y_crs + c2_y_crs) / 2.0
    center_x_mm, center_y_mm = _crs_point_to_model_xy(
        center_x_crs, center_y_crs, ref_transform, rows, cols, x_size_mm, model_y_mm)

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

    box_mesh = trimesh.Trimesh(vertices=box_verts_rot, faces=box_faces, process=False)
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

    # Convert center to model mm coordinates (exact, no pixel snapping)
    transformer = Transformer.from_crs("EPSG:4326", ref_crs, always_xy=True)
    center_x_crs, center_y_crs = transformer.transform(center_lon, center_lat)
    center_x_mm, center_y_mm = _crs_point_to_model_xy(
        center_x_crs, center_y_crs, ref_transform, rows, cols, x_size_mm, model_y_mm)

    # Convert radius to model mm (grid spans cols-1 pixel spacings)
    terrain_width_m = (cols - 1) * px_size_x
    scale = x_size_mm / terrain_width_m  # mm per meter
    radius_mm = radius_m * scale

    # Generate n-gon vertices at exact radius
    angles = np.linspace(0, 2 * np.pi, n_gon_sides, endpoint=False)
    ngon_x = center_x_mm + radius_mm * np.cos(angles)
    ngon_y = center_y_mm + radius_mm * np.sin(angles)

    # Build rectangular DEM mesh for boolean intersection
    vertices_dem, faces_dem, _ = _build_rectangular_mesh(rows, cols, X, Y, z_surface_mm, valid_mask)

    # Boolean intersection with n-gon cylinder for smooth walls
    dem_mesh = trimesh.Trimesh(vertices=vertices_dem, faces=faces_dem, process=False)

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

    cylinder_mesh = trimesh.Trimesh(vertices=cylinder_verts, faces=cylinder_faces, process=False)

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
    # The mesh spans first-to-last pixel centers: cols-1 / rows-1 spacings.
    aspect_ratio = ((rows - 1) * px_size_y) / ((cols - 1) * px_size_x)
    model_y_mm = x_size_mm * aspect_ratio

    valid_mask = np.isfinite(dem)
    if not valid_mask.any():
        raise ValueError("DEM contains no valid data (all NaN/infinite)")

    valid_data = dem[valid_mask]
    min_elev = float(np.min(valid_data))
    max_elev = float(np.max(valid_data))
    height_range = max_elev - min_elev

    if use_true_scale:
        terrain_width_m = (cols - 1) * px_size_x
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

    water_faces_array: Optional[np.ndarray] = None
    if lake_mask is not None and lake_mask.any():
        cell_mask = lake_mask[:-1, :-1] & lake_mask[1:, :-1] & lake_mask[:-1, 1:] & lake_mask[1:, 1:]
        ci, cj = np.where(cell_mask)
        if ci.size:
            v00 = vertex_map[ci, cj].astype(np.int64)
            v10 = vertex_map[ci + 1, cj].astype(np.int64)
            v11 = vertex_map[ci + 1, cj + 1].astype(np.int64)
            v01 = vertex_map[ci, cj + 1].astype(np.int64)
            b00, b10, b11, b01 = (
                v00 + base_offset, v10 + base_offset,
                v11 + base_offset, v01 + base_offset,
            )

            blocks = [
                # Top and base surface of every lake cell
                np.column_stack((v00, v10, v11)),
                np.column_stack((v00, v11, v01)),
                np.column_stack((b00, b11, b10)),
                np.column_stack((b00, b01, b11)),
            ]

            # Walls where the neighboring cell is not a lake cell; the 1-cell
            # zero-padded mask reads out-of-bounds neighbors as non-lake.
            P = np.zeros((rows + 1, cols + 1), dtype=bool)
            P[1:rows, 1:cols] = cell_mask
            wall_north = ~P[ci, cj + 1]
            wall_south = ~P[ci + 2, cj + 1]
            wall_west = ~P[ci + 1, cj]
            wall_east = ~P[ci + 1, cj + 2]

            if wall_north.any():
                m = wall_north
                blocks.append(np.column_stack((v00[m], v01[m], b00[m])))
                blocks.append(np.column_stack((v01[m], b01[m], b00[m])))
            if wall_south.any():
                m = wall_south
                blocks.append(np.column_stack((v10[m], v11[m], b11[m])))
                blocks.append(np.column_stack((v10[m], b11[m], b10[m])))
            if wall_west.any():
                m = wall_west
                blocks.append(np.column_stack((v00[m], v10[m], b00[m])))
                blocks.append(np.column_stack((v10[m], b10[m], b00[m])))
            if wall_east.any():
                m = wall_east
                blocks.append(np.column_stack((v01[m], b01[m], v11[m])))
                blocks.append(np.column_stack((v11[m], b01[m], b11[m])))

            water_faces_array = np.concatenate(blocks, axis=0)

    max_z = float(np.max(z_surface_mm[valid_mask]))
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
) -> Tuple[Optional[trimesh.Trimesh], Optional[ShapelyPolygon]]:
    """Build the cutout shape (cylinder or rotated box) as a trimesh solid.

    Returns ``(mesh, footprint)`` where ``footprint`` is the cutout's 2D outline
    in model-mm coordinates (used to clip overlay polygons in 2D, so the inserts
    never go through a solid-vs-solid boolean that would subdivide their vertical
    walls).  Both are None if cutout_type is not recognized.
    """
    rows, cols = dem_shape
    max_terrain_z = float(np.max(z_surface_mm[valid_mask]))
    box_height = max(max_terrain_z * 2, base_thickness_mm * 3)

    transformer = Transformer.from_crs("EPSG:4326", ref_crs, always_xy=True)

    model_y_mm = float(Y[0, 0])

    if cutout_type == "circular":
        center_x_crs, center_y_crs = transformer.transform(cutout_center_lon, cutout_center_lat)
        center_x_mm, center_y_mm = _crs_point_to_model_xy(
            center_x_crs, center_y_crs, ref_transform, rows, cols, x_size_mm, model_y_mm)

        terrain_width_m = (cols - 1) * px_size_x
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

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        mesh.fix_normals()
        footprint = ShapelyPolygon(zip(ngon_x, ngon_y))
        return mesh, footprint

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

        terrain_width_m = abs(ref_transform.a) * (cols - 1)
        dem_scale = x_size_mm / terrain_width_m
        half_w = AB_length_m * dem_scale / 2.0
        half_h = AD_length_m * dem_scale / 2.0

        center_x_crs = (c1_x + c2_x) / 2.0
        center_y_crs = (c1_y + c2_y) / 2.0
        center_x_mm, center_y_mm = _crs_point_to_model_xy(
            center_x_crs, center_y_crs, ref_transform, rows, cols, x_size_mm, model_y_mm)

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
        mesh = trimesh.Trimesh(vertices=box_verts_rot, faces=box_faces, process=False)
        mesh.fix_normals()
        footprint = ShapelyPolygon([(v[0], v[1]) for v in box_verts_rot[:4]])
        return mesh, footprint

    return None, None


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

    terrain_width_m = abs(ref_transform.a) * (cols - 1)
    dem_scale = x_size_mm / terrain_width_m
    final_scale = x_size_mm / AB_length_m

    rect_width_mm_final = x_size_mm
    rect_height_mm_final = AD_length_m * final_scale

    center_x_crs = (c1_x_crs + c2_x_crs) / 2.0
    center_y_crs = (c1_y_crs + c2_y_crs) / 2.0
    model_y_mm = float(Y[0, 0])
    center_x_mm, center_y_mm = _crs_point_to_model_xy(
        center_x_crs, center_y_crs, ref_transform, rows, cols, x_size_mm, model_y_mm)

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
        if not p.is_valid:
            # Self-intersecting rings are common in OSM data; repair instead of
            # dropping the whole feature.  make_valid may return a collection —
            # keep only the polygonal parts.
            p = shapely.make_valid(p)
            if p.geom_type == "GeometryCollection":
                parts = [g2 for g2 in p.geoms if g2.geom_type in ("Polygon", "MultiPolygon")]
                if not parts:
                    continue
                p = unary_union(parts)
            if p.geom_type not in ("Polygon", "MultiPolygon"):
                continue
        if not p.is_empty:
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


def _region_triangles(piece: ShapelyPolygon):
    """Constrained-Delaunay triangulate a polygon piece -> list of xy coord triples.

    Uses only the piece's own vertices (no Steiner points), so a boundary shared
    with a neighbour keeps identical vertices on both sides.
    """
    tc = shapely.constrained_delaunay_triangles(piece)
    if tc.is_empty:
        return []
    return [list(t.exterior.coords)[:3] for t in tc.geoms]


def _dem_sampler(z_grid_asc: np.ndarray, xs: np.ndarray, ys: np.ndarray):
    """Bilinear DEM sampler on an ascending-x, ascending-y grid.

    Returns a callable mapping an (N,2) array of model-mm xy to interpolated Z.
    ``z_grid_asc[i, j]`` is the surface at (xs[j], ys[i]).
    """
    x0 = float(xs[0]); y0 = float(ys[0])
    dx = (float(xs[-1]) - x0) / (len(xs) - 1)
    dy = (float(ys[-1]) - y0) / (len(ys) - 1)
    nx, ny = len(xs), len(ys)

    def sample(xy: np.ndarray) -> np.ndarray:
        fx = (xy[:, 0] - x0) / dx
        fy = (xy[:, 1] - y0) / dy
        j = np.clip(np.floor(fx).astype(np.int64), 0, nx - 2)
        i = np.clip(np.floor(fy).astype(np.int64), 0, ny - 2)
        tx = np.clip(fx - j, 0.0, 1.0)
        ty = np.clip(fy - i, 0.0, 1.0)
        z00 = z_grid_asc[i, j]; z01 = z_grid_asc[i, j + 1]
        z10 = z_grid_asc[i + 1, j]; z11 = z_grid_asc[i + 1, j + 1]
        return (z00 * (1 - tx) * (1 - ty) + z01 * tx * (1 - ty)
                + z10 * (1 - tx) * ty + z11 * tx * ty)

    return sample


def _dem_min_over(poly, sampler, xs: np.ndarray, ys: np.ndarray) -> Optional[float]:
    """Exact minimum of the extruded top surface over a footprint.

    Reads the minimum from the SAME vertex set build_region_prism_fast emits for
    this region (via _region_top_surface): grid points inside the region PLUS the
    polygon-edge x grid-line crossing points the boundary cells' CDT introduces.
    A crossing point is a bilinear DEM sample that can dip below every interior grid
    point and every polygon vertex, so sampling only those (the old approximation)
    overestimated the minimum -- placing a flat floor at ``min - offset`` a hair too
    high and thinning the wall below ``offset`` at that crossing.  Only vertices the
    top faces reference are considered (boundary cells also carry their outside grid
    corners as unused vertices, which must not pull the minimum down).
    """
    xy, top_faces = _region_top_surface(poly, xs, ys)
    if xy is None:
        return None
    used = np.unique(top_faces)
    return float(np.min(sampler(xy[used])))


def _region_top_surface(poly, xs, ys, key_scale=1e6):
    """Top-surface vertices + triangulation for one 2D region over the DEM grid.

    Returns ``(xy, top_faces)``: ``xy`` is the (N, 2) array of every top-surface
    vertex the region emits -- grid points of the used cells plus the polygon-edge x
    grid-line crossing points the boundary cells' CDT introduces -- and ``top_faces``
    (M, 3) indexes them CCW-from-above.  ``(None, None)`` if the region covers no
    cell.  Note ``xy`` also carries the outside corners of boundary cells (needed so
    the prism's walls close), which no top face references; callers that want the
    true surface extent must restrict to ``np.unique(top_faces)``.

    Factored out of build_region_prism_fast so the flat-floor placement reads the
    surface minimum from the identical vertex set the mesh is built on.
    """
    minx, miny, maxx, maxy = poly.bounds
    j0 = max(int(np.searchsorted(xs, minx)) - 1, 0)
    j1 = min(int(np.searchsorted(xs, maxx)) + 1, len(xs) - 1)
    i0 = max(int(np.searchsorted(ys, miny)) - 1, 0)
    i1 = min(int(np.searchsorted(ys, maxy)) + 1, len(ys) - 1)
    gx = xs[j0:j1 + 1]; gy = ys[i0:i1 + 1]
    ni, nj = len(gy), len(gx)
    if ni < 2 or nj < 2:
        return None, None
    GX, GY = np.meshgrid(gx, gy)
    inside = shapely.contains_xy(poly, GX, GY)

    c00 = inside[:-1, :-1]; c10 = inside[1:, :-1]
    c11 = inside[1:, 1:];   c01 = inside[:-1, 1:]
    all_in = c00 & c10 & c11 & c01
    bnd = (c00 | c10 | c11 | c01) & ~all_in
    if not all_in.any() and not bnd.any():
        return None, None

    usedcell = all_in | bnd
    ptused = np.zeros((ni, nj), bool)
    ptused[:-1, :-1] |= usedcell; ptused[1:, :-1] |= usedcell
    ptused[1:, 1:] |= usedcell;   ptused[:-1, 1:] |= usedcell
    gridvid = np.full((ni, nj), -1, np.int64)
    pi, pj = np.where(ptused)
    n_grid = len(pi)
    gridvid[pi, pj] = np.arange(n_grid)
    grid_xy = np.column_stack((gx[pj], gy[pi]))

    gxkey = {round(v * key_scale): k for k, v in enumerate(gx)}
    gykey = {round(v * key_scale): k for k, v in enumerate(gy)}
    cross_xy = []
    cross_tbl = {}

    def vid(x, y):
        kx = round(x * key_scale); ky = round(y * key_scale)
        j = gxkey.get(kx); i = gykey.get(ky)
        if i is not None and j is not None and gridvid[i, j] >= 0:
            return int(gridvid[i, j])
        k = (kx, ky)
        got = cross_tbl.get(k)
        if got is None:
            got = cross_tbl[k] = n_grid + len(cross_xy)
            cross_xy.append((x, y))
        return got

    ci, cj = np.where(all_in)
    v00 = gridvid[ci, cj]; v10 = gridvid[ci + 1, cj]
    v11 = gridvid[ci + 1, cj + 1]; v01 = gridvid[ci, cj + 1]
    top_faces = [np.column_stack((v00, v01, v11)),
                 np.column_stack((v00, v11, v10))]

    bci, bcj = np.where(bnd)
    bnd_tris = []
    for i, j in zip(bci.tolist(), bcj.tolist()):
        inter = poly.intersection(shapely.geometry.box(gx[j], gy[i], gx[j + 1], gy[i + 1]))
        geoms = inter.geoms if inter.geom_type in (
            "MultiPolygon", "GeometryCollection") else [inter]
        for g in geoms:
            if g.geom_type != "Polygon" or g.is_empty or g.area <= 0:
                continue
            for t in _region_triangles(g):
                if (t[1][0] - t[0][0]) * (t[2][1] - t[0][1]) - \
                   (t[1][1] - t[0][1]) * (t[2][0] - t[0][0]) < 0:
                    t = [t[0], t[2], t[1]]
                bnd_tris.append([vid(*t[0]), vid(*t[1]), vid(*t[2])])
    if bnd_tris:
        top_faces.append(np.array(bnd_tris, np.int64))
    top_faces = np.vstack(top_faces)

    xy = grid_xy if not cross_xy else np.vstack((grid_xy, np.array(cross_xy)))
    return xy, top_faces


def build_region_prism_fast(poly, top_fn, bottom_fn, xs, ys, key_scale=1e6):
    """Watertight prism for one 2D region, extruded over the shared DEM grid.

    ``poly`` is a shapely (Multi)Polygon in model mm; ``top_fn``/``bottom_fn`` are
    callables (N,2 xy)->z for the two surfaces (a constant lambda for a flat face,
    the DEM sampler for a draped one, or ``DEM - t`` for a uniform-thickness floor).
    Interior grid cells are meshed vectorized; only the ~perimeter boundary cells
    are clipped with shapely (CDT).  Neighbours share boundary vertices (same grid +
    same polygon edge) so separately-built regions abut with no crack and no
    boolean.  Winding is correct by construction (CCW-from-above top, reversed
    bottom, walls oriented from the top surface's boundary edges) so no
    fix_normals/merge is needed (those dominate runtime on multi-million-face meshes).
    Returns a trimesh.Trimesh or None if the region covers no cells.
    """
    xy, top_faces = _region_top_surface(poly, xs, ys, key_scale)
    if xy is None:
        return None
    n = len(xy)
    verts = np.empty((2 * n, 3))
    verts[:n, :2] = xy; verts[:n, 2] = top_fn(xy)
    verts[n:, :2] = xy; verts[n:, 2] = bottom_fn(xy)

    de = np.vstack((top_faces[:, [0, 1]], top_faces[:, [1, 2]], top_faces[:, [2, 0]]))
    key = np.minimum(de[:, 0], de[:, 1]) * np.int64(2 ** 32) + np.maximum(de[:, 0], de[:, 1])
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    wall_de = de[cnt[inv] == 1]
    a, b = wall_de[:, 0], wall_de[:, 1]
    faces = np.vstack((top_faces, top_faces[:, ::-1] + n,
                       np.column_stack((a, b + n, b)),
                       np.column_stack((a, a + n, b + n))))
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def build_all_terrain_meshes(
    dem: np.ndarray,
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
    terrain_types: Optional[List[str]] = None,
    base_class: Optional[int] = None,
    insert_xy_clearance_mm: float = 0.0,
    insert_z_clearance_mm: float = 0.0,
    insert_corner_relief_mm: float = 0.0,
    insert_corner_min_angle_deg: float = 45.0,
) -> dict:
    """Build the base plate mesh and terrain overlay (insert) meshes.

    Args:
        class_geometries: dict mapping terrain class int → list of GeoJSON
            geometry dicts in CRS coordinates (from classify_terrain).
        base_class: terrain class used as the leftover base plate (never
            rasterized — it is whatever the overlays don't cover). Defaults to
            rock. The satellite-inverted Ararat print passes foliage, which turns
            rock into an ordinary insert class (its polygons must then be present
            in class_geometries). Overlay classes and their precedence come from
            terrain_classifier.TERRAIN_PRECEDENCE with the base removed.
        insert_xy_clearance_mm: per-side horizontal gap for separately-printed
            inserts.  The insert walls are inset by this amount while the rock
            pocket keeps the full polygon, yielding a uniform XY gap.  0 gives a
            touching fit for one-piece multimaterial printing.
        insert_z_clearance_mm: vertical relief at the hidden pocket floor.  The
            pocket is deepened by this amount while the insert keeps its full
            height, so the insert seats flush on its walls instead of bottoming
            out.  0 gives a touching fit.
        insert_corner_relief_mm: extra clearance at sharp corners (delta), on top
            of the flat XY clearance, to defeat FDM inside-corner over-extrusion
            that otherwise locks the fit.  Applied to the rock pocket at convex
            corners and to the insert at reflex corners.  0 disables corner relief.
        insert_corner_min_angle_deg: minimum boundary turn angle for a corner to
            receive relief.  Near-straight vertices are skipped so the flat
            clearance is preserved.

    Returns:
        dict mapping terrain name to (vertices, faces, max_z) or None.
    """
    from terrain_classifier import (TERRAIN_ROCK, TERRAIN_NAMES,
                                     TERRAIN_PRECEDENCE, overlay_precedence)
    from terrain_compose import (resolve_layers, fretted_bit_moves,
                                 MIN_THICKNESS_MM, MIN_BLOB_MM)

    # The base plate is the "leftover" class (never rasterized: it is whatever the
    # overlays don't cover). Rock is the base for normal prints; the satellite
    # inversion passes base_class=FOLIAGE. Overlay/insert classes and their
    # precedence come from the shared TERRAIN_PRECEDENCE (terrain_classifier).
    if base_class is None:
        base_class = TERRAIN_ROCK
    PRIORITY_ORDER = overlay_precedence(base_class)
    base_name = TERRAIN_NAMES[base_class]

    # Determine active (meshed) vs excluded terrain classes
    if terrain_types is not None:
        name_to_class = {TERRAIN_NAMES[c]: c for c in PRIORITY_ORDER}
        active_set = set()
        for t in terrain_types:
            t = t.strip()
            if t not in name_to_class:
                raise ValueError(f"Unknown terrain type '{t}'. Valid types: {list(name_to_class.keys())}")
            active_set.add(name_to_class[t])
    else:
        active_set = set(PRIORITY_ORDER)

    all_overlay_classes = [tc for tc in PRIORITY_ORDER if tc in active_set]

    rows, cols = dem.shape

    # Compute model coordinates once (shared by rock base and overlays)
    X, Y, z_surface_mm, valid_mask, _, model_y_mm = _compute_model_coordinates(
        dem, px_size_x, px_size_y, x_size_mm, max_height_mm,
        z_exaggeration, base_thickness_mm,
        lake_range_percent=0.0, lake_lowering_mm=0.0,
        use_true_scale=use_true_scale,
    )

    # Convert ALL class geometries to shapely in model mm (including the base
    # class's mask -- the resolver needs it to carve the complement).
    class_polys_mm = {}
    for tc in TERRAIN_PRECEDENCE:
        geoms = class_geometries.get(tc, [])
        poly = _polygons_to_model_mm(
            geoms, ref_transform, rows, cols, x_size_mm, model_y_mm,
        )
        class_polys_mm[tc] = poly

    # For excluded types, merge each component polygon into the active type
    # that spatially contains it.  Components not inside any active type
    # fall through to rock (no action needed).
    for tc in PRIORITY_ORDER:
        if tc in active_set or class_polys_mm[tc] is None:
            continue
        candidates = [c for c in PRIORITY_ORDER
                      if c in active_set and class_polys_mm[c] is not None]
        # Prepared geometries make the many point-in-polygon tests cheap, and
        # absorbed components are unioned once per candidate at the end rather
        # than re-unioning the whole candidate after every component.
        # (Equivalent to the incremental version: components of one class union
        # are disjoint, so absorbing one can never make another contained.)
        for c in candidates:
            shapely.prepare(class_polys_mm[c])
        absorbed = {c: [] for c in candidates}
        for component in _iter_polygon_components(class_polys_mm[tc]):
            pt = component.representative_point()
            for candidate in candidates:
                if class_polys_mm[candidate].contains(pt):
                    absorbed[candidate].append(component)
                    break
            # If no active type contains it, it becomes rock — nothing to do
        for candidate, comps in absorbed.items():
            if comps:
                class_polys_mm[candidate] = unary_union([class_polys_mm[candidate], *comps])
        class_polys_mm[tc] = None

    # Collapse boundary detail finer than the surface grid BEFORE the layers are
    # combined. A satellite mask can carry huge sub-pixel wiggle (~76k vertices,
    # 90% below one DEM sample) and every vertex becomes extruded walls. Half the
    # model-space DEM pitch is below what the sampled surface can represent, so
    # simplifying to it changes no printable shape (area drift < 0.1%) while
    # cutting vertices ~20x. All per-layer boundary processing must happen here,
    # before resolve_layers: the resolver's exact boolean differences are the last
    # thing to touch shared boundaries, so the circle stays exactly partitioned
    # (simplifying afterwards would move each layer's chords independently,
    # creating overlaps and orphan slivers along every shared boundary).
    grid_pitch_mm = (float(X.max()) - float(X.min())) / max(cols - 1, 1)
    simplify_tol_mm = 0.5 * grid_pitch_mm
    for tc in TERRAIN_PRECEDENCE:
        poly = class_polys_mm.get(tc)
        if poly is None:
            continue
        poly = shapely.simplify(poly, simplify_tol_mm, preserve_topology=True).buffer(0)
        class_polys_mm[tc] = poly if not poly.is_empty else None

    # Resolve masks -> mutually-exclusive inserts + base plate. Model mm is print
    # mm, so scale_m_per_mm=1.0. The oversized domain leaves inserts unclipped (the
    # cutout trims later); the complement it carves is clipped there too.
    margin = float(X.max()) - float(X.min())
    domain = shapely.geometry.box(float(X.min()) - margin, float(Y.min()) - margin,
                                  float(X.max()) + margin, float(Y.max()) + margin)
    masks = {tc: class_polys_mm[tc] for tc in TERRAIN_PRECEDENCE
             if class_polys_mm.get(tc) is not None}
    base_poly_mm, resolved_inserts = resolve_layers(
        domain, masks, base_class,
        min_thickness_mm=MIN_THICKNESS_MM, min_blob_mm=MIN_BLOB_MM, scale_m_per_mm=1.0)
    for tc in TERRAIN_PRECEDENCE:
        class_polys_mm[tc] = resolved_inserts.get(tc)

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

    # Build cutout shape once (the 3D mesh trims the rock base; the 2D footprint
    # clips overlay/insert polygons before any solid is built)
    cutout_shape, cutout_footprint = _build_cutout_shape(
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

    # Absorb sub-printable bits the cutout rim frets off any layer into a neighbour,
    # by the same rule the satellite interface uses in resolve_layers. The circular
    # cutout slices the resolved layers, and where it pinches a piece touching the
    # rim thinner than min_thickness it would print as a fragile edge sliver; that
    # bit is handed to whichever other layer it borders most. Bits are found on the
    # in-cutout footprint (so the rim is a real edge) but removed from / added to the
    # full (pre-cutout) polygons, so kept components keep their true outline and
    # insert inset/relief still seat flush against the rim.
    if cutout_footprint is not None:
        full = {base_class: base_poly_mm}
        for tc in all_overlay_classes:
            if class_polys_mm.get(tc) is not None:
                full[tc] = class_polys_mm[tc]
        clipped = {k: g.intersection(cutout_footprint).buffer(0)
                   for k, g in full.items()}
        moves = fretted_bit_moves(clipped, cutout_footprint.boundary,
                                  MIN_BLOB_MM, scale_m_per_mm=1.0)
        for frm, to, bit in moves:
            full[frm] = full[frm].difference(bit).buffer(0)
            full[to] = unary_union([full[to], bit]).buffer(0)
        base_poly_mm = full[base_class]
        for tc in all_overlay_classes:
            g = full.get(tc)
            class_polys_mm[tc] = g if g is not None and not g.is_empty else None

    # Pre-compute per overlay component: the recess flat_z (used by the rock
    # pocket) and, in flat mode, the cutout-clipped surface solid that the flat_z
    # is read from — reused below for the insert when there is no XY clearance.
    # The polygon is clipped to the cutout footprint in 2D first, so the surface
    # solid is built once with clean vertical walls.  The depth comes from the
    # true surface minimum (outline points included), independent of XY clearance.
    # List of (terrain_class, component_polygon, flat_z, full_solid).
    # Downsample polygons to the float32 STL output resolution before 2D->3D.  The
    # np.float32 cast makes np.spacing report the 32-bit ULP (~1.5e-5 mm), taken at
    # the model's largest coordinate where float32's absolute step is coarsest.
    output_resolution = np.spacing(np.float32(max(X.max(), Y.max())))

    # 2D-first setup: ascending model grid + a bilinear DEM sampler (model mm ==
    # print mm), and helpers to flatten a Z and to finalize a body.
    xs = np.asarray(X[0, :], dtype=float)
    ys = np.asarray(Y[::-1, 0], dtype=float)          # Y descends with row -> reverse
    z_grid_asc = np.asarray(z_surface_mm[::-1, :], dtype=float)
    if not np.isfinite(z_grid_asc).all():
        fill = float(np.nanmin(z_grid_asc))
        z_grid_asc = np.where(np.isfinite(z_grid_asc), z_grid_asc, fill)
        print("[WARN] DEM has voids inside the cutout; the base plate fills them "
              "with the minimum height (no spikes).", flush=True)
    sample_dem = _dem_sampler(z_grid_asc, xs, ys)
    max_terrain_z = float(np.max(z_grid_asc))

    def _flat(z):
        return lambda xy: np.full(len(xy), float(z))

    def _finalize(mesh):
        verts = mesh.vertices
        if cutout_type == "rectangular" and c1_x_crs is not None:
            verts = _apply_rect_cutout_transform(
                verts, dem.shape, px_size_x, x_size_mm, ref_transform, X, Y, bearing,
                c1_x_crs, c1_y_crs, c2_x_crs, c2_y_crs)
        v = verts.astype(np.float32)
        return v, mesh.faces.astype(np.int64), float(np.max(v[:, 2]))

    base_outline = (cutout_footprint if cutout_footprint is not None
                    else shapely.geometry.box(float(X.min()), float(Y.min()),
                                              float(X.max()), float(Y.max())))

    # Per insert component, in 2D: the pocket carved in the base (component +
    # convex corner relief) and the printed insert footprint (inset + reflex
    # relief), both clipped to the cutout. Floors come from the DEM minimum over the
    # footprint (pocket: min - thickness; insert part: min - (thickness -
    # z_clearance)); uniform mode drapes DEM - thickness instead of a flat floor.
    overlay_specs = []                                     # (tc, pocket_poly, flat_z|None)
    insert_specs = {tc: [] for tc in all_overlay_classes}  # tc -> [(part, floor|None)]
    for tc in all_overlay_classes:
        union_poly = class_polys_mm[tc]
        if union_poly is None:
            continue
        for component in _iter_polygon_components(union_poly):
            # Clip to the cutout, then drop small rim bits BEFORE the inset clearance
            # step, so the insert and its pocket are built from one cleaned footprint.
            # The rim clip can shear a sub-min_blob lobe off a component; left in, the
            # inset erases it from the insert while the pocket keeps it as an empty
            # recessed notch. Removing it here means no seat is carved where no insert
            # seats; the dropped area stays base terrain (base = cutout - pockets).
            insert_src = _clip_to_footprint(component, base_outline)
            insert_src = _drop_small_rim_bits(
                insert_src, base_outline.boundary, MIN_BLOB_MM ** 2)
            if insert_src is None:
                continue

            # Corner relief comes off the cleaned, clipped footprint -- not the raw
            # component -- so a convex corner that lies outside the cutout cannot drop
            # a relief disc that floats just inside the rim as an orphan pocket blob.
            pocket_extra, insert_cut = _corner_reliefs(
                insert_src, insert_xy_clearance_mm, insert_corner_relief_mm,
                insert_corner_min_angle_deg)

            # Pocket = the cleaned footprint (the seat) + convex corner relief.
            pocket_component = (unary_union([insert_src, pocket_extra])
                                if pocket_extra is not None else insert_src)
            pocket_poly = _clip_to_footprint(pocket_component, base_outline)
            if pocket_poly is not None:
                pocket_poly = _quantize_to_f32(pocket_poly, output_resolution)
            if pocket_poly is not None:
                if recess_mode == "uniform":
                    overlay_specs.append((tc, pocket_poly, None))
                else:
                    tmin = _dem_min_over(pocket_poly, sample_dem, xs, ys)
                    if tmin is not None:
                        overlay_specs.append(
                            (tc, pocket_poly, max(tmin - overlay_thickness_mm, 0.01)))

            # Printed insert footprint: inset by the clearance + reflex corner relief.
            if insert_xy_clearance_mm > 0 or insert_cut is not None:
                insert_poly = _inset_polygon(insert_src, insert_xy_clearance_mm)
                if insert_poly is None:
                    continue
                if insert_cut is not None:
                    insert_poly = insert_poly.difference(insert_cut).buffer(0)
                    if insert_poly.is_empty:
                        continue
            else:
                insert_poly = insert_src
            insert_poly = _quantize_to_f32(insert_poly, output_resolution)
            if insert_poly is None:
                continue
            for part in _iter_polygon_components(insert_poly):
                if recess_mode == "uniform":
                    insert_specs[tc].append((part, None))
                else:
                    pmin = _dem_min_over(part, sample_dem, xs, ys)
                    if pmin is not None:
                        insert_specs[tc].append(
                            (part, max(pmin - (overlay_thickness_mm - insert_z_clearance_mm), 0.01)))

    # --- Base plate: terrain top (base class) everywhere an insert doesn't seat, plus
    # each pocket recess floor. Base = cutout - pockets, one DEM-topped prism per
    # component; the pockets are the only recessed regions. Because a pocket is only
    # carved where an insert seats (small rim bits were dropped before the inset), the
    # base has no empty recessed notches. drop_unprintable's opened-off leftovers sit
    # in no pocket, so they stay at terrain height, flush with the surrounding base (a
    # sliver flush on both sides is not a fin). No solid-vs-solid boolean anywhere.
    pocket_polys = [pk for _, pk, _ in overlay_specs]
    pocket_union = unary_union(pocket_polys) if pocket_polys else None
    base_bodies = []

    def _add_base(poly, top_fn, bottom_fn):
        if poly is None or poly.is_empty:
            return
        for part in _iter_polygon_components(poly):
            m = build_region_prism_fast(part, top_fn, bottom_fn, xs, ys)
            if m is not None and len(m.faces) > 0:
                base_bodies.append(m)

    for tc, pocket_poly, flat_z in overlay_specs:
        if recess_mode == "uniform":
            top_fn = lambda xy: sample_dem(xy) - overlay_thickness_mm
        else:
            top_fn = _flat(flat_z)
        _add_base(pocket_poly, top_fn, _flat(0.0))

    # Base terrain: the cutout minus the insert pockets, draped at DEM height. The
    # base class (foliage when inverted, rock otherwise) is the terrain surface, and
    # the pockets are the only recessed regions -- so this holds for both scenes.
    base_fp = base_outline
    if pocket_union is not None:
        base_fp = base_fp.difference(pocket_union).buffer(0)
    if base_fp is not None:
        base_fp = _quantize_to_f32(base_fp, output_resolution)
    _add_base(base_fp, sample_dem, _flat(0.0))

    if base_bodies:
        result = {base_name: _finalize(trimesh.util.concatenate(base_bodies))}
    else:
        result = {base_name: None}

    # --- Inserts: each part a DEM-topped prism with a flat bottom (flat mode) or a
    # DEM-(thickness-clearance) bottom (uniform), concatenated as multiple bodies.
    # The 2D inset/relief/clip already gave the seating clearance; no boolean here.
    for tc in all_overlay_classes:
        name = TERRAIN_NAMES[tc]
        bodies = []
        for part, floor in insert_specs.get(tc, []):
            if recess_mode == "uniform":
                bottom_fn = (lambda xy: sample_dem(xy)
                             - (overlay_thickness_mm - insert_z_clearance_mm))
            else:
                bottom_fn = _flat(floor)
            m = build_region_prism_fast(part, sample_dem, bottom_fn, xs, ys)
            if m is not None and len(m.faces) > 0:
                bodies.append(m)
        if not bodies:
            result[name] = None
            continue
        result[name] = _finalize(trimesh.util.concatenate(bodies))

    return result


# Binary STL triangle record: 3xfloat32 normal + 3x3 float32 vertices + uint16 attr = 50 bytes
_STL_RECORD = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])


def save_stl(vertices: np.ndarray, faces: np.ndarray, output_path: str) -> None:
    """Write vertices/faces to a binary STL file.

    Vectorized writer that builds the binary STL directly with NumPy. This
    avoids constructing a trimesh.Trimesh, whose merge_vertices pass is a no-op
    on our already-indexed mesh (verified: 0 vertices merged on real DEM data)
    yet dominates save time on large meshes. Produces identical geometry and
    same-direction outward face normals as trimesh's exporter (verified: normal
    dot product = 1.0 across all faces on a 41M-face mesh).
    """
    tris = vertices[faces].astype("<f4")                       # (F, 3, 3)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    norm[norm == 0] = 1.0                                      # guard zero-area faces
    records = np.zeros(len(faces), dtype=_STL_RECORD)
    records["normal"] = (n / norm).astype("<f4")
    records["vertices"] = tris
    with open(output_path, "wb") as fh:
        fh.write(b"\x00" * 80)                                 # 80-byte header
        fh.write(np.array(len(faces), dtype="<u4").tobytes())  # uint32 triangle count
        fh.flush()
        # tofile() writes the record buffer directly, avoiding the full in-memory
        # copy that records.tobytes() would create (~2 GB on a 40M-face mesh).
        records.tofile(fh)
