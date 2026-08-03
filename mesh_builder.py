"""Mesh generation and STL export.

One entry point: ``build_terrain_meshes`` extrudes a finished
``terrain_layout.TerrainLayout`` into the base plate + insert bodies. It does no
polygon work at all -- the layout's boundaries are final and already snapped to the
float32 export grid, so anything here that moved one would break the shared seams.
A print with no mask providers reaches it too: the layout resolves to a single base
layer covering the cutout, and this stage drapes it over the DEM.
"""

from typing import Tuple, Optional

import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_origin
import trimesh

import shapely
from shapely.ops import unary_union


def _compute_model_coordinates(
    dem: np.ndarray,
    px_size_x: float,
    px_size_y: float,
    x_size_mm: float,
    max_height_mm: float,
    z_exaggeration: float,
    base_thickness_mm: float,
    use_true_scale: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute model-space coordinates from DEM data.

    Returns:
        (X, Y, z_surface_mm, valid_mask, model_y_mm)
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

    xs = np.linspace(0, x_size_mm, cols)
    ys = np.linspace(model_y_mm, 0, rows)
    X, Y = np.meshgrid(xs, ys)

    return X, Y, z_surface_mm, valid_mask, model_y_mm


def _dem_sampler(z_grid_asc: np.ndarray, frame):
    """Bilinear DEM sampler on an ascending-x, ascending-y grid.

    Returns a callable mapping an (N,2) array of PRINT-mm xy to interpolated Z.
    ``z_grid_asc[i, j]`` is the surface at grid point (xs[j], ys[i]).

    The lattice is axis-aligned in grid space, so the fractional index is two
    divisions -- but the xy handed in is print space, which a rotated cutout turns
    relative to the grid. Mapping back through ``frame.to_grid`` is a READ: it decides
    where to look up a height, and never moves the vertex whose height it is.
    """
    xs, ys = frame.grid_xs, frame.grid_ys
    x0 = float(xs[0]); y0 = float(ys[0])
    dx = (float(xs[-1]) - x0) / (len(xs) - 1)
    dy = (float(ys[-1]) - y0) / (len(ys) - 1)
    nx, ny = len(xs), len(ys)

    def sample(xy: np.ndarray) -> np.ndarray:
        g = frame.to_grid(xy)
        fx = (g[:, 0] - x0) / dx
        fy = (g[:, 1] - y0) / dy
        j = np.clip(np.floor(fx).astype(np.int64), 0, nx - 2)
        i = np.clip(np.floor(fy).astype(np.int64), 0, ny - 2)
        tx = np.clip(fx - j, 0.0, 1.0)
        ty = np.clip(fy - i, 0.0, 1.0)
        z00 = z_grid_asc[i, j]; z01 = z_grid_asc[i, j + 1]
        z10 = z_grid_asc[i + 1, j]; z11 = z_grid_asc[i + 1, j + 1]
        return (z00 * (1 - tx) * (1 - ty) + z01 * tx * (1 - ty)
                + z10 * (1 - tx) * ty + z11 * tx * ty)

    return sample


def _dem_min_over(poly, sampler, frame, resolution: float) -> Optional[float]:
    """Exact minimum of the extruded top surface over a footprint.

    Reads the minimum from the SAME vertex set build_region_prism_fast emits for
    this region (via _region_top_surface), so a flat floor at ``min - offset`` leaves
    a wall exactly ``offset`` thick at its thinnest point: each top triangle is a
    linear interpolant, whose minimum over the triangle is attained at a corner, so
    the minimum over the vertices IS the minimum over the built surface.  It is not
    the minimum of the underlying DEM over the footprint -- a boundary edge spanning
    several cells is one chord, riding above whatever the terrain does between its
    endpoints.  Densifying region boundaries on the grid lines would close that gap
    and belongs in the 2D stage, not here.

    Only vertices the top faces reference are considered: _region_top_surface also
    emits grid points that its CDT left outside the region, and those must not pull
    the minimum down.
    """
    xy, top_faces = _region_top_surface(poly, frame, resolution)
    if xy is None:
        return None
    used = np.unique(top_faces)
    return float(np.min(sampler(xy[used])))


def _interior_grid_points(region, constraints, frame, resolution):
    """DEM sampling points inside ``region`` that the export can tell from its edges.

    ``region`` and ``constraints`` are print-space geometry; the returned points are
    print-space too. The windowing and the proximity test run in GRID space, where the
    lattice is axis-aligned so a bisection over ``grid_xs``/``grid_ys`` is valid. The
    motion between the two is rigid, so it preserves both containment and the
    distances the proximity test compares.

    A grid point closer to a constraint edge than one float32 step of the output is
    not a usable sample: written to the STL it keeps its own value on one axis but
    collapses onto the edge's on the other, flattening the sliver triangle it
    anchors into a zero-area face. It carries nothing the edge does not already --
    boundaries are densified on these same grid lines -- so it is dropped rather
    than snapped, which would move a boundary vertex.

    Only the constraints are used for the proximity test, not ``region.bounds``, so
    the base plate excludes points hugging a pocket seam as well as the rim.

    The test is driven from the SEGMENTS, not from the points. ``resolution`` is
    four orders below the grid pitch, so each segment can only endanger the handful
    of nodes in its own bounding box, and the constraints are densified -- no
    segment spans more than one cell. That makes the work proportional to the
    boundary, not to the interior: a shapely ``dwithin`` over every interior point
    costs 45s on the Ararat base plate and finds nothing, where this costs
    milliseconds.
    """
    xs, ys = frame.grid_xs, frame.grid_ys
    minx, miny, maxx, maxy = frame.geom_to_grid(region).bounds
    j0 = max(int(np.searchsorted(xs, minx)) - 1, 0)
    j1 = min(int(np.searchsorted(xs, maxx)) + 1, len(xs) - 1)
    i0 = max(int(np.searchsorted(ys, miny)) - 1, 0)
    i1 = min(int(np.searchsorted(ys, maxy)) + 1, len(ys) - 1)
    wx, wy = xs[j0:j1 + 1], ys[i0:i1 + 1]
    GX, GY = np.meshgrid(wx, wy)
    P = frame.to_print(np.column_stack((GX.ravel(), GY.ravel())))
    ins = shapely.contains_xy(region, P[:, 0], P[:, 1]).reshape(GX.shape)
    if not ins.any():
        return np.empty((0, 2))

    A, B = _constraint_segments(frame.geom_to_grid(constraints))
    if len(A):
        too_close = _nodes_near_segments(A, B, wx, wy, resolution)
        ins &= ~too_close
        if not ins.any():
            return np.empty((0, 2))
    return P.reshape(GX.shape + (2,))[ins]


def _cells_crossed_by(geom_grid, frame):
    """Boolean (rows-1, cols-1) mask of grid cells a boundary passes through.

    ``geom_grid`` is in GRID space -- the cells only exist there -- so the caller does
    the mapping back from print space.

    Complete only because the layout densifies boundaries on the grid lines: no
    segment crosses a cell any more, so a segment's own bounding box reaches every
    cell it can touch. Over-marking is harmless (the cell just gets the exact test);
    under-marking is not, so the box is taken generously on both sides.
    """
    xs, ys = frame.grid_xs, frame.grid_ys
    nj, ni = len(xs) - 1, len(ys) - 1
    mask = np.zeros((ni, nj), bool)
    for g in (geom_grid.geoms if geom_grid.geom_type == "MultiPolygon" else [geom_grid]):
        for ring in [g.exterior] + list(g.interiors):
            c = np.asarray(ring.coords, float)
            if len(c) < 2:
                continue
            a, b = c[:-1], c[1:]
            j0 = np.clip(np.searchsorted(xs, np.minimum(a[:, 0], b[:, 0]), "left") - 1,
                         0, nj - 1)
            j1 = np.clip(np.searchsorted(xs, np.maximum(a[:, 0], b[:, 0]), "right") - 1,
                         0, nj - 1)
            i0 = np.clip(np.searchsorted(ys, np.minimum(a[:, 1], b[:, 1]), "left") - 1,
                         0, ni - 1)
            i1 = np.clip(np.searchsorted(ys, np.maximum(a[:, 1], b[:, 1]), "right") - 1,
                         0, ni - 1)
            for dj in range(3):
                for di in range(3):
                    mask[np.minimum(i0 + di, i1), np.minimum(j0 + dj, j1)] = True
    return mask


def _assign_pockets(cen, pockets, frame):
    """Index of the first pocket containing each triangle centroid; -1 for none.

    Centroids and pockets arrive in print space and are mapped back to grid space
    once, because the raster shortcut needs cells that are axis-aligned. The motion is
    rigid, so which pocket contains which centroid is the same question in either.

    Testing every centroid against every candidate pocket is what the base plate
    used to do, and on a real build it was 35 s of a 41 s mesh stage: ~900k
    point-in-polygon tests against boundaries carrying tens of thousands of
    coordinates.

    Almost all of that work is redundant. A grid cell that no pocket boundary
    crosses lies wholly inside or wholly outside every pocket, so one sample
    answers for every triangle in it -- and those cells are ~94% of the plate.
    Rasterising the pockets onto the cell grid answers them in one pass; only
    triangles in a crossed cell get the real test.

    Ties follow the same rule as before: the FIRST pocket in the list wins, which
    is the documented precedence in TerrainLayout.pockets. rasterize paints later
    shapes over earlier ones, so the list is burned in reverse.
    """
    xs, ys = frame.grid_xs, frame.grid_ys
    ni, nj = len(ys) - 1, len(xs) - 1
    out = np.full(len(cen), -1, np.int64)
    if ni < 1 or nj < 1 or not len(cen):
        return out

    pockets = [frame.geom_to_grid(pk) for pk in pockets]
    cen = frame.to_grid(cen)
    dx = float(xs[1] - xs[0])
    dy = float(ys[1] - ys[0])
    burn = rasterize([(pk, k + 1) for k, pk in enumerate(pockets)][::-1],
                     out_shape=(ni, nj), transform=from_origin(xs[0], ys[-1], dx, dy),
                     fill=0, all_touched=False, dtype="int32")[::-1]

    crossed = np.zeros((ni, nj), bool)
    for pk in pockets:
        crossed |= _cells_crossed_by(pk, frame)

    jj = np.clip(np.searchsorted(xs, cen[:, 0], "right") - 1, 0, nj - 1)
    ii = np.clip(np.searchsorted(ys, cen[:, 1], "right") - 1, 0, ni - 1)
    out[:] = burn[ii, jj].astype(np.int64) - 1

    exact = crossed[ii, jj]
    if exact.any():
        hits = shapely.STRtree(pockets).query(shapely.points(cen[exact]),
                                              predicate="within")
        got = np.full(int(exact.sum()), -1, np.int64)
        if hits.size:
            o = np.lexsort((hits[1], hits[0]))
            pc, pp = hits[0][o], hits[1][o]
            fst = np.ones(len(pc), bool)
            fst[1:] = pc[1:] != pc[:-1]
            got[pc[fst]] = pp[fst]
        out[exact] = got
    return out


def _constraint_segments(geom):
    """(A, B) endpoint arrays for every segment of a line or polygon boundary."""
    parts = [geom] if geom.geom_type in ("LineString", "LinearRing") \
        else list(shapely.get_parts(geom))
    A, B = [], []
    for ls in parts:
        if ls.geom_type not in ("LineString", "LinearRing"):
            continue
        c = np.asarray(ls.coords, float)
        if len(c) >= 2:
            A.append(c[:-1])
            B.append(c[1:])
    if not A:
        return np.empty((0, 2)), np.empty((0, 2))
    return np.vstack(A), np.vstack(B)


def _nodes_near_segments(A, B, wx, wy, resolution):
    """Boolean (len(wy), len(wx)) mask of nodes within ``resolution`` of a segment."""
    lox = np.minimum(A[:, 0], B[:, 0]) - resolution
    hix = np.maximum(A[:, 0], B[:, 0]) + resolution
    loy = np.minimum(A[:, 1], B[:, 1]) - resolution
    hiy = np.maximum(A[:, 1], B[:, 1]) + resolution
    j0 = np.searchsorted(wx, lox, "left")
    j1 = np.searchsorted(wx, hix, "right")
    i0 = np.searchsorted(wy, loy, "left")
    i1 = np.searchsorted(wy, hiy, "right")

    nj = np.maximum(j1 - j0, 0)
    ni = np.maximum(i1 - i0, 0)
    cnt = nj * ni
    keep = cnt > 0
    mask = np.zeros((len(wy), len(wx)), bool)
    if not keep.any():
        return mask

    # Expand each surviving segment's node box into flat (segment, i, j) triples.
    seg = np.repeat(np.flatnonzero(keep), cnt[keep])
    within = np.arange(cnt[keep].sum()) - np.repeat(
        np.concatenate(([0], np.cumsum(cnt[keep])[:-1])), cnt[keep])
    jj = j0[seg] + within % nj[seg]
    ii = i0[seg] + within // nj[seg]

    P = np.column_stack((wx[jj], wy[ii]))
    a, b = A[seg], B[seg]
    d = b - a
    dd = (d * d).sum(1)
    t = np.where(dd > 0, ((P - a) * d).sum(1) / np.where(dd > 0, dd, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    foot = a + t[:, None] * d
    near = np.hypot(P[:, 0] - foot[:, 0], P[:, 1] - foot[:, 1]) < resolution
    mask[ii[near], jj[near]] = True
    return mask


def _region_top_surface(poly, frame, resolution):
    """Top-surface vertices + triangulation for one 2D region over the DEM grid.

    Returns ``(xy, top_faces)``: ``xy`` is the (N, 2) array of top-surface vertices
    -- the region's own boundary coordinates plus every DEM grid point inside it --
    and ``top_faces`` (M, 3) indexes them CCW-from-above.  ``(None, None)`` if the
    region has no area.

    Built the same way ``build_base_solid`` builds the base plate: ONE constrained
    Delaunay triangulation whose constraint segments are the region's boundary rings
    and whose free vertices are the interior grid points.  The boundary is therefore
    reproduced exactly instead of being sampled, and the surface still follows the
    DEM at full grid resolution.

    The grid-walk this replaces classified each grid CELL by testing its four corner
    samples for containment, and skipped cells with no corner inside.  That is a
    proxy for "does this cell meet the region", and it fails wherever the boundary
    cuts a corner-free path through a cell: such a part of the region was silently
    dropped from the surface, whatever the region's overall size (a sub-cell bump on
    a large polygon vanished; so did any region smaller than one cell).  The pocket
    the insert seats into is built by the exact CDT path, so the two disagreed by up
    to a cell -- several times the designed XY clearance.  Constraining on the
    boundary removes the failure mode rather than narrowing it; measured at Ararat
    scale the two paths cost the same.

    ``-p`` takes the PSLG, ``-Y`` forbids Steiner points on the segments (so a
    boundary shared with a neighbour keeps identical vertices on both sides), ``-Q``
    is quiet.  No quality or area flags, so no interior Steiner points either.
    """
    import triangle as _triangle

    rings = []
    for p in (poly.geoms if poly.geom_type == "MultiPolygon" else [poly]):
        rings.append(np.asarray(p.exterior.coords, float)[:-1])
        for h in p.interiors:
            rings.append(np.asarray(h.coords, float)[:-1])
    rings = [r for r in rings if len(r) >= 3]
    if not rings:
        return None, None

    # Interior grid points, over the region's bbox window only.
    grid_xy = _interior_grid_points(poly, poly.boundary, frame, resolution)

    ring_all = np.vstack(rings)
    allxy = (np.vstack((ring_all, grid_xy)) if len(grid_xy) else ring_all).astype(float)

    # Intern at the float32 export resolution: two coordinates that round to the same
    # float32 (as the binary STL will) MUST become one vertex, or a sub-float32 sliver
    # triangle survives construction and collapses to a degenerate, non-manifold face
    # on export.  float32 keying can never over-merge: distinct grid lines are a whole
    # pitch (~0.18 mm at Ararat scale) apart.
    _, first, inv = np.unique(allxy.astype(np.float32), axis=0,
                              return_index=True, return_inverse=True)
    V = allxy[first]

    segs = []
    off = 0
    for r in rings:
        m = len(r)
        idx = inv[off:off + m]; off += m
        segs.append(np.column_stack((idx, np.roll(idx, -1))))   # ring closes on itself
    seg = np.vstack(segs)
    seg = seg[seg[:, 0] != seg[:, 1]]           # rings that f32-interning collapsed
    seg = np.unique(np.sort(seg, axis=1), axis=0)
    if not len(seg):
        return None, None

    B = _triangle.triangulate({"vertices": V, "segments": seg}, "pYQ")
    Vt = B["vertices"]; T = B["triangles"]

    # The triangulation fills the segments' convex hull; keep the triangles actually
    # inside the region, which drops both the concavities and the holes.
    cen = Vt[T].mean(axis=1)
    T = T[shapely.contains_xy(poly, cen[:, 0], cen[:, 1])]
    if not len(T):
        return None, None

    a, b, c = Vt[T[:, 0]], Vt[T[:, 1]], Vt[T[:, 2]]
    cw = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
          - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])) < 0
    T[cw] = T[cw][:, ::-1]
    return Vt, T


def build_region_prism_fast(poly, top_fn, bottom_fn, frame, resolution):
    """Watertight prism for one 2D region, extruded over the shared DEM grid.

    ``poly`` is a shapely (Multi)Polygon in model mm; ``top_fn``/``bottom_fn`` are
    callables (N,2 xy)->z for the two surfaces (a constant lambda for a flat face,
    the DEM sampler for a draped one, or ``DEM - t`` for a uniform-thickness floor).
    The top surface comes from ``_region_top_surface``, which constrains the region's
    own boundary into one CDT, so neighbours share boundary vertices (same segments +
    same grid) and separately-built regions abut with no crack and no boolean.
    Winding is correct by construction (CCW-from-above top, reversed bottom, walls
    oriented from the top surface's boundary edges) so no fix_normals/merge is needed
    (those dominate runtime on multi-million-face meshes).
    Returns a trimesh.Trimesh or None if the region has no area on the grid.
    """
    xy, top_faces = _region_top_surface(poly, frame, resolution)
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


def build_base_solid(base_outline, boundary_geoms, pockets, pocket_top_fns,
                     base_top_fn, frame, resolution):
    """One watertight, manifold terraced solid for the whole base plate.

    The plate is a 2.5D terrain with vertical cliffs: the base class is draped at the
    DEM everywhere an insert does not seat, and each pocket is a flat (or DEM-minus-
    thickness) recess floor.  It is built from ONE constrained Delaunay triangulation
    of the whole cutout -- every region border in ``boundary_geoms`` is a constraint
    segment and every DEM grid point inside ``base_outline`` is an input vertex -- so
    each interior edge is shared by exactly two triangles by construction (no per-region
    re-triangulation, hence no cracks).  Each triangle's top z comes from the pocket
    (highest priority = lowest index in ``pockets``) whose interior holds its centroid,
    else ``base_top_fn``; top vertices are keyed by (column, f32 z) so equal-z seams
    weld flush and differing-z seams get a single vertical cliff.  A cliff's vertical
    sides run through every plateau level present at their column, so the T-vertex where
    a third plateau meets is built in, not repaired.  Every cliff's low side is flat (a
    pocket floor, or z=0 for the outline), giving a shared-bottom convex vertical polygon
    triangulated by a branchless zip.  Fully vectorized; returns a Trimesh or None.
    """
    import triangle as _triangle

    reg_fns = list(pocket_top_fns)
    # ---- 1. PSLG: region-border segments + all interior DEM grid points (f32-deduped)
    ring_xy = []
    for g in boundary_geoms:
        parts = ([g] if g.geom_type == "LineString"
                 else shapely.get_parts(shapely.line_merge(g)))
        for ls in parts:
            c = np.asarray(ls.coords, float)
            if len(c) >= 2:
                ring_xy.append(c)
    if not ring_xy:
        return None
    ring_all = np.vstack(ring_xy)
    # Constraints here are the whole noded arrangement, not just the rim: a grid
    # point hugging a pocket seam collapses onto it on export exactly as one
    # hugging the outline would.
    grid_xy = _interior_grid_points(base_outline, unary_union(list(boundary_geoms)),
                                    frame, resolution)
    allxy = np.vstack((ring_all, grid_xy)).astype(float)
    _, first, inv = np.unique(allxy.astype(np.float32), axis=0,
                              return_index=True, return_inverse=True)
    V = allxy[first]
    n_ring = len(ring_all)
    off = 0
    segs = []
    for c in ring_xy:
        m = len(c)
        idx = inv[off:off + m]; off += m
        segs.append(np.column_stack((idx[:-1], idx[1:])))
    seg = np.vstack(segs)
    seg = seg[seg[:, 0] != seg[:, 1]]
    seg = np.unique(np.sort(seg, axis=1), axis=0)

    # ---- 2. one constrained Delaunay triangulation (no Steiner points added) ----
    B = _triangle.triangulate({"vertices": V, "segments": seg}, "pYQ")
    Vt = B["vertices"]; T = B["triangles"]
    cen = Vt[T].mean(axis=1)
    inb = shapely.contains_xy(base_outline, cen[:, 0], cen[:, 1])
    T = T[inb]; cen = cen[inb]
    if not len(T):
        return None

    # ---- 3. per-triangle top_fn: highest-priority pocket holding its centroid ----
    tri_fn = (_assign_pockets(cen, pockets, frame) if pockets
              else np.full(len(T), -1, np.int64))      # -1 = base

    # ---- 4. top vertices keyed by (column vertex, f32 z): equal z welds, diff steps
    corner_v = T.reshape(-1)
    corner_fn = np.repeat(tri_fn, 3)
    corner_xy = Vt[corner_v]
    cz = np.empty(len(corner_v))
    for fi in np.unique(corner_fn):
        m = corner_fn == fi
        cz[m] = (base_top_fn if fi == -1 else reg_fns[fi])(corner_xy[m])
    czf = cz.astype(np.float32)
    topkey = np.column_stack((corner_v.astype(np.int64),
                              czf.view(np.int32).astype(np.int64)))
    uk, uinv = np.unique(topkey, axis=0, return_inverse=True)
    n_top = len(uk)
    top_z = uk[:, 1].astype(np.int32).view(np.float32).astype(float)
    TOPV = np.column_stack((Vt[uk[:, 0], 0], Vt[uk[:, 0], 1], top_z))
    top_faces = uinv.reshape(-1, 3)

    # ---- 5. single z=0 bottom (one vertex per used column) ----
    used_v = np.unique(T)
    bot_map = np.full(len(Vt), -1, np.int64)
    bot_map[used_v] = np.arange(len(used_v)) + n_top
    BOTV = np.column_stack((Vt[used_v, 0], Vt[used_v, 1], np.zeros(len(used_v))))
    bot_faces = bot_map[T[:, ::-1]]

    P = np.vstack((TOPV, BOTV))
    base_of = np.empty(len(P), np.int64)
    base_of[:n_top] = uk[:, 0]
    base_of[n_top:] = used_v

    # per-column sorted top vertices (colcount>=3 marks a tripoint)
    o = np.lexsort((top_z, uk[:, 0]))
    scol = uk[o, 0]; sz = top_z[o]; sid = o.astype(np.int64)
    col_l = np.searchsorted(scol, np.arange(len(Vt)), "left")
    col_r = np.searchsorted(scol, np.arange(len(Vt)), "right")
    colcount = col_r - col_l

    # ---- 6. edge pool over the shared CDT (each interior edge -> exactly 2 tris) ----
    tv = uinv.reshape(-1, 3)
    ei = np.array([0, 1, 2]); ej = np.array([1, 2, 0])
    va = T[:, ei].reshape(-1); vb = T[:, ej].reshape(-1)
    ta = tv[:, ei].reshape(-1); tb = tv[:, ej].reshape(-1)
    tri_of = np.repeat(np.arange(len(T)), 3)
    swap = va > vb
    lo = np.where(swap, vb, va); hi = np.where(swap, va, vb)
    tlo = np.where(swap, tb, ta); thi = np.where(swap, ta, tb)
    ekey = lo.astype(np.int64) * (2 ** 32) + hi
    sidx = np.argsort(ekey, kind="stable")
    sk = ekey[sidx]
    _, start, cnt = np.unique(sk, return_index=True, return_counts=True)
    if (cnt > 2).any():
        raise ValueError(f"base edge shared by >2 triangles ({int((cnt > 2).sum())})")

    # ---- 7. cliffs: an upper (high) top edge over a lower one (a lower plateau, or
    # the z=0 bottom for an outline edge) ----
    ul = []; uh = []; ll = []; lh = []; rf = []
    b1 = cnt == 1                                       # outline edge -> down to z=0
    if b1.any():
        r = sidx[start[b1]]
        ul.append(tlo[r]); uh.append(thi[r])
        ll.append(bot_map[lo[r]]); lh.append(bot_map[hi[r]]); rf.append(tri_of[r])
    i2 = cnt == 2                                       # seam -> step where z differs
    if i2.any():
        s0 = start[i2]; r0 = sidx[s0]; r1 = sidx[s0 + 1]
        stp = ~((tlo[r0] == tlo[r1]) & (thi[r0] == thi[r1]))
        r0 = r0[stp]; r1 = r1[stp]
        hi0 = P[tlo[r0], 2] + P[thi[r0], 2] >= P[tlo[r1], 2] + P[thi[r1], 2]
        hR = np.where(hi0, r0, r1); lR = np.where(hi0, r1, r0)
        ul.append(tlo[hR]); uh.append(thi[hR])
        ll.append(tlo[lR]); lh.append(thi[lR]); rf.append(tri_of[hR])

    wall_faces = np.empty((0, 3), np.int64)
    if ul:
        Alo = np.concatenate(ul); Ahi = np.concatenate(uh)
        Blo = np.concatenate(ll); Bhi = np.concatenate(lh)
        refc = np.concatenate(rf)
        N = len(Alo)
        cp = base_of[Blo]; cq = base_of[Bhi]
        zAlo = P[Alo, 2]; zBlo = P[Blo, 2]; zAhi = P[Ahi, 2]; zBhi = P[Bhi, 2]

        def _mids(col, zl, zh):
            # ragged, loop-free: per cliff, its column's top vertices with zl<z<zh
            k = colcount[col]
            seg_ = np.repeat(np.arange(N), k)
            within = np.arange(int(k.sum())) - np.repeat(np.cumsum(k) - k, k)
            src = np.repeat(col_l[col], k) + within
            z = sz[src]
            m = (z > np.repeat(zl, k) + 1e-9) & (z < np.repeat(zh, k) - 1e-9)
            return seg_[m], sid[src][m]

        pseg, pmid = _mids(cp, zBlo, zAlo)
        qseg, qmid = _mids(cq, zBhi, zAhi)

        # every cliff vertex tagged by cliff, side (p=0/q=1); each side is a vertical
        # chain [floor, mids..., top].  The floor is flat -> a shared-bottom convex
        # polygon triangulated by: for each vertex above the two bottom corners emit
        # (v, prevSameSide, prevOppSide) -- two verts from one column, one from the
        # other, never the 3-collinear sliver a fan makes at a subdivided side.
        cvid = np.concatenate([Blo, Alo, Bhi, Ahi, pmid, qmid])
        cclf = np.concatenate([np.arange(N), np.arange(N), np.arange(N), np.arange(N),
                               pseg, qseg])
        cside = np.concatenate([np.zeros(2 * N, np.int8), np.ones(2 * N, np.int8),
                                np.zeros(len(pmid), np.int8), np.ones(len(qmid), np.int8)])
        oz = np.lexsort((P[cvid, 2], cclf))
        mclf = cclf[oz]; mside = cside[oz]; mvid = cvid[oz]
        nn = len(mvid); pos = np.arange(nn)
        lastA = np.maximum.accumulate(np.where(mside == 0, pos, -1))
        lastB = np.maximum.accumulate(np.where(mside == 1, pos, -1))
        lastA_pre = np.concatenate(([-1], lastA[:-1]))
        lastB_pre = np.concatenate(([-1], lastB[:-1]))
        isA = mside == 0
        prevSame = np.where(isA, lastA_pre, lastB_pre)
        prevOpp = np.where(isA, lastB, lastA)
        ok = (prevSame >= 0) & (prevOpp >= 0)
        ok[ok] &= (mclf[prevSame[ok]] == mclf[ok]) & (mclf[prevOpp[ok]] == mclf[ok])
        k = np.where(ok)[0]
        tris = np.column_stack((mvid[k], mvid[prevSame[k]], mvid[prevOpp[k]]))
        ck = mclf[k]
        out = 0.5 * (P[Alo[ck], :2] + P[Ahi[ck], :2]) - cen[refc[ck]]
        nrm = np.cross(P[tris[:, 1]] - P[tris[:, 0]], P[tris[:, 2]] - P[tris[:, 0]])
        flip = (nrm[:, 0] * out[:, 0] + nrm[:, 1] * out[:, 1]) < 0
        tris[flip] = tris[flip][:, ::-1]
        wall_faces = tris

    faces = np.vstack([top_faces, bot_faces, wall_faces])
    return trimesh.Trimesh(vertices=P, faces=faces, process=False)


def build_terrain_meshes(
    layout: "TerrainLayout",
    frame: "ModelFrame",
    dem: np.ndarray,
    max_height_mm: float,
    z_exaggeration: float,
    base_thickness_mm: float,
    overlay_thickness_mm: float,
    use_true_scale: bool = False,
    recess_mode: str = "flat",
    insert_z_clearance_mm: float = 0.0,
    water_lowering_mm: float = 0.0,
) -> dict:
    """Extrude a finished 2D terrain layout into the base plate + insert meshes.

    Pure assembly: every polygon in ``layout`` is final. Nothing here moves a
    boundary, and nothing may -- the layout's shared seams are already noded and
    rounded to the float32 export grid as one arrangement, and any further 2D op
    would part the copies again (see ``terrain_layout.snap_arrangement``). This
    stage only adds Z: the DEM surface on top, and a recess floor underneath.

    Args:
        layout: the finished 2D geometry (``terrain_layout.build_terrain_layout``).
        frame: the DEM -> model-mm mapping the layout was built in.
        dem: elevations on that frame's grid.
        overlay_thickness_mm: insert thickness -- the pocket floor sits this far
            below the terrain surface.
        insert_z_clearance_mm: vertical relief at the hidden pocket floor. The
            pocket is deepened by this amount while the insert keeps its full
            height, so the insert seats flush on its walls instead of bottoming out.
            0 gives a touching fit.
        recess_mode: "flat" places each recess floor at a single Z (the surface
            minimum over the footprint, so the insert has a flat underside);
            "uniform" drapes the floor at DEM - thickness for a constant-thickness
            insert.
        water_lowering_mm: how far the WATER class sinks below the ground it sits
            in, so a lake reads as a pool instead of a flush plug. It moves the
            whole water column -- surface, insert underside and pocket floor --
            by the same amount, leaving the seating fit untouched. A lowered water
            body is also FLAT-topped in either recess mode (its surface is one Z,
            the lowest ground under it minus the drop): a water surface draped over
            terrain is not water. 0 leaves water an ordinary insert, drawn at the
            DEM like every other class.

    Returns:
        dict mapping terrain name to (vertices, faces, max_z) or None.
    """
    from masks import TERRAIN_NAMES, TERRAIN_WATER

    # Model coordinates + surface heights. The horizontal half of this duplicates
    # `frame`; the Z half (exaggeration, base thickness, true scale) is what is
    # actually wanted here.
    X, Y, z_surface_mm, valid_mask, model_y_mm = _compute_model_coordinates(
        dem, frame.px_size_x, frame.px_size_y, frame.x_size_mm, max_height_mm,
        z_exaggeration, base_thickness_mm,
        use_true_scale=use_true_scale,
    )

    # 2D-first setup: a bilinear DEM sampler and helpers to flatten a Z and to finalize
    # a body. The lattice itself is read from the FRAME rather than from the X/Y
    # meshgrids, so the 2D stage and the 3D stage cannot end up on grids that differ in
    # the last bit; `z_surface_mm` is indexed to that same lattice, row 0 first.
    resolution = float(frame.output_resolution)
    z_grid_asc = np.asarray(z_surface_mm[::-1, :], dtype=float)
    if not np.isfinite(z_grid_asc).all():
        fill = float(np.nanmin(z_grid_asc))
        z_grid_asc = np.where(np.isfinite(z_grid_asc), z_grid_asc, fill)
        print("[WARN] DEM has voids inside the cutout; the base plate fills them "
              "with the minimum height (no spikes).", flush=True)
    sample_dem = _dem_sampler(z_grid_asc, frame)

    def _flat(z):
        return lambda xy: np.full(len(xy), float(z))

    def _finalize(mesh):
        # Cast and hand over -- no xy adjustment. Every coordinate here came from the
        # layout, in print space, already snapped to this float32 grid; a rectangular
        # cutout's turn onto the print axes is part of the frame the layout worked in
        # (terrain_layout.frame_with_print_motion), not something applied afterwards.
        v = mesh.vertices.astype(np.float32)
        return v, mesh.faces.astype(np.int64), float(np.max(v[:, 2]))

    # --- Pocket floors. Every pocket the layout emitted is built: it owns the 2D
    # geometry, including which pieces are viable, so nothing is filtered here.
    # The floor comes off the insert part seated in the pocket when the layout
    # named one (pocket_floor_refs): floor = dem_min(part) - thickness, so the
    # seating gap is exactly the z clearance. A recess with no insert floors off
    # its own ground.
    raw_pockets = [pocket_poly for _tc, pocket_poly in layout.pockets]
    pocket_classes = [tc for tc, _pocket_poly in layout.pockets]
    floor_refs = list(layout.pocket_floor_refs)
    if len(floor_refs) != len(raw_pockets):     # layouts predating floor refs
        floor_refs = [None] * len(raw_pockets)
    pocket_top_fns = []
    floors = []
    for tc, pocket_poly, ref in zip(pocket_classes, raw_pockets, floor_refs):
        drop = water_lowering_mm if tc == TERRAIN_WATER else 0.0
        if recess_mode == "uniform" and drop == 0.0:
            pocket_top_fns.append(lambda xy: sample_dem(xy) - overlay_thickness_mm)
            floors.append(np.inf)               # drapes; nothing to order it by
            continue
        tmin = _dem_min_over(ref if ref is not None else pocket_poly,
                             sample_dem, frame, resolution)
        if tmin is None:
            raise ValueError(
                "pocket has no top surface; the layout must not emit a degenerate "
                "region, and the mesh stage must not drop one")
        floors.append(max(tmin - drop - overlay_thickness_mm, 0.01))
        pocket_top_fns.append(_flat(floors[-1]))
    # Where pockets overlap (per-part holes share their connector web, relief
    # discs bulge across a neighbour), the overlap gets the LOWER floor.
    # build_base_solid takes the first containing pocket, so hand it the pockets
    # ordered deepest-first. Draped floors sort last (+inf): they follow the terrain
    # and have no single Z to compare, so a flat floor overlapping one wins.
    if floors:
        order = np.argsort(np.asarray(floors), kind="stable")
        raw_pockets = [raw_pockets[k] for k in order]
        pocket_top_fns = [pocket_top_fns[k] for k in order]

    # --- Base plate: base-class terrain at the DEM everywhere an insert doesn't seat,
    # plus each pocket recess floor -- one watertight, manifold terraced solid.
    # build_base_solid triangulates the whole cutout ONCE with every region border as a
    # constraint (so shared edges are bit-identical, no cracks) and drapes/floors each
    # triangle by the first pocket holding its centroid, else the DEM. The pockets
    # arrive ordered deepest-first (above), so where they overlap the lower floor
    # wins, and pockets only floor the recess (insert seating is built from
    # layout.insert_parts) so an overlap never touches the insert fit.
    base_mesh = build_base_solid(layout.base_outline, [layout.noded_boundaries],
                                 raw_pockets,
                                 pocket_top_fns, sample_dem, frame, resolution)
    if base_mesh is not None and len(base_mesh.faces) > 0:
        result = {layout.base_name: _finalize(base_mesh)}
    else:
        result = {layout.base_name: None}

    # --- Inserts: each part a DEM-topped prism with a flat bottom (flat mode) or a
    # DEM-(thickness-clearance) bottom (uniform), concatenated as multiple bodies.
    # The layout's inset/relief/clip already gave the seating clearance; no boolean here.
    # A lowered water part is the one exception to the DEM top: it is a flat slab,
    # sunk by the drop its pocket floor was sunk by, so it seats exactly as before.
    for tc in layout.overlay_classes:
        name = TERRAIN_NAMES[tc]
        drop = water_lowering_mm if tc == TERRAIN_WATER else 0.0
        bodies = []
        clamped = False
        for part in layout.insert_parts.get(tc, []):
            top_fn = sample_dem
            if recess_mode == "uniform" and drop == 0.0:
                bottom_fn = (lambda xy: sample_dem(xy)
                             - (overlay_thickness_mm - insert_z_clearance_mm))
            else:
                pmin = _dem_min_over(part, sample_dem, frame, resolution)
                if pmin is None:
                    raise ValueError(
                        "insert part has no top surface; the layout must not emit a "
                        "degenerate region, and the mesh stage must not drop one")
                want = pmin - drop - (overlay_thickness_mm - insert_z_clearance_mm)
                bottom_z = max(want, 0.01)
                if drop > 0.0:
                    if pmin - drop <= bottom_z:
                        raise ValueError(
                            f"water lowered by {drop} mm leaves no body above the "
                            f"model floor (surface would sit at {pmin - drop:.2f} mm); "
                            "reduce --lake-lowering-mm or raise --base-thickness-mm")
                    if want < bottom_z and not clamped:
                        clamped = True
                        print(f"[WARN] the lowered water body reaches the model floor: "
                              f"it is {bottom_z:.2f} mm thick instead of "
                              f"{overlay_thickness_mm - insert_z_clearance_mm:.2f} mm "
                              f"and seats with no z clearance. Raise "
                              f"--base-thickness-mm or lower --lake-lowering-mm.",
                              flush=True)
                    top_fn = _flat(pmin - drop)
                bottom_fn = _flat(bottom_z)
            m = build_region_prism_fast(part, top_fn, bottom_fn, frame,
                                        resolution)
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
