"""Terrain layer masks -> the final 2D polygons the mesh builder extrudes.

This is the whole polygon stage, and it is source-agnostic: it takes cleaned masks
from whichever provider produced them (OSM, Sentinel-2) and a ``ModelFrame``, and
returns a ``TerrainLayout`` -- the base outline, the pockets carved in it, and the
printed insert footprints, all in model mm. Nothing here reads DEM elevations; the
mesh builder consumes the layout as-is and only adds Z.

The ORDER of the steps below is load-bearing, and keeping it in one place is the
point of this module:

  1. masks -> model mm, then per-layer boundary work (excluded-type absorption,
     collapsing sub-grid wiggle) -- everything that moves a single layer's
     boundary must happen BEFORE the layers are combined;
  2. ``resolve_layers`` cuts them into a disjoint partition. Its exact boolean
     differences are the last thing to touch shared boundaries, so the cutout stays
     exactly partitioned. Simplifying afterwards would move each layer's chords
     independently and open overlaps and orphan slivers along every shared seam;
  3. the cutout rim frets sub-printable bits off the partition; each is handed to a
     neighbour rather than dropped, so the partition stays exact;
  4. per component: clip to the rim, drop rim runts, apply insert fit clearances;
  5. the base outline and every pocket boundary are snapped to the float32 export
     grid as ONE arrangement (``node_and_snap``). Snapping polygon-by-polygon
     diverges the two copies of every shared seam by up to a pixel, and the base
     solid then drapes the hairline cell between them to the full DEM height -- a
     zero-width, full-height razor fin.

Model mm are print mm, so the printable-feature rules (``MIN_THICKNESS_MM``,
``MIN_BLOB_MM``) apply directly and ``scale_m_per_mm`` is 1.0 throughout.
"""

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import shapely
from pyproj import Transformer
from shapely.geometry import (MultiPolygon, Point, Polygon as ShapelyPolygon, box)
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from bearing_utils import rotate_from_bearing_frame, rotate_to_bearing_frame
from model_frame import ModelFrame
from masks import (TERRAIN_NAMES, TERRAIN_PRECEDENCE, TERRAIN_ROCK,
                   overlay_precedence)
from terrain_compose import (MIN_BLOB_MM, MIN_THICKNESS_MM, fretted_bit_moves,
                             resolve_layers)


@dataclass(frozen=True)
class CutoutSpec:
    """The region of interest that trims the model to the printed shape."""

    cutout_type: Optional[str] = None            # "circular" | "rectangular" | None
    center_lat: Optional[float] = None
    center_lon: Optional[float] = None
    radius_m: Optional[float] = None
    side_length_km: Optional[float] = None
    n_gon_sides: int = 64
    bearing: float = 0.0
    rect_corner1_lat: Optional[float] = None
    rect_corner1_lon: Optional[float] = None
    rect_corner2_lat: Optional[float] = None
    rect_corner2_lon: Optional[float] = None


@dataclass(frozen=True)
class InsertFit:
    """Fit tolerances for separately-printed inserts (see build_terrain_layout)."""

    xy_clearance_mm: float = 0.0
    z_clearance_mm: float = 0.0
    corner_relief_mm: float = 0.0
    corner_min_angle_deg: float = 45.0


@dataclass
class TerrainLayout:
    """The finished 2D geometry of one print, in model mm.

    Every xy coordinate the print will ever have is fixed here. The mesh stage may
    only give each of these vertices a z and raise walls between them: it must not
    move, add, merge or drop a boundary vertex, and it must not drop a region. So
    the boundaries arrive already densified on the DEM grid lines, already noded
    and snapped into one arrangement, already quantized to the export's float32
    resolution, and already free of degenerate pieces.

    ``pockets`` is ORDERED: where two pockets overlap (a convex corner-relief disc
    can bulge across a neighbouring layer), the base solid resolves the overlap by
    the first pocket whose region holds the triangle centroid, so this order is
    the precedence and must not be rearranged.

    ``noded_boundaries`` is the constraint set the base plate is triangulated
    against -- the outline and every pocket boundary as one snapped arrangement.
    It is built here, from exactly the pockets in ``pockets``, so the constraint
    edges and the draped regions can never describe different sets.
    """

    base_class: int
    overlay_classes: List[int]
    base_outline: ShapelyPolygon
    pockets: List[Tuple[int, Union[ShapelyPolygon, MultiPolygon]]] = field(default_factory=list)
    insert_parts: Dict[int, List[ShapelyPolygon]] = field(default_factory=dict)
    noded_boundaries: object = None

    @property
    def base_name(self) -> str:
        return TERRAIN_NAMES[self.base_class]


# --------------------------------------------------------------------------
# Cutout
# --------------------------------------------------------------------------

def rect_crs_corners(ref_crs, spec: CutoutSpec):
    """The rectangular cutout's two opposite corners in CRS coords, or Nones.

    Pure geodesy in the DEM's CRS -- it needs no model scale, which is what lets the
    caller size the model FROM the rectangle (see ``rect_extent_m``) instead of the
    other way round.
    """
    if spec is None or spec.cutout_type != "rectangular" or not ref_crs:
        return None, None, None, None
    transformer = Transformer.from_crs("EPSG:4326", ref_crs, always_xy=True)
    bearing_rad = np.radians(spec.bearing)
    if spec.rect_corner1_lat is not None:
        c1_x, c1_y = transformer.transform(spec.rect_corner1_lon, spec.rect_corner1_lat)
        c2_x, c2_y = transformer.transform(spec.rect_corner2_lon, spec.rect_corner2_lat)
    else:
        cx, cy = transformer.transform(spec.center_lon, spec.center_lat)
        half = spec.side_length_km * 1000.0 / 2.0
        de1, dn1 = rotate_from_bearing_frame(-half, -half, bearing_rad)
        c1_x, c1_y = cx + de1, cy + dn1
        de2, dn2 = rotate_from_bearing_frame(half, half, bearing_rad)
        c2_x, c2_y = cx + de2, cy + dn2
    return c1_x, c1_y, c2_x, c2_y


def rect_extent_m(ref_crs, spec: CutoutSpec):
    """The rectangular cutout's (width, height) in metres, or (None, None).

    Width runs perpendicular to the bearing (the AB edge, the one ``--x-size-mm``
    sizes) and height along it. The corners arrive as a diagonal, so the bearing
    frame is what separates the two edges.

    This is smaller than the cropped raster: ``crop_to_cutout`` bounds the rectangle
    with a CRS-axis-aligned box rounded out to whole pixels, and a rotated rectangle
    needs a wider box still. The caller pins the model scale to THIS length so the
    printed rectangle is exactly ``--x-size-mm`` wide.
    """
    c1_x, c1_y, c2_x, c2_y = rect_crs_corners(ref_crs, spec)
    if c1_x is None:
        return None, None
    width_m, height_m = rotate_to_bearing_frame(c2_x - c1_x, c2_y - c1_y,
                                                np.radians(spec.bearing))
    return abs(width_m), abs(height_m)


def frame_with_print_motion(frame: ModelFrame,
                            spec: Optional[CutoutSpec]) -> ModelFrame:
    """``frame`` carrying the grid -> print motion this cutout needs.

    Only a rectangular cutout needs one: its edges become the print axes, so the model
    is turned by -bearing about the rectangle's centre and shifted until the
    rectangle's corner is the origin. A disc is rotation-invariant and a whole-raster
    model is already axis-aligned, so both keep the identity.

    No scale: the caller has already pinned ``x_size_mm`` to the cutout, which is what
    makes this a rigid motion and therefore safe to apply in the 2D stage. A rescale
    would have to wait until the mesh existed, and by then the float32 quantization has
    happened and moving the coordinates invalidates it.
    """
    if spec is None or spec.cutout_type != "rectangular":
        return frame
    c1_x, c1_y, c2_x, c2_y = rect_crs_corners(frame.ref_crs, spec)
    if c1_x is None:
        return frame
    width_m, height_m = rect_extent_m(frame.ref_crs, spec)
    scale = frame.x_size_mm / ((frame.cols - 1) * frame.px_size_x)
    return replace(
        frame,
        print_bearing=float(spec.bearing),
        print_pivot_mm=frame.point_to_mm((c1_x + c2_x) / 2.0, (c1_y + c2_y) / 2.0),
        print_origin_mm=(width_m * scale / 2.0, height_m * scale / 2.0),
    )


def cutout_footprint(frame: ModelFrame,
                     spec: Optional[CutoutSpec]) -> Optional[ShapelyPolygon]:
    """The cutout's 2D outline in PRINT mm, or None when there is no cutout.

    Overlay polygons are clipped to this in 2D so the cutout boundary becomes
    ordinary prism walls (one vertical segment) instead of a solid-vs-solid boolean
    that would subdivide the existing walls with intermediate vertices. Needs no DEM
    elevations -- only the frame's georeference.
    """
    if spec is None or spec.cutout_type is None:
        return None
    if not frame.ref_transform or not frame.ref_crs:
        return None

    transformer = Transformer.from_crs("EPSG:4326", frame.ref_crs, always_xy=True)
    cols = frame.cols

    if spec.cutout_type == "circular":
        center_x_crs, center_y_crs = transformer.transform(spec.center_lon, spec.center_lat)
        center_x_mm, center_y_mm = frame.point_to_mm(center_x_crs, center_y_crs)

        terrain_width_m = (cols - 1) * frame.px_size_x
        scale = frame.x_size_mm / terrain_width_m
        radius_mm = spec.radius_m * scale

        angles = np.linspace(0, 2 * np.pi, spec.n_gon_sides, endpoint=False)
        ngon_x = center_x_mm + radius_mm * np.cos(angles)
        ngon_y = center_y_mm + radius_mm * np.sin(angles)
        return ShapelyPolygon(frame.to_print(np.column_stack((ngon_x, ngon_y))))

    if spec.cutout_type == "rectangular":
        bearing_rad = np.radians(spec.bearing)
        c1_x, c1_y, c2_x, c2_y = rect_crs_corners(frame.ref_crs, spec)
        width_m, height_m = rect_extent_m(frame.ref_crs, spec)

        # Same scale as the circular branch above. The caller has pinned x_size_mm to
        # the cutout, so this is the final print scale.
        terrain_width_m = (cols - 1) * frame.px_size_x
        dem_scale = frame.x_size_mm / terrain_width_m
        half_w = width_m * dem_scale / 2.0
        half_h = height_m * dem_scale / 2.0

        center_x_mm, center_y_mm = frame.point_to_mm((c1_x + c2_x) / 2.0,
                                                     (c1_y + c2_y) / 2.0)
        corners_local = [(-half_w, -half_h), (half_w, -half_h),
                         (half_w, half_h), (-half_w, half_h)]
        corners = []
        for vx, vy in corners_local:
            de, dn = rotate_from_bearing_frame(vx, vy, bearing_rad)
            corners.append((de + center_x_mm, dn + center_y_mm))
        # Through the frame's own motion rather than straight to (0, 0, W, H): the rim
        # is a seam shared with every insert clipped against it, so its coordinates
        # must come out of the same arithmetic as theirs, to the bit.
        return ShapelyPolygon(frame.to_print(corners))

    return None


# --------------------------------------------------------------------------
# 2D helpers
# --------------------------------------------------------------------------

def iter_polygon_components(geom: shapely.Geometry):
    """Yield individual Polygon components from a Polygon or MultiPolygon."""
    if geom.geom_type == "Polygon":
        yield geom
    elif geom.geom_type == "MultiPolygon":
        yield from geom.geoms


def _inset_polygon(polygon_mm, distance_mm: float):
    """Shrink a polygon inward by ``distance_mm`` on every side.

    Gives a separately-printed insert horizontal clearance from its pocket.
    ``buffer(-d)`` may return a (Multi)Polygon or empty geometry when a feature is
    narrower than 2*distance; both are handled downstream. Returns the original
    polygon when distance is non-positive, or None if the inset erases it entirely.
    """
    if distance_mm <= 0:
        return polygon_mm
    inset = polygon_mm.buffer(-distance_mm)
    if inset.is_empty:
        return None
    return inset


def _corner_reliefs(component, clearance_mm: float, relief_mm: float,
                    min_turn_deg: float):
    """Corner-relief geometry for a separately-printed insert.

    FDM over-extrudes reentrant (inside) corners, so a sharp corner seats tighter
    than the flat clearance and can lock. This adds ``relief_mm`` of extra clearance
    at sharp corners, on whichever body owns the inside corner:

      * convex footprint corner -> the pocket has the inside corner, so enlarge the
        pocket: a disc of radius (clearance+relief) centred on the insert's inset
        corner point (concentric with the insert corner, making the corner gap
        clearance+relief).
      * reflex footprint corner -> the insert has the inside corner, so shrink the
        insert: the same disc, centred on the nominal vertex.

    Only corners turning by at least ``min_turn_deg`` are relieved; a disc at a
    near-straight vertex would push the flat wall out by ``relief_mm`` and wreck the
    flat clearance. The signed turn angle classifies convex vs reflex and catches
    needle-sharp corners (turn near 180 deg) that a sine test would miss.

    Returns (pocket_extra, insert_cut): the union of convex discs to add to the
    pocket (or None) and the union of reflex discs to subtract from the insert.
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


def _clip_to_footprint(polygon_mm, footprint):
    """Clip a polygon to the cutout footprint in 2D.

    Returns the original polygon when there is no cutout, a cleaned (Multi)Polygon
    when the clip is non-empty, or None when nothing remains.
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


def _drop_small_rim_bits(polygon_mm, rim, min_area_mm2: float):
    """Drop connected components smaller than ``min_area_mm2`` that touch the rim.

    Clipping an insert footprint to the cutout can shear a sub-printable lobe off at
    the rim. Removed here -- BEFORE the inset clearance step -- so the insert and its
    pocket are built from one cleaned footprint: otherwise the inset erases the lobe
    from the insert (``buffer(-clearance)``) while the pocket keeps it, leaving an
    empty recessed notch in the perimeter. The dropped area sits in no pocket, so it
    stays ordinary base terrain (base = cutout - pockets). Interior components are
    untouched (resolve_layers already enforced min_blob on whole components); only
    rim-touching runts go. Returns the cleaned geometry, or None if nothing remains.
    """
    if polygon_mm is None or polygon_mm.is_empty:
        return None
    keep = [c for c in iter_polygon_components(polygon_mm)
            if not (c.area < min_area_mm2 and rim.intersects(c))]
    if not keep:
        return None
    return keep[0] if len(keep) == 1 else unary_union(keep)


def _quantize_to_f32(polygon_mm, output_resolution):
    """Quantize polygon coordinates to float32 precision, before 2D->3D.

    Binary STL stores vertices as 32-bit floats, so any polygon feature finer than
    float32 resolution (a near-pinch where two boundary points are a few nm apart)
    collapses to coincident vertices on export and becomes a zero-area triangle that
    breaks slicing. ``set_precision`` rounds coordinates to ``output_resolution``
    *and* removes the resulting duplicate/collapsed vertices (the default
    "valid_output" mode), so no sub-resolution sliver is ever built. Returns the
    quantized (Multi)Polygon, or None if it collapses to nothing.
    """
    quantized = shapely.set_precision(polygon_mm, output_resolution)
    if quantized.is_empty or quantized.geom_type not in ("Polygon", "MultiPolygon"):
        return None
    # set_precision leaves a *persistent* fixed precision model on the geometry. Any
    # later GEOS overlay (the per-cell poly.intersection in _region_top_surface) would
    # then run OverlayNG under that model and snap-round every output vertex to the
    # grid -- moving exact grid corners off-grid and producing duplicate vertices,
    # spurious interior walls, and non-manifold edges. The coordinates are already
    # quantized; clear the model back to floating so downstream intersections preserve
    # pass-through vertices exactly (grid_size=0 leaves coordinates bit-identical).
    return shapely.set_precision(quantized, 0.0)


def densify_on_grid(geom, frame):
    """Insert a boundary vertex wherever a segment crosses a DEM grid line.

    Pure vertex insertion: every ring passes through exactly the points it did
    before, so the footprint is the same shape and the same area. What changes is
    how well the mesh stage can follow the terrain along that boundary.

    It has to happen here because the 2D stage owns every xy coordinate. The mesh
    stage reads a z for each boundary vertex and interpolates linearly between
    them, so a boundary edge spanning several cells becomes one chord riding over
    whatever the DEM does in between -- even though the DEM's bilinear interpolant
    already defines that profile exactly. Splitting at the crossings puts the
    breaks where the interpolant's own breaks are, and matches the interior, whose
    every grid node is already a vertex.

    Two rings sharing a seam traverse it in OPPOSITE directions, so the crossings
    must not depend on which way the segment is walked: ``y0 + t * (y1 - y0)`` and
    ``y1 + (1 - t) * (y0 - y1)`` differ in the last bits, and a seam whose two
    copies sit an ULP apart is precisely the hairline cell the base solid drapes to
    full DEM height as a razor fin. Each segment is therefore oriented canonically
    (lexicographically smaller endpoint first) before its crossings are computed.

    ``geom`` is print-space, and the crossings are computed on those rotated
    coordinates: each vertex's grid coordinates are read off only to find which grid
    lines the segment spans and the crossing fraction along it; the crossing point
    itself is built on the print-space segment, so it sits within the intersection
    arithmetic's own rounding of the boundary -- the same as on an unrotated frame.
    Building it in grid space and mapping it back would stack the motion's rounding
    on top. Either way the residual is ~1e-16, far below the float32 snap that
    follows; the vertices the ring already had are copied through untouched.
    """
    xs, ys = frame.grid_xs, frame.grid_ys

    def ring(coords):
        c = np.asarray(coords, float)
        g = frame.to_grid(c)
        out = []
        for k in range(len(c) - 1):
            p0 = c[k]
            flip = (c[k + 1][0], c[k + 1][1]) < (p0[0], p0[1])
            i0, i1 = (k + 1, k) if flip else (k, k + 1)
            (x0, y0), (x1, y1) = c[i0], c[i1]
            (u0, v0), (u1, v1) = g[i0], g[i1]
            cx = np.empty(0)
            cy = np.empty(0)
            if u1 != u0:
                lo, hi = (u0, u1) if u0 < u1 else (u1, u0)
                t = (xs[(xs > lo) & (xs < hi)] - u0) / (u1 - u0)
                cx = np.concatenate((cx, x0 + t * (x1 - x0)))
                cy = np.concatenate((cy, y0 + t * (y1 - y0)))
            if v1 != v0:
                lo, hi = (v0, v1) if v0 < v1 else (v1, v0)
                t = (ys[(ys > lo) & (ys < hi)] - v0) / (v1 - v0)
                cx = np.concatenate((cx, x0 + t * (x1 - x0)))
                cy = np.concatenate((cy, y0 + t * (y1 - y0)))
            if len(cx):
                # Order along the canonical direction, then drop the duplicate a
                # segment through a grid NODE produces (it crosses one line of each
                # family at the same point).
                d = (cx - x0) * (x1 - x0) + (cy - y0) * (y1 - y0)
                o = np.argsort(d, kind="stable")
                pts = np.column_stack((cx[o], cy[o]))
                keep = np.ones(len(pts), bool)
                keep[1:] = np.any(pts[1:] != pts[:-1], axis=1)
                pts = pts[keep]
                if flip:
                    pts = pts[::-1]
            else:
                pts = np.empty((0, 2))
            out.append(np.vstack((p0[None, :], pts)))
        out.append(c[-1:])
        return np.vstack(out)

    def one(p):
        return ShapelyPolygon(ring(p.exterior.coords),
                              [ring(h.coords) for h in p.interiors])

    if geom is None or geom.is_empty:
        return geom
    if geom.geom_type == "MultiPolygon":
        return MultiPolygon([one(p) for p in geom.geoms])
    return one(geom)


def node_and_snap(base_outline, pockets, output_resolution):
    """Node the base outline + pocket boundaries into ONE snapped arrangement.

    The result is the constraint set the base plate is triangulated against. Every
    boundary is snapped together, in a single ``set_precision`` call: a global
    snap-round collapses every near-coincident copy of a shared seam onto the same
    grid points, whereas snapping each polygon separately snap-rounds it against its
    own hot pixels and diverges the two copies of every shared seam (pocket/pocket
    and pocket/rim) by up to a pixel. The noded arrangement then keeps both copies
    and the base solid drapes the degenerate ribbon cell between them to the DEM: a
    zero-width, full-height razor fin.

    ``unary_union`` afterwards dissolves the now-duplicate segments so each seam is
    exactly one constraint edge, and ``set_precision(_, 0.0)`` clears the persistent
    precision model (see ``_quantize_to_f32``).

    ``pockets`` must be the pockets that will actually be built, so the constraints
    and the draped regions agree.
    """
    boundaries = [base_outline.boundary] + [pk.boundary for pk in pockets]
    noded = unary_union(boundaries)
    noded = shapely.set_precision(noded, output_resolution)
    noded = shapely.set_precision(noded, 0.0)
    return unary_union(noded)


# --------------------------------------------------------------------------
# The layout
# --------------------------------------------------------------------------

def build_terrain_layout(
    frame: ModelFrame,
    class_geometries: Dict[int, List[dict]],
    outline: Optional[ShapelyPolygon] = None,
    base_class: Optional[int] = None,
    terrain_types: Optional[List[str]] = None,
    fit: InsertFit = InsertFit(),
) -> TerrainLayout:
    """Resolve terrain masks into the final printable 2D geometry.

    Args:
        frame: the DEM -> model-mm mapping.
        class_geometries: terrain class -> list of GeoJSON geometry dicts in
            ``frame.ref_crs``, already cleaned by their provider (satellite masks
            arrive despeckled + APCSF-smoothed; OSM rings arrive raw). Must NOT
            contain ``TERRAIN_ROCK``: rock is the one class with no mask, derived
            by ``resolve_layers`` as the leftover of all the others.
        outline: the cutout footprint in model mm (``cutout_footprint``), or None
            for the whole grid. Clipped to the DEM extent, so no part of the print
            can be placed where there is no elevation data.
        base_class: the terrain class used as the base plate the inserts seat into.
            Defaults to rock. The satellite-inverted Ararat print passes foliage,
            which makes rock an ordinary insert -- still derived, not supplied.
        terrain_types: names of the overlay classes to actually build. Components of
            an excluded class are merged into whichever active class spatially
            contains them; the rest fall through to the derived leftover.
        fit: insert clearances. ``xy_clearance_mm`` is the per-side horizontal gap
            (the insert walls are inset by it while the pocket keeps the full
            polygon); ``corner_relief_mm`` adds extra clearance at sharp corners to
            defeat FDM inside-corner over-extrusion. Zero gives a touching fit for
            one-piece multimaterial printing. ``z_clearance_mm`` is applied by the
            mesh builder, which owns the floors.
    """
    if base_class is None:
        base_class = TERRAIN_ROCK
    if class_geometries.get(TERRAIN_ROCK):
        raise ValueError(
            "TERRAIN_ROCK must not be supplied as a mask: it is the derived "
            "leftover class (TERRAIN_PRECEDENCE[-1]), and resolve_layers ignores "
            "any geometry passed for it."
        )

    priority_order = overlay_precedence(base_class)

    # Determine active (meshed) vs excluded terrain classes
    if terrain_types is not None:
        name_to_class = {TERRAIN_NAMES[c]: c for c in priority_order}
        active_set = set()
        for t in terrain_types:
            t = t.strip()
            if t not in name_to_class:
                raise ValueError(
                    f"Unknown terrain type '{t}'. Valid types: {list(name_to_class.keys())}")
            active_set.add(name_to_class[t])
    else:
        active_set = set(priority_order)

    all_overlay_classes = [tc for tc in priority_order if tc in active_set]

    # Convert ALL class geometries to shapely in model mm (including the base
    # class's mask -- the resolver needs it to carve the complement).
    class_polys_mm = {tc: frame.geojsons_to_mm(class_geometries.get(tc, []))
                      for tc in TERRAIN_PRECEDENCE}

    # For excluded types, merge each component polygon into the active type that
    # spatially contains it. Components not inside any active type fall through to
    # the derived leftover (no action needed).
    for tc in priority_order:
        if tc in active_set or class_polys_mm[tc] is None:
            continue
        candidates = [c for c in priority_order
                      if c in active_set and class_polys_mm[c] is not None]
        # Prepared geometries make the many point-in-polygon tests cheap, and
        # absorbed components are unioned once per candidate at the end rather than
        # re-unioning the whole candidate after every component. (Equivalent to the
        # incremental version: components of one class union are disjoint, so
        # absorbing one can never make another contained.)
        for c in candidates:
            shapely.prepare(class_polys_mm[c])
        absorbed = {c: [] for c in candidates}
        for component in iter_polygon_components(class_polys_mm[tc]):
            pt = component.representative_point()
            for candidate in candidates:
                if class_polys_mm[candidate].contains(pt):
                    absorbed[candidate].append(component)
                    break
        for candidate, comps in absorbed.items():
            if comps:
                class_polys_mm[candidate] = unary_union([class_polys_mm[candidate], *comps])
        class_polys_mm[tc] = None

    # Collapse boundary detail finer than the surface grid BEFORE the layers are
    # combined. A satellite mask can carry huge sub-pixel wiggle (~76k vertices, 90%
    # below one DEM sample) and every vertex becomes extruded walls. Half the
    # model-space DEM pitch is below what the sampled surface can represent, so
    # simplifying to it changes no printable shape (area drift < 0.1%) while cutting
    # vertices ~20x. See the module docstring on why this cannot happen later.
    simplify_tol_mm = 0.5 * frame.grid_pitch_mm
    for tc in TERRAIN_PRECEDENCE:
        poly = class_polys_mm.get(tc)
        if poly is None:
            continue
        poly = shapely.simplify(poly, simplify_tol_mm, preserve_topology=True).buffer(0)
        class_polys_mm[tc] = poly if not poly.is_empty else None

    # Nothing may be built beyond the DEM: the height sampler clamps outside the grid,
    # so terrain there would come out flat at the nearest edge value. Bounding the
    # OUTLINE bounds everything, because the pockets, the insert footprints and the
    # base plate are all cut from it. A cutout reaching past the raster is not a
    # geometry case but a coverage one -- crop_to_cutout clamps its window to the tiles
    # it was handed (dem_processing.py:261-264), so a cutout larger than the supplied
    # DEM coverage passes through silently. Clipped only when it actually overhangs, so
    # the ordinary case keeps the outline's own coordinates untouched.
    dem_extent = frame.print_footprint()
    if outline is not None and not dem_extent.covers(outline):
        outline = outline.intersection(dem_extent)

    # Resolve masks -> mutually-exclusive inserts + base plate. Model mm is print mm,
    # so scale_m_per_mm=1.0. The oversized domain leaves inserts unclipped (the cutout
    # trims later); the complement it carves is clipped there too.
    minx, miny, maxx, maxy = frame.print_bounds_mm
    margin = maxx - minx
    domain = box(minx - margin, miny - margin, maxx + margin, maxy + margin)
    masks = {tc: class_polys_mm[tc] for tc in TERRAIN_PRECEDENCE
             if class_polys_mm.get(tc) is not None}
    base_poly_mm, resolved_inserts = resolve_layers(
        domain, masks, base_class,
        min_thickness_mm=MIN_THICKNESS_MM, min_blob_mm=MIN_BLOB_MM, scale_m_per_mm=1.0)
    for tc in TERRAIN_PRECEDENCE:
        class_polys_mm[tc] = resolved_inserts.get(tc)

    # Absorb sub-printable bits the cutout rim frets off any layer into a neighbour,
    # by the same rule the satellite interface uses in resolve_layers. The cutout
    # slices the resolved layers, and where it pinches a piece touching the rim
    # thinner than min_thickness it would print as a fragile edge sliver; that bit is
    # handed to whichever other layer it borders most. Bits are found on the in-cutout
    # footprint (so the rim is a real edge) but removed from / added to the full
    # (pre-cutout) polygons, so kept components keep their true outline and insert
    # inset/relief still seat flush against the rim.
    if outline is not None:
        full = {base_class: base_poly_mm}
        for tc in all_overlay_classes:
            if class_polys_mm.get(tc) is not None:
                full[tc] = class_polys_mm[tc]
        clipped = {k: g.intersection(outline).buffer(0) for k, g in full.items()}
        moves = fretted_bit_moves(clipped, outline.boundary, MIN_BLOB_MM,
                                  scale_m_per_mm=1.0)
        for frm, to, bit in moves:
            full[frm] = full[frm].difference(bit).buffer(0)
            full[to] = unary_union([full[to], bit]).buffer(0)
        base_poly_mm = full[base_class]
        for tc in all_overlay_classes:
            g = full.get(tc)
            class_polys_mm[tc] = g if g is not None and not g.is_empty else None

    # Downsample polygons to the float32 STL output resolution before 2D->3D.
    output_resolution = frame.output_resolution
    base_outline = outline if outline is not None else dem_extent
    # Densify BEFORE quantizing, so the f32 snap absorbs any inserted vertex that
    # lands within a ULP of a corner instead of leaving a sub-resolution sliver.
    base_outline = densify_on_grid(base_outline, frame)
    # The pockets cut from this outline are snapped to the f32 output grid (as one
    # arrangement, in node_and_snap); the outline they share their rim edge with must
    # live on the same grid, or the rim exists in two near-coincident copies and the
    # base solid drapes the hairline cell between them to the full DEM -- a
    # zero-width, full-height razor fin along the rim.
    base_outline = _quantize_to_f32(base_outline, output_resolution)

    # Per insert component: the pocket carved in the base (component + convex corner
    # relief) and the printed insert footprint (inset + reflex relief), both clipped
    # to the cutout.
    pockets: List[Tuple[int, object]] = []
    insert_parts: Dict[int, List[ShapelyPolygon]] = {tc: [] for tc in all_overlay_classes}
    for tc in all_overlay_classes:
        union_poly = class_polys_mm[tc]
        if union_poly is None:
            continue
        for component in iter_polygon_components(union_poly):
            # Clip to the cutout, then drop small rim bits BEFORE the inset clearance
            # step, so the insert and its pocket are built from one cleaned footprint.
            insert_src = _clip_to_footprint(component, base_outline)
            insert_src = _drop_small_rim_bits(
                insert_src, base_outline.boundary, MIN_BLOB_MM ** 2)
            if insert_src is None:
                continue

            # Corner relief comes off the cleaned, clipped footprint -- not the raw
            # component -- so a convex corner that lies outside the cutout cannot drop
            # a relief disc that floats just inside the rim as an orphan pocket blob.
            pocket_extra, insert_cut = _corner_reliefs(
                insert_src, fit.xy_clearance_mm, fit.corner_relief_mm,
                fit.corner_min_angle_deg)

            # Pocket = the cleaned footprint (the seat) + convex corner relief. NOT
            # quantized here: the whole arrangement is snapped in one call later (see
            # node_and_snap).
            pocket_component = (unary_union([insert_src, pocket_extra])
                                if pocket_extra is not None else insert_src)
            pocket_poly = _clip_to_footprint(pocket_component, base_outline)
            if pocket_poly is not None and pocket_poly.area > 0:
                # Corner-relief arcs and the inset are new curves, not clipped from
                # anything already densified, so they need their own crossings.
                pockets.append((tc, densify_on_grid(pocket_poly, frame)))

            # Printed insert footprint: inset by the clearance + reflex corner relief.
            if fit.xy_clearance_mm > 0 or insert_cut is not None:
                insert_poly = _inset_polygon(insert_src, fit.xy_clearance_mm)
                if insert_poly is None:
                    continue
                if insert_cut is not None:
                    insert_poly = insert_poly.difference(insert_cut).buffer(0)
                    if insert_poly.is_empty:
                        continue
            else:
                insert_poly = insert_src
            insert_poly = densify_on_grid(insert_poly, frame)
            insert_poly = _quantize_to_f32(insert_poly, output_resolution)
            if insert_poly is None:
                continue
            # A part with no area yields no triangles and would be silently skipped
            # downstream -- i.e. the mesh stage dropping a region. Drop it here.
            insert_parts[tc].extend(p for p in iter_polygon_components(insert_poly)
                                    if p.area > 0)

    return TerrainLayout(
        base_class=base_class,
        overlay_classes=all_overlay_classes,
        base_outline=base_outline,
        pockets=pockets,
        insert_parts=insert_parts,
        noded_boundaries=node_and_snap(base_outline, [pk for _tc, pk in pockets],
                                       output_resolution),
    )
