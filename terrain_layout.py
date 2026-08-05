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
  3. the cutout rim -- densified and quantized onto the export grid at birth, the
     ONLY copy of that curve in the pipeline (a second raw copy disagrees with it
     by a few um, and clipping one against the other writes that disagreement into
     the boundary as micro-kinks sharp enough to fire corner reliefs) -- frets
     sub-printable bits off the partition; each is handed to a neighbour rather
     than dropped, so the partition stays exact;
  4. per component: clip to the rim, drop small rim bits, apply insert fit
     clearances -- measured to in-print neighbours only, so inserts sit flush at
     the rim, and corner reliefs never fire on clip-manufactured rim vertices. A
     bit the fit itself severs is dropped at footprint level too (its zone
     removed, the fit redone), never after the inset -- a region returned to the
     base behind the inset stands at full height with no clearance computed
     against it;
  5. the base outline and every pocket boundary are noded into ONE arrangement and
     rounded onto the float32 export grid in one pass (``snap_arrangement``), and
     the pockets are then rebuilt as unions of that arrangement's cells. Every
     shared seam (pocket/pocket and pocket/rim) is therefore ONE vertex list before
     anything is rounded; quantizing the copies separately lets a vertex one copy
     has and the other lacks come off the shared line, and the base solid drapes
     the hairline cell between the copies to the full DEM height -- a zero-width,
     full-height razor fin.

Model mm are print mm, so the printable-feature rules (``MIN_THICKNESS_MM``,
``MIN_BLOB_MM``) apply directly and ``scale_m_per_mm`` is 1.0 throughout.
"""

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import shapely
from pyproj import Transformer
from shapely.geometry import (LineString, MultiPolygon, Point,
                              Polygon as ShapelyPolygon, box)
from shapely.geometry.polygon import orient
from shapely.ops import polygonize, unary_union

from bearing_utils import rotate_from_bearing_frame, rotate_to_bearing_frame
from model_frame import ModelFrame
from masks import (TERRAIN_NAMES, TERRAIN_PRECEDENCE, TERRAIN_ROCK,
                   overlay_precedence)
from terrain_compose import (MIN_BLOB_MM, MIN_THICKNESS_MM, MIN_WALL_MM,
                             fretted_bit_moves, resolve_layers)


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


# Body-relief ramp endpoints, in perimeter (boundary length, holes included).
# Zero relief at the perimeter of the smallest printable compact blob -- parts
# that small are compliant and keep their full body. Full relief from
# BODY_RELIEF_PMAX_MM up: on the Ararat test print every insert with >= ~150 mm
# of boundary already needed real force to seat, and relief past "the body never
# touches" costs nothing (the width gate keeps every wall printable), so the
# exact endpoint is uncritical.
BODY_RELIEF_P0_MM = 4.0 * MIN_BLOB_MM
BODY_RELIEF_PMAX_MM = 150.0


@dataclass(frozen=True)
class InsertFit:
    """Fit tolerances for separately-printed inserts (see build_terrain_layout)."""

    xy_clearance_mm: float = 0.0
    z_clearance_mm: float = 0.0
    corner_relief_mm: float = 0.0
    corner_min_angle_deg: float = 45.0
    body_relief_max_mm: float = 0.0


@dataclass
class TerrainLayout:
    """The finished 2D geometry of one print, in model mm.

    Every xy coordinate the print will ever have is fixed here. The mesh stage may
    only give each of these vertices a z and raise walls between them: it must not
    move, add, merge or drop a boundary vertex, and it must not drop a region. So
    the boundaries arrive already densified on the DEM grid lines, already noded
    and snapped into one arrangement, already quantized to the export's float32
    resolution, and already free of degenerate pieces.

    Pockets may OVERLAP: per-part holes cut from one connected recess share their
    connector web, and a convex corner-relief disc can bulge across a neighbouring
    layer. The mesh floors an overlap at the LOWER of the overlapping floors.

    ``pocket_floor_refs`` runs parallel to ``pockets``: the insert part whose
    ground sets the pocket's flat floor (``dem_min(part) - thickness``, making the
    seating gap exactly the z clearance), or None for a recess with no insert,
    which floors off its own ground.

    ``noded_boundaries`` is the constraint set the base plate is triangulated
    against -- the outline and every pocket boundary noded and rounded as one
    arrangement (``snap_arrangement``). The outline and every pocket are unions of
    that arrangement's cells, so the constraint edges and the draped regions can
    never describe different line work.

    ``insert_bodies`` runs parallel to each ``insert_parts`` list: the part's
    relieved body footprint (everything below the collar band, see
    ``_insert_body``), or None when the part is printed as one full-footprint
    prism. Body boundaries are the one line work deliberately OUTSIDE the shared
    arrangement: they seat against nothing (the relief exists so they never
    touch), and where the width gate keeps a thin feature they reuse the part's
    own already-final boundary coordinates rather than deriving a second copy.
    """

    base_class: int
    overlay_classes: List[int]
    base_outline: ShapelyPolygon
    pockets: List[Tuple[int, Union[ShapelyPolygon, MultiPolygon]]] = field(default_factory=list)
    insert_parts: Dict[int, List[ShapelyPolygon]] = field(default_factory=dict)
    pocket_floor_refs: List[Optional[ShapelyPolygon]] = field(default_factory=list)
    noded_boundaries: object = None
    insert_bodies: Dict[int, List[Optional[Union[ShapelyPolygon, MultiPolygon]]]] = \
        field(default_factory=dict)

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


def _inset_polygon(polygon_mm, distance_mm: float, outline=None):
    """Drop every point of the polygon within ``distance_mm`` of its neighbours.

    Gives a separately-printed insert horizontal clearance from its pocket. With
    ``outline`` given, the neighbours are the IN-PRINT complement
    (``outline - polygon``) only: outside the cutout rim nothing exists, so the
    insert keeps its rim edge and the print's perimeter stays a full circle --
    a plain ``buffer(-d)`` would also retreat from the rim, recessing the whole
    perimeter by a clearance-wide slit. Along real walls the two are identical.

    The result may be a (Multi)Polygon (a neck thinner than 2*distance severs);
    both are handled downstream. Returns the original polygon when distance is
    non-positive, or None if the clearance erases it entirely.
    """
    if distance_mm <= 0:
        return polygon_mm
    if outline is None:
        inset = polygon_mm.buffer(-distance_mm)
    else:
        # Only neighbours within reach matter: collar the complement to a band
        # around the polygon so the buffer runs on it, not on the whole print.
        # Material beyond the collar is farther than the distance already and
        # could at most touch the polygon tangentially.
        collar = outline.intersection(polygon_mm.buffer(distance_mm))
        neighbours = collar.difference(polygon_mm)
        inset = polygon_mm.difference(neighbours.buffer(distance_mm)).buffer(0)
    if inset.is_empty:
        return None
    return inset


def _insert_body(part, relief_max_mm: float, frame, output_resolution):
    """The relieved body footprint below a printed insert part's collar band.

    Only the collar -- the top band of the insert, at the part's own footprint --
    needs the designed fit: it alone sets the visible seam. Everything below it is
    inset by an extra relief so the body never touches the pocket wall, cutting
    the mating contact to the collar band and giving the insert a lead-in.

    The relief ramps with the part's boundary length (holes included): position
    error and insertion force both grow with how much wall a part carries, while
    the smallest parts comply elastically and need none. Width gate: where the
    local width cannot afford the full relief and still leave a printable wall
    (``MIN_WALL_MM``), the footprint is kept unchanged -- a thin fin keeps its
    full width instead of thinning below what the nozzle can lay down.

    Returns the body (Multi)Polygon, densified and quantized like all final line
    work, or None when the part gets no relief (below the ramp, or thin
    everywhere) and is printed as one full-footprint prism.
    """
    t = ((part.length - BODY_RELIEF_P0_MM)
         / (BODY_RELIEF_PMAX_MM - BODY_RELIEF_P0_MM))
    relief = relief_max_mm * min(max(t, 0.0), 1.0)
    if relief <= 0:
        return None
    # The width test is an opening, but with a MITRE dilation, not
    # ``open_min_width``'s disc: a disc opening rounds every convex corner, so
    # each sharp corner tip would test "thin" and stay a full-footprint sliver.
    # The mitre rebuilds corners exactly (up to the default limit -- a spike too
    # sharp for it really is thin at the tip and belongs in the keep).
    h = 0.5 * (MIN_WALL_MM + 2.0 * relief)
    wide = part.buffer(-h).buffer(h, join_style="mitre")

    # A keep is a wall the printer lays down at full width, so a keep piece must
    # have a MIN_WALL-wide core somewhere; what fails that test is not a thin
    # feature but junction debris of the difference (hairline quads along the
    # walls, well above f32 resolution yet armless razor slivers that leave the
    # prism builder with open shells). Selection is by PIECE, so a surviving
    # keep's outline stays the part's own reused coordinates.
    def _has_wall_core(p):
        return not p.buffer(-0.5 * MIN_WALL_MM).is_empty

    keep = [p for p in iter_polygon_components(part.difference(wide))
            if _has_wall_core(p)]
    body = unary_union([part.buffer(-relief), *keep]).buffer(0)
    # The same rule prunes the union: an eroded remnant narrower than MIN_WALL
    # everywhere (a region between the gate and vanishing) supports nothing.
    kept = [p for p in iter_polygon_components(body) if _has_wall_core(p)]
    if not kept:
        return None
    body = kept[0] if len(kept) == 1 else MultiPolygon(kept)
    if part.area - body.area <= 0:
        return None
    body = densify_on_grid(body, frame)
    return _quantize_to_f32(body, output_resolution)


def _corner_reliefs(component, clearance_mm: float, relief_mm: float,
                    min_turn_deg: float, within=None, solid=None):
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

    ``component`` is the PRE-clip footprint. Clipping to the cutout manufactures
    vertices ON the rim (wherever a mask edge meets the rim arc); a disc fired
    there bites a dent into the print's perimeter and relieves nothing -- the
    corner is open to the outside, there is no material to seat past. Those
    vertices do not exist before the clip, so taking the corners pre-clip
    excludes them without any "is this vertex on the rim" tolerance. Two vertex
    filters keep the rest sane: ``within`` (the cutout footprint) drops corners
    on or beyond the rim, and ``solid`` (the cleaned, clipped footprint) drops
    corners whose piece the clip or the rim-bit cleanup removed -- either kind
    of disc would otherwise carve a pocket where no insert exists.

    Vertex tests alone do not protect the perimeter, for two reasons: the rim
    fret welds bit boundaries into the pre-clip polygons, whose junction corners
    sit ON the rim but register as "strictly inside" (a GEOS crossing point lands
    within a few ULP of either curve), and a genuine reflex corner a fraction of
    the disc radius inside the rim fires a disc that reaches through to it. So
    the perimeter rule is enforced on the discs themselves: any disc that would
    cross the rim is dropped whole. The print's perimeter stays a full circle,
    tight corner or not -- a locked corner can be sanded, a notched rim cannot
    be filled.

    Returns (pocket_extra, insert_cut): the union of convex discs to add to the
    pocket (or None) and the union of reflex discs to subtract from the insert.
    """
    if relief_mm <= 0:
        return None, None
    r = clearance_mm + relief_mm
    rim = within.boundary if within is not None else None
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
            vertex = Point(V)
            if within is not None and not within.contains(vertex):
                continue                 # on or past the rim: open corner, no seat
            if solid is not None and not solid.covers(vertex):
                continue                 # its piece was clipped or dropped away
            if turn > 0.0:                                       # convex -> relieve pocket
                nu = np.array([-u[1], u[0]]); nw = np.array([-w[1], w[0]])  # inward normals
                denom = max(1.0 + float(nu @ nw), 0.1)           # cap miter length at needles
                P = V + clearance_mm * (nu + nw) / denom
                disc = Point(P).buffer(r)
                if rim is None or not disc.intersects(rim):
                    convex.append(disc)
            else:                                                # reflex -> relieve insert
                disc = Point(V).buffer(r)
                if rim is None or not disc.intersects(rim):
                    reflex.append(disc)
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
    rim-touching bits go. Returns the cleaned geometry, or None if nothing remains.
    """
    if polygon_mm is None or polygon_mm.is_empty:
        return None
    keep = [c for c in iter_polygon_components(polygon_mm)
            if not (c.area < min_area_mm2 and rim.intersects(c))]
    if not keep:
        return None
    return keep[0] if len(keep) == 1 else unary_union(keep)


def _round_lines(lines, output_resolution):
    """Round every vertex of a line arrangement onto the output grid.

    A per-vertex round is a pure function of the coordinate -- no GEOS hot pixels,
    no dependence on which geometry a point sits in -- so it is safe to apply to
    the single shared copy of every seam that ``snap_arrangement`` maintains.
    Vertices brought together by the round are dropped, as is any line left with
    fewer than two distinct points. Returns a list of LineStrings.
    """
    out = []
    for ls in (lines.geoms if hasattr(lines, "geoms") else [lines]):
        c = np.asarray(ls.coords, float)
        c = np.round(c / output_resolution) * output_resolution
        keep = np.ones(len(c), bool)
        keep[1:] = np.any(c[1:] != c[:-1], axis=1)
        c = c[keep]
        if len(c) >= 2:
            out.append(LineString(c))
    return out


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


def snap_arrangement(base_outline, pocket_polys, output_resolution):
    """Node the outline + pocket boundaries into ONE on-grid arrangement.

    This is the single quantization of all the shared 2D line work, and the order
    inside it is the point. ``unary_union`` runs first, while the coordinates are
    still doubles: it dissolves the two copies of every shared seam (pocket/pocket
    and pocket/rim) into one polyline carrying the union of their vertices, and
    nodes every genuine crossing. Only THEN is anything rounded -- and by a plain
    per-vertex round, not ``set_precision``: rounding is position-independent, so
    with one copy of each seam in existence there is nothing left to diverge,
    whereas snap-rounding works against each geometry's own hot pixels and cannot
    promise that. Rounding the copies separately is exactly what built razor fins:
    a vertex present in one copy and absent from the other comes off the shared
    straight line by a fraction of a grid step, and the base solid drapes the
    sliver between the copies to the full DEM height.

    Rounding can slide a segment across a nearby vertex, un-noding the arrangement,
    so the round + re-node pair repeats until a pass leaves every coordinate on the
    grid (re-noding on-grid input inserts no new vertices, so that pass is the
    fixpoint; each round only moves the crossings the previous union created, and
    in practice one or two passes suffice). Spikes that collapse onto themselves
    dissolve in the union; ``polygonize`` then never sees a cell there.

    The result is both the constraint set the base plate is triangulated against
    and the source the pockets are rebuilt from, so the constraint edges and the
    draped regions can never describe different line work.
    """
    noded = unary_union([base_outline.boundary]
                        + [pk.boundary for pk in pocket_polys])
    for _ in range(100):
        noded = unary_union(_round_lines(noded, output_resolution))
        coords = shapely.get_coordinates(noded)
        if np.array_equal(coords,
                          np.round(coords / output_resolution) * output_resolution):
            return noded
    raise RuntimeError(
        "snap_arrangement did not reach an on-grid noded fixpoint")


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
            mesh builder, which owns the floors. ``body_relief_max_mm`` caps the
            extra inset of each part's below-collar body (``_insert_body``); 0
            keeps every insert a single full-footprint prism.
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

    # THE one copy of the cutout rim, final from birth: densified (BEFORE
    # quantizing, so the f32 snap absorbs any inserted vertex that lands within a
    # ULP of a corner instead of leaving a sub-resolution sliver), then rounded
    # onto the export grid. Every consumer below -- the rim fret, the clips, the
    # clearance inset, snap_arrangement -- reads this object, so no coordinate of
    # the rim ever exists in two versions. The raw cutout is deleted: clipping
    # against one copy what was cut against another writes their disagreement (a
    # few um) into the boundary as micro-kinks sharp enough to fire corner
    # reliefs, each disc a visible dent in the print's perimeter.
    output_resolution = frame.output_resolution
    has_cutout = outline is not None
    base_outline = densify_on_grid(outline if has_cutout else dem_extent, frame)
    base_outline = _quantize_to_f32(base_outline, output_resolution)
    del outline

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
    if has_cutout:
        full = {base_class: base_poly_mm}
        for tc in all_overlay_classes:
            if class_polys_mm.get(tc) is not None:
                full[tc] = class_polys_mm[tc]
        clipped = {k: g.intersection(base_outline).buffer(0) for k, g in full.items()}
        moves = fretted_bit_moves(clipped, base_outline.boundary, MIN_BLOB_MM,
                                  scale_m_per_mm=1.0)
        for frm, to, bit in moves:
            full[frm] = full[frm].difference(bit).buffer(0)
            full[to] = unary_union([full[to], bit]).buffer(0)
        base_poly_mm = full[base_class]
        for tc in all_overlay_classes:
            g = full.get(tc)
            class_polys_mm[tc] = g if g is not None and not g.is_empty else None

    # Per insert component: the pocket carved in the base (component + convex corner
    # relief) and the printed insert footprint (inset + reflex relief), both clipped
    # to the cutout. Pockets are only COLLECTED here; their final geometry comes out
    # of the one shared arrangement below.
    pocket_records: List[Tuple[int, object, bool, Optional[ShapelyPolygon]]] = []
    # (class, polygon, share, floor-setting insert part or None)
    part_records: List[Tuple[int, ShapelyPolygon]] = []
    insert_parts: Dict[int, List[ShapelyPolygon]] = {tc: [] for tc in all_overlay_classes}
    rim = base_outline.boundary
    for tc in all_overlay_classes:
        union_poly = class_polys_mm[tc]
        if union_poly is None:
            continue
        for component in iter_polygon_components(union_poly):
            # Clip to the cutout, then drop small rim bits BEFORE the inset clearance
            # step, so the insert and its pocket are built from one cleaned footprint.
            insert_src = _clip_to_footprint(component, base_outline)
            insert_src = _drop_small_rim_bits(insert_src, rim, MIN_BLOB_MM ** 2)
            if insert_src is None:
                continue
            # Fit loop: corner reliefs + clearance inset, redone whenever the fit
            # severs a small rim bit off the footprint. The rule is the same as
            # _drop_small_rim_bits, per severed PART: dropping the part but
            # keeping its hole is exactly the "small bit removed, hole left in
            # the polygon" perimeter dent. And the drop must happen at FOOTPRINT
            # level, before the fit -- a part dropped after the inset returns
            # its region to the base at full height with no clearance ever
            # computed against it, and the neighbouring insert jams. So each
            # bit's zone (the region of the footprint it can reach without
            # crossing another part -- the per-part hole rule, applied to the
            # footprint) is removed and the fit is rebuilt from the cleaned
            # footprint; removing a zone moves the inset, which can sever a new
            # bit, hence the loop. Every pass strictly shrinks the footprint.
            for _ in range(100):
                # Corner relief comes off the PRE-clip component: clipping
                # manufactures vertices on the rim, and a disc fired there is a
                # dent in the print's perimeter (see _corner_reliefs). The
                # filters drop corners on or past the rim and corners on
                # clipped/dropped pieces, so no disc can carve a pocket where no
                # insert exists.
                pocket_extra, insert_cut = _corner_reliefs(
                    component, fit.xy_clearance_mm, fit.corner_relief_mm,
                    fit.corner_min_angle_deg, within=base_outline,
                    solid=insert_src)

                # Printed insert footprint: inset by the clearance (flush at the
                # rim: the clearance comes from in-print neighbours only) +
                # reflex corner relief. The inset (a neck thinner than twice the
                # clearance) and the reflex discs can sever it, so one footprint
                # can yield several parts.
                parts_local: List[ShapelyPolygon] = []
                insert_poly = _inset_polygon(insert_src, fit.xy_clearance_mm,
                                             outline=base_outline)
                if insert_poly is not None and insert_cut is not None:
                    insert_poly = insert_poly.difference(insert_cut).buffer(0)
                if insert_poly is not None and not insert_poly.is_empty:
                    # A part with no area yields no triangles and would be
                    # silently skipped downstream -- i.e. the mesh stage
                    # dropping a region.
                    parts_local = [p for p in iter_polygon_components(
                                       densify_on_grid(insert_poly, frame))
                                   if p.area > 0]
                rim_bits = [p for p in parts_local
                            if p.area < MIN_BLOB_MM ** 2 and p.intersects(rim)]
                if not rim_bits:
                    break
                zones = []
                for p in rim_bits:
                    cut = insert_src.difference(unary_union(
                        [q for q in parts_local if q is not p])).buffer(0)
                    rep = p.representative_point()
                    zones.append(next(c for c in iter_polygon_components(cut)
                                      if c.contains(rep)))
                insert_src = insert_src.difference(
                    unary_union(zones)).buffer(0)
                if insert_src.is_empty:
                    insert_src = None
                    break
            else:
                raise RuntimeError(
                    "rim-bit removal did not reach a bit-free fit")
            if insert_src is None:
                continue

            # Pocket = the cleaned footprint (the seat) + convex corner relief. NOT
            # quantized here: every pocket boundary joins one global arrangement,
            # rounded onto the export grid in a single pass, and the pocket is then
            # rebuilt FROM that arrangement (see snap_arrangement below).
            if pocket_extra is None:
                # Already clipped and rim-cleaned above; clipping again is another
                # overlay op on coordinates that are final, and every such op can
                # move them. Only the relief arcs can reach past the rim.
                pocket_poly = insert_src
            else:
                pocket_poly = _clip_to_footprint(
                    unary_union([insert_src, pocket_extra]), base_outline)
            share = fit.xy_clearance_mm <= 0 and insert_cut is None
            if pocket_poly is not None and pocket_poly.area <= 0:
                pocket_poly = None
            if share:
                # Touching fit (one-piece multimaterial): the insert IS the pocket,
                # so it takes the pocket's rebuilt pieces verbatim, below. Deriving
                # the same curve twice is what must never happen: two copies of a
                # seam that must be identical can part by up to a grid step.
                if pocket_poly is not None:
                    pocket_records.append(
                        (tc, densify_on_grid(pocket_poly, frame), True, None))
                continue

            # Parts are NOT quantized here, for the same reason the pockets are
            # not: their boundaries join the one shared arrangement and the
            # printed parts are rebuilt from its cells below. Quantizing a flush
            # part now would move the densify-inserted vertices off the rim
            # chords -- a um-different second copy of the rim that reaches the
            # arrangement through the hole cuts and leaves unclaimed razor cells
            # on the rim, which the base drapes as zero-width double walls.
            part_records.extend((tc, p) for p in parts_local)
            if pocket_poly is None:
                continue

            if len(parts_local) <= 1:
                # One insert (or none) in this recess: the whole recess is its
                # hole. The part sets the floor; with no insert (the inset erased
                # it entirely) the recess floors off its own ground.
                pocket_records.append(
                    (tc, densify_on_grid(pocket_poly, frame), False,
                     parts_local[0] if parts_local else None))
                continue

            # Several inserts in one connected recess: hole and insert must be
            # 1:1, each hole flat at its own part's floor. hole_i is the connected
            # region of the recess the part can reach without crossing another
            # part -- built purely from curves that already exist (the recess
            # outline and the other parts' outlines), so no re-derived curve can
            # run near-parallel to an existing one. Holes OVERLAP on the shared
            # connector web; the mesh gives an overlap the lower floor.
            claimed = []
            for part in parts_local:
                cut = pocket_poly.difference(
                    unary_union([q for q in parts_local if q is not part])).buffer(0)
                rep = part.representative_point()
                hole = next((c for c in iter_polygon_components(cut)
                             if c.contains(rep)), None)
                if hole is None:
                    raise ValueError(
                        "insert part lies in no component of its own recess; the "
                        "layout must not emit an insert without a hole")
                pocket_records.append(
                    (tc, densify_on_grid(hole, frame), False, part))
                claimed.append(hole)
            # Web pieces no part can reach (sealed off behind other parts) still
            # belong to the recess; they hold no insert and floor off their own
            # ground, like a recess whose insert vanished.
            leftover = pocket_poly.difference(unary_union(claimed)).buffer(0)
            for c in iter_polygon_components(leftover):
                if c.area > 0:
                    pocket_records.append(
                        (tc, densify_on_grid(c, frame), False, None))

    # One arrangement, one quantization, THEN the split into pockets and parts.
    # Both come back as unions of the arrangement's own cells, so a pocket
    # boundary IS a set of constraint edges the base is triangulated against, and
    # the printed part, its seat and the rim are the same on-grid polyline --
    # sharing by construction, not by reconciling copies afterwards.
    noded_boundaries = snap_arrangement(
        base_outline,
        [poly for _tc, poly, _s, _r in pocket_records]
        + [poly for _tc, poly in part_records],
        output_resolution)
    cells = [c for c in polygonize(noded_boundaries) if c.area > 0]

    # Printed insert parts: the arrangement round above IS their export
    # quantization. ``rebuilt_part`` maps each pre-arrangement part (the hole
    # records' floor refs) to its rebuilt self.
    rebuilt_part: Dict[int, ShapelyPolygon] = {}
    for tc, raw in part_records:
        shapely.prepare(raw)
        mine = [c for c in cells if raw.contains(c.representative_point())]
        if not mine:
            continue
        rebuilt = shapely.coverage_union_all(mine)
        pieces = [p for p in iter_polygon_components(rebuilt) if p.area > 0]
        insert_parts[tc].extend(pieces)
        rep = raw.representative_point()
        for p in pieces:
            if p.contains(rep):
                rebuilt_part[id(raw)] = p

    pockets: List[Tuple[int, object]] = []
    pocket_floor_refs: List[Optional[ShapelyPolygon]] = []
    for tc, raw, share, ref in pocket_records:
        if ref is not None:
            # The floor ref must be the PRINTED part (its rebuilt self), or None
            # when the part did not survive the rebuild.
            ref = rebuilt_part.get(id(ref))
        shapely.prepare(raw)
        mine = [c for c in cells if raw.contains(c.representative_point())]
        if not mine:
            continue
        # coverage_union of cells sharing exact edges: dissolves the interior
        # edges, moves no vertex. Pockets may overlap (per-part holes share their
        # connector web, and a convex corner-relief disc bulges across a
        # neighbour), so a cell can belong to several records -- the mesh floors
        # an overlap at the lower of the overlapping floors.
        rebuilt = shapely.coverage_union_all(mine)
        # One entry per CONNECTED piece. A component is connected in the unclipped
        # plane, but its arms may join only through terrain outside the print, so
        # the cutout clip hands back several pieces. The mesh stage floors a whole
        # entry at one Z, taken over everything in it -- so a rim arm high on the
        # mountain would be sunk to the floor of the lowest piece anywhere in the
        # print, and the insert seated in it would hang that far above its seat.
        pieces = [p for p in iter_polygon_components(rebuilt) if p.area > 0]
        rep = ref.representative_point() if ref is not None else None
        for p in pieces:
            pockets.append((tc, p))
            pocket_floor_refs.append(
                ref if rep is not None and p.contains(rep) else None)
        if share:
            insert_parts[tc].extend(pieces)

    # The rim gains a vertex (rounded with everything else) wherever a pocket seam
    # meets it; rebuild the outline from the same cells, so the rim the mesh walls
    # follow is the arrangement's own, not a stale pre-noding copy.
    covered = shapely.coverage_union_all(cells)
    if not covered.is_empty:
        base_outline = covered

    # Relieved body footprints, one per printed part, from the parts' FINAL
    # (arrangement-rebuilt) geometry. A touching fit has no separately-printed
    # walls to relieve, so bodies exist only alongside a real clearance.
    insert_bodies: Dict[int, List[Optional[ShapelyPolygon]]] = {}
    for tc in all_overlay_classes:
        if fit.body_relief_max_mm > 0 and fit.xy_clearance_mm > 0:
            insert_bodies[tc] = [
                _insert_body(p, fit.body_relief_max_mm, frame, output_resolution)
                for p in insert_parts[tc]]
        else:
            insert_bodies[tc] = [None] * len(insert_parts[tc])

    return TerrainLayout(
        base_class=base_class,
        overlay_classes=all_overlay_classes,
        base_outline=base_outline,
        pockets=pockets,
        insert_parts=insert_parts,
        pocket_floor_refs=pocket_floor_refs,
        noded_boundaries=noded_boundaries,
        insert_bodies=insert_bodies,
    )
