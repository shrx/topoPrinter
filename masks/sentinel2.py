"""Load a satellite-derived snow polygon and prepare it as a terrain-overlay class.

The raw NDSI polygon (e.g. ararat_snow_2026-07-14.geojson, in UTM metres) has
hundreds of tiny specks and a dense filigree of hair-thin finger tips that cannot
be printed as a standalone insert. Cleanup is done entirely in VECTOR space (the
polygon becomes extruded mesh walls downstream, so raster round-trips would
staircase and fragment it):

  1. despeckle  -- drop polygons and interior rings below MIN_FEATURE_M2 (the
     2x2 mm insert threshold at print scale); exact-area, no boundary change.
  2. area-preserving curve-shortening flow (APCSF) -- the surface-tension model.
     Each step curve-shortens every vertex using unit edge vectors (true
     curvature flow: thin finger tips, being high-curvature, retract fastest),
     then restores the lost area with a uniform outward buffer offset. Total area
     is conserved throughout; fingers retract, ridges bulge, tiny components
     vanish, nearby blobs coalesce -- driving the shape toward printable,
     handleable blobs while keeping the enclosed snow area fixed.

The default APCSF_ITERS stop was chosen visually for the Ararat snow: it leaves
the main ice mass plus one satellite, with essentially no strand thinner than
MIN_STRAND_MM once printed. Reproject to the DEM CRS as GeoJSON geometry dicts,
matching masks.osm.classify_terrain's output so it drops into the glacier class.

The same loading + cleaning serves the NDVI foliage layer (load_and_clean_veg),
just with far fewer APCSF iterations.
"""
import json
from dataclasses import dataclass
from typing import Optional

import numpy as np
from shapely.geometry import shape, mapping, Polygon
from shapely.ops import unary_union, transform as shp_transform
from pyproj import Transformer, CRS

from masks import TERRAIN_FOLIAGE, TERRAIN_GLACIER

# Feature-size design rules are in MILLIMETRES (fixed by the 0.4 mm nozzle); the
# metric thresholds below are those mm sizes times a print scale. SCALE_M_PER_MM
# is only the NOMINAL scale (Ararat 20 km -> 150 mm = 1:133,333) used as a default
# by the standalone preview scripts. The production pipeline derives the true scale
# from --diameter/--x-size-mm and passes it in -- do not assume this nominal value.
SCALE_M_PER_MM = 133.333
MIN_STRAND_MM = 2.0                                   # min standalone-insert width
MIN_FEATURE_MM = 2.0                                  # compact-feature despeckle size
STRAND_R = SCALE_M_PER_MM * MIN_STRAND_MM / 2.0       # opening radius at nominal scale
MIN_FEATURE_M2 = (MIN_FEATURE_MM * SCALE_M_PER_MM) ** 2   # 2x2 mm at nominal scale


RESAMPLE_M = 15.0                                     # APCSF ring resampling
DT = 4.0                                              # APCSF curve-shortening step (m)
APCSF_ITERS = 220                                     # sub-1mm < 1% -> 2 pieces (at RESAMPLE_M=15)
VEG_ITERS = 100         # foliage APCSF iterations (sub-1mm base slivers <1% at RESAMPLE_M=15)


def despeckle_area_m2(scale_m_per_mm=SCALE_M_PER_MM):
    """Min compact-feature area (m^2) — a MIN_FEATURE_MM square at the print scale."""
    return (MIN_FEATURE_MM * scale_m_per_mm) ** 2


def _src_epsg(fc: dict) -> int:
    name = fc.get("crs", {}).get("properties", {}).get("name", "EPSG:4326")
    return int(name.split(":")[-1])


def npoly(g) -> int:
    return len(g.geoms) if g.geom_type == "MultiPolygon" else 1


def despeckle(geom, min_area):
    """Drop polygons and interior rings smaller than min_area (exact-area, vector)."""
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    kept = []
    for p in polys:
        if p.area < min_area:
            continue
        holes = [r for r in p.interiors if Polygon(r).area >= min_area]
        kept.append(Polygon(p.exterior.coords, [h.coords for h in holes]))
    return unary_union(kept) if kept else geom


def resample_ring(coords, spacing=RESAMPLE_M):
    """Resample a closed ring to ~uniform arc-length spacing. Returns open (M,2)."""
    pts = np.asarray(coords, float)
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    if len(pts) < 3:
        return pts
    loop = np.vstack([pts, pts[0]])
    seg = np.hypot(*np.diff(loop, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    n = max(3, int(round(total / spacing)))
    s = np.linspace(0.0, total, n, endpoint=False)
    return np.column_stack([np.interp(s, cum, loop[:, 0]),
                            np.interp(s, cum, loop[:, 1])])


def _csf_ring(coords, dt, spacing=RESAMPLE_M):
    """One geometric curve-shortening step on a resampled closed ring.

    The unit-vector sum (up+un) ~= spacing*kappa*N, so the flow advanced per iteration
    scales as dt*spacing. RESAMPLE_M and the iteration counts are therefore tuned
    together: changing RESAMPLE_M changes how far a given iteration count flows (and
    the sub-2*RESAMPLE_M detail floor), so re-run the iteration sweep if you change it.
    """
    p = resample_ring(coords, spacing)
    if len(p) < 5:
        return p
    prev, nxt = np.roll(p, 1, 0), np.roll(p, -1, 0)
    up = prev - p
    un = nxt - p
    up /= np.linalg.norm(up, axis=1, keepdims=True) + 1e-12
    un /= np.linalg.norm(un, axis=1, keepdims=True) + 1e-12
    return p + dt * (up + un)          # -dt * dL/dp,  dL/dp = -(up + un)


def _csf_polygon(poly, dt, spacing=RESAMPLE_M):
    ext = _csf_ring(poly.exterior.coords, dt, spacing)
    holes = [_csf_ring(r.coords, dt, spacing) for r in poly.interiors]
    holes = [h for h in holes if len(h) >= 4]
    if len(ext) < 4:
        return None
    return Polygon(ext, holes).buffer(0)


def apcsf_step(geom, a0, dt, min_area=MIN_FEATURE_M2, spacing=RESAMPLE_M):
    """Curve-shorten every component, restore area a0 by uniform offset, despeckle.

    The despeckle drops sub-insert pieces AND the degenerate near-zero slivers that
    buffer/union leave behind, so the working (and saved/drawn) geometry only ever
    holds real insert-sized polygons. Their area migrates into the survivors via the
    fixed-a0 offset, exactly as a shrinking component would if left to vanish."""
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    moved = [g for g in (_csf_polygon(p, dt, spacing) for p in polys)
             if g is not None and not g.is_empty]
    geom = unary_union(moved)
    if geom.is_empty:
        return geom
    length = geom.length
    if length > 0:
        geom = geom.buffer((a0 - geom.area) / length, join_style=1).buffer(0)
    return despeckle(geom, min_area)


def thin_fraction(geom, r=STRAND_R):
    """Fraction of area removed by an opening at radius r (strands thinner than 2r)."""
    opened = geom.buffer(-r).buffer(r)
    return max(geom.area - opened.area, 0.0) / geom.area


def _apcsf_batch(geom, a0, dt, min_area, spacing, steps):
    """Flow every ring `steps` times purely in numpy, then one GEOS cleanup.

    Equivalent to `steps` back-to-back `apcsf_step` calls, except the GEOS work
    (validity fix, component union, area-preserving offset, despeckle) is deferred
    to the batch boundary instead of running every step. Between cleanups the rings
    stay as numpy arrays: each step resamples + curve-shortens without leaving
    array-land, so the type boundary is crossed once per batch, not once per step.
    With a small dt a handful of steps barely self-intersect, so the single
    end-of-batch buffer(0) resolves the same topology the per-step version would.
    """
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    rebuilt = []
    for poly in polys:
        ext = np.asarray(poly.exterior.coords, float)
        holes = [np.asarray(r.coords, float) for r in poly.interiors]
        for _ in range(steps):
            ext = _csf_ring(ext, dt, spacing)
            holes = [_csf_ring(h, dt, spacing) for h in holes]
        holes = [h for h in holes if len(h) >= 4]
        if len(ext) < 4:
            continue
        p = Polygon(ext, holes)
        rebuilt.append(p if p.is_valid else p.buffer(0))
    moved = [g for g in rebuilt if not g.is_empty]
    if not moved:
        return unary_union([])          # empty -> caller stops the flow
    geom = unary_union(moved)
    if geom.is_empty:
        return geom
    length = geom.length
    if length > 0:
        geom = geom.buffer((a0 - geom.area) / length, join_style=1).buffer(0)
    return despeckle(geom, min_area)


def apcsf_clean(geom, iterations, dt=DT, min_feature_m2=MIN_FEATURE_M2,
                resample_m=RESAMPLE_M, geos_every=10):
    """Despeckle + area-preserving curve-shortening flow on a (Multi)Polygon.

    Total area is preserved from the despeckled input; `iterations` sets how far
    the surface-tension flow runs (fingers retract, boundaries smooth). Generic
    over terrain class — snow uses many iterations, foliage far fewer.

    The flow runs in numpy batches of `geos_every` steps, dropping into GEOS only
    at each batch boundary for validity/union/area-restore/despeckle. This keeps
    the per-vertex motion in array-land and crosses into GEOS ~iterations/geos_every
    times instead of every step (an order-of-magnitude fewer GEOS round-trips).
    """
    geom = despeckle(geom, min_feature_m2)
    a0 = geom.area
    remaining = int(iterations)
    while remaining > 0:
        steps = min(geos_every, remaining)
        geom = _apcsf_batch(geom, a0, dt, min_feature_m2, resample_m, steps)
        if geom.is_empty:
            break
        remaining -= steps
    return geom


def load_geojson_layer(geojson_path):
    """Load a GeoJSON FeatureCollection into a unioned (Multi)Polygon + src EPSG."""
    with open(geojson_path) as fh:
        fc = json.load(fh)
    geom = unary_union([shape(f["geometry"]) for f in fc["features"]])
    return geom, _src_epsg(fc)


def load_and_clean_snow(geojson_path, iterations=APCSF_ITERS, dt=DT,
                        min_feature_m2=MIN_FEATURE_M2, resample_m=RESAMPLE_M):
    """Load snow polygons (native CRS) and clean via despeckle + APCSF.

    Returns (cleaned_geometry, src_epsg): a shapely (Multi)Polygon in the source
    CRS with total snow area preserved from the despeckled input.
    """
    geom, src_epsg = load_geojson_layer(geojson_path)
    return apcsf_clean(geom, iterations, dt, min_feature_m2, resample_m), src_epsg


def load_and_clean_veg(geojson_path, iterations=VEG_ITERS,
                       min_feature_m2=MIN_FEATURE_M2):
    """Load a satellite foliage polygon and clean it (despeckle + APCSF)."""
    geom, src_epsg = load_geojson_layer(geojson_path)
    return apcsf_clean(geom, iterations, min_feature_m2=min_feature_m2), src_epsg


def snow_to_ref_geoms(geom, src_epsg, ref_crs):
    """Reproject a shapely geom (src_epsg) to ref_crs; return GeoJSON geom dicts."""
    tr = Transformer.from_crs(CRS.from_epsg(src_epsg), ref_crs, always_xy=True)
    reproj = shp_transform(lambda xs, ys, zs=None: tr.transform(xs, ys), geom)
    geoms = reproj.geoms if reproj.geom_type == "MultiPolygon" else [reproj]
    return [mapping(g) for g in geoms]


@dataclass(frozen=True)
class SnowMasks:
    """Mask provider: an NDSI snow GeoJSON, cleaned, carried as GLACIER."""
    geojson_path: str
    iterations: int = APCSF_ITERS
    dt: float = DT
    min_feature_m2: Optional[float] = None    # None -> 2x2 mm at the print scale

    def __call__(self, frame):
        min_feature_m2 = (self.min_feature_m2 if self.min_feature_m2 is not None
                          else despeckle_area_m2(frame.scale_m_per_mm))
        geom, src_epsg = load_and_clean_snow(
            self.geojson_path, iterations=self.iterations,
            dt=self.dt, min_feature_m2=min_feature_m2)
        print(f"[INFO] snow -> glacier: {geom.area / 1e6:.1f} km^2", flush=True)
        return {TERRAIN_GLACIER: snow_to_ref_geoms(geom, src_epsg, frame.ref_crs)}


@dataclass(frozen=True)
class FoliageMasks:
    """Mask provider: an NDVI foliage GeoJSON, cleaned, carried as FOLIAGE."""
    geojson_path: str
    iterations: int = VEG_ITERS
    min_feature_m2: Optional[float] = None    # None -> 2x2 mm at the print scale

    def __call__(self, frame):
        min_feature_m2 = (self.min_feature_m2 if self.min_feature_m2 is not None
                          else despeckle_area_m2(frame.scale_m_per_mm))
        geom, src_epsg = load_and_clean_veg(
            self.geojson_path, iterations=self.iterations,
            min_feature_m2=min_feature_m2)
        print(f"[INFO] foliage -> foliage: {geom.area / 1e6:.1f} km^2", flush=True)
        return {TERRAIN_FOLIAGE: snow_to_ref_geoms(geom, src_epsg, frame.ref_crs)}


def draw(ax, geom, facecolor="#c026d3"):
    """Fill a (Multi)Polygon on a matplotlib axis (holes in white). Preview helper."""
    import numpy as _np
    from matplotlib.patches import Polygon as MplPoly
    from matplotlib.collections import PatchCollection
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    ax.add_collection(PatchCollection(
        [MplPoly(_np.asarray(p.exterior.coords)) for p in polys],
        facecolor=facecolor, edgecolor="none"))
    holes = [MplPoly(_np.asarray(r.coords)) for p in polys for r in p.interiors]
    if holes:
        ax.add_collection(PatchCollection(holes, facecolor="white", edgecolor="none"))
