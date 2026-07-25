"""Resolve satellite terrain layers into one base + inserts for separate printing.

The design is a vertical layer hierarchy, bottom -> top:

  * the BOTTOM layer is the base plate the inserts seat into. It is the
    complement -- whatever the layers above it don't claim -- so it spans the
    whole cutout and carries the raw, uncleaned boundary detail.
  * every layer ABOVE it is a printable insert, made mutually exclusive in
    TERRAIN_PRECEDENCE order (earlier = higher = claimed first).

Which terrain type sits at the bottom is a per-scene choice: for a
vegetation-dominated scene (Ararat) foliage is the base, rock + snow are inserts;
for a rock-dominated scene (Bishorn) rock is the base, foliage + snow are inserts.
So the code never special-cases a terrain type -- it works in hierarchy terms
(bottom vs. inserts) and the caller names which class is the bottom.

Because the bottom is the raw complement, it is the layer that carries thin necks
and slivers; it alone is passed through drop_unprintable, and the opened-off
sub-printable bits are absorbed by the inserts that border them. Explicit insert
outlines (satellite snow/foliage) arrive already cleaned (despeckle + APCSF).

All geometry is in the cutout CRS (metres).
"""
from shapely.geometry import Polygon
from shapely.ops import unary_union

from snow_overlay import (despeckle, apcsf_clean, load_geojson_layer,
                          MIN_FEATURE_M2, SCALE_M_PER_MM)
from terrain_classifier import (TERRAIN_PRECEDENCE, TERRAIN_GLACIER,
                                TERRAIN_FOLIAGE, TERRAIN_ROCK)

VEG_ITERS = 100         # foliage APCSF iterations (sub-1mm base slivers <1% at RESAMPLE_M=15)
# Printable-feature rules, by feature shape (at 1:133k, 0.4 mm nozzle):
MIN_THICKNESS_MM = 1.0  # ELONGATED features must be >= this wide (thin ridges/slivers)
MIN_BLOB_MM = 2.0       # COMPACT features must be >= this across (small islands)


def load_and_clean_veg(geojson_path, iterations=VEG_ITERS,
                       min_feature_m2=MIN_FEATURE_M2):
    """Load a satellite foliage polygon and clean it (despeckle + APCSF)."""
    geom, src_epsg = load_geojson_layer(geojson_path)
    return apcsf_clean(geom, iterations, min_feature_m2=min_feature_m2), src_epsg


def open_min_width(geom, min_width_m):
    """Drop features thinner than min_width_m via a morphological opening."""
    r = min_width_m / 2.0
    return geom.buffer(-r).buffer(r).buffer(0)


def drop_unprintable(geom, min_thickness_mm=MIN_THICKNESS_MM, min_blob_mm=MIN_BLOB_MM,
                     scale_m_per_mm=SCALE_M_PER_MM):
    """Remove features too small/thin to print & handle as a standalone insert.

    Two shape-dependent rules:
      * elongated features (long thin ridges/slivers) must be >= min_thickness_mm
        wide  -> morphological opening at that width;
      * compact features (small islands) must be >= min_blob_mm across, even if
        they survived the opening -> drop polygons below that area.
    """
    opened = open_min_width(geom, min_thickness_mm * scale_m_per_mm)
    min_blob_m2 = (min_blob_mm * scale_m_per_mm) ** 2
    return despeckle(opened, min_blob_m2)


def resolve_layers(cutout, layer_geoms, base_class,
                   min_thickness_mm=MIN_THICKNESS_MM, min_blob_mm=MIN_BLOB_MM,
                   scale_m_per_mm=SCALE_M_PER_MM):
    """Cut terrain layers into disjoint printable inserts + one base plate.

    `layer_geoms` maps a terrain class -> polygon (in the cutout's CRS) for every
    class with an explicit satellite outline (e.g. snow=GLACIER, foliage=FOLIAGE).
    `base_class` names the BOTTOM layer: the base plate the inserts seat into.

    Classes are made mutually exclusive in TERRAIN_PRECEDENCE order (earlier =
    higher in the stack, claimed first). Exactly one class carries no satellite
    outline (TERRAIN_PRECEDENCE[-1], the unclassified ground); its mask is derived
    as the leftover of the others. That unmasked layer is the raw complement --
    built by subtraction, so it is the only layer carrying thin dendritic slivers
    (the satellite layers arrive APCSF-cleaned from their extractor).

    drop_unprintable is applied to that raw complement, and only when it is an
    INSERT: its slivers (thinner than min_thickness_mm) and specks (smaller than
    min_blob_mm) are opened/despeckled off so it yields chunky, seatable inserts,
    and the removed area falls through into the base below. When the raw complement
    IS the base it is left whole -- a continuous base plate prints fine and needs no
    opening. Satellite insert outlines are never re-opened (re-opening the APCSF
    snow severs its interior necks and shaves ~0.6% of it).

    The BOTTOM (`base_class`) is the complement of the finished inserts -- the
    cutout minus every insert above it -- so it spans the whole disc.

    Returns (base_geom, inserts) where inserts is a {terrain_class: geom} dict,
    all clipped to cutout and mutually disjoint.
    """
    unmasked_cls = TERRAIN_PRECEDENCE[-1]     # the one class with no satellite mask
    remaining = cutout.buffer(0)
    layers = {}
    for tc in TERRAIN_PRECEDENCE:
        if tc == unmasked_cls:
            continue                          # filled from the remainder below
        g = layer_geoms.get(tc)
        if g is None:
            continue
        piece = g.intersection(remaining).buffer(0)
        layers[tc] = piece
        remaining = remaining.difference(piece).buffer(0)
    layers[unmasked_cls] = remaining          # unmasked = cutout - every mask

    inserts = {}
    for tc, g in layers.items():
        if tc == base_class or g.is_empty:
            continue
        # The raw-complement insert is opened to chunky pieces; its removed slivers
        # fall through to the base. Satellite inserts arrive pre-cleaned.
        gi = (drop_unprintable(g, min_thickness_mm, min_blob_mm, scale_m_per_mm)
              if tc == unmasked_cls else g)
        if not gi.is_empty:
            inserts[tc] = gi

    base = cutout.buffer(0)
    for gi in inserts.values():
        base = base.difference(gi).buffer(0)
    return base, inserts


def resolve_foliage_base(cutout, foliage, snow, min_thickness_mm=MIN_THICKNESS_MM,
                         min_blob_mm=MIN_BLOB_MM, scale_m_per_mm=SCALE_M_PER_MM):
    """Ararat inversion: foliage base, rock + snow inserts (see resolve_layers).

    Thin wrapper over resolve_layers with base_class=FOLIAGE. Returns
    (foliage_base, rock_inserts, snow_inserts) for the preview scripts.
    """
    base, inserts = resolve_layers(
        cutout, {TERRAIN_GLACIER: snow, TERRAIN_FOLIAGE: foliage},
        base_class=TERRAIN_FOLIAGE,
        min_thickness_mm=min_thickness_mm, min_blob_mm=min_blob_mm,
        scale_m_per_mm=scale_m_per_mm)
    empty = Polygon()
    return base, inserts.get(TERRAIN_ROCK, empty), inserts.get(TERRAIN_GLACIER, empty)
