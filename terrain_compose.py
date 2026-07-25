"""Resolve satellite terrain layers into one base + inserts for separate printing.

For Ararat the vegetation dominates the cutout (~62%), so the printable design is
inverted from the usual rock base: FOLIAGE is the base plate and rock + snow are
seated inserts. Rock forms a dendritic network of ridges/valleys, so its thin
"river slivers" (thinner than SLIVER_MM once printed) would be fragile inserts;
they are dissolved into the foliage base, leaving only chunky rock inserts.

All geometry is in the cutout CRS (metres). Foliage/snow are expected already
cleaned (despeckle + APCSF) via snow_overlay.apcsf_clean.
"""
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
    class with an explicit outline (e.g. satellite snow=GLACIER, foliage=FOLIAGE).
    Classes are made mutually exclusive in the shared TERRAIN_PRECEDENCE order
    (earlier wins). Rock (the last, unclassified-ground entry of the precedence)
    is the COMPLEMENT: it is materialised as `cutout` minus every explicit class,
    so it never needs an outline of its own. `base_class` becomes the base plate:
    the leftover area not claimed by any printable insert.

    The raw COMPLEMENT (rock leftover) is passed through drop_unprintable, so its
    elongated slivers (thinner than min_thickness_mm) and compact islands (smaller
    than min_blob_mm) fall through into the base plate rather than becoming fragile
    inserts. Explicit layer outlines are assumed already cleaned by their own
    extractor (snow/foliage via APCSF) and are NOT re-opened -- verified that
    re-running drop_unprintable on the APCSF snow severs interior necks and shaves
    ~0.6% of it, so it would degrade the locked snow polygon rather than tidy it.

    Returns (base_geom, inserts) where inserts is a {terrain_class: geom} dict,
    all clipped to cutout and mutually disjoint.
    """
    complement_cls = TERRAIN_PRECEDENCE[-1]   # rock: whatever isn't classified
    remaining = cutout.buffer(0)
    resolved = {}
    for tc in TERRAIN_PRECEDENCE:
        if tc == complement_cls:
            continue                          # filled from the remainder below
        g = layer_geoms.get(tc)
        if g is None:
            continue
        piece = g.intersection(remaining).buffer(0)
        resolved[tc] = piece
        remaining = remaining.difference(piece).buffer(0)
    resolved[complement_cls] = remaining      # complement = cutout - all explicit

    inserts = {}
    for tc, g in resolved.items():
        if tc == base_class or g.is_empty:
            continue
        # Only the raw complement (rock) needs sliver removal: its thin dendritic
        # "river" ridges (< min_thickness) are dissolved into the foliage base by
        # a morphological opening, and its small islands (< min_blob) despeckled
        # away, leaving only chunky rock inserts. Explicit outlines arrive already
        # cleaned by their extractor.
        gi = (drop_unprintable(g, min_thickness_mm, min_blob_mm, scale_m_per_mm)
              if tc == complement_cls else g)
        if not gi.is_empty:
            inserts[tc] = gi

    base = cutout.buffer(0)
    for gi in inserts.values():
        base = base.difference(gi).buffer(0)

    # Snow-interface cleanup (the ONLY interface touched here). The locked snow
    # boundary frets the foliage base into thin necks between its tendrils. Each
    # thin foliage bit that touches the snow boundary is merged into its non-snow
    # neighbour: where it also borders the rock complement it is given to rock; a
    # bit bordered only by snow + foliage stays foliage (it is base-joined, so it
    # prints as supported base rather than a fragile island). Rock's own bits are
    # already resolved -- its isolated sub-blob specks went to the base via the
    # despeckle above, and a strand joined to the main rock stays rock. Only bits
    # meeting the snow boundary are touched, so the base interior and the
    # base<->rock interface are left alone.
    snow_geoms = [g for tc, g in inserts.items()
                  if tc not in (complement_cls, base_class)]
    rock = inserts.get(complement_cls)
    if snow_geoms and rock is not None and not rock.is_empty:
        snow = unary_union(snow_geoms)
        r = min_thickness_mm * scale_m_per_mm / 2.0
        thin = base.difference(base.buffer(-r).buffer(r)).buffer(0)
        pieces = thin.geoms if thin.geom_type == "MultiPolygon" else [thin]
        give = [p for p in pieces if not p.is_empty
                and p.intersects(snow.boundary) and p.intersects(rock)]
        if give:
            rock = rock.union(unary_union(give)).buffer(0)
            inserts[complement_cls] = rock
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
    from shapely.geometry import Polygon
    empty = Polygon()
    return base, inserts.get(TERRAIN_ROCK, empty), inserts.get(TERRAIN_GLACIER, empty)
