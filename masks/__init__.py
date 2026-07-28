"""Mask providers: where the terrain-class polygons come from.

A mask provider is a callable taking the ModelFrame and returning
``{terrain_class: [GeoJSON geometry dicts]}`` with coordinates in
``frame.ref_crs`` (CRS metres). Everything source-specific -- API queries,
file formats, cleanup tuned to the sensor -- lives behind that call; the
layout stage only ever sees the merged dict. Providers read the true print
scale off ``frame.scale_m_per_mm`` to apply feature-size rules stated in mm.

Providers: ``masks.osm.OsmMasks`` (Overpass query + rasterizable polygons),
``masks.sentinel2.SnowMasks`` / ``masks.sentinel2.FoliageMasks`` (satellite
GeoJSON layers, cleaned by despeckle + APCSF).

This package also owns the terrain-class vocabulary the providers key their
results by, shared by every downstream stage.
"""

# Terrain class constants. The integer VALUE is just an id, not a priority — see
# TERRAIN_PRECEDENCE below for the actual layer ordering.
TERRAIN_ROCK = 0
TERRAIN_GLACIER = 1
TERRAIN_WATER = 2
TERRAIN_FOLIAGE = 3

TERRAIN_NAMES = {
    TERRAIN_ROCK: "rock",
    TERRAIN_GLACIER: "glacier",
    TERRAIN_WATER: "water",
    TERRAIN_FOLIAGE: "foliage",
}
NAME_TO_TERRAIN = {name: cls for cls, name in TERRAIN_NAMES.items()}

# Single source of truth for layer ordering, used by every stage (OSM mesh build
# and satellite composition). Where two classes overlap, the earlier entry wins
# over the later one. The LAST entry is the default "leftover" base: any area not
# claimed by an earlier class becomes it. Normal prints use rock as that base
# (rock is never rasterized — it is whatever the overlays don't cover); the
# satellite-inverted Ararat print instead designates foliage as the base and
# rasterizes rock as an insert class (see terrain_compose.resolve_layers).
# Satellite "snow" (NDSI) is carried as the GLACIER class — the same physical
# frozen-ground layer OSM tags as glacier.
TERRAIN_PRECEDENCE = [TERRAIN_WATER, TERRAIN_GLACIER, TERRAIN_FOLIAGE, TERRAIN_ROCK]


def overlay_precedence(base_class=TERRAIN_ROCK):
    """Precedence-ordered overlay (insert) classes for a given base class.

    The base class is removed from TERRAIN_PRECEDENCE; the rest keep their order,
    highest priority first. With base_class=ROCK this is [WATER, GLACIER, FOLIAGE]
    (the historical order); with base_class=FOLIAGE it is [WATER, GLACIER, ROCK].
    """
    return [c for c in TERRAIN_PRECEDENCE if c != base_class]


def merge_masks(frame, providers):
    """Run mask providers in order and merge their class dicts.

    Where two providers contribute the same class, their geometry lists
    concatenate (e.g. satellite snow unions with OSM glacier downstream).
    """
    merged = {}
    for provider in providers:
        for terrain_class, geoms in provider(frame).items():
            merged.setdefault(terrain_class, []).extend(geoms)
    return merged
