"""Mask provider: lakes read off the DEM itself, carried as the WATER class.

No survey layer is involved. The rule is the one the plain relief block used
before water became an ordinary insert class: every sample within
``range_percent`` of the DEM's total relief above its minimum is water. On a
print whose lowest ground IS a lake -- the reason to reach for this at all --
that picks the lake out; on anything else it picks out the valley floor, which
is what the threshold literally says.

The polygon is the boundary between wet and dry SAMPLES: ``rasterio.features``
traces pixel edges, and a pixel edge lies exactly midway between the two pixel
centres it separates, which is where a linearly interpolated surface crosses
the threshold. No smoothing follows -- a DEM lake's shoreline IS that crossing
-- only the 2x2 mm despeckle every provider applies at the print scale, so the
layout is never handed a speck too small to seat as an insert.

Merges with OSM water (``masks.osm``) through ``merge_masks`` when both are
given; the layout unions same-class geometry from every provider.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from rasterio.features import shapes
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

from masks import TERRAIN_WATER
from masks.sentinel2 import despeckle, despeckle_area_m2


def dem_water_geoms(dem: np.ndarray, ref_transform, range_percent: float,
                    min_area_m2: float) -> list:
    """GeoJSON geometry dicts (in the DEM's CRS) for the low ground below the
    relief-percent threshold, despeckled to ``min_area_m2``."""
    valid = np.isfinite(dem)
    if not valid.any():
        return []
    lo = float(np.min(dem[valid]))
    hi = float(np.max(dem[valid]))
    threshold = lo + (hi - lo) * (range_percent / 100.0)

    wet = valid & (dem <= threshold)
    if not wet.any():
        return []
    polys = [shape(g) for g, _v in shapes(wet.astype(np.uint8), mask=wet,
                                          transform=ref_transform)]
    polys = [p for p in polys if p.area >= min_area_m2]
    if not polys:
        return []
    # Every survivor is above the threshold already, so despeckle only reaches
    # the interior rings (islands too small to print as a hole).
    geom = despeckle(unary_union(polys), min_area_m2)
    return [mapping(g) for g in
            (geom.geoms if geom.geom_type == "MultiPolygon" else [geom])]


@dataclass(frozen=True, eq=False)
class LakeMasks:
    """Mask provider: DEM ground below a relief-percent threshold, as WATER."""
    dem: np.ndarray
    range_percent: float
    min_feature_m2: Optional[float] = None    # None -> 2x2 mm at the print scale

    def __call__(self, frame):
        min_area_m2 = (self.min_feature_m2 if self.min_feature_m2 is not None
                       else despeckle_area_m2(frame.scale_m_per_mm))
        geoms = dem_water_geoms(self.dem, frame.ref_transform,
                                self.range_percent, min_area_m2)
        print(f"[INFO] DEM lakes -> water: {len(geoms)} polygon(s) below "
              f"{self.range_percent}% of the relief", flush=True)
        return {TERRAIN_WATER: geoms}
