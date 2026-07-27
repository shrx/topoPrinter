"""The DEM -> print-space mapping, with no elevations in it.

``ModelFrame`` carries everything needed to place a CRS coordinate in model
millimetres: the merged DEM's georeference plus the requested print width. It is
deliberately the HORIZONTAL half of what ``mesh_builder._compute_model_coordinates``
computes -- no heights, no z exaggeration, no base thickness -- so the whole 2D
polygon stage (``terrain_layout``) can run without ever touching the raster.

Model space is a plain affine image of the DEM grid: the mesh spans first-to-last
pixel CENTRE, so ``cols - 1`` spacings cover ``x_size_mm``, and the model extent is
exactly ``(0, 0) - (x_size_mm, model_y_mm)``. That is why the derived quantities
below are pure scalar arithmetic rather than reductions over the X/Y meshgrids:
``np.linspace`` reproduces the endpoints exactly, so ``X.max()`` IS ``x_size_mm``.

Model mm are print mm (1 model mm prints as 1 mm), so feature-size rules stated in
mm apply directly here. ``scale_m_per_mm`` converts the other way, for the mask
providers that clean their geometry in CRS metres before it ever reaches this frame.
"""

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import shapely
from shapely.geometry import (MultiPolygon, Polygon as ShapelyPolygon,
                              shape as shapely_shape)
from shapely.ops import unary_union


@dataclass(frozen=True)
class ModelFrame:
    """CRS <-> model-mm mapping for one merged DEM at one print size."""

    ref_transform: object
    ref_crs: object
    rows: int
    cols: int
    px_size_x: float      # metres (already converted from degrees for geographic CRS)
    px_size_y: float
    x_size_mm: float

    @classmethod
    def from_dem(cls, dem_shape: Tuple[int, int], px_size_x: float, px_size_y: float,
                 x_size_mm: float, ref_transform, ref_crs) -> "ModelFrame":
        rows, cols = dem_shape
        return cls(ref_transform, ref_crs, int(rows), int(cols),
                   float(px_size_x), float(px_size_y), float(x_size_mm))

    # --- derived scalars -------------------------------------------------

    @property
    def model_y_mm(self) -> float:
        """Model height in mm, from the DEM's pixel-centre-to-pixel-centre aspect."""
        aspect_ratio = (((self.rows - 1) * self.px_size_y)
                        / ((self.cols - 1) * self.px_size_x))
        return self.x_size_mm * aspect_ratio

    @property
    def bounds_mm(self) -> Tuple[float, float, float, float]:
        """(minx, miny, maxx, maxy) of the model grid, in mm."""
        return (0.0, 0.0, self.x_size_mm, self.model_y_mm)

    @property
    def grid_pitch_mm(self) -> float:
        """Model-space distance between adjacent DEM samples along X."""
        return self.x_size_mm / max(self.cols - 1, 1)

    @property
    def grid_xs(self) -> np.ndarray:
        """Model-space X of each DEM column, ascending.

        Bit-identical to the X meshgrid ``_compute_model_coordinates`` builds, so a
        boundary densified against this lands exactly on the mesh stage's grid lines.
        """
        return np.linspace(0.0, self.x_size_mm, self.cols)

    @property
    def grid_ys(self) -> np.ndarray:
        """Model-space Y of each DEM row, ascending.

        Row 0 of the DEM is the TOP row, so the mesh stage builds Y descending; this
        reverses that array rather than re-running ``linspace`` the other way round,
        because the two are not bit-identical in the interior.
        """
        return np.linspace(self.model_y_mm, 0.0, self.rows)[::-1]

    @property
    def output_resolution(self) -> np.float32:
        """The float32 ULP at the model's largest coordinate.

        Binary STL stores vertices as float32, so this is the finest distance the
        export can represent; it is the grid every polygon boundary is snapped to
        before any solid is built. Taken at the largest coordinate because that is
        where float32's absolute step is coarsest.
        """
        return np.spacing(np.float32(max(self.x_size_mm, self.model_y_mm)))

    @property
    def scale_m_per_mm(self) -> float:
        """Real terrain metres per printed mm -- the true print scale.

        Mask providers clean in CRS metres and need this to apply feature-size
        rules that are stated in mm; the layout stage works in mm and uses 1.0.
        """
        return ((self.cols - 1) * self.px_size_x) / self.x_size_mm

    # --- CRS -> model mm -------------------------------------------------

    def point_to_mm(self, x_crs: float, y_crs: float) -> Tuple[float, float]:
        """Map one CRS point to model mm exactly (pixel-centre convention).

        Grid vertex (i, j) carries the DEM sample of pixel (i, j), whose CRS
        location is the pixel CENTRE (col + 0.5, row + 0.5 in pixel space), so the
        mapping must subtract that half-pixel before scaling.
        """
        col_frac = (x_crs - self.ref_transform.c) / self.ref_transform.a - 0.5
        row_frac = (y_crs - self.ref_transform.f) / self.ref_transform.e - 0.5
        model_x = col_frac / (self.cols - 1) * self.x_size_mm
        model_y = self.model_y_mm * (1 - row_frac / (self.rows - 1))
        return model_x, model_y

    def coords_to_mm(self, crs_coords) -> np.ndarray:
        """Vectorized ``point_to_mm`` over an (N, 2) array.

        OSM polygon rings can carry tens of thousands of points, so whole rings are
        transformed at once.
        """
        arr = np.asarray(crs_coords, dtype=np.float64)
        model_y_mm = self.model_y_mm
        col_frac = (arr[:, 0] - self.ref_transform.c) / self.ref_transform.a - 0.5
        row_frac = (arr[:, 1] - self.ref_transform.f) / self.ref_transform.e - 0.5
        model_x = col_frac / (self.cols - 1) * self.x_size_mm
        model_y = model_y_mm * (1 - row_frac / (self.rows - 1))
        return np.column_stack((model_x, model_y))

    def geojson_to_mm(self, geojson_geom: dict) -> shapely.Geometry:
        """Convert one GeoJSON geometry (CRS coords) to shapely in model mm."""
        geom = shapely_shape(geojson_geom)

        def _ring(ring):
            return self.coords_to_mm(np.asarray(ring.coords))

        if geom.geom_type == "Polygon":
            return ShapelyPolygon(_ring(geom.exterior),
                                  [_ring(h) for h in geom.interiors])
        if geom.geom_type == "MultiPolygon":
            return MultiPolygon([
                ShapelyPolygon(_ring(p.exterior), [_ring(h) for h in p.interiors])
                for p in geom.geoms
            ])
        return geom

    def geojsons_to_mm(self, geojson_geoms: Iterable[dict]) -> Optional[shapely.Geometry]:
        """Convert a list of GeoJSON geometries to one unioned geometry in model mm."""
        geoms: List[dict] = list(geojson_geoms or [])
        if not geoms:
            return None
        polys_mm = []
        for g in geoms:
            p = self.geojson_to_mm(g)
            if not p.is_valid:
                # Self-intersecting rings are common in OSM data; repair instead of
                # dropping the whole feature. make_valid may return a collection --
                # keep only the polygonal parts.
                p = shapely.make_valid(p)
                if p.geom_type == "GeometryCollection":
                    parts = [g2 for g2 in p.geoms
                             if g2.geom_type in ("Polygon", "MultiPolygon")]
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
        return None if union.is_empty else union
