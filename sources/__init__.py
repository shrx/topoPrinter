"""DEM source adapters.

Each source module owns everything specific to one DEM provider:

- a tile lookup CLI (``python -m sources.swiss`` / ``sources.arso`` /
  ``sources.tandemx``) that turns a region into download URLs, and
- a ``prepare(path) -> Optional[str]`` hook that recognises the provider's
  downloaded artifact and converts it into a raster rasterio can read
  (TanDEM-X zip -> EGM GeoTIFF, ARSO XYZ point cloud -> gridded GeoTIFF),
  returning None for files it does not claim.

The rest of the pipeline stays source-agnostic: the CLI downloads the URL
list, runs the files through prepare_dem_files(), and merges them with
load_dem() into a DemProduct.
"""

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import rasterio


@dataclass(frozen=True)
class DemProduct:
    """A merged DEM ready for the model pipeline.

    Pixel sizes are in metres regardless of the CRS (geographic DEMs are
    converted at their centre latitude); nodata cells are NaN in ``array``.
    """

    array: np.ndarray
    transform: object
    crs: object
    px_size_x: float
    px_size_y: float


def prepare_dem_files(paths: Iterable[str]) -> List[str]:
    """Convert downloaded files into rasters, dispatching to source adapters.

    Order matters and mirrors the sniffing the pipeline has always done: an
    archive can never be opened as a raster, so TanDEM-X goes first; a plain
    raster must not be mistaken for a point cloud (GDAL has an XYZ driver),
    so rasterio gets a try before the ARSO gridder.
    """
    from sources import arso, tandemx

    prepared = []
    for path in paths:
        converted = tandemx.prepare(path)
        if converted is not None:
            prepared.append(converted)
            continue
        try:
            with rasterio.open(path):
                prepared.append(path)
                continue
        except Exception:
            pass
        converted = arso.prepare(path)
        if converted is not None:
            prepared.append(converted)
            continue
        raise ValueError(f"Unable to read file as raster or XYZ point cloud: {path}")

    return prepared


def load_dem(paths: Iterable[str], downsample: int) -> DemProduct:
    """Merge prepared rasters into one DemProduct."""
    from dem_processing import load_and_merge

    arr, px_size_x, px_size_y, crs, transform = load_and_merge(paths, downsample)
    return DemProduct(array=arr, transform=transform, crs=crs,
                      px_size_x=px_size_x, px_size_y=px_size_y)
