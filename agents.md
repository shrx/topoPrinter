# Agents / Responsibilities

- `dem_batch_to_stl.py`: Orchestrates the end-to-end batch run (downloads with caching, mosaics, meshes), argument parsing, reporting, and error handling.
- `downloader.py`: Prepares directories, derives base filenames from URLs (auto-detects extension), reads the URL list (text, CSV, standard `.xlsx`, or falls back to text for plain `.xls`), and downloads DEM files (GeoTIFF or ASC) via HTTP with caching into `cache/`.
- `dem_processing.py`: Opens and merges DEMs with rasterio (format-agnostic), fills nodata cells (minimum of valid values), and applies optional grid downsampling.
- `mesh_builder.py`: Extrudes a finished 2D layout over the DEM into watertight bodies (base plate with its pocket floors, one prism per insert part) and exports binary STL files; adds z only, and sinks the water class by the lake lowering.
