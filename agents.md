# Agents / Responsibilities

- `dem_batch_to_stl.py`: Orchestrates the end-to-end batch run (downloads with caching, mosaics, meshes), argument parsing, reporting, and error handling.
- `downloader.py`: Prepares directories, derives base filenames from URLs (auto-detects extension), reads the URL list (text, CSV, standard `.xlsx`, or falls back to text for plain `.xls`), and downloads DEM files (GeoTIFF or ASC) via HTTP with caching into `cache/`.
- `dem_processing.py`: Opens and merges DEMs with rasterio (format-agnostic), fills nodata cells (minimum of valid values), and applies optional grid downsampling.
- `mesh_builder.py`: Converts processed DEM grids into watertight meshes (top surface, base, side walls) and exports binary STL files; supports lake lowering by range and fixed mm offset.
