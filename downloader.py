"""
Helpers for reading URL lists (text, CSV, or Excel) and downloading DEMs with caching.
"""

import os
import shutil
import sys
import zipfile
import xml.etree.ElementTree as ET
from typing import List
from urllib.parse import urlsplit

import requests
import csv

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


def _read_text_urls(path: str) -> List[str]:
    urls: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            url = line.strip()
            if not url or url.startswith("#"):
                continue
            urls.append(url)
    return urls


def _read_csv_urls(path: str) -> List[str]:
    urls: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            for cell in row:
                text = cell.strip()
                if text.lower().startswith("http"):
                    urls.append(text)
    return urls


def _read_xlsx_urls(path: str) -> List[str]:
    # Minimal XLSX reader using stdlib to avoid extra dependencies.
    urls: List[str] = []
    ns_main = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

    with zipfile.ZipFile(path, "r") as zf:
        wb_xml = ET.fromstring(zf.read("xl/workbook.xml"))
        rels_xml = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels_xml.findall("m:Relationship", {"m": "http://schemas.openxmlformats.org/package/2006/relationships"})}

        first_sheet = wb_xml.find("m:sheets/m:sheet", ns_main)
        if first_sheet is None:
            return urls
        rid = first_sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        sheet_target = rel_map.get(rid)
        if not sheet_target:
            return urls
        sheet_path = sheet_target if sheet_target.startswith("xl/") else f"xl/{sheet_target}"
        sheet_xml = ET.fromstring(zf.read(sheet_path))

        shared_strings = []
        if "xl/sharedStrings.xml" in zf.namelist():
            ss_xml = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in ss_xml.findall("m:si", ns_main):
                texts = [t.text or "" for t in si.findall(".//m:t", ns_main)]
                shared_strings.append("".join(texts))

        def cell_text(cell) -> str:
            cell_type = cell.attrib.get("t")
            value_el = cell.find("m:v", ns_main)
            if value_el is None or value_el.text is None:
                return ""
            if cell_type == "s":
                try:
                    return shared_strings[int(value_el.text)]
                except (ValueError, IndexError):
                    return ""
            if cell_type in ("str", "inlineStr"):
                return value_el.text
            return value_el.text

        for row in sheet_xml.findall(".//m:sheetData/m:row", ns_main):
            for cell in row.findall("m:c", ns_main):
                text = cell_text(cell).strip()
                if text.lower().startswith("http"):
                    urls.append(text)
    return urls


def read_url_list(path: str) -> List[str]:
    """Load URLs from text/CSV/XLSX (or text-like XLS) list files."""
    lower = path.lower()
    try:
        with open(path, "rb") as f:
            header = f.read(4)
    except FileNotFoundError:
        raise

    if lower.endswith(".csv"):
        return _read_csv_urls(path)

    is_excelish = lower.endswith((".xlsx", ".xls")) or header.startswith(b"PK\x03\x04")
    if is_excelish:
        try:
            return _read_xlsx_urls(path)
        except (KeyError, zipfile.BadZipFile, ET.ParseError, ValueError):
            # Likely legacy .xls or misnamed text; fall back to text parsing.
            print(
                "[WARN] Failed to parse Excel structure; falling back to plain-text URL reading.",
                file=sys.stderr,
            )
    # Fallback to text (also covers legacy .xls that are plain text lists).
    return _read_text_urls(path)


def ensure_dir(path: str) -> None:
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def derive_base_name(url: str, fallback_index: int) -> str:
    """Derive a safe base filename from a URL, with index fallback."""
    parsed = urlsplit(url)
    candidate = os.path.basename(parsed.path)
    if not candidate:
        return f"tile_{fallback_index}"

    # Auto-detect extension and strip it
    name, _ext = os.path.splitext(candidate)
    return name or f"tile_{fallback_index}"


def extract_tandemx_edem_tif(zip_path: str) -> str:
    """Extract the usable DEM GeoTIFF from a TanDEM-X EDEM zip into the cache.

    TanDEM-X EDEM tiles ship as a zip whose payload is a product directory with
    two elevation rasters — ellipsoidal WGS84 heights (``_W84.tif``) and
    geoid/orthometric heights (``_EGM.tif``) — plus auxiliary quality masks
    under ``EDEM_AUXFILES/``. We pick the EGM raster (metres above sea level)
    for a physically meaningful relief, falling back to W84, then any GeoTIFF.

    Returns the path to the extracted GeoTIFF (cached, so re-runs skip the work).
    """
    with zipfile.ZipFile(zip_path) as zf:
        tifs = [n for n in zf.namelist() if n.lower().endswith(".tif")]
        # Prefer the main elevation layers over the auxiliary masks.
        elevation = [n for n in tifs if "auxfiles" not in n.lower()]
        pool = elevation or tifs
        egm = [n for n in pool if n.lower().endswith("_egm.tif")]
        w84 = [n for n in pool if n.lower().endswith("_w84.tif")]
        chosen = egm or w84 or pool
        if not chosen:
            raise RuntimeError(f"No GeoTIFF found inside archive: {zip_path}")
        member = chosen[0]

        out_path = os.path.join(CACHE_DIR, os.path.basename(member))
        if os.path.exists(out_path):
            return out_path

        # Extract to a temp file and rename on success, mirroring download_dem,
        # so an interrupted extraction never leaves a truncated cached tif.
        part_path = out_path + ".part"
        try:
            with zf.open(member) as src, open(part_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            os.replace(part_path, out_path)
        except Exception:
            if os.path.exists(part_path):
                os.remove(part_path)
            raise
    return out_path


def download_dem(url: str, fallback_index: int) -> str:
    """Download DEM file directly to cache."""
    ensure_dir(CACHE_DIR)

    parsed = urlsplit(url)
    filename_from_url = os.path.basename(parsed.path)
    base_name, ext = os.path.splitext(filename_from_url)

    if not base_name:
        base_name = f"tile_{fallback_index}"
    if not ext:
        ext = ".tif"

    file_name = f"{base_name}{ext}"
    cache_path = os.path.join(CACHE_DIR, file_name)

    if os.path.exists(cache_path):
        if zipfile.is_zipfile(cache_path):
            return extract_tandemx_edem_tif(cache_path)
        return cache_path

    # Download to a temp file and rename on success, so an interrupted
    # download never leaves a truncated file that later runs treat as cached.
    part_path = cache_path + ".part"
    try:
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(part_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        os.replace(part_path, cache_path)
    except Exception as exc:  # noqa: BLE001
        if os.path.exists(part_path):
            os.remove(part_path)
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc

    if zipfile.is_zipfile(cache_path):
        return extract_tandemx_edem_tif(cache_path)
    return cache_path
