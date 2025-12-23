"""
Helpers for reading URL lists (text, CSV, or Excel) and downloading DEMs with caching.
"""

import os
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
        return cache_path

    try:
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(cache_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc

    return cache_path
