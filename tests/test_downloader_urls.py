"""Tests for the URL-list readers and filename derivation.

``read_url_list`` accepts four things that all arrive as "the tile list": a plain
text file, a CSV, a real XLSX, and a legacy ``.xls`` that is actually plain text.
It picks between them by extension AND by sniffing the zip magic, with a fallback
when the Excel parse fails -- so the interesting cases are the mismatches between
what a file is named and what it contains. The XLSX reader is hand-rolled over
stdlib ``zipfile``/``ElementTree`` rather than openpyxl, so it is exercised against
a real (minimal) workbook built here.
"""

import zipfile

import pytest

from downloader import derive_base_name, read_url_list


URL_A = "https://data.geo.admin.ch/tiles/a_1m.tif"
URL_B = "http://gis.arso.gov.si/lidar/DMR1_456_100.asc"

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
DOC_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def _make_xlsx(path, rows, *, use_shared_strings=True, sheet_target="worksheets/sheet1.xml"):
    """Write a minimal but structurally valid XLSX containing `rows` of text."""
    shared, cells_xml = [], []
    for r, row in enumerate(rows, start=1):
        cs = []
        for c, text in enumerate(row):
            ref = f"{chr(ord('A') + c)}{r}"
            if use_shared_strings:
                shared.append(text)
                cs.append(f'<c r="{ref}" t="s"><v>{len(shared) - 1}</v></c>')
            else:
                cs.append(f'<c r="{ref}" t="str"><v>{text}</v></c>')
        cells_xml.append(f'<row r="{r}">{"".join(cs)}</row>')

    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("xl/workbook.xml",
                    f'<workbook xmlns="{MAIN_NS}" xmlns:r="{DOC_REL_NS}">'
                    f'<sheets><sheet name="S1" sheetId="1" r:id="rId1"/></sheets>'
                    f'</workbook>')
        zf.writestr("xl/_rels/workbook.xml.rels",
                    f'<Relationships xmlns="{PKG_REL_NS}">'
                    f'<Relationship Id="rId1" Type="{DOC_REL_NS}/worksheet" '
                    f'Target="{sheet_target}"/></Relationships>')
        zf.writestr("xl/worksheets/sheet1.xml",
                    f'<worksheet xmlns="{MAIN_NS}"><sheetData>'
                    f'{"".join(cells_xml)}</sheetData></worksheet>')
        if use_shared_strings:
            sis = "".join(f"<si><t>{s}</t></si>" for s in shared)
            zf.writestr("xl/sharedStrings.xml",
                        f'<sst xmlns="{MAIN_NS}">{sis}</sst>')
    return str(path)


class TestTextList:
    def test_reads_one_url_per_line(self, tmp_path):
        p = tmp_path / "urls.txt"
        p.write_text(f"{URL_A}\n{URL_B}\n")
        assert read_url_list(str(p)) == [URL_A, URL_B]

    def test_skips_blank_lines_and_comments(self, tmp_path):
        p = tmp_path / "urls.txt"
        p.write_text(f"# a comment\n\n{URL_A}\n   \n#{URL_B}\n{URL_B}\n")
        assert read_url_list(str(p)) == [URL_A, URL_B]

    def test_strips_surrounding_whitespace(self, tmp_path):
        p = tmp_path / "urls.txt"
        p.write_text(f"  {URL_A}  \n\t{URL_B}\t\n")
        assert read_url_list(str(p)) == [URL_A, URL_B]

    def test_duplicates_are_preserved(self, tmp_path):
        """Reading does not dedupe; the download cache is what avoids refetching."""
        p = tmp_path / "urls.txt"
        p.write_text(f"{URL_A}\n{URL_A}\n")
        assert read_url_list(str(p)) == [URL_A, URL_A]

    def test_empty_file_gives_an_empty_list(self, tmp_path):
        p = tmp_path / "urls.txt"
        p.write_text("")
        assert read_url_list(str(p)) == []

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_url_list(str(tmp_path / "nope.txt"))


class TestCsvList:
    def test_picks_url_cells_out_of_a_row(self, tmp_path):
        p = tmp_path / "urls.csv"
        p.write_text(f"tile name,link,notes\nAlpha,{URL_A},keep\nBeta,{URL_B},keep\n")
        assert read_url_list(str(p)) == [URL_A, URL_B]

    def test_header_and_non_url_cells_are_ignored(self, tmp_path):
        """Selection is by the http prefix, so no header row handling is needed."""
        p = tmp_path / "urls.csv"
        p.write_text(f"name,url\n,\nfoo,not-a-url\nbar,{URL_A}\n")
        assert read_url_list(str(p)) == [URL_A]

    def test_several_urls_in_one_row_are_all_taken(self, tmp_path):
        p = tmp_path / "urls.csv"
        p.write_text(f"{URL_A},{URL_B}\n")
        assert read_url_list(str(p)) == [URL_A, URL_B]

    def test_quoted_cells_are_unquoted_by_the_csv_reader(self, tmp_path):
        p = tmp_path / "urls.csv"
        p.write_text(f'"tile, with comma","{URL_A}"\n')
        assert read_url_list(str(p)) == [URL_A]

    def test_prefix_match_is_case_insensitive(self, tmp_path):
        p = tmp_path / "urls.csv"
        p.write_text("x,HTTPS://EXAMPLE.COM/T.TIF\n")
        assert read_url_list(str(p)) == ["HTTPS://EXAMPLE.COM/T.TIF"]

    def test_csv_extension_wins_over_content_sniffing(self, tmp_path):
        """A .csv is parsed as CSV before the zip-magic check ever runs."""
        p = tmp_path / "urls.csv"
        p.write_text(f"{URL_A}\n")
        assert read_url_list(str(p)) == [URL_A]


class TestXlsxList:
    def test_reads_shared_string_cells(self, tmp_path):
        p = _make_xlsx(tmp_path / "urls.xlsx", [["Alpha", URL_A], ["Beta", URL_B]])
        assert read_url_list(p) == [URL_A, URL_B]

    def test_reads_inline_string_cells(self, tmp_path):
        p = _make_xlsx(tmp_path / "urls.xlsx", [["Alpha", URL_A]],
                       use_shared_strings=False)
        assert read_url_list(p) == [URL_A]

    def test_non_url_cells_are_ignored(self, tmp_path):
        p = _make_xlsx(tmp_path / "urls.xlsx",
                       [["header", "link"], ["Alpha", URL_A], ["note", "n/a"]])
        assert read_url_list(p) == [URL_A]

    def test_zip_magic_is_sniffed_without_an_excel_extension(self, tmp_path):
        """A workbook saved as .txt must still be read as a workbook."""
        p = _make_xlsx(tmp_path / "urls.txt", [["Alpha", URL_A]])
        assert read_url_list(p) == [URL_A]

    def test_absolute_sheet_target_is_handled(self, tmp_path):
        """Some writers emit Target="xl/worksheets/..." instead of a relative path."""
        p = _make_xlsx(tmp_path / "urls.xlsx", [["Alpha", URL_A]],
                       sheet_target="xl/worksheets/sheet1.xml")
        assert read_url_list(p) == [URL_A]

    def test_workbook_with_no_sheets_gives_an_empty_list(self, tmp_path):
        path = tmp_path / "empty.xlsx"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("xl/workbook.xml",
                        f'<workbook xmlns="{MAIN_NS}"><sheets/></workbook>')
            zf.writestr("xl/_rels/workbook.xml.rels",
                        f'<Relationships xmlns="{PKG_REL_NS}"/>')
        assert read_url_list(str(path)) == []

    def test_a_text_file_named_xls_falls_back_to_text(self, tmp_path):
        """Legacy .xls exports from this project are plain URL lists."""
        p = tmp_path / "urls.xls"
        p.write_text(f"{URL_A}\n{URL_B}\n")
        assert read_url_list(str(p)) == [URL_A, URL_B]

    def test_a_corrupt_xlsx_falls_back_to_text(self, tmp_path, capsys):
        """Named .xlsx but not a zip -- parse must fail soft, not raise."""
        p = tmp_path / "urls.xlsx"
        p.write_text(f"{URL_A}\n")
        assert read_url_list(str(p)) == [URL_A]
        assert "falling back" in capsys.readouterr().err

    @pytest.mark.xfail(strict=True, reason=(
        "KNOWN BUG: a real zip that is not a workbook (a TanDEM-X tile zip passed by "
        "mistake, or a truncated xlsx download) has its KeyError caught and is then "
        "handed to the plain-TEXT fallback, which opens the binary as utf-8 and dies "
        "with UnicodeDecodeError. The fallback exists for legacy .xls files that are "
        "really text, so it should not be reached once the zip magic has matched."))
    def test_a_zip_without_workbook_parts_fails_cleanly(self, tmp_path):
        """Real zip magic, but no ``xl/workbook.xml``.

        Whatever the outcome -- an empty list or a clear error naming the file -- it
        must not be a utf-8 decode error from re-reading the zip as text.
        """
        path = tmp_path / "urls.xlsx"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("something/else.txt", "not a workbook")
        try:
            read_url_list(str(path))
        except UnicodeDecodeError:
            pytest.fail("binary zip content reached the utf-8 text fallback")
        except ValueError:
            pass                      # a clear, deliberate rejection is fine


class TestDeriveBaseName:
    def test_strips_a_query_string(self):
        assert derive_base_name("https://x.com/tile_7.tif?token=abc", 0) == "tile_7"

    def test_trailing_slash_falls_back_to_the_index(self):
        assert derive_base_name("https://x.com/dir/", 4) == "tile_4"

    def test_a_name_with_no_extension_is_kept(self):
        assert derive_base_name("https://x.com/N46E008", 0) == "N46E008"

    def test_only_the_last_extension_is_stripped(self):
        """Swiss tiles carry dots in the name itself, e.g. ...2_2056_5728.tif."""
        assert derive_base_name("https://x.com/swissalti3d_2019_2742-1234_2.tif",
                                0) == "swissalti3d_2019_2742-1234_2"

    def test_an_extension_only_name_is_kept_verbatim(self):
        """``splitext(".tif")`` is ``(".tif", "")``, not ``("", ".tif")``.

        Python treats a leading dot as a dotfile stem rather than an extension, so
        nothing is stripped and the name comes back as ``.tif``. That also makes the
        ``or f"tile_{index}"`` guard on the return unreachable: for any non-empty
        basename splitext yields a non-empty stem, and an empty basename has already
        returned above. Pinned because it looks like the fallback covers this.
        """
        assert derive_base_name("https://x.com/.tif", 9) == ".tif"
