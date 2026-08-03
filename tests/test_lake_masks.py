"""The DEM-thresholded water mask.

It is the one mask provider with no survey data behind it -- the rule is a
percentage of the DEM's own relief -- so what needs pinning is where it puts the
shoreline (midway between a wet and a dry SAMPLE, not on a pixel corner), that it
drops specks too small to seat as an insert, and that it stays silent rather than
inventing a lake when nothing is below the threshold.
"""
import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import shape

from masks import TERRAIN_WATER
from masks.lake import LakeMasks, dem_water_geoms
from model_frame import ModelFrame

ROWS, COLS, PX = 20, 24, 10.0
ORIGIN_X, ORIGIN_Y = 500000.0, 5000000.0
TRANSFORM = from_origin(ORIGIN_X, ORIGIN_Y, PX, PX)


def _dem(low_rows=slice(4, 9), low_cols=slice(6, 12), low=100.0, high=200.0):
    dem = np.full((ROWS, COLS), high)
    dem[low_rows, low_cols] = low
    return dem


def _geom(dem, range_percent=10.0, min_area_m2=1.0):
    geoms = dem_water_geoms(dem, TRANSFORM, range_percent, min_area_m2)
    assert len(geoms) == 1
    return shape(geoms[0])


class TestThreshold:
    def test_the_low_block_is_the_lake(self):
        poly = _geom(_dem())
        assert poly.area == pytest.approx(5 * 6 * PX * PX)

    def test_the_shoreline_lies_between_wet_and_dry_samples(self):
        """A pixel edge is exactly halfway between the two pixel centres it parts,
        which is where a linearly interpolated surface crosses the threshold."""
        minx, miny, maxx, maxy = _geom(_dem()).bounds
        assert minx == pytest.approx(ORIGIN_X + 6 * PX)     # centres 5.5 | 6.5
        assert maxx == pytest.approx(ORIGIN_X + 12 * PX)
        assert maxy == pytest.approx(ORIGIN_Y - 4 * PX)
        assert miny == pytest.approx(ORIGIN_Y - 9 * PX)

    def test_the_percentage_is_of_the_relief_not_of_the_elevation(self):
        dem = _dem(low=100.0, high=200.0)
        dem[0, 0] = 145.0                       # 45% up the relief
        assert len(dem_water_geoms(dem, TRANSFORM, 40.0, 1.0)) == 1, "45% > 40%"
        assert len(dem_water_geoms(dem, TRANSFORM, 50.0, 1.0)) == 2, "45% < 50%"

    def test_nothing_below_the_threshold_yields_no_water(self):
        assert dem_water_geoms(np.full((ROWS, COLS), 100.0), TRANSFORM, 0.0, 1.0) != []
        assert dem_water_geoms(_dem(), TRANSFORM, -1.0, 1.0) == []

    def test_voids_are_not_water(self):
        dem = _dem()
        dem[0, 0] = np.nan
        poly = _geom(dem)
        assert poly.area == pytest.approx(5 * 6 * PX * PX)

    def test_all_void_yields_no_water(self):
        assert dem_water_geoms(np.full((ROWS, COLS), np.nan), TRANSFORM, 10.0, 1.0) == []


class TestDespeckle:
    def test_specks_below_the_insert_size_are_dropped(self):
        dem = _dem()
        dem[15, 20] = 100.0                     # one lone wet sample
        big = PX * PX * 2
        assert len(dem_water_geoms(dem, TRANSFORM, 10.0, 1.0)) == 2
        assert len(dem_water_geoms(dem, TRANSFORM, 10.0, big)) == 1

    def test_an_island_too_small_to_print_is_filled_in(self):
        dem = _dem()
        dem[6, 9] = 200.0                       # a one-sample island in the lake
        assert _geom(dem, min_area_m2=1.0).interiors, "kept when it is printable"
        assert not _geom(dem, min_area_m2=PX * PX * 2).interiors


class TestProvider:
    def test_it_reports_water_at_the_frames_print_scale(self):
        frame = ModelFrame.from_dem((ROWS, COLS), PX, PX, 100.0, TRANSFORM,
                                    "EPSG:32633")
        out = LakeMasks(_dem(), 10.0)(frame)
        assert list(out) == [TERRAIN_WATER]
        assert shape(out[TERRAIN_WATER][0]).area == pytest.approx(5 * 6 * PX * PX)

    def test_the_print_scale_sets_the_speck_size(self):
        """2x2 mm at the print scale: the same lake is a speck on a small print."""
        dem = _dem()
        big_print = ModelFrame.from_dem((ROWS, COLS), PX, PX, 1000.0, TRANSFORM,
                                        "EPSG:32633")
        small_print = ModelFrame.from_dem((ROWS, COLS), PX, PX, 5.0, TRANSFORM,
                                          "EPSG:32633")
        assert LakeMasks(dem, 10.0)(big_print)[TERRAIN_WATER]
        assert LakeMasks(dem, 10.0)(small_print)[TERRAIN_WATER] == []
