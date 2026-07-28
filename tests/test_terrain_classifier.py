"""Tests for the masks.osm provider module."""

from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from rasterio.transform import from_bounds

from masks import (
    TERRAIN_ROCK,
    TERRAIN_GLACIER,
    TERRAIN_WATER,
    TERRAIN_FOLIAGE,
)
from masks.osm import (
    _classify_element,
    _build_overpass_query,
    _way_to_geojson,
    _relation_to_geojson,
    _join_ring_segments,
    classify_terrain,
)


class TestClassifyElement:
    def test_glacier(self):
        assert _classify_element({"natural": "glacier"}) == TERRAIN_GLACIER

    def test_glacier_rock_type_is_rock(self):
        assert _classify_element({"natural": "glacier", "glacier:type": "rock"}) == TERRAIN_ROCK

    def test_glacier_valley_type(self):
        assert _classify_element({"natural": "glacier", "glacier:type": "valley"}) == TERRAIN_GLACIER

    def test_water(self):
        assert _classify_element({"natural": "water"}) == TERRAIN_WATER
        assert _classify_element({"natural": "water", "water": "lake"}) == TERRAIN_WATER

    def test_water_landuse(self):
        assert _classify_element({"landuse": "reservoir"}) == TERRAIN_WATER
        assert _classify_element({"landuse": "basin"}) == TERRAIN_WATER

    def test_foliage_natural(self):
        for val in ["wood", "scrub", "heath", "grassland", "fell", "tundra", "moor", "wetland"]:
            assert _classify_element({"natural": val}) == TERRAIN_FOLIAGE, f"Failed for natural={val}"

    def test_foliage_landuse(self):
        for val in ["forest", "meadow", "grass", "farmland", "orchard", "vineyard"]:
            assert _classify_element({"landuse": val}) == TERRAIN_FOLIAGE, f"Failed for landuse={val}"

    def test_rock_default(self):
        assert _classify_element({}) == TERRAIN_ROCK
        assert _classify_element({"natural": "bare_rock"}) == TERRAIN_ROCK
        assert _classify_element({"landuse": "residential"}) == TERRAIN_ROCK

    def test_glacier_priority_over_water(self):
        # If something is tagged as both glacier and water, glacier wins
        assert _classify_element({"natural": "glacier", "landuse": "reservoir"}) == TERRAIN_GLACIER

    def test_water_priority_over_foliage(self):
        # Water tag takes priority (natural=water checked before foliage)
        assert _classify_element({"natural": "water", "landuse": "forest"}) == TERRAIN_WATER


class TestBuildOverpassQuery:
    def test_valid_query_structure(self):
        query = _build_overpass_query((46.0, 7.0, 47.0, 8.0))
        assert "[out:json]" in query
        assert "[timeout:120]" in query
        assert "out geom;" in query
        assert "46.0,7.0,47.0,8.0" in query

    def test_contains_all_tag_filters(self):
        query = _build_overpass_query((46.0, 7.0, 47.0, 8.0))
        # Check key tags are present
        assert '"natural"="glacier"' in query
        assert '"natural"="water"' in query
        assert '"landuse"="forest"' in query
        assert '"natural"="wetland"' in query
        assert '"landuse"="vineyard"' in query

    def test_queries_ways_and_relations(self):
        query = _build_overpass_query((46.0, 7.0, 47.0, 8.0))
        assert "way[" in query
        assert "relation[" in query


class TestWayToGeojson:
    def test_simple_polygon(self):
        element = {
            "geometry": [
                {"lon": 7.0, "lat": 46.0},
                {"lon": 8.0, "lat": 46.0},
                {"lon": 8.0, "lat": 47.0},
                {"lon": 7.0, "lat": 47.0},
                {"lon": 7.0, "lat": 46.0},
            ]
        }
        geom = _way_to_geojson(element)
        assert geom is not None
        assert geom["type"] == "Polygon"
        assert len(geom["coordinates"]) == 1  # one ring
        assert len(geom["coordinates"][0]) == 5  # 4 points + closing

    def test_auto_closes_ring(self):
        element = {
            "geometry": [
                {"lon": 7.0, "lat": 46.0},
                {"lon": 8.0, "lat": 46.0},
                {"lon": 8.0, "lat": 47.0},
                {"lon": 7.0, "lat": 47.0},
            ]
        }
        geom = _way_to_geojson(element)
        assert geom is not None
        coords = geom["coordinates"][0]
        assert coords[0] == coords[-1]

    def test_too_few_points_returns_none(self):
        element = {"geometry": [{"lon": 7.0, "lat": 46.0}, {"lon": 8.0, "lat": 46.0}]}
        assert _way_to_geojson(element) is None

    def test_empty_geometry_returns_none(self):
        assert _way_to_geojson({"geometry": []}) is None
        assert _way_to_geojson({}) is None


class TestJoinRingSegments:
    def test_single_closed_ring(self):
        seg = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
        rings = _join_ring_segments([seg])
        assert len(rings) == 1
        assert rings[0][0] == rings[0][-1]

    def test_join_two_segments(self):
        s1 = [[0, 0], [1, 0], [1, 1]]
        s2 = [[1, 1], [0, 1], [0, 0]]
        rings = _join_ring_segments([s1, s2])
        assert len(rings) == 1
        assert rings[0][0] == rings[0][-1]

    def test_empty_input(self):
        assert _join_ring_segments([]) == []


class TestRelationToGeojson:
    def test_simple_relation(self):
        element = {
            "members": [
                {
                    "role": "outer",
                    "geometry": [
                        {"lon": 7.0, "lat": 46.0},
                        {"lon": 8.0, "lat": 46.0},
                        {"lon": 8.0, "lat": 47.0},
                        {"lon": 7.0, "lat": 47.0},
                        {"lon": 7.0, "lat": 46.0},
                    ],
                }
            ]
        }
        geom = _relation_to_geojson(element)
        assert geom is not None
        assert geom["type"] == "Polygon"

    def test_no_members_returns_none(self):
        assert _relation_to_geojson({"members": []}) is None
        assert _relation_to_geojson({}) is None


class TestClassifyTerrain:
    def _make_overpass_response(self):
        """Create a mock Overpass response with a water polygon and a forest polygon."""
        return {
            "elements": [
                {
                    "type": "way",
                    "tags": {"natural": "water", "water": "lake"},
                    "geometry": [
                        {"lon": 7.40, "lat": 46.40},
                        {"lon": 7.60, "lat": 46.40},
                        {"lon": 7.60, "lat": 46.60},
                        {"lon": 7.40, "lat": 46.60},
                        {"lon": 7.40, "lat": 46.40},
                    ],
                },
                {
                    "type": "way",
                    "tags": {"landuse": "forest"},
                    "geometry": [
                        {"lon": 7.00, "lat": 46.00},
                        {"lon": 7.30, "lat": 46.00},
                        {"lon": 7.30, "lat": 46.30},
                        {"lon": 7.00, "lat": 46.30},
                        {"lon": 7.00, "lat": 46.00},
                    ],
                },
            ]
        }

    @patch("masks.osm.requests.post")
    def test_classify_terrain_basic(self, mock_post):
        """Test that classification produces a valid array with correct shape."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = self._make_overpass_response()
        mock_post.return_value = mock_resp

        # Small DEM covering roughly 46-47°N, 7-8°E in Swiss CRS (EPSG:2056)
        dem_shape = (10, 10)
        # Approximate transform: 1km pixels, origin near (2580000, 1180000) in LV95
        transform = from_bounds(2580000, 1180000, 2590000, 1190000, 10, 10)

        class_geoms = classify_terrain(dem_shape, transform, "EPSG:2056")

        assert isinstance(class_geoms, dict)
        assert TERRAIN_GLACIER in class_geoms
        assert TERRAIN_WATER in class_geoms
        assert TERRAIN_FOLIAGE in class_geoms

    @patch("masks.osm.time.sleep")  # skip retry delays
    @patch("masks.osm.requests.post")
    def test_classify_terrain_api_failure_returns_all_rock(self, mock_post, mock_sleep):
        """On API failure, should return all-rock classification."""
        mock_post.side_effect = Exception("Network error")

        dem_shape = (5, 5)
        transform = from_bounds(2580000, 1180000, 2585000, 1185000, 5, 5)

        class_geoms = classify_terrain(dem_shape, transform, "EPSG:2056")

        assert all(len(v) == 0 for v in class_geoms.values())

    @patch("masks.osm.requests.post")
    def test_classify_terrain_empty_response(self, mock_post):
        """Empty Overpass response should return all rock."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"elements": []}
        mock_post.return_value = mock_resp

        dem_shape = (5, 5)
        transform = from_bounds(2580000, 1180000, 2585000, 1185000, 5, 5)

        class_geoms = classify_terrain(dem_shape, transform, "EPSG:2056")

        assert all(len(v) == 0 for v in class_geoms.values())
