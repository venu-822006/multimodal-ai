"""Tests for OpenStreetMap exposure handling.

The central guarantee is that an unavailable external lookup is never reported as a
count of zero. Successful, empty and failed responses must be distinguishable.
"""

from __future__ import annotations

import httpx
import pytest
from shapely.geometry import Polygon, box

from app.core.config import Settings
from app.processing.validators import AreaOfInterest
from app.services.damage_assessment_service import (
    OverpassClient,
    OverpassUnavailableError,
    assess_exposure,
)

AOI = AreaOfInterest(
    min_lon=85.10,
    min_lat=27.80,
    max_lon=85.40,
    max_lat=28.00,
    area_m2=1_000_000_000.0,
)


def _settings(tmp_path, **overrides) -> Settings:
    base = {
        "environment": "test",
        "database_url": "postgresql+psycopg://unused:unused@localhost/unused",
        "artifact_root": tmp_path / "artifacts",
        "model_repo_path": tmp_path,
        "model_path": None,
        "osm_enabled": True,
        "osm_endpoint": "https://overpass.test/api/interpreter",
        "osm_fallback_endpoints": [],
        "osm_timeout_seconds": 5.0,
        "osm_max_elements": 6000,
        "log_format": "text",
        "log_level": "ERROR",
    }
    base.update(overrides)
    return Settings(**base)


def _building_way(lon: float, lat: float, size: float = 0.002):
    return {
        "type": "way",
        "id": 1,
        "tags": {"building": "yes"},
        "geometry": [
            {"lon": lon, "lat": lat},
            {"lon": lon + size, "lat": lat},
            {"lon": lon + size, "lat": lat + size},
            {"lon": lon, "lat": lat + size},
            {"lon": lon, "lat": lat},
        ],
    }


def _road_way():
    return {
        "type": "way",
        "id": 2,
        "tags": {"highway": "primary"},
        "geometry": [
            {"lon": 85.15, "lat": 27.85},
            {"lon": 85.35, "lat": 27.95},
        ],
    }


def _place_node(lon: float, lat: float):
    return {
        "type": "node",
        "id": 3,
        "tags": {"place": "town", "name": "Testville"},
        "geometry": {"lon": lon, "lat": lat},
    }


class FakeTransport:
    """Minimal httpx transport that answers by URL and returns canned payloads."""

    def __init__(self, responses: dict[str, object]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.calls.append(url)
        response = self.responses.get(url)
        if isinstance(response, Exception):
            raise response
        if response is None:
            return httpx.Response(503, text="no route")
        status, payload = response
        if isinstance(payload, str):
            return httpx.Response(status, text=payload)
        return httpx.Response(status, json=payload)


def _patch_client(monkeypatch, responses: dict[str, object]) -> FakeTransport:
    """Route OverpassClient's httpx calls through a fake transport."""
    transport = FakeTransport(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        return transport.handle_request(request)

    original_init = httpx.Client.__init__

    def patched_init(self, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        original_init(self, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", patched_init)
    return transport


def _classify(payload: dict) -> dict:
    """Return the payload the Overpass parser expects for a category."""
    return payload


class TestSuccessfulResponse:
    def test_counts_intersections(self, tmp_path, monkeypatch):
        url = "https://overpass.test/api/interpreter"
        flood = box(85.20, 27.88, 85.30, 27.96)
        monkeypatch.setattr(
            "app.services.damage_assessment_service.OverpassClient.query",
            lambda self, ql: _category_payload(ql, flood),
        )
        result = assess_exposure(AOI, flood, settings=_settings(tmp_path))

        assert result.available is True
        assert result.buildings_intersecting == 1
        assert result.buildings_total == 1
        assert result.road_segments_intersecting == 1
        assert result.settlements_in_flood_zone == 1
        assert result.summary["status"] == "ok"

    def test_populated_areas_are_counted(self, tmp_path, monkeypatch):
        flood = box(85.20, 27.88, 85.30, 27.96)
        monkeypatch.setattr(
            "app.services.damage_assessment_service.OverpassClient.query",
            lambda self, ql: _category_payload(ql, flood),
        )
        result = assess_exposure(AOI, flood, settings=_settings(tmp_path))
        assert result.settlements_total == 1
        assert result.settlements_near_flood is not None

    def test_layers_are_returned(self, tmp_path, monkeypatch):
        flood = box(85.20, 27.88, 85.30, 27.96)
        monkeypatch.setattr(
            "app.services.damage_assessment_service.OverpassClient.query",
            lambda self, ql: _category_payload(ql, flood),
        )
        result = assess_exposure(AOI, flood, settings=_settings(tmp_path))
        assert "infrastructure" in result.layers
        assert "settlements" in result.layers
        assert result.layers["settlements"]["type"] == "FeatureCollection"


def _category_payload(ql: str, flood) -> dict:
    """Return elements appropriate to whichever category query was issued."""
    if 'way["building"]' in ql:
        return {"elements": [_building_way(85.21, 27.89)]}
    if '"highway"' in ql:
        return {"elements": [_road_way()]}
    if 'node["place"' in ql:
        return {"elements": [_place_node(85.25, 27.92)]}
    return {"elements": []}


class TestEmptyResponse:
    def test_empty_result_is_a_real_zero_not_unavailable(self, tmp_path, monkeypatch):
        """A successful query with no features is a genuine zero."""
        monkeypatch.setattr(
            "app.services.damage_assessment_service.OverpassClient.query",
            lambda self, ql: {"elements": []},
        )
        result = assess_exposure(AOI, None, settings=_settings(tmp_path))

        assert result.available is True
        assert result.summary["status"] == "ok"
        assert result.buildings_intersecting == 0
        assert result.road_segments_intersecting == 0
        assert result.settlements_total == 0
        assert result.settlements_in_flood_zone == 0
        assert result.roads_length_km_flooded == 0.0
        # A real zero must not be reported as unavailable.
        assert result.summary["unavailable_categories"] == []


class TestFailedResponse:
    def test_total_failure_reports_none_not_zero(self, tmp_path, monkeypatch):
        def boom(self, ql):
            raise OverpassUnavailableError("all endpoints failed")

        monkeypatch.setattr(
            "app.services.damage_assessment_service.OverpassClient.query", boom
        )
        result = assess_exposure(AOI, None, settings=_settings(tmp_path))

        assert result.available is False
        assert result.summary["status"] == "unavailable"
        assert result.buildings_intersecting is None
        assert result.road_segments_intersecting is None
        assert result.settlements_in_flood_zone is None
        assert result.settlements_total is None
        assert any("did not complete" in w for w in result.warnings)

    def test_one_category_failing_leaves_others_intact(self, tmp_path, monkeypatch):
        """Exposure must degrade per category, not fail wholesale."""

        def selective(self, ql):
            if 'node["place"' in ql:
                raise OverpassUnavailableError("places unavailable")
            return _category_payload(ql, None)

        monkeypatch.setattr(
            "app.services.damage_assessment_service.OverpassClient.query", selective
        )
        result = assess_exposure(AOI, None, settings=_settings(tmp_path))

        assert result.available is True
        assert result.summary["status"] == "partial"
        assert result.buildings_total == 1
        assert result.settlements_total is None
        assert result.settlements_in_flood_zone is None
        assert "settlements" in result.summary["unavailable_categories"]
        assert any("for settlements failed" in w for w in result.warnings)

    def test_truncated_response_is_unavailable_not_zero(self, tmp_path, monkeypatch):
        limit = 3

        def truncating(self, ql):
            if '"highway"' in ql:
                return {"elements": [_road_way()] * limit}
            return _category_payload(ql, None)

        settings = _settings(tmp_path, osm_max_elements=limit)
        monkeypatch.setattr(
            "app.services.damage_assessment_service.OverpassClient.query", truncating
        )
        result = assess_exposure(AOI, None, settings=settings)

        assert result.available is True
        assert result.road_segments_intersecting is None
        assert result.summary["unavailable_categories"] == ["roads"]
        assert any("incomplete" in w for w in result.warnings)

    def test_disabled_configuration_reports_none(self, tmp_path, monkeypatch):
        def never(self, ql):
            raise AssertionError("must not query when disabled")

        monkeypatch.setattr(
            "app.services.damage_assessment_service.OverpassClient.query", never
        )
        result = assess_exposure(AOI, None, settings=_settings(tmp_path, osm_enabled=False))
        assert result.available is False
        assert result.summary["status"] == "disabled"
        assert result.buildings_intersecting is None
        assert result.settlements_in_flood_zone is None


class TestClientFallback:
    def test_falls_back_to_the_next_endpoint(self, tmp_path, monkeypatch):
        good = "https://fallback.test/api/interpreter"
        transport = _patch_client(
            monkeypatch,
            {
                good: (200, {"elements": [_place_node(85.25, 27.92)]}),
            },
        )
        settings = _settings(
            tmp_path,
            osm_endpoint="https://primary.test/api/interpreter",
            osm_fallback_endpoints=[good],
        )
        payload = OverpassClient(settings).query("node;out;")
        assert payload["elements"][0]["id"] == 3
        assert len(transport.calls) == 2, "primary failure must be followed by a fallback"

    def test_raises_when_every_endpoint_fails(self, tmp_path, monkeypatch):
        _patch_client(monkeypatch, {})
        settings = _settings(
            tmp_path,
            osm_endpoint="https://primary.test/api/interpreter",
            osm_fallback_endpoints=["https://fallback.test/api/interpreter"],
        )
        with pytest.raises(OverpassUnavailableError):
            OverpassClient(settings).query("node;out;")

    def test_rejects_a_non_overpass_response(self, tmp_path, monkeypatch):
        """A 200 that is not an Overpass result must not be treated as data."""
        url = "https://overpass.test/api/interpreter"
        _patch_client(monkeypatch, {url: (200, "<html>maintenance</html>")})
        settings = _settings(tmp_path)
        with pytest.raises(OverpassUnavailableError):
            OverpassClient(settings).query("node;out;")


class TestQueryConstruction:
    def test_categories_are_queried_separately(self, tmp_path, monkeypatch):
        seen: list[str] = []

        def record(self, ql):
            seen.append(ql)
            return {"elements": []}

        monkeypatch.setattr(
            "app.services.damage_assessment_service.OverpassClient.query", record
        )
        assess_exposure(AOI, None, settings=_settings(tmp_path))

        assert len(seen) == 3
        assert any('way["building"]' in q for q in seen)
        assert any('"highway"' in q for q in seen)
        assert any('node["place"' in q for q in seen)
        # Each query targets exactly one category, so a dense category cannot
        # exhaust the element limit and starve the others.
        assert not any(
            'way["building"]' in q and '"highway"' in q and 'node["place"' in q for q in seen
        ), "categories must not be combined into one query"

    def test_bbox_clause_is_south_west_first(self, tmp_path):
        from app.services.damage_assessment_service import _bbox_clause

        assert _bbox_clause(AOI) == "27.8,85.1,28.0,85.4"

    def test_geometry_outside_the_extent_is_not_counted(self, tmp_path, monkeypatch):
        # Contains the building at (85.21, 27.89) but not the place at (85.25, 27.92).
        flood = Polygon([(85.20, 27.88), (85.22, 27.88), (85.22, 27.90), (85.20, 27.90)])
        monkeypatch.setattr(
            "app.services.damage_assessment_service.OverpassClient.query",
            lambda self, ql: _category_payload(ql, flood),
        )
        result = assess_exposure(AOI, flood, settings=_settings(tmp_path))
        assert result.buildings_intersecting == 1
        assert result.settlements_total == 1
        assert result.settlements_in_flood_zone == 0
        assert result.settlements_near_flood == 0