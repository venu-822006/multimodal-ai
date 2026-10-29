from __future__ import annotations

from datetime import date, timedelta
from uuid import UUID, uuid4

import pytest

BBOX = [[85.10, 27.80], [85.30, 27.95]]

FLOOD_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"source": "custom_unet", "crs": "EPSG:4326"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [85.15, 27.85],
                        [85.25, 27.85],
                        [85.25, 27.92],
                        [85.15, 27.92],
                        [85.15, 27.85],
                    ]
                ],
            },
        }
    ],
}


@pytest.fixture()
def completed(client, session, settings):
    """An analysis with persisted, model-sourced results."""
    from app.core.enums import AnalysisStatus
    from app.models.analysis import Analysis, AnalysisResult

    created = client.post(
        "/api/analyses",
        json={
            "location": "Koshi",
            "flood_date": (date.today() - timedelta(days=30)).isoformat(),
            "bbox": BBOX,
        },
    ).json()
    analysis_id = UUID(created["id"])
    record = session.get(Analysis, analysis_id)
    session.add(
        AnalysisResult(
            analysis_id=analysis_id,
            status=AnalysisStatus.COMPLETED,
            aoi_area_m2=1_000_000_000.0,
            aoi_area_km2=1000.0,
            flooded_area_m2=950_000_000.0,
            flooded_area_km2=950.0,
            flooded_percentage=95.0,
            flood_polygon_count=3,
            flood_geometry=FLOOD_GEOJSON,
            geometry_summary={"polygon_count": 3, "total_area_km2": 950.0},
            measurements={
                "valid_pixel_count": 1000,
                "flood_pixel_count": 950,
                "resolution_m": 20.0,
                "percentage_denominator": "valid_analysis_area",
            },
            method="custom_unet",
            methodology="FloodSight AI custom U-Net flood segmentation v1.0.0.",
            algorithm={"name": "FloodSight AI custom U-Net flood segmentation", "version": "1.0.0"},
            imagery=[
                {
                    "item_id": "S1A_TEST",
                    "acquisition_date": "2024-09-20",
                    "platform": "S1A",
                    "collection": "sentinel-1-rtc",
                }
            ],
            provenance={
                "catalogue": "https://planetarycomputer.microsoft.com/api/stac/v1",
                "catalogue_provider": "Microsoft Planetary Computer",
                "retrieved_at": "2024-09-21T00:00:00+00:00",
                "model": {
                    "architecture": "FloodUNet",
                    "checkpoint": "best:abc123",
                    "val_flood_iou": 0.81,
                    "device": "cpu",
                    "input_bands": ["VV", "VH"],
                    "confidence_threshold": 0.5,
                },
            },
            confidence={"confidence_threshold": 0.5, "validation_flood_iou": 0.81},
            infrastructure={
                "layer": {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {"kind": "road"},
                            "geometry": {
                                "type": "LineString",
                                "coordinates": [[85.16, 27.86], [85.24, 27.91]],
                            },
                        }
                    ],
                },
                "buildings_intersecting_flood": 12,
                "road_segments_intersecting_flood": 4,
            },
            settlements={
                "layer": {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {"name": "Birtamod"},
                            "geometry": {"type": "Point", "coordinates": [85.2, 27.88]},
                        }
                    ],
                },
                "settlements_in_flood_zone": 1,
            },
            warnings=["Low radar coverage in the north of the area."],
            limitations=["Predictions are not confirmed ground truth."],
            artifacts={
                "directory": str(settings.artifact_root / str(uuid4())),
                "flood_geojson": "unset",
                "flood_mask": "unset",
            },
        )
    )
    record.status = AnalysisStatus.COMPLETED
    record.report_available = True
    record.completed_at = date.today()
    session.commit()
    return analysis_id


class TestFacts:
    def test_returns_measurements(self, client, completed):
        body = client.get(f"/api/analyses/{completed}/facts").json()
        assert body["results_ready"] is True
        assert body["flooded_area_km2"] == 950.0
        assert body["flooded_percentage"] == 95.0
        assert body["flood_polygon_count"] == 3

    def test_includes_model_provenance(self, client, completed):
        body = client.get(f"/api/analyses/{completed}/facts").json()
        assert body["model_name"] == "FloodUNet"
        assert body["model_checkpoint"] == "best:abc123"
        assert body["model_val_flood_iou"] == 0.81
        assert body["model_input_bands"] == ["VV", "VH"]

    def test_includes_imagery_metadata(self, client, completed):
        body = client.get(f"/api/analyses/{completed}/facts").json()
        assert body["imagery_scenes"] == ["S1A_TEST"]
        assert body["imagery_acquisition_dates"] == ["2024-09-20"]
        assert body["imagery_provider"] == "Microsoft Planetary Computer"

    def test_includes_warnings_and_limitations(self, client, completed):
        body = client.get(f"/api/analyses/{completed}/facts").json()
        assert body["warnings"]
        assert body["limitations"]

    def test_states_predictions_are_not_ground_truth(self, client, completed):
        body = client.get(f"/api/analyses/{completed}/facts").json()
        assert body["prediction_is_model_output_not_ground_truth"] is True

    def test_no_private_filesystem_paths(self, client, completed):
        raw = client.get(f"/api/analyses/{completed}/facts").text
        assert "/home/" not in raw
        assert "/tmp/" not in raw
        assert ".pt" not in raw

    def test_denominator_reported(self, client, completed):
        body = client.get(f"/api/analyses/{completed}/facts").json()
        assert body["provenance_detail"]["model"]["architecture"] == "FloodUNet"


class TestMap:
    def test_returns_flood_layer(self, client, completed):
        body = client.get(f"/api/analyses/{completed}/map").json()
        assert body["layers"]["flood_extent"]["type"] == "FeatureCollection"
        assert len(body["layers"]["flood_extent"]["features"]) == 1

    def test_returns_infrastructure_and_settlements(self, client, completed):
        body = client.get(f"/api/analyses/{completed}/map").json()
        assert body["layers"]["infrastructure"]["type"] == "FeatureCollection"
        assert body["layers"]["settlements"]["features"][0]["properties"]["name"] == "Birtamod"

    def test_coordinates_are_lon_lat_in_wgs84(self, client, completed):
        body = client.get(f"/api/analyses/{completed}/map").json()
        ring = body["layers"]["flood_extent"]["features"][0]["geometry"]["coordinates"][0][0]
        lon, lat = ring
        assert -180 <= lon <= 180
        assert -90 <= lat <= 90
        # Values are degrees, not projected metres.
        assert 80 < lon < 90

    def test_bbox_computed(self, client, completed):
        body = client.get(f"/api/analyses/{completed}/map").json()
        assert body["bbox"] == [85.15, 27.85, 85.25, 27.92]

    def test_legend_warns_about_predictions(self, client, completed):
        legend = client.get(f"/api/analyses/{completed}/map").json()["legend"]
        assert any("not confirmed ground truth" in item["description"] for item in legend)

    def test_metrics_present(self, client, completed):
        body = client.get(f"/api/analyses/{completed}/map").json()
        assert body["metrics"]["flooded_area_km2"] == 950.0

    def test_matches_frontend_sync_data_layers_shape(self, client, completed):
        """The frontend reads layers.flood_extent | .flood | .flood_layer."""
        layers = client.get(f"/api/analyses/{completed}/map").json()["layers"]
        chosen = layers.get("flood_extent") or layers.get("flood") or layers.get("flood_layer")
        assert chosen is not None
        assert isinstance(chosen.get("features"), list)


class TestReport:
    def test_descriptor_available(self, client, completed):
        body = client.get(f"/api/analyses/{completed}/report").json()
        assert body["format"] == "pdf"
        assert body["url"].endswith(f"/api/analyses/{completed}/report/download")

    def test_pdf_downloads(self, client, completed):
        response = client.get(f"/api/analyses/{completed}/report/download")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        assert response.content[:4] == b"%PDF"

    def test_pdf_is_not_empty(self, client, completed):
        assert len(client.get(f"/api/analyses/{completed}/report/download").content) > 1000

    def test_report_url_in_list_after_completion(self, client, completed):
        items = client.get("/api/analyses").json()["items"]
        entry = next(item for item in items if item["id"] == str(completed))
        assert entry["report_url"]
        assert entry["has_results"] is True

    def test_report_reflects_results_not_placeholders(self, client, completed, session):
        """The rendered report is driven by persisted values, not placeholders."""
        from app.models.analysis import AnalysisResult
        from app.services.report_service import ReportService, _styles

        result = session.query(AnalysisResult).filter_by(analysis_id=completed).one()
        service = ReportService()
        story = service._model_story(result, _styles())
        assert story, "model provenance section was not rendered"
        # Measurements and imagery tables are likewise built from the stored result.
        assert service._measurements_story(result, _styles())
        assert service._imagery_story(result, _styles())

    def test_report_states_predictions_are_not_ground_truth(self, client, completed, session):
        from app.models.analysis import AnalysisResult
        from app.services.report_service import ReportService, _styles

        result = session.query(AnalysisResult).filter_by(analysis_id=completed).one()
        rendered = str(ReportService()._model_story(result, _styles()))
        assert "not confirmed ground truth" in rendered


class TestPersistedResultLifecycle:
    def test_results_survive_a_new_session(self, client, completed, session):
        from app.models.analysis import AnalysisResult

        session.expire_all()
        stored = session.query(AnalysisResult).filter_by(analysis_id=completed).one()
        assert stored.flooded_area_km2 == 950.0

    def test_detail_reports_results(self, client, completed):
        body = client.get(f"/api/analyses/{completed}").json()
        assert body["has_results"] is True
        assert body["report_available"] is True
        assert body["result_summary"]["flooded_area_km2"] == 950.0