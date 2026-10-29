"""End-to-end pipeline tests using the synthetic test provider.

These run the real pipeline (imagery discovery, preprocessing, polygonisation,
measurement, exposure and reporting) with no trained checkpoint and no network.
They assert the two properties that matter most:

* synthetic output is labelled as synthetic everywhere it surfaces, and
* filtering small polygons never rewrites the raw prediction mask.
"""

from __future__ import annotations

import dataclasses
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from app.core.enums import AnalysisStatus
from app.schemas.analysis import AnalysisCreateRequest
from app.services.analysis_service import AnalysisService

#: The flood date must sit inside the supported archive window, so the fixture scene
#: is dated relative to today rather than hard coded.
FLOOD_DATE = date.today() - timedelta(days=20)


def _synthetic_settings(settings, **overrides):
    """Clone the session settings with the synthetic provider selected."""
    from app.core.config import Settings

    base = settings.model_dump()
    base.update({"prediction_provider": "synthetic_test"})
    base.update(overrides)
    return Settings(**base)


@pytest.fixture()
def synthetic(settings, db_schema):
    return _synthetic_settings(settings)


@pytest.fixture()
def local_imagery(monkeypatch, local_scene):
    """Serve the local GeoTIFF scene in place of a STAC catalogue search.

    The scene is re-dated onto the flood date so the pipeline sees a pre-event and a
    post-event observation inside the requested window.
    """
    from app.core.enums import ImagerySource
    from app.services.imagery_service import PlanetaryComputerStacClient

    acquired = datetime(
        FLOOD_DATE.year, FLOOD_DATE.month, FLOOD_DATE.day, tzinfo=timezone.utc
    ) + timedelta(days=3)
    scene = dataclasses.replace(local_scene, acquisition=acquired)
    pre = dataclasses.replace(
        scene,
        item_id="TEST_S1_ITEM_PRE",
        acquisition=acquired - timedelta(days=12),
    )

    def fake_search(self, source, aoi, window):
        if source == ImagerySource.SENTINEL1_RTC:
            return [pre, scene]
        return []

    monkeypatch.setattr(PlanetaryComputerStacClient, "search", fake_search)
    return [pre, scene]


def _read_mask(analysis, key="flood_mask"):
    """Read a written mask raster, returning its array and metadata."""
    import rasterio

    filename = "flood_mask.tif" if key == "flood_mask" else "flood_extent_mask.tif"
    path = Path(analysis.artifact_directory) / filename
    with rasterio.open(path) as dataset:
        return {
            "array": dataset.read(1),
            "crs": dataset.crs,
            "transform": dataset.transform,
            "nodata": dataset.nodata,
            "tags": dataset.tags(),
            "bounds": dataset.bounds,
            "width": dataset.width,
            "height": dataset.height,
        }


def _pdf_text(pdf: bytes) -> bytes:
    """Best-effort extraction of drawn text from a reportlab PDF."""
    import base64
    import re
    import zlib

    out = b""
    for match in re.finditer(rb"stream\r?\n(.*?)endstream", pdf, re.S):
        raw = match.group(1).strip(b"\r\n")
        try:
            out += zlib.decompress(raw)
            continue
        except zlib.error:
            pass
        try:
            out += zlib.decompress(base64.a85decode(raw, adobe=True))
        except Exception:  # noqa: BLE001
            continue
    return out


def _run_pipeline(session, settings, **request_overrides):
    """Create an analysis and drive the pipeline to a terminal state."""
    from app.processing.pipeline import AnalysisPipeline

    payload = {
        "flood_date": FLOOD_DATE,
        "bbox": [[85.10, 27.80], [85.30, 27.95]],
        "sources": ["sentinel-1-rtc"],
    }
    payload.update(request_overrides)
    analysis, _ = AnalysisService(session, settings).create(AnalysisCreateRequest(**payload))
    session.commit()

    seen: list[str] = []

    def transition(status):
        seen.append(str(status))
        analysis.status = AnalysisStatus(status)

    outcome = AnalysisPipeline(session, analysis, settings=settings).run(transition)
    session.commit()
    return analysis, outcome, seen


class TestSyntheticPipelineRunsWithoutWeights:
    def test_completes_and_records_the_provider(self, session, synthetic, local_imagery):
        analysis, outcome, seen = _run_pipeline(session, synthetic)

        assert analysis.status == AnalysisStatus.COMPLETED.value
        assert outcome.result is not None
        assert outcome.result.method == "synthetic_test_data"
        assert seen[-1] == "completed"
        assert "flood_mapping" in seen

    def test_records_provider_in_the_request_payload(self, session, synthetic, local_imagery):
        analysis, _, _ = _run_pipeline(session, synthetic)
        assert analysis.request_payload["prediction_provider"] == "synthetic_test"

    def test_no_checkpoint_is_required(self, synthetic, local_imagery):
        from app.services.prediction_providers import get_prediction_provider

        provider = get_prediction_provider(synthetic)
        descriptor = provider.describe()
        assert descriptor.available is True
        assert descriptor.synthetic is True


class TestSyntheticLabelling:
    def test_measurements_declare_synthetic(self, session, synthetic, local_imagery):
        _, outcome, _ = _run_pipeline(session, synthetic)
        measurements = outcome.result.measurements
        assert measurements["predictions_are_synthetic"] is True
        assert measurements["prediction_data_label"] == "synthetic_test_data"
        assert measurements["prediction_provider"] == "synthetic_test"

    def test_methodology_never_claims_a_model(self, session, synthetic, local_imagery):
        _, outcome, _ = _run_pipeline(session, synthetic)
        methodology = outcome.result.methodology
        assert "synthetic_test_data" in methodology
        assert "No trained model weights were loaded" in methodology
        assert "FloodUNet" not in methodology

    def test_warnings_disclose_the_substitution(self, session, synthetic, local_imagery):
        _, outcome, _ = _run_pipeline(session, synthetic)
        assert any("synthetic test provider" in w.lower() for w in outcome.result.warnings)
        assert any("synthetic_test_data" in limitation for limitation in outcome.result.limitations)

    def test_confidence_expresses_no_probability(self, session, synthetic, local_imagery):
        _, outcome, _ = _run_pipeline(session, synthetic)
        confidence = outcome.result.confidence
        assert confidence["confidence_threshold"] is None
        assert confidence["validation_flood_iou"] is None

    def test_facts_flag_synthetic_metrics(self, client, session, synthetic, local_imagery):
        analysis, _, _ = _run_pipeline(session, synthetic)
        from app.services.result_service import build_facts
        from app.repositories.analyses import AnalysisResultRepository

        session.refresh(analysis)
        result = AnalysisResultRepository(session).get(analysis.id)
        facts = build_facts(analysis, result)
        assert facts["predictions_are_synthetic"] is True
        assert facts["metrics_are_synthetic"] is True
        assert facts["prediction_data_label"] == "synthetic_test_data"
        # Synthetic numbers are not model output, so the ground-truth flag is cleared.
        assert facts["prediction_is_model_output_not_ground_truth"] is False

    def test_map_payload_flags_synthetic(self, session, synthetic, local_imagery):
        from app.services.result_service import build_map_payload
        from app.repositories.analyses import AnalysisResultRepository

        analysis, _, _ = _run_pipeline(session, synthetic)
        session.refresh(analysis)
        result = AnalysisResultRepository(session).get(analysis.id)
        payload = build_map_payload(analysis, result)
        assert payload["predictions_are_synthetic"] is True
        assert payload["legend"][0]["label"] == "Synthetic test extent"
        assert "not a flood prediction" in payload["legend"][0]["description"].lower()

    def test_report_carries_a_banner(self, session, synthetic, local_imagery):
        from app.services.report_service import ReportService

        analysis, outcome, _ = _run_pipeline(session, synthetic)
        pdf = ReportService().render_pdf(analysis, outcome.result)
        assert pdf.startswith(b"%PDF")
        assert len(pdf) > 1000

    def test_report_banner_text_is_rendered(self, session, synthetic, local_imagery):
        """The banner is drawn content, so assert it survives into the PDF stream."""
        from app.services.report_service import ReportService

        analysis, outcome, _ = _run_pipeline(session, synthetic)
        pdf = ReportService().render_pdf(analysis, outcome.result)
        assert b"SYNTHETIC" in _pdf_text(pdf)


class TestGeospatialIntegrity:
    def test_raw_mask_keeps_flood_pixels(self, session, synthetic, local_imagery):
        analysis, outcome, _ = _run_pipeline(session, synthetic)
        raster = _read_mask(analysis)
        assert (raster["array"] == 1).sum() == outcome.result.measurements["flood_pixel_count"]
        assert raster["nodata"] == 255

    def test_mask_carries_crs_and_transform(self, session, synthetic, local_imagery):
        from pyproj import CRS

        analysis, _, _ = _run_pipeline(session, synthetic)
        raster = _read_mask(analysis)
        assert raster["crs"] == CRS.from_epsg(32645)
        assert raster["transform"].a == pytest.approx(20.0)
        assert raster["transform"].e == pytest.approx(-20.0)

    def test_mask_is_tagged_synthetic(self, session, synthetic, local_imagery):
        analysis, _, _ = _run_pipeline(session, synthetic)
        tags = _read_mask(analysis)["tags"]
        assert tags["predictions_are_synthetic"] == "true"
        assert tags["prediction_data_label"] == "synthetic_test_data"
        assert tags["method"] == "synthetic_test_data"

    def test_geotiff_round_trips_to_the_requested_bbox(self, session, synthetic, local_imagery):
        from rasterio.warp import transform_bounds

        analysis, _, _ = _run_pipeline(session, synthetic)
        raster = _read_mask(analysis)
        bounds = transform_bounds(
            raster["crs"], "EPSG:4326", *raster["bounds"], densify_pts=21
        )
        west, south, east, north = bounds
        assert west == pytest.approx(85.10, abs=0.05)
        assert east == pytest.approx(85.30, abs=0.05)
        assert south == pytest.approx(27.80, abs=0.05)
        assert north == pytest.approx(27.95, abs=0.05)

    def test_geojson_is_well_formed(self, session, synthetic, local_imagery):
        import json

        analysis, _, _ = _run_pipeline(session, synthetic)
        payload = json.loads(
            (Path(analysis.artifact_directory) / "flood_extent.geojson").read_text()
        )
        assert payload["type"] == "FeatureCollection"
        for feature in payload["features"]:
            assert feature["properties"]["predictions_are_synthetic"] is True
            assert feature["properties"]["prediction_data_label"] == "synthetic_test_data"


class TestPolygonFilteringDoesNotTouchTheRawMask:
    def test_filtering_everything_leaves_the_raw_mask_intact(
        self, session, synthetic, local_imagery
    ):
        """A threshold that discards all polygons must not erase raw model pixels."""
        analysis, outcome, _ = _run_pipeline(
            session, synthetic, min_flood_area_m2=50_000_000.0
        )
        measurements = outcome.result.measurements

        assert measurements["flood_pixel_count"] > 0
        assert measurements["retained_polygons"] == 0
        # The reported area follows the filtered geometry...
        assert outcome.result.flooded_area_km2 is None
        assert outcome.result.flood_polygon_count == 0
        # ...but the raw raster still records the prediction.
        raw = _read_mask(analysis)["array"]
        assert (raw == 1).sum() == measurements["flood_pixel_count"]

    def test_extent_mask_reflects_filtering_not_the_raw_mask(self, session, synthetic, local_imagery):
        analysis, outcome, _ = _run_pipeline(
            session, synthetic, min_flood_area_m2=50_000_000.0
        )
        raw = _read_mask(analysis, "flood_mask")["array"]
        extent = _read_mask(analysis, "flood_extent_mask")["array"]
        assert (raw == 1).sum() > 0
        assert (extent == 1).sum() == 0

    def test_per_request_minimum_area_is_applied(self, session, synthetic, local_imagery):
        """The stored min_flood_area_m2 must actually drive polygonisation."""
        _, strict, _ = _run_pipeline(session, synthetic, min_flood_area_m2=50_000_000.0)
        _, loose, _ = _run_pipeline(session, synthetic, min_flood_area_m2=1.0)
        assert strict.result.measurements["retained_polygons"] == 0
        assert loose.result.measurements["retained_polygons"] > 0
        assert loose.result.measurements["min_flood_polygon_area_m2"] == 1.0


class TestMeasurementConsistency:
    def test_counts_and_areas_are_coherent(self, session, synthetic, local_imagery):
        _, outcome, _ = _run_pipeline(session, synthetic, min_flood_area_m2=1.0)
        result = outcome.result
        measurements = result.measurements

        assert measurements["flood_pixel_count"] >= measurements["retained_flood_pixel_count"]
        assert (
            measurements["filtered_out_pixel_count"]
            == measurements["flood_pixel_count"] - measurements["retained_flood_pixel_count"]
        )
        assert result.flooded_area_m2 == pytest.approx(measurements["flooded_area_m2"], rel=1e-6)
        assert result.flooded_area_km2 == pytest.approx(
            measurements["flooded_area_km2"], rel=1e-6
        )

    def test_percentage_uses_the_valid_area(self, session, synthetic, local_imagery):
        _, outcome, _ = _run_pipeline(session, synthetic, min_flood_area_m2=1.0)
        measurements = outcome.result.measurements
        expected = (
            measurements["flooded_area_m2"]
            / (measurements["valid_pixel_count"] * measurements["resolution_m"] ** 2)
            * 100.0
        )
        assert outcome.result.flooded_percentage == pytest.approx(expected, rel=1e-3)

    def test_summary_facts_and_map_agree(self, session, synthetic, local_imagery):
        from app.repositories.analyses import AnalysisResultRepository
        from app.services.result_service import build_facts, build_map_payload

        analysis, outcome, _ = _run_pipeline(session, synthetic, min_flood_area_m2=1.0)
        session.refresh(analysis)
        result = AnalysisResultRepository(session).get(analysis.id)
        facts = build_facts(analysis, result)
        payload = build_map_payload(analysis, result)

        assert facts["flooded_area_km2"] == outcome.result.flooded_area_km2
        assert facts["flooded_percentage"] == outcome.result.flooded_percentage
        assert facts["flood_polygon_count"] == outcome.result.flood_polygon_count
        assert payload["metrics"]["flooded_area_km2"] == outcome.result.flooded_area_km2
        assert payload["metrics"]["flood_pixel_count"] == facts["flood_pixel_count"]

    def test_report_contains_the_persisted_figures(self, session, synthetic, local_imagery):
        from app.services.report_service import ReportService

        analysis, outcome, _ = _run_pipeline(session, synthetic, min_flood_area_m2=1.0)
        pdf = ReportService().render_pdf(analysis, outcome.result)
        assert b"synthetic" in _pdf_text(pdf)


class TestMissingCheckpointStillFails:
    def test_trained_provider_analysis_fails_without_a_checkpoint(
        self, session, settings, local_imagery
    ):
        """With the trained provider selected and no weights, the analysis must fail."""
        from app.core.exceptions import PermanentProcessingError

        with pytest.raises(PermanentProcessingError) as excinfo:
            _run_pipeline(session, settings)
        assert "MODEL_PATH" in str(excinfo.value)

    def test_facts_for_a_missing_checkpoint_analysis_refuse_to_invent_numbers(
        self, session, settings, local_imagery
    ):
        from app.services.result_service import build_facts

        analysis, _ = AnalysisService(session, settings).create(
            AnalysisCreateRequest(
                flood_date=FLOOD_DATE,
                bbox=[[85.10, 27.80], [85.30, 27.95]],
                sources=["sentinel-1-rtc"],
            )
        )
        session.commit()
        analysis.status = AnalysisStatus.FAILED
        analysis.failure_message = "No trained checkpoint is configured."
        session.commit()

        facts = build_facts(analysis, None)
        assert facts["results_ready"] is False
        assert facts["flooded_area_km2"] is None
        assert facts["flooded_percentage"] is None
        assert facts.get("flood_pixel_count") is None
        assert facts.get("predictions_are_synthetic") is not True
        assert "unavailable" in facts["unavailable_measurements"][0]


class TestOsmIndependence:
    def test_analysis_completes_when_exposure_is_disabled(self, session, synthetic, local_imagery):
        _, outcome, _ = _run_pipeline(session, synthetic, min_flood_area_m2=1.0)
        # osm_enabled is False in the test settings.
        assert outcome.result.status == AnalysisStatus.COMPLETED.value
        assert not outcome.result.infrastructure
        assert outcome.result.settlements is None

    def test_unavailable_exposure_is_not_reported_as_zero(self, session, synthetic, local_imagery):
        from app.repositories.analyses import AnalysisResultRepository
        from app.services.result_service import build_facts

        analysis, _, _ = _run_pipeline(session, synthetic, min_flood_area_m2=1.0)
        session.refresh(analysis)
        result = AnalysisResultRepository(session).get(analysis.id)
        facts = build_facts(analysis, result)
        assert facts["buildings_intersecting_flood"] is None
        assert facts["road_segments_intersecting_flood"] is None
        assert facts["settlements_in_flood_zone"] is None
        assert any("OpenStreetMap" in w for w in facts["warnings"])

    def test_exposure_failure_does_not_fail_the_analysis(
        self, session, synthetic, local_imagery, monkeypatch
    ):
        from app.services.damage_assessment_service import OverpassUnavailableError

        def boom(*args, **kwargs):
            raise OverpassUnavailableError("no endpoint responded")

        monkeypatch.setattr(
            "app.services.damage_assessment_service.OverpassClient.query", boom
        )
        osm_settings = _synthetic_settings(synthetic, osm_enabled=True)
        _, outcome, _ = _run_pipeline(session, osm_settings, min_flood_area_m2=1.0)
        assert outcome.result.status == AnalysisStatus.COMPLETED.value
        assert outcome.result.flooded_area_km2 is not None