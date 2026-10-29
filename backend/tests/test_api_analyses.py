from __future__ import annotations

from datetime import date, timedelta
from uuid import uuid4

import pytest

BBOX = [[85.10, 27.80], [85.30, 27.95]]


def _payload(**overrides):
    payload = {
        "location": "Koshi River, Nepal",
        "flood_date": (date.today() - timedelta(days=30)).isoformat(),
        "bbox": BBOX,
    }
    payload.update(overrides)
    return payload


class TestHealth:
    def test_live_is_ok(self, client):
        response = client.get("/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_ready_reports_database(self, client):
        response = client.get("/health/ready")
        assert response.status_code == 200
        assert response.json()["checks"]["database"]["status"] == "ok"

    def test_model_endpoint_does_not_require_a_gpu(self, client):
        response = client.get("/health/model")
        assert response.status_code == 200
        assert response.json()["checks"]["model"]["status"] in {"ok", "unavailable"}

    def test_capabilities_reports_acceptance(self, client):
        response = client.get("/health/ready/capabilities")
        assert response.status_code == 200
        assert "accepts_new_analyses" in response.json()["checks"]

    def test_openapi_documents_the_contract(self, client):
        schema = client.get("/openapi.json").json()
        assert "/api/analyses" in schema["paths"]
        assert "/api/analyses/{analysis_id}/map" in schema["paths"]
        post = schema["paths"]["/api/analyses"]["post"]
        assert "202" in post["responses"]


class TestCreateAnalysis:
    def test_returns_202_with_id_and_job(self, client):
        response = client.post("/api/analyses", json=_payload())
        assert response.status_code == 202, response.text
        body = response.json()
        assert body["id"]
        assert body["status"] == "queued"
        assert body["job_id"]
        assert body["links"]["status"].endswith("/status")

    def test_accepts_analysis_id_alias_shape(self, client):
        response = client.post("/api/analyses", json=_payload())
        body = response.json()
        assert isinstance(body["id"], str)

    def test_persists_a_job(self, client, session):
        from app.models.analysis import AnalysisJob

        response = client.post("/api/analyses", json=_payload())
        job = session.query(AnalysisJob).one()
        assert str(job.analysis_id) == response.json()["id"]
        assert job.status == "pending"
        assert job.max_attempts >= 1

    def test_location_is_optional(self, client):
        assert client.post("/api/analyses", json=_payload(location=None)).status_code == 202


class TestValidation:
    @pytest.mark.parametrize(
        "bbox,reason",
        [
            ([[86.0, 27.8], [85.0, 28.0]], "reversed"),
            ([[85.0, 27.8], [85.3, 27.4]], "reversed"),
            ([[200.0, 27.8], [200.5, 28.0]], "out of range"),
            ([[85.0, 100.0], [85.3, 100.5]], "out of range"),
            ([[85.0, 27.8]], "too few corners"),
            ([[85.0, 27.8, 1.0], [85.3, 28.0, 1.0]], "wrong shape"),
        ],
    )
    def test_invalid_bbox_rejected(self, client, bbox, reason):
        response = client.post("/api/analyses", json=_payload(bbox=bbox))
        assert response.status_code == 422, f"{reason}: {response.status_code}"

    def test_non_finite_coordinates_rejected(self, client):
        response = client.post(
            "/api/analyses", json=_payload(bbox=[["NaN", 27.8], [85.3, 28.0]])
        )
        assert response.status_code == 422

    def test_future_flood_date_rejected(self, client):
        response = client.post(
            "/api/analyses", json=_payload(flood_date=(date.today() + timedelta(days=5)).isoformat())
        )
        assert response.status_code == 422

    def test_oversized_area_rejected(self, client):
        response = client.post(
            "/api/analyses",
            json=_payload(bbox=[[80.0, 10.0], [84.0, 13.0]]),
        )
        assert response.status_code == 422

    def test_unknown_field_rejected(self, client):
        response = client.post("/api/analyses", json=_payload(unexpected="x"))
        assert response.status_code == 422

    def test_error_body_is_sanitised(self, client):
        body = client.post("/api/analyses", json=_payload(bbox=[[86, 28], [85, 27]])).json()
        assert body["error"]["code"]
        assert "Traceback" not in body["error"]["message"]


class TestListing:
    def test_empty_list(self, client):
        body = client.get("/api/analyses").json()
        assert body["items"] == []
        assert body["total_items"] == 0

    def test_pagination_and_deterministic_order(self, client):
        for index in range(3):
            client.post("/api/analyses", json=_payload())
        body = client.get("/api/analyses?page=1&page_size=2").json()
        assert len(body["items"]) == 2
        assert body["total_items"] == 3
        assert body["has_next"] is True
        assert body["has_previous"] is False
        second = client.get("/api/analyses?page=2&page_size=2").json()
        ids_first = {item["id"] for item in body["items"]}
        ids_second = {item["id"] for item in second["items"]}
        assert not (ids_first & ids_second)

    def test_status_filter(self, client):
        client.post("/api/analyses", json=_payload())
        body = client.get("/api/analyses?status=completed").json()
        assert body["total_items"] == 0
        body = client.get("/api/analyses?status=queued").json()
        assert body["total_items"] == 1

    def test_multiple_status_filter(self, client):
        client.post("/api/analyses", json=_payload())
        body = client.get("/api/analyses?status=queued,failed").json()
        assert body["total_items"] == 1

    def test_unknown_status_rejected(self, client):
        assert client.get("/api/analyses?status=bogus").status_code == 422

    def test_page_size_bound_enforced(self, client):
        assert client.get("/api/analyses?page_size=100000").status_code == 422

    def test_list_items_expose_frontend_fields(self, client):
        client.post("/api/analyses", json=_payload())
        item = client.get("/api/analyses").json()["items"][0]
        for field in ("id", "name", "status", "flood_date", "location", "report_url"):
            assert field in item
        assert item["report_url"] is None

    def test_collection_response_has_no_large_artifacts(self, client):
        client.post("/api/analyses", json=_payload())
        raw = client.get("/api/analyses").text
        assert "FeatureCollection" not in raw


class TestDetailAndStatus:
    def test_detail(self, client):
        created = client.post("/api/analyses", json=_payload()).json()
        response = client.get(f"/api/analyses/{created['id']}")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "queued"
        assert body["bbox"]["southWest"]["lon"] == 85.10
        assert body["links"]["map"].endswith("/map")

    def test_status_reports_persisted_lifecycle(self, client):
        created = client.post("/api/analyses", json=_payload()).json()
        body = client.get(f"/api/analyses/{created['id']}/status").json()
        assert body["status"] == "queued"
        assert body["is_terminal"] is False
        assert body["attempt"] == 0
        assert body["progress_percent"] == 0
        assert body["recommended_poll_seconds"] > 0

    def test_unknown_id_returns_404(self, client):
        missing = uuid4()
        for suffix in ("", "/status", "/facts", "/map", "/report"):
            assert client.get(f"/api/analyses/{missing}{suffix}").status_code == 404

    def test_malformed_id_returns_422(self, client):
        assert client.get("/api/analyses/not-a-uuid").status_code == 422

    def test_status_survives_restart(self, client, session):
        from app.core.enums import AnalysisStatus
        from app.models.analysis import Analysis

        created = client.post("/api/analyses", json=_payload()).json()
        record = session.get(Analysis, uuid4().__class__(created["id"]))
        record.status = AnalysisStatus.PREPROCESSING
        session.commit()
        body = client.get(f"/api/analyses/{created['id']}/status").json()
        assert body["status"] == "preprocessing"
        assert body["progress_percent"] > 0


class TestUnfinishedResults:
    @pytest.fixture()
    def queued(self, client):
        return client.post("/api/analyses", json=_payload()).json()["id"]

    def test_facts_not_ready(self, client, queued):
        response = client.get(f"/api/analyses/{queued}/facts")
        assert response.status_code == 200
        body = response.json()
        assert body["results_ready"] is False
        assert body["message"]

    def test_map_not_ready(self, client, queued):
        body = client.get(f"/api/analyses/{queued}/map").json()
        assert body["message"]
        assert "layers" not in body or body["layers"] == {}

    def test_report_not_ready(self, client, queued):
        body = client.get(f"/api/analyses/{queued}/report").json()
        assert body["message"]

    def test_report_download_not_ready(self, client, queued):
        response = client.get(f"/api/analyses/{queued}/report/download")
        assert response.status_code == 202

    def test_no_empty_map_concealing_failure(self, client, queued):
        body = client.get(f"/api/analyses/{queued}/map").json()
        assert body["status"] == "queued"
        assert "features" not in body


class TestFailedAnalysis:
    @pytest.fixture()
    def failed(self, client, session):
        from app.core.enums import AnalysisStatus
        from app.models.analysis import Analysis
        from uuid import UUID

        created = client.post("/api/analyses", json=_payload()).json()
        record = session.get(Analysis, UUID(created["id"]))
        record.status = AnalysisStatus.FAILED
        record.failure_code = "no_imagery"
        record.failure_message = "No Sentinel-1 observations overlap the requested area."
        session.commit()
        return created["id"]

    def test_facts_report_failure_not_zeroes(self, client, failed):
        body = client.get(f"/api/analyses/{failed}/facts").json()
        assert body["results_ready"] is False
        assert "fail" in body["message"].lower()

    def test_report_refuses_failed_analysis(self, client, failed):
        assert client.get(f"/api/analyses/{failed}/report/download").status_code == 202

    def test_status_is_terminal(self, client, failed):
        body = client.get(f"/api/analyses/{failed}/status").json()
        assert body["is_terminal"] is True
        assert body["failure_code"] == "no_imagery"


class TestSecurity:
    def test_artifact_traversal_blocked(self, client):
        created = client.post("/api/analyses", json=_payload()).json()
        for attempt in ("../../../../etc/passwd", "..%2f..%2fetc%2fpasswd"):
            response = client.get(f"/api/analyses/{created['id']}/artifacts/{attempt}")
            assert response.status_code in {202, 404}, attempt

    def test_unknown_artifact_key(self, client):
        created = client.post("/api/analyses", json=_payload()).json()
        assert (
            client.get(f"/api/analyses/{created['id']}/artifacts/nonexistent").status_code
            in {202, 404}
        )

    def test_error_responses_hide_internals(self, client):
        response = client.get(f"/api/analyses/{uuid4()}")
        assert "sqlalchemy" not in response.text.lower()
        assert "/home/" not in response.text

    def test_request_id_header_present(self, client):
        assert client.get("/health/live").headers.get("X-Request-ID")