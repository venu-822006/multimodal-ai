from __future__ import annotations

from datetime import date, timedelta
from uuid import UUID

import pytest

from app.core.enums import AnalysisStatus, JobStatus
from app.models.analysis import Analysis, AnalysisJob
from app.repositories.jobs import JobRepository
from app.schemas.analysis import AnalysisCreateRequest
from app.services.analysis_service import AnalysisService


def _create(session, settings, **overrides):
    payload = {
        "flood_date": (date.today() - timedelta(days=30)).isoformat(),
        "bbox": [[85.10, 27.80], [85.30, 27.95]],
    }
    payload.update(overrides)
    analysis, _ = AnalysisService(session, settings).create(AnalysisCreateRequest(**payload))
    session.commit()
    return analysis


@pytest.fixture()
def worker(settings, db_schema):
    from app.db.session import get_session_factory
    from app.workers.analysis_worker import AnalysisWorker

    get_session_factory.cache_clear() if hasattr(get_session_factory, "cache_clear") else None
    worker = AnalysisWorker(settings, worker_id="test-worker")
    yield worker


class TestJobCreation:
    def test_job_created_with_analysis(self, session, settings):
        analysis = _create(session, settings)
        job = JobRepository(session).get_for_analysis(analysis.id)
        assert job is not None
        assert job.status == JobStatus.PENDING
        assert job.attempts == 0
        assert job.max_attempts == settings.worker_max_attempts

    def test_job_is_transactional_with_analysis(self, session, settings):
        analysis = _create(session, settings)
        assert session.query(AnalysisJob).filter_by(analysis_id=analysis.id).count() == 1


class TestClaiming:
    def test_claims_pending_job(self, session, settings):
        analysis = _create(session, settings)
        job = JobRepository(session).claim_next("worker-a")
        session.commit()
        assert job is not None
        assert job.status == JobStatus.RUNNING
        assert job.locked_by == "worker-a"
        assert job.attempts == 1
        assert job.lease_expires_at is not None

    def test_second_worker_cannot_claim(self, session, settings):
        _create(session, settings)
        first = JobRepository(session).claim_next("worker-a")
        session.commit()
        second = JobRepository(session).claim_next("worker-b")
        session.commit()
        assert first is not None
        assert second is None

    def test_no_jobs_returns_none(self, session, settings):
        assert JobRepository(session).claim_next("worker-a") is None

    def test_future_available_at_not_claimed(self, session, settings):
        from datetime import datetime, timedelta, timezone

        analysis = _create(session, settings)
        job = JobRepository(session).get_for_analysis(analysis.id)
        job.available_at = datetime.now(tz=timezone.utc) + timedelta(hours=1)
        session.commit()
        assert JobRepository(session).claim_next("worker-a") is None

    def test_claim_is_orderly(self, session, settings):
        from datetime import datetime, timedelta, timezone

        first = _create(session, settings)
        second = _create(session, settings)
        now = datetime.now(tz=timezone.utc)
        JobRepository(session).get_for_analysis(first.id).available_at = now + timedelta(seconds=5)
        JobRepository(session).get_for_analysis(second.id).available_at = now
        session.commit()
        claimed = JobRepository(session).claim_next("worker-a")
        session.commit()
        assert claimed.analysis_id == second.id


class TestRetryAndFailure:
    def test_reschedule_back_to_pending(self, session, settings):
        analysis = _create(session, settings)
        job = JobRepository(session).claim_next("worker-a")
        session.commit()
        JobRepository(session).reschedule(job, delay_seconds=10, error_code="x", message="boom")
        session.commit()
        assert job.status == JobStatus.PENDING
        assert job.last_error_code == "x"
        assert job.locked_by is None

    def test_permanent_failure_marks_failed(self, session, settings):
        analysis = _create(session, settings)
        job = JobRepository(session).claim_next("worker-a")
        session.commit()
        JobRepository(session).mark_failed(job, error_code="no_imagery", message="none found")
        session.commit()
        assert job.status == JobStatus.FAILED
        assert job.finished_at is not None

    def test_exhausted_attempts_expire(self, session, settings):
        analysis = _create(session, settings)
        job = JobRepository(session).get_for_analysis(analysis.id)
        job.attempts = job.max_attempts
        session.commit()
        expired = JobRepository(session).expire_exhausted()
        session.commit()
        assert expired == 1
        assert job.status == JobStatus.FAILED

    def test_bounded_retries_no_infinite_loop(self, session, settings):
        analysis = _create(session, settings)
        job = JobRepository(session).get_for_analysis(analysis.id)
        session.commit()
        for _ in range(10):
            if job.status != JobStatus.PENDING:
                break
            if job.attempts >= job.max_attempts:
                break
            JobRepository(session).claim_next("worker-a")
            session.commit()
            JobRepository(session).reschedule(job, delay_seconds=0, error_code="x", message="y")
            session.commit()
        assert job.attempts <= job.max_attempts

    def test_succeeded_not_reclaimable(self, session, settings):
        _create(session, settings)
        job = JobRepository(session).claim_next("worker-a")
        session.commit()
        JobRepository(session).mark_succeeded(job)
        session.commit()
        assert JobRepository(session).claim_next("worker-b") is None


class TestRecovery:
    def test_stale_running_job_recovered(self, session, settings):
        from datetime import datetime, timedelta, timezone

        analysis = _create(session, settings)
        job = JobRepository(session).get_for_analysis(analysis.id)
        job.status = JobStatus.RUNNING
        job.attempts = 1
        job.locked_at = datetime.now(tz=timezone.utc) - timedelta(hours=2)
        job.locked_by = "dead-worker"
        session.commit()
        recovered = JobRepository(session).recover_stale(stale_after_seconds=60)
        session.commit()
        assert recovered == 1
        assert job.status == JobStatus.PENDING
        assert job.locked_by is None
        assert job.last_error_code == "worker_lost"

    def test_fresh_running_job_not_recovered(self, session, settings):
        from datetime import datetime, timezone

        analysis = _create(session, settings)
        job = JobRepository(session).get_for_analysis(analysis.id)
        job.status = JobStatus.RUNNING
        job.locked_at = datetime.now(tz=timezone.utc)
        session.commit()
        assert JobRepository(session).recover_stale(stale_after_seconds=3600) == 0

    def test_exhausted_stale_job_not_requeued(self, session, settings):
        from datetime import datetime, timedelta, timezone

        analysis = _create(session, settings)
        job = JobRepository(session).get_for_analysis(analysis.id)
        job.status = JobStatus.RUNNING
        job.attempts = job.max_attempts
        job.locked_at = datetime.now(tz=timezone.utc) - timedelta(hours=2)
        session.commit()
        assert JobRepository(session).recover_stale(stale_after_seconds=60) == 0

    def test_worker_recovers_on_startup(self, session, settings, worker):
        from datetime import datetime, timedelta, timezone

        analysis = _create(session, settings)
        job = JobRepository(session).get_for_analysis(analysis.id)
        job.status = JobStatus.RUNNING
        job.locked_at = datetime.now(tz=timezone.utc) - timedelta(hours=2)
        session.commit()
        worker._recover_interrupted_jobs()
        session.expire_all()
        assert JobRepository(session).get_for_analysis(analysis.id).status == JobStatus.PENDING


class TestWorkerRunOnce:
    def test_no_job_returns_false(self, session, settings, worker):
        assert worker.run_once() is False

    def test_failure_persists_terminal_state(self, session, settings, worker, monkeypatch):
        analysis = _create(session, settings)
        from app.processing.pipeline import AnalysisPipeline
        from app.core.exceptions import PermanentProcessingError

        def explode(self, transition):
            transition(AnalysisStatus.VALIDATING)
            raise PermanentProcessingError(
                "No imagery found.", failure_code="no_imagery"
            )

        monkeypatch.setattr(AnalysisPipeline, "run", explode)
        worker.run_once()
        session.expire_all()
        record = session.get(Analysis, analysis.id)
        assert record.status == AnalysisStatus.FAILED
        assert record.failure_code == "no_imagery"
        job = JobRepository(session).get_for_analysis(analysis.id)
        assert job.status == JobStatus.FAILED
        assert record.completed_at is not None

    def test_retryable_failure_reschedules(self, session, settings, worker, monkeypatch):
        analysis = _create(session, settings)
        from app.processing.pipeline import AnalysisPipeline
        from app.core.exceptions import RetryableProcessingError

        def explode(self, transition):
            transition(AnalysisStatus.VALIDATING)
            raise RetryableProcessingError("Catalogue unreachable.", failure_code="imagery_unavailable")

        monkeypatch.setattr(AnalysisPipeline, "run", explode)
        worker.run_once()
        session.expire_all()
        job = JobRepository(session).get_for_analysis(analysis.id)
        assert job.status == JobStatus.PENDING
        assert job.attempts == 1
        assert job.last_error_code == "imagery_unavailable"
        assert session.get(Analysis, analysis.id).status != AnalysisStatus.COMPLETED

    def test_incomplete_job_not_marked_complete_after_crash(
        self, session, settings, worker, monkeypatch
    ):
        analysis = _create(session, settings)
        from app.processing.pipeline import AnalysisPipeline
        from app.core.exceptions import RetryableProcessingError

        def explode(self, transition):
            transition(AnalysisStatus.VALIDATING)
            raise RetryableProcessingError("Process died mid-run.", failure_code="processing_error")

        monkeypatch.setattr(AnalysisPipeline, "run", explode)
        worker.run_once()
        session.expire_all()
        record = session.get(Analysis, analysis.id)
        assert record.status != AnalysisStatus.COMPLETED
        assert record.result is None

    def test_stage_transitions_are_recorded(self, session, settings, worker, monkeypatch):
        analysis = _create(session, settings)
        from app.processing.pipeline import AnalysisPipeline
        from app.core.exceptions import PermanentProcessingError

        def explode(self, transition):
            transition(AnalysisStatus.VALIDATING)
            transition(AnalysisStatus.SEARCHING_DATA)
            raise PermanentProcessingError("stop", failure_code="processing_error")

        monkeypatch.setattr(AnalysisPipeline, "run", explode)
        worker.run_once()
        session.expire_all()
        from app.models.analysis import AnalysisExecution

        stages = session.query(AnalysisExecution).filter_by(analysis_id=analysis.id).all()
        assert {item.stage for item in stages} >= {
            AnalysisStatus.VALIDATING,
            AnalysisStatus.SEARCHING_DATA,
        }
        assert all(item.status in {"succeeded", "failed", "running"} for item in stages)

    def test_second_run_is_idempotent_on_completed(self, session, settings, worker, monkeypatch):
        analysis = _create(session, settings)
        from app.processing.pipeline import AnalysisPipeline

        calls = {"n": 0}

        def succeed(self, transition):
            calls["n"] += 1
            for status in (
                AnalysisStatus.VALIDATING,
                AnalysisStatus.SEARCHING_DATA,
                AnalysisStatus.DOWNLOADING,
                AnalysisStatus.PREPROCESSING,
                AnalysisStatus.FLOOD_MAPPING,
                AnalysisStatus.DAMAGE_ANALYSIS,
                AnalysisStatus.CONNECTIVITY_ANALYSIS,
                AnalysisStatus.GENERATING_REPORT,
                AnalysisStatus.COMPLETED,
            ):
                transition(status)

        monkeypatch.setattr(AnalysisPipeline, "run", succeed)
        worker.run_once()
        session.expire_all()
        assert session.get(Analysis, analysis.id).status == AnalysisStatus.COMPLETED
        assert worker.run_once() is False
        assert calls["n"] == 1

    def test_duplicate_execution_prevented(self, session, settings, worker, monkeypatch):
        _create(session, settings)
        from app.processing.pipeline import AnalysisPipeline

        runs = {"n": 0}

        def count(self, transition):
            runs["n"] += 1
            transition(AnalysisStatus.VALIDATING)

        monkeypatch.setattr(AnalysisPipeline, "run", count)
        worker.run_once()
        session.expire_all()
        # The single job stays claimed, so a second pass finds nothing.
        assert worker.run_once() is False
        assert runs["n"] == 1

    def test_cancelled_analysis_skipped(self, session, settings, worker, monkeypatch):
        analysis = _create(session, settings)
        from app.processing.pipeline import AnalysisPipeline

        record = session.get(Analysis, analysis.id)
        record.status = AnalysisStatus.CANCELLED
        session.commit()
        ran = {"n": 0}

        def never(self, transition):
            ran["n"] += 1

        monkeypatch.setattr(AnalysisPipeline, "run", never)
        worker.run_once()
        assert ran["n"] == 0
        session.expire_all()
        assert JobRepository(session).get_for_analysis(analysis.id).status == JobStatus.CANCELLED


class TestStatusTransitions:
    def test_illegal_transition_rejected(self, session, settings):
        from app.core.exceptions import ConflictError

        analysis = _create(session, settings)
        service = AnalysisService(session, settings)
        record = session.get(Analysis, analysis.id)
        record.status = AnalysisStatus.COMPLETED
        session.commit()
        with pytest.raises(ConflictError):
            service.transition(record, AnalysisStatus.PREPROCESSING)

    def test_legal_transition_allowed(self, session, settings):
        analysis = _create(session, settings)
        service = AnalysisService(session, settings)
        service.transition(session.get(Analysis, analysis.id), AnalysisStatus.VALIDATING)
        session.commit()
        assert session.get(Analysis, analysis.id).status == AnalysisStatus.VALIDATING

    def test_started_at_recorded_once(self, session, settings):
        analysis = _create(session, settings)
        service = AnalysisService(session, settings)
        record = session.get(Analysis, analysis.id)
        service.transition(record, AnalysisStatus.VALIDATING)
        session.commit()
        first = record.started_at
        service.transition(record, AnalysisStatus.SEARCHING_DATA)
        session.commit()
        assert record.started_at == first