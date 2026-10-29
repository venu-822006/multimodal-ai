from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.core.enums import JobStatus
from app.models.analysis import AnalysisJob


class JobRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self, analysis_id: UUID, *, max_attempts: int, available_at: datetime | None = None
    ) -> AnalysisJob:
        now = datetime.now(tz=timezone.utc)
        job = AnalysisJob(
            analysis_id=analysis_id,
            status=JobStatus.PENDING,
            attempts=0,
            max_attempts=max_attempts,
            available_at=available_at or now,
            priority=100,
            created_at=now,
            updated_at=now,
        )
        self._session.add(job)
        self._session.flush()
        return job

    def get_for_analysis(self, analysis_id: UUID) -> AnalysisJob | None:
        return self._session.execute(
            select(AnalysisJob).where(AnalysisJob.analysis_id == analysis_id)
        ).scalar_one_or_none()

    def pending_count(self) -> int:
        now = datetime.now(tz=timezone.utc)
        return int(
            self._session.execute(
                select(func.count())
                .select_from(AnalysisJob)
                .where(AnalysisJob.status == JobStatus.PENDING, AnalysisJob.available_at <= now)
            ).scalar_one()
        )

    def claim_next(self, worker_id: str, *, lease_seconds: int = 900) -> AnalysisJob | None:
        now = datetime.now(tz=timezone.utc)
        candidate = self._session.execute(
            select(AnalysisJob.id)
            .where(
                AnalysisJob.status == JobStatus.PENDING,
                AnalysisJob.available_at <= now,
            )
            .order_by(AnalysisJob.priority.asc(), AnalysisJob.available_at.asc(),
                      AnalysisJob.created_at.asc())
            .with_for_update(skip_locked=True)
            .limit(1)
        ).scalar_one_or_none()
        if candidate is None:
            return None
        self._session.execute(
            update(AnalysisJob)
            .where(AnalysisJob.id == candidate)
            .values(
                status=JobStatus.RUNNING,
                attempts=AnalysisJob.attempts + 1,
                locked_at=now,
                locked_by=worker_id,
                lease_expires_at=now + timedelta(seconds=lease_seconds),
                updated_at=now,
            )
        )
        self._session.flush()
        return self.get(candidate)

    def get(self, job_id: UUID) -> AnalysisJob | None:
        return self._session.get(AnalysisJob, job_id)

    def mark_succeeded(self, job: AnalysisJob) -> None:
        now = datetime.now(tz=timezone.utc)
        job.status = JobStatus.SUCCEEDED
        job.finished_at = now
        job.updated_at = now
        job.locked_by = None
        job.locked_at = None
        job.lease_expires_at = None
        self._session.flush()

    def mark_failed(self, job: AnalysisJob, *, error_code: str, message: str) -> None:
        now = datetime.now(tz=timezone.utc)
        job.status = JobStatus.FAILED
        job.finished_at = now
        job.updated_at = now
        job.locked_by = None
        job.locked_at = None
        job.lease_expires_at = None
        job.last_error_code = error_code
        job.last_error_message = message
        self._session.flush()

    def mark_cancelled(self, job: AnalysisJob, *, message: str) -> None:
        now = datetime.now(tz=timezone.utc)
        job.status = JobStatus.CANCELLED
        job.finished_at = now
        job.updated_at = now
        job.locked_by = None
        job.locked_at = None
        job.lease_expires_at = None
        job.last_error_code = "cancelled"
        job.last_error_message = message
        self._session.flush()

    def reschedule(
        self, job: AnalysisJob, *, delay_seconds: float, error_code: str, message: str
    ) -> None:
        now = datetime.now(tz=timezone.utc)
        job.status = JobStatus.PENDING
        job.available_at = now + timedelta(seconds=delay_seconds)
        job.updated_at = now
        job.locked_by = None
        job.locked_at = None
        job.lease_expires_at = None
        job.last_error_code = error_code
        job.last_error_message = message
        self._session.flush()

    def recover_stale(self, *, stale_after_seconds: int) -> int:
        now = datetime.now(tz=timezone.utc)
        cutoff = now - timedelta(seconds=stale_after_seconds)
        result = self._session.execute(
            update(AnalysisJob)
            .where(
                AnalysisJob.status == JobStatus.RUNNING,
                AnalysisJob.locked_at.is_not(None),
                AnalysisJob.locked_at < cutoff,
                AnalysisJob.attempts < AnalysisJob.max_attempts,
            )
            .values(
                status=JobStatus.PENDING,
                available_at=now,
                updated_at=now,
                locked_at=None,
                locked_by=None,
                lease_expires_at=None,
                last_error_code="worker_lost",
                last_error_message="Previous worker stopped before the job finished.",
            )
            .returning(AnalysisJob.id)
        )
        return len(list(result.scalars()))

    def expire_exhausted(self) -> int:
        now = datetime.now(tz=timezone.utc)
        result = self._session.execute(
            update(AnalysisJob)
            .where(
                AnalysisJob.status == JobStatus.PENDING,
                AnalysisJob.attempts >= AnalysisJob.max_attempts,
            )
            .values(
                status=JobStatus.FAILED,
                finished_at=now,
                updated_at=now,
                last_error_code="retry_limit_exhausted",
                last_error_message="Maximum processing attempts were exhausted.",
            )
            .returning(AnalysisJob.id)
        )
        return len(list(result.scalars()))

    def bump_max_attempts(self, analysis_id: UUID, max_attempts: int) -> None:
        self._session.execute(
            update(AnalysisJob)
            .where(AnalysisJob.analysis_id == analysis_id)
            .values(max_attempts=max_attempts, updated_at=datetime.now(tz=timezone.utc))
        )
        self._session.flush()

    def pending_ids_for_analysis(self, analysis_id: UUID) -> list[UUID]:
        rows = self._session.execute(
            select(AnalysisJob.id).where(
                AnalysisJob.analysis_id == analysis_id,
                AnalysisJob.status == JobStatus.PENDING,
            )
        ).scalars()
        return list(rows)


__all__ = ["JobRepository", "pg_insert"]