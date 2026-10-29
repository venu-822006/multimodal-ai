from __future__ import annotations

import argparse
import os
import signal
import socket
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.enums import AnalysisStatus, ExecutionStatus, FailureCode, JobStatus
from app.core.exceptions import (
    AnalysisCancelledError,
    FloodSightError,
    RetryableProcessingError,
    sanitise_public_message,
)
from app.core.logging import analysis_id_var, configure_logging, get_logger, job_id_var
from app.db.session import get_session_factory
from app.models.analysis import Analysis, AnalysisJob, AnalysisExecution
from app.processing.pipeline import AnalysisPipeline
from app.repositories.analyses import AnalysisRepository, ExecutionRepository
from app.repositories.jobs import JobRepository
from app.services.analysis_service import AnalysisService

logger = get_logger("app.worker")


class AnalysisWorker:
    """Single-process PostgreSQL backed analysis worker.

    Jobs are claimed atomically with ``FOR UPDATE SKIP LOCKED`` so several worker
    processes can run safely against the same database.
    """

    def __init__(self, settings: Settings | None = None, worker_id: str | None = None) -> None:
        self._settings = settings or get_settings()
        self._worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}"
        self._stop_event = threading.Event()
        self._session_factory = get_session_factory(self._settings)

    def request_stop(self, signum: int | None = None, _frame: Any = None) -> None:
        logger.info(
            "worker_stop_requested",
            extra={
                "event": "worker_stop_requested",
                "signal": signum,
                "worker_id": self._worker_id,
            },
        )
        self._stop_event.set()

    def run_forever(self) -> int:
        logger.info(
            "worker_started",
            extra={
                "event": "worker_started",
                "worker_id": self._worker_id,
                "max_attempts": self._settings.worker_max_attempts,
                "poll_interval_seconds": self._settings.worker_poll_interval_seconds,
            },
        )
        self._recover_interrupted_jobs()
        last_recovery = time.monotonic()
        while not self._stop_event.is_set():
            processed = self.run_once()
            if not processed:
                self._stop_event.wait(self._settings.worker_poll_interval_seconds)
            if time.monotonic() - last_recovery >= self._settings.worker_stale_after_seconds:
                self._recover_interrupted_jobs()
                last_recovery = time.monotonic()
        logger.info(
            "worker_stopped", extra={"event": "worker_stopped", "worker_id": self._worker_id}
        )
        return 0

    def run_once(self) -> bool:
        with self._session_factory() as session:
            jobs = JobRepository(session)
            job = jobs.claim_next(
                self._worker_id, lease_seconds=self._settings.worker_stale_after_seconds
            )
            if job is None:
                session.rollback()
                return False
            session.commit()
            job_id = job.id
            job_snapshot = (job.analysis_id, job.attempts)
            logger.info(
                "job_claimed",
                extra={
                    "event": "job_claimed",
                    "job_id": str(job_id),
                    "analysis_id": str(job_snapshot[0]),
                    "attempt": job_snapshot[1],
                    "worker_id": self._worker_id,
                },
            )
        self._process_job(job_id)
        return True

    def _process_job(self, job_id: UUID) -> None:
        started = time.perf_counter()
        analysis_id: UUID | None = None
        with self._session_factory() as session:
            jobs = JobRepository(session)
            job = jobs.get(job_id)
            if job is None or job.status != JobStatus.RUNNING:
                session.rollback()
                return
            analysis_id = job.analysis_id
            job_id_var.set(str(job.id))
            analysis_id_var.set(str(analysis_id))
            analysis = session.get(Analysis, analysis_id)
            if analysis is None:
                jobs.mark_failed(
                    job,
                    error_code="analysis_missing",
                    message="The analysis record referenced by this job no longer exists.",
                )
                session.commit()
                return
            if analysis.status == AnalysisStatus.CANCELLED:
                jobs.mark_cancelled(job, message="Analysis was cancelled.")
                session.commit()
                return
            if analysis.status == AnalysisStatus.COMPLETED:
                jobs.mark_succeeded(job)
                session.commit()
                logger.info(
                    "job_skipped_already_complete",
                    extra={
                        "event": "job_skipped_already_complete",
                        "job_id": str(job.id),
                        "analysis_id": str(analysis_id),
                    },
                )
                return
            repository = AnalysisRepository(session)
            repository.bump_attempt(analysis)
            session.commit()
            self._execute(session, analysis, job, job.attempts)
        logger.info(
            "job_finished",
            extra={
                "event": "job_finished",
                "job_id": str(job_id),
                "analysis_id": str(analysis_id) if analysis_id else None,
                "duration_ms": int((time.perf_counter() - started) * 1000),
            },
        )

    def _execute(
        self, session: Session, analysis: Analysis, job: AnalysisJob, attempt: int
    ) -> None:
        jobs = JobRepository(session)
        executions = ExecutionRepository(session)
        repository = AnalysisRepository(session)
        service = AnalysisService(session, self._settings)
        stage_state: dict[str, Any] = {"stage": None, "execution": None}

        def cancel_check() -> bool:
            session.expire(analysis, ["cancel_requested"])
            return bool(analysis.cancel_requested)

        def transition(target: str) -> None:
            current_execution = stage_state["execution"]
            current_stage = stage_state["stage"]
            if current_execution is not None and current_stage:
                executions.finish_stage(
                    current_execution,
                    status=ExecutionStatus.SUCCEEDED,
                    message=f"Stage {current_stage} completed.",
                )
            service.transition(analysis, target)
            stage_state["stage"] = target
            stage_state["execution"] = executions.start_stage(
                analysis.id, job.id, attempt, target, self._worker_id
            )
            session.commit()

        pipeline = AnalysisPipeline(
            session,
            analysis,
            settings=self._settings,
            attempt=attempt,
            worker_id=self._worker_id,
            cancel_check=cancel_check,
        )

        try:
            outcome = pipeline.run(transition)
            final_execution = stage_state["execution"]
            if final_execution is not None:
                executions.finish_stage(
                    final_execution,
                    status=ExecutionStatus.SUCCEEDED,
                    message="Analysis results persisted.",
                )
            jobs.mark_succeeded(job)
            session.commit()
            logger.info(
                "analysis_completed",
                extra={
                    "event": "analysis_completed",
                    "analysis_id": str(analysis.id),
                    "warnings": len(outcome.warnings),
                    "attempt": attempt,
                },
            )
        except AnalysisCancelledError as exc:
            self._handle_cancelled(
                session, jobs, executions, repository, analysis, job, stage_state, exc
            )
        except FloodSightError as exc:
            self._handle_failure(
                session, jobs, executions, repository, analysis, job, stage_state, exc
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "pipeline_unhandled_error",
                extra={
                    "event": "pipeline_unhandled_error",
                    "analysis_id": str(analysis.id),
                    "exception": type(exc).__name__,
                },
                exc_info=exc,
            )
            self._handle_failure(
                session,
                jobs,
                executions,
                repository,
                analysis,
                job,
                stage_state,
                RetryableProcessingError(
                    f"An unexpected processing error occurred ({type(exc).__name__}).",
                    failure_code=FailureCode.INTERNAL_ERROR,
                ),
            )

    def _finish_open_stage(
        self, executions: ExecutionRepository, stage_state: dict[str, Any],
        status: str, message: str, error_code: str | None
    ) -> None:
        execution: AnalysisExecution | None = stage_state["execution"]
        if execution is not None and stage_state["stage"]:
            executions.finish_stage(
                execution, status=status, message=message, error_code=error_code
            )

    def _handle_cancelled(
        self, session, jobs, executions, repository, analysis, job, stage_state, exc
    ) -> None:
        self._finish_open_stage(
            executions, stage_state, ExecutionStatus.SKIPPED,
            "Cancelled before completion.", str(FailureCode.CANCELLED),
        )
        session.refresh(analysis)
        if analysis.status != AnalysisStatus.CANCELLED:
            repository.set_status(
                analysis,
                AnalysisStatus.CANCELLED,
                failure_code=str(FailureCode.CANCELLED),
                failure_message="The analysis was cancelled before results were produced.",
            )
        jobs.mark_cancelled(job, message="Cancelled by request before completion.")
        session.commit()
        logger.info(
            "analysis_cancelled",
            extra={
                "event": "analysis_cancelled",
                "analysis_id": str(analysis.id),
                "reason": sanitise_public_message(exc.public_message),
            },
        )

    def _handle_failure(
        self, session, jobs, executions, repository, analysis, job, stage_state, exc
    ) -> None:
        message = sanitise_public_message(exc.public_message)
        code = str(getattr(exc, "failure_code", None) or FailureCode.PROCESSING_ERROR)
        retryable = bool(getattr(exc, "retryable", False))
        self._finish_open_stage(
            executions, stage_state, ExecutionStatus.FAILED, message, code
        )
        session.refresh(analysis)
        if retryable and job.attempts < job.max_attempts:
            delay = min(
                self._settings.worker_retry_max_seconds,
                self._settings.worker_retry_base_seconds * (2 ** max(0, job.attempts - 1)),
            )
            jobs.reschedule(job, delay_seconds=delay, error_code=code, message=message)
            session.commit()
            logger.warning(
                "job_rescheduled",
                extra={
                    "event": "job_rescheduled",
                    "analysis_id": str(analysis.id),
                    "attempt": job.attempts,
                    "max_attempts": job.max_attempts,
                    "retry_in_seconds": delay,
                    "error_code": code,
                },
            )
            return
        repository.set_status(
            analysis,
            AnalysisStatus.FAILED,
            failure_code=code,
            failure_message=message,
        )
        jobs.mark_failed(job, error_code=code, message=message)
        session.commit()
        logger.error(
            "analysis_failed",
            extra={
                "event": "analysis_failed",
                "analysis_id": str(analysis.id),
                "error_code": code,
                "retryable": retryable,
                "attempts": job.attempts,
                "reason": message,
            },
        )

    def _recover_interrupted_jobs(self) -> None:
        with self._session_factory() as session:
            jobs = JobRepository(session)
            recovered = jobs.recover_stale(
                stale_after_seconds=self._settings.worker_stale_after_seconds
            )
            expired = jobs.expire_exhausted()
            session.commit()
        if recovered or expired:
            logger.warning(
                "stale_jobs_recovered",
                extra={
                    "event": "stale_jobs_recovered",
                    "recovered": recovered,
                    "expired": expired,
                    "worker_id": self._worker_id,
                },
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FloodSight AI analysis worker")
    parser.add_argument("--worker-id", default=None)
    parser.add_argument("--once", action="store_true", help="Claim at most one job then exit")
    parser.add_argument("--poll-interval", type=float, default=None)
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.poll_interval:
        settings.worker_poll_interval_seconds = args.poll_interval
    configure_logging(settings)

    worker = AnalysisWorker(settings, worker_id=args.worker_id)
    if args.once:
        worker.run_once()
        return 0
    signal.signal(signal.SIGINT, worker.request_stop)
    signal.signal(signal.SIGTERM, worker.request_stop)
    return worker.run_forever()


if __name__ == "__main__":
    sys.exit(main())