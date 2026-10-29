from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Sequence
from uuid import UUID

from sqlalchemy import Select, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session, selectinload

from app.core.enums import TERMINAL_STATUSES, AnalysisStatus
from app.models.analysis import Analysis, AnalysisExecution, AnalysisResult


class AnalysisRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, analysis: Analysis) -> Analysis:
        self._session.add(analysis)
        self._session.flush()
        return analysis

    def get(self, analysis_id: UUID, *, with_relations: bool = False) -> Analysis | None:
        statement: Select[Any] = select(Analysis).where(Analysis.id == analysis_id)
        if with_relations:
            statement = statement.options(
                selectinload(Analysis.result),
                selectinload(Analysis.executions),
            )
        return self._session.execute(statement).scalar_one_or_none()

    def list(
        self,
        *,
        page: int,
        page_size: int,
        statuses: Sequence[str] | None = None,
        order_by: str = "created_at",
        descending: bool = True,
    ) -> tuple[list[Analysis], int]:
        filters = []
        if statuses:
            filters.append(Analysis.status.in_(list(statuses)))
        total = self._session.execute(
            select(func.count()).select_from(Analysis).where(*filters)
        ).scalar_one()

        column = {
            "created_at": Analysis.created_at,
            "updated_at": Analysis.updated_at,
            "flood_date": Analysis.flood_date,
            "status": Analysis.status,
        }.get(order_by, Analysis.created_at)
        ordering = [column.desc(), Analysis.id.desc()] if descending else [
            column.asc(),
            Analysis.id.asc(),
        ]
        statement = (
            select(Analysis)
            .options(selectinload(Analysis.result))
            .where(*filters)
            .order_by(*ordering)
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        items = list(self._session.execute(statement).scalars().unique().all())
        return items, int(total)

    def count_by_status(self, statuses: Sequence[str]) -> int:
        return int(
            self._session.execute(
                select(func.count())
                .select_from(Analysis)
                .where(Analysis.status.in_(list(statuses)))
            ).scalar_one()
        )

    def set_status(
        self,
        analysis: Analysis,
        status: str,
        *,
        failure_code: str | None = None,
        failure_message: str | None = None,
    ) -> None:
        now = datetime.now(tz=timezone.utc)
        analysis.status = status
        analysis.updated_at = now
        if failure_code is not None:
            analysis.failure_code = failure_code
        if failure_message is not None:
            analysis.failure_message = failure_message
        if status == AnalysisStatus.VALIDATING and analysis.started_at is None:
            analysis.started_at = now
        if status in TERMINAL_STATUSES:
            analysis.completed_at = analysis.completed_at or now
        self._session.flush()

    def request_cancellation(self, analysis: Analysis, reason: str | None = None) -> None:
        if analysis.status in TERMINAL_STATUSES:
            return
        analysis.cancel_requested = True
        analysis.cancel_reason = reason
        analysis.updated_at = datetime.now(tz=timezone.utc)
        self._session.flush()

    def cancel_requested(self, analysis_id: UUID) -> bool:
        value = self._session.execute(
            select(Analysis.cancel_requested).where(Analysis.id == analysis_id)
        ).scalar_one_or_none()
        return bool(value)

    def bump_attempt(self, analysis: Analysis) -> None:
        analysis.attempt_count = (analysis.attempt_count or 0) + 1
        self._session.flush()


class AnalysisResultRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def upsert(self, analysis_id: UUID, payload: dict[str, Any]) -> AnalysisResult:
        values = {"analysis_id": analysis_id, **payload}
        statement = (
            pg_insert(AnalysisResult)
            .values(**values)
            .on_conflict_do_update(
                constraint="uq_analysis_results_analysis_id", set_=values
            )
            .returning(AnalysisResult)
        )
        result = self._session.execute(statement).scalar_one()
        self._session.flush()
        return result

    def get(self, analysis_id: UUID) -> AnalysisResult | None:
        return self._session.execute(
            select(AnalysisResult).where(AnalysisResult.analysis_id == analysis_id)
        ).scalar_one_or_none()


class ExecutionRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def start_stage(
        self,
        analysis_id: UUID,
        job_id: UUID | None,
        attempt: int,
        stage: str,
        worker_id: str | None,
    ) -> AnalysisExecution:
        existing = self._session.execute(
            select(AnalysisExecution).where(
                AnalysisExecution.analysis_id == analysis_id,
                AnalysisExecution.attempt == attempt,
                AnalysisExecution.stage == stage,
            )
        ).scalar_one_or_none()
        now = datetime.now(tz=timezone.utc)
        if existing is not None:
            existing.status = "running"
            existing.started_at = now
            existing.finished_at = None
            existing.duration_ms = None
            existing.error_code = None
            existing.message = None
            existing.worker_id = worker_id
            execution = existing
        else:
            execution = AnalysisExecution(
                analysis_id=analysis_id,
                job_id=job_id,
                attempt=attempt,
                stage=stage,
                status="running",
                worker_id=worker_id,
                started_at=now,
            )
            self._session.add(execution)
        self._session.flush()
        return execution

    def finish_stage(
        self,
        execution: AnalysisExecution,
        *,
        status: str,
        message: str | None = None,
        error_code: str | None = None,
    ) -> None:
        now = datetime.now(tz=timezone.utc)
        execution.status = status
        execution.finished_at = now
        execution.message = message
        execution.error_code = error_code
        if execution.started_at is not None:
            execution.duration_ms = max(
                0, int((now - execution.started_at).total_seconds() * 1000)
            )
        self._session.flush()

    def list_for_analysis(self, analysis_id: UUID) -> list[AnalysisExecution]:
        return list(
            self._session.execute(
                select(AnalysisExecution)
                .where(AnalysisExecution.analysis_id == analysis_id)
                .order_by(AnalysisExecution.started_at.asc(), AnalysisExecution.stage.asc())
            ).scalars()
        )

    def last_stage(self, analysis_id: UUID) -> AnalysisExecution | None:
        return self._session.execute(
            select(AnalysisExecution)
            .where(AnalysisExecution.analysis_id == analysis_id)
            .order_by(AnalysisExecution.started_at.desc())
            .limit(1)
        ).scalar_one_or_none()


class AnalysisView:
    @staticmethod
    def duration_seconds(analysis: Analysis) -> float | None:
        if analysis.started_at is None:
            return None
        end = analysis.completed_at or datetime.now(tz=timezone.utc)
        return round((end - analysis.started_at).total_seconds(), 3)

    @staticmethod
    def bbox(analysis: Analysis) -> dict[str, Any]:
        return {
            "southWest": {"lon": analysis.min_lon, "lat": analysis.min_lat},
            "northEast": {"lon": analysis.max_lon, "lat": analysis.max_lat},
        }

    @staticmethod
    def geojson_bbox(analysis: Analysis) -> dict[str, Any]:
        return {
            "type": "Feature",
            "properties": {"analysis_id": str(analysis.id), "kind": "area_of_interest"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [analysis.min_lon, analysis.min_lat],
                        [analysis.max_lon, analysis.min_lat],
                        [analysis.max_lon, analysis.max_lat],
                        [analysis.min_lon, analysis.max_lat],
                        [analysis.min_lon, analysis.min_lat],
                    ]
                ],
            },
        }

    @staticmethod
    def link_prefix(analysis_id: UUID, prefix: str) -> str:
        return f"{prefix}/analyses/{analysis_id}"

    @staticmethod
    def links(analysis_id: UUID, prefix: str) -> dict[str, str]:
        base = f"{prefix}/analyses/{analysis_id}"
        return {
            "self": base,
            "status": f"{base}/status",
            "facts": f"{base}/facts",
            "map": f"{base}/map",
            "report": f"{base}/report",
        }


__all__ = [
    "AnalysisRepository",
    "AnalysisResultRepository",
    "AnalysisView",
    "ExecutionRepository",
    "update",
]