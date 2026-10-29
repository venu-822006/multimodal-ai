from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Sequence
from uuid import UUID, uuid4

from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.enums import (
    ACTIVE_STATUSES,
    PIPELINE_STAGES,
    TERMINAL_STATUSES,
    AnalysisStatus,
    DetectionMethod,
    ImagerySource,
    is_valid_transition,
)
from app.core.exceptions import (
    CapacityError,
    ConflictError,
    DomainValidationError,
    NotFoundError,
)
from app.core.logging import get_logger
from app.models.analysis import Analysis
from app.processing.validators import (
    AreaOfInterest,
    build_date_window,
    geodesic_area_m2,
    validate_bbox,
    validate_flood_date,
)
from app.repositories.analyses import (
    AnalysisRepository,
    AnalysisResultRepository,
    AnalysisView,
    ExecutionRepository,
)
from app.repositories.jobs import JobRepository
from app.schemas.analysis import AnalysisCreateRequest
from app.services.prediction_providers import resolve_provider_id

logger = get_logger("app.analysis_service")


class AnalysisService:
    def __init__(self, session: Session, settings: Settings | None = None) -> None:
        self._session = session
        self._settings = settings or get_settings()
        self.analyses = AnalysisRepository(session)
        self.results = AnalysisResultRepository(session)
        self.executions = ExecutionRepository(session)
        self.jobs = JobRepository(session)

    def create(self, request: AnalysisCreateRequest) -> tuple[Analysis, int]:
        aoi = validate_bbox(request.bbox, self._settings)
        flood_date = validate_flood_date(request.flood_date, self._settings)
        sources = self._resolve_sources(request)
        method = self._resolve_method(request)

        queued = self.analyses.count_by_status([AnalysisStatus.QUEUED])
        active = self.analyses.count_by_status([AnalysisStatus.QUEUED, *ACTIVE_STATUSES])
        if queued >= self._settings.max_queued_analyses or (
            active >= self._settings.max_active_analyses
        ):
            raise CapacityError(
                "The analysis queue is at capacity. Wait for running analyses to finish "
                "before submitting another request.",
                details={
                    "active_analyses": active,
                    "max_active_analyses": self._settings.max_active_analyses,
                },
            )

        name = request.name or self._default_name(flood_date, aoi)
        lookback = request.lookback_days or self._settings.default_lookback_days
        lookahead = request.lookahead_days or self._settings.default_lookahead_days
        window = build_date_window(flood_date, self._settings, lookback, lookahead)
        # Recorded so the worker runs the same provider the request asked for.
        provider_id = resolve_provider_id(self._settings, request.prediction_provider)
        analysis = Analysis(
            id=uuid4(),
            name=name,
            description=request.description,
            location_label=request.location,
            flood_date=flood_date,
            min_lon=aoi.min_lon,
            min_lat=aoi.min_lat,
            max_lon=aoi.max_lon,
            max_lat=aoi.max_lat,
            aoi_area_m2=aoi.area_m2,
            aoi_geometry=aoi.to_geojson_polygon(),
            requested_method=method.value,
            requested_sources=[source.value for source in sources],
            request_payload={
                "lookback_days": lookback,
                "lookahead_days": lookahead,
                "min_flood_area_m2": request.min_flood_area_m2
                or self._settings.min_flood_polygon_area_m2,
                "window_start": window.window_start.isoformat(),
                "window_end": window.window_end.isoformat(),
                "target_resolution_m": self._settings.raster_target_resolution_m,
                "stac_endpoint": self._settings.stac_endpoint,
                "prediction_provider": provider_id,
            },
            status=AnalysisStatus.QUEUED,
        )
        self.analyses.add(analysis)
        job = self.jobs.create(analysis.id, max_attempts=self._settings.worker_max_attempts)
        queue_position = self.jobs.pending_count()
        logger.info(
            "analysis_created",
            extra={
                "event": "analysis_created",
                "analysis_id": str(analysis.id),
                "analysis_name": name,
                "method": method.value,
                "sources": [source.value for source in sources],
                "flood_date": flood_date.isoformat(),
                "aoi_area_km2": round(aoi.area_m2 / 1_000_000.0, 3),
                "job_id": str(job.id),
                "queue_position": queue_position,
                "prediction_provider": provider_id,
            },
        )
        return analysis, queue_position

    def list_analyses(
        self,
        *,
        page: int,
        page_size: int,
        statuses: Sequence[str] | None = None,
        order_by: str = "created_at",
        descending: bool = True,
    ) -> tuple[list[Analysis], int]:
        return self.analyses.list(
            page=page,
            page_size=page_size,
            statuses=statuses,
            order_by=order_by,
            descending=descending,
        )

    def get(self, analysis_id: UUID) -> Analysis:
        analysis = self.analyses.get(analysis_id, with_relations=True)
        if analysis is None:
            raise NotFoundError(f"Analysis {analysis_id} was not found.")
        return analysis

    def get_result(self, analysis_id: UUID):
        analysis = self.get(analysis_id)
        result = self.analyses.get(analysis_id)
        return analysis, self.results.get(analysis_id) if result else None

    def cancel(self, analysis_id: UUID, reason: str | None = None) -> Analysis:
        analysis = self.get(analysis_id)
        if analysis.status in TERMINAL_STATUSES:
            raise ConflictError(
                f"Analysis {analysis_id} already finished with status '{analysis.status}' and "
                "cannot be cancelled."
            )
        self.analyses.request_cancellation(analysis, reason)
        logger.info(
            "analysis_cancellation_requested",
            extra={
                "event": "analysis_cancellation_requested",
                "analysis_id": str(analysis_id),
                "status": analysis.status,
            },
        )
        return analysis

    def transition(self, analysis: Analysis, target: str) -> None:
        current = analysis.status
        if current == target:
            return
        if not is_valid_transition(current, target):
            raise ConflictError(
                f"Invalid analysis status transition from '{current}' to '{target}'."
            )
        self.analyses.set_status(analysis, target)
        logger.info(
            "analysis_status_changed",
            extra={
                "event": "analysis_status_changed",
                "analysis_id": str(analysis.id),
                "from_status": current,
                "to_status": target,
            },
        )

    def progress_percent(self, analysis: Analysis) -> int:
        if analysis.status == AnalysisStatus.COMPLETED:
            return 100
        if analysis.status in TERMINAL_STATUSES:
            return 100
        if analysis.status == AnalysisStatus.QUEUED:
            return 0
        stages = (AnalysisStatus.QUEUED, *PIPELINE_STAGES)
        try:
            index = stages.index(analysis.status)
        except ValueError:
            return 0
        return int(round(index / len(PIPELINE_STAGES) * 100))

    def resolve_aoi(self, analysis: Analysis) -> AreaOfInterest:
        return validate_bbox(
            [
                [analysis.min_lon, analysis.min_lat],
                [analysis.max_lon, analysis.max_lat],
            ],
            self._settings,
        )

    @property
    def settings(self) -> Settings:
        return self._settings

    def _resolve_sources(self, request: AnalysisCreateRequest) -> list[ImagerySource]:
        if request.sources:
            return list(dict.fromkeys(request.sources))
        return [ImagerySource.SENTINEL1_RTC, ImagerySource.SENTINEL2_L2A]

    def _resolve_method(self, request: AnalysisCreateRequest) -> DetectionMethod:
        if request.method is DetectionMethod.AUTO:
            return DetectionMethod.AUTO
        return request.method

    def _default_name(self, flood_date, aoi: AreaOfInterest) -> str:
        area = geodesic_area_m2(aoi.geometry) / 1_000_000.0
        return (
            f"Flood extent {aoi.min_lat:.2f},{aoi.min_lon:.2f} to "
            f"{aoi.max_lat:.2f},{aoi.max_lon:.2f} ({flood_date.isoformat()}, {area:.0f} km2)"
        )


def utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def utcnow_plus(seconds: float) -> datetime:
    return utcnow() + timedelta(seconds=seconds)


__all__ = ["AnalysisService", "AnalysisView", "utcnow", "utcnow_plus"]