from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Response, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.api.deps import (
    database_session,
    get_analysis_service,
    public_base_url,
    settings_dependency,
)
from app.core.config import Settings
from app.core.enums import ACTIVE_STATUSES, PIPELINE_STAGES, AnalysisStatus
from app.core.exceptions import DomainValidationError, NotFoundError, ResultNotReadyError
from app.core.logging import get_logger, request_id_var
from app.models.analysis import Analysis
from app.repositories.analyses import AnalysisView
from app.repositories.jobs import JobRepository
from app.schemas.analysis import (
    AnalysisCreateOut,
    AnalysisCreateRequest,
    AnalysisDetailOut,
    AnalysisListOut,
    AnalysisStatusOut,
    AnalysisSummaryOut,
    LinksOut,
)
from app.schemas.common import ApiModel
from app.schemas.reports import (
    FactsOut,
    FactsPendingOut,
    MapPendingOut,
    MapResponse,
    ReportDescriptorOut,
)
from app.services.analysis_service import AnalysisService
from app.services.model_service import get_segmentation_model
from app.services.report_service import ReportService
from app.services.result_service import build_facts, build_map_payload, results_pending

logger = get_logger("app.api.analyses")
router = APIRouter(prefix="/analyses", tags=["analyses"])

VALID_STATUSES = {str(item) for item in AnalysisStatus}


def _links(analysis_id: UUID, prefix: str) -> dict[str, str]:
    return AnalysisView.links(analysis_id, prefix)


def _report_url(analysis_id: UUID, base: str, prefix: str) -> str:
    return f"{base}{prefix}/analyses/{analysis_id}/report/download"


def _report_public_base(analysis: Analysis, base: str, prefix: str) -> str:
    """Absolute URL for a completed analysis report, else None."""
    if not analysis.report_available or analysis.status != AnalysisStatus.COMPLETED:
        return None
    return _report_url(analysis.id, base, prefix)


def _to_summary(
    analysis: Analysis, *, base: str, prefix: str
) -> AnalysisSummaryOut:
    result = analysis.result
    return AnalysisSummaryOut(
        id=analysis.id,
        name=analysis.name,
        description=analysis.description,
        location=analysis.location_label,
        flood_date=analysis.flood_date,
        status=analysis.status,
        method=analysis.requested_method,
        bbox=AnalysisView.bbox(analysis),
        created_at=analysis.created_at,
        updated_at=analysis.updated_at,
        started_at=analysis.started_at,
        completed_at=analysis.completed_at,
        duration_seconds=AnalysisView.duration_seconds(analysis),
        has_results=result is not None and analysis.status == AnalysisStatus.COMPLETED,
        report_url=_report_public_base(analysis, base, prefix),
        failure_code=analysis.failure_code,
        failure_message=analysis.failure_message,
        flooded_area_km2=result.flooded_area_km2 if result else None,
        flooded_percentage=result.flooded_percentage if result else None,
    )


@router.post(
    "",
    response_model=AnalysisCreateOut,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create and schedule a flood analysis",
)
def create_analysis(
    payload: AnalysisCreateRequest,
    response: Response,
    session: Session = Depends(database_session),
    settings: Settings = Depends(settings_dependency),
    base_url: str = Depends(public_base_url),
) -> AnalysisCreateOut:
    """Persist an analysis and schedule a worker job.

    A 202 response means the job was recorded and scheduled. It does not mean
    flood detection has succeeded.
    """
    service = AnalysisService(session, settings)
    if settings.model_require_checkpoint and not get_segmentation_model(settings).is_available():
        logger.warning(
            "analysis_accepted_without_model",
            extra={
                "event": "analysis_accepted_without_model",
                "reason": get_segmentation_model(settings).describe().reason,
            },
        )
    analysis, queue_position = service.create(payload)
    session.commit()
    job = JobRepository(session).get_for_analysis(analysis.id)
    request_id = request_id_var.get()
    logger.info(
        "analysis_request_accepted",
        extra={
            "event": "analysis_request_accepted",
            "analysis_id": str(analysis.id),
            "job_id": str(job.id) if job else None,
            "queue_position": queue_position,
            "request_id": request_id,
        },
    )
    response.headers["Location"] = f"{settings.api_prefix}/analyses/{analysis.id}"
    return AnalysisCreateOut(
        id=analysis.id,
        name=analysis.name,
        status=analysis.status,
        flood_date=analysis.flood_date,
        location=analysis.location_label,
        bbox=AnalysisView.bbox(analysis),
        created_at=analysis.created_at,
        job_id=job.id if job else UUID(int=0),
        queue_position=queue_position,
        links=LinksOut(**_links(analysis.id, settings.api_prefix)),
    )


@router.get("", response_model=AnalysisListOut, summary="List analyses")
def list_analyses(
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int | None, Query(ge=1, le=500)] = None,
    status_filter: Annotated[str | None, Query(alias="status")] = None,
    order_by: str = "created_at",
    descending: bool = True,
    session: Session = Depends(database_session),
    settings: Settings = Depends(settings_dependency),
    base_url: str = Depends(public_base_url),
) -> AnalysisListOut:
    size = page_size or settings.default_page_size
    if size > settings.max_page_size:
        raise DomainValidationError(
            f"page_size must not exceed {settings.max_page_size}."
        )
    if order_by not in {"created_at", "updated_at", "flood_date", "status"}:
        raise DomainValidationError(
            "order_by must be one of created_at, updated_at, flood_date, status."
        )
    statuses: list[str] | None = None
    if status_filter:
        statuses = [item.strip() for item in status_filter.split(",") if item.strip()]
        unknown = [item for item in statuses if item not in VALID_STATUSES]
        if unknown:
            raise DomainValidationError(
                f"Unknown status value(s): {', '.join(unknown)}."
            )

    service = AnalysisService(session, settings)
    items, total = service.list_analyses(
        page=page,
        page_size=size,
        statuses=statuses,
        order_by=order_by,
        descending=descending,
    )
    total_pages = (total + size - 1) // size if total else 0
    session.commit()
    return AnalysisListOut(
        items=[
            _to_summary(analysis, base=base_url, prefix=settings.api_prefix) for analysis in items
        ],
        total_items=total,
        page=page,
        page_size=size,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_previous=page > 1,
    )


@router.get("/{analysis_id}", response_model=AnalysisDetailOut, summary="Analysis detail")
def get_analysis(
    analysis_id: UUID,
    session: Session = Depends(database_session),
    settings: Settings = Depends(settings_dependency),
    base_url: str = Depends(public_base_url),
) -> AnalysisDetailOut:
    service = AnalysisService(session, settings)
    analysis = service.get(analysis_id)
    result = analysis.result
    session.commit()
    summary: dict[str, Any] | None = None
    if result is not None:
        summary = {
            "flooded_area_km2": result.flooded_area_km2,
            "flooded_percentage": result.flooded_percentage,
            "flood_polygon_count": result.flood_polygon_count,
            "method": result.method,
        }
    return AnalysisDetailOut(
        id=analysis.id,
        name=analysis.name,
        description=analysis.description,
        location=analysis.location_label,
        flood_date=analysis.flood_date,
        bbox=AnalysisView.bbox(analysis),
        aoi_area_km2=(
            round(analysis.aoi_area_m2 / 1_000_000.0, 4) if analysis.aoi_area_m2 else None
        ),
        aoi_geometry=analysis.aoi_geometry,
        status=analysis.status,
        method=analysis.requested_method,
        requested_sources=list(analysis.requested_sources or []),
        parameters=dict(analysis.request_payload or {}),
        created_at=analysis.created_at,
        updated_at=analysis.updated_at,
        started_at=analysis.started_at,
        completed_at=analysis.completed_at,
        duration_seconds=AnalysisView.duration_seconds(analysis),
        attempt_count=analysis.attempt_count,
        cancel_requested=analysis.cancel_requested,
        failure_code=analysis.failure_code,
        failure_message=analysis.failure_message,
        has_results=result is not None,
        report_available=bool(analysis.report_available),
        report_url=_report_public_base(analysis, base_url, settings.api_prefix),
        executions=[
            {
                "stage": execution.stage,
                "status": execution.status,
                "attempt": execution.attempt,
                "started_at": execution.started_at,
                "finished_at": execution.finished_at,
                "duration_ms": execution.duration_ms,
                "message": execution.message,
                "error_code": execution.error_code,
            }
            for execution in analysis.executions
        ],
        result_summary=summary,
        links=LinksOut(**_links(analysis.id, settings.api_prefix)),
    )


@router.get("/{analysis_id}/status", response_model=AnalysisStatusOut, summary="Job status")
def get_status(
    analysis_id: UUID,
    session: Session = Depends(database_session),
    settings: Settings = Depends(settings_dependency),
) -> AnalysisStatusOut:
    service = AnalysisService(session, settings)
    analysis = service.get(analysis_id)
    job = JobRepository(session).get_for_analysis(analysis_id)
    stages = [
        {
            "stage": execution.stage,
            "status": execution.status,
            "attempt": execution.attempt,
            "started_at": execution.started_at,
            "finished_at": execution.finished_at,
            "duration_ms": execution.duration_ms,
            "message": execution.message,
            "error_code": execution.error_code,
        }
        for execution in analysis.executions
    ]
    progress = service.progress_percent(analysis)
    session.commit()
    return AnalysisStatusOut(
        id=analysis.id,
        analysis_id=analysis.id,
        status=analysis.status,
        stage=stages[-1]["stage"] if stages else str(analysis.status),
        attempt=analysis.attempt_count,
        max_attempts=job.max_attempts if job else settings.worker_max_attempts,
        is_terminal=analysis.status
        in {AnalysisStatus.COMPLETED, AnalysisStatus.FAILED, AnalysisStatus.CANCELLED},
        progress_percent=progress,
        next_stage=_next_stage(analysis.status),
        created_at=analysis.created_at,
        updated_at=analysis.updated_at,
        started_at=analysis.started_at,
        completed_at=analysis.completed_at,
        duration_seconds=AnalysisView.duration_seconds(analysis),
        failure_code=analysis.failure_code,
        failure_message=analysis.failure_message,
        cancel_requested=analysis.cancel_requested,
        recommended_poll_seconds=5 if analysis.status in ACTIVE_STATUSES or
        analysis.status == AnalysisStatus.QUEUED else 60,
        stages=stages,
    )


def _next_stage(current: str) -> str | None:
    stages = [str(AnalysisStatus.QUEUED), *PIPELINE_STAGES]
    try:
        index = stages.index(str(current))
    except ValueError:
        return None
    if index + 1 < len(stages):
        return stages[index + 1]
    return None


@router.get(
   ("/{analysis_id}/facts"),
    response_model=FactsOut | FactsPendingOut,
    summary="Computed measurements and provenance",
)
def get_facts(
    analysis_id: UUID,
    session: Session = Depends(database_session),
    settings: Settings = Depends(settings_dependency),
) -> FactsOut | FactsPendingOut:
    service = AnalysisService(session, settings)
    analysis = service.get(analysis_id)
    result = analysis.result
    session.commit()
    pending, message = results_pending(analysis)
    if pending:
        return FactsPendingOut(
            analysis_id=analysis.id,
            status=analysis.status,
            message=message,
            next_poll_seconds=5,
        )
    return FactsOut.model_validate(build_facts(analysis, result))


@router.get(
   ("/{analysis_id}/map"),
    response_model=MapResponse | MapPendingOut,
    summary="Generated GeoJSON map layers",
)
def get_map(
    analysis_id: UUID,
    session: Session = Depends(database_session),
    settings: Settings = Depends(settings_dependency),
) -> MapResponse | MapPendingOut:
    service = AnalysisService(session, settings)
    analysis = service.get(analysis_id)
    result = analysis.result
    session.commit()
    pending, message = results_pending(analysis)
    if pending:
        return MapPendingOut(
            analysis_id=analysis.id,
            status=analysis.status,
            message=message,
            retry_after_seconds=5,
        )
    return MapResponse.model_validate(build_map_payload(analysis, result))


@router.get(
   ("/{analysis_id}/report"),
    response_model=ReportDescriptorOut | MapPendingOut,
    summary="Situation report descriptor",
)
def get_report(
    analysis_id: UUID,
    session: Session = Depends(database_session),
    settings: Settings = Depends(settings_dependency),
    base_url: str = Depends(public_base_url),
) -> ReportDescriptorOut | MapPendingOut:
    service = AnalysisService(session, settings)
    analysis = service.get(analysis_id)
    pending, message = results_pending(analysis)
    session.commit()
    if pending:
        return MapPendingOut(
            analysis_id=analysis.id,
            status=analysis.status,
            message=message,
            retry_after_seconds=5,
        )
    reports = ReportService(page_size=settings.report_page_size)
    geojson_url = None
    if analysis.result and (analysis.result.artifacts or {}).get("flood_geojson"):
        geojson_url = (
            f"{base_url}{settings.api_prefix}/analyses/{analysis.id}/artifacts/flood_geojson"
        )
    return reports.descriptor(
        analysis,
        url=_report_url(analysis.id, base_url, settings.api_prefix),
        pdf_url=_report_url(analysis.id, base_url, settings.api_prefix),
        geojson_url=geojson_url,
    )


@router.get(
    "/{analysis_id}/report/download",
    summary="Download the situation report as PDF",
    response_class=FileResponse,
    responses={200: {"content": {"application/pdf": {}}, 202: {"description": "Not ready"}}},
)
def download_report(
    analysis_id: UUID,
    session: Session = Depends(database_session),
    settings: Settings = Depends(settings_dependency),
) -> FileResponse:
    service = AnalysisService(session, settings)
    analysis = service.get(analysis_id)
    result = analysis.result
    session.commit()
    pending, message = results_pending(analysis)
    if pending or result is None:
        raise ResultNotReadyError(message)
    pdf = ReportService(page_size=settings.report_page_size).render_pdf(analysis, result)
    filename = f"floodsight-{analysis.id}.pdf"
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "private, max-age=300",
        },
    )


@router.get("/{analysis_id}/artifacts/{artifact_key}", summary="Download a generated artifact")
def get_artifact(
    analysis_id: UUID,
    artifact_key: str,
    session: Session = Depends(database_session),
    settings: Settings = Depends(settings_dependency),
) -> FileResponse:
    """Serve a persisted artifact by database reference only.

    The request path never contributes to the resolved file location, so path
    traversal is not reachable from user input.
    """
    service = AnalysisService(session, settings)
    analysis = service.get(analysis_id)
    result = analysis.result
    session.commit()
    if result is None or not analysis.status:
        raise ResultNotReadyError("No artifacts exist for this analysis yet.")
    artifacts = result.artifacts or {}
    relative = artifacts.get(artifact_key)
    if not relative:
        raise NotFoundError(f"No artifact named '{artifact_key}' exists for this analysis.")
    root = Path(settings.artifact_root).resolve()
    target = Path(relative).resolve()
    if root not in target.parents:
        raise NotFoundError(f"No artifact named '{artifact_key}' exists for this analysis.")
    if not target.is_file():
        raise NotFoundError(f"The artifact '{artifact_key}' is no longer available on disk.")
    if target.stat().st_size > settings.max_artifact_bytes:
        raise NotFoundError("This artifact exceeds the configured maximum download size.")
    media = "application/json" if target.suffix == ".geojson" else "image/tiff"
    return FileResponse(
        target,
        media_type=media,
        filename=target.name,
        headers={"Cache-Control": "private, max-age=300"},
    )


@router.post("/{analysis_id}/cancel", response_model=AnalysisSummaryOut, summary="Cancel analysis")
def cancel_analysis(
    analysis_id: UUID,
    session: Session = Depends(database_session),
    settings: Settings = Depends(settings_dependency),
    base_url: str = Depends(public_base_url),
) -> AnalysisSummaryOut:
    service = AnalysisService(session, settings)
    analysis = service.cancel(analysis_id)
    session.commit()
    return _to_summary(analysis, base=base_url, prefix=settings.api_prefix)


__all__ = ["router"]