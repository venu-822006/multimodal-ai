from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.enums import AnalysisStatus, JobStatus
from app.db.base import Base


def utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


class Analysis(TimestampMixin, Base):
    __tablename__ = "analyses"
    __table_args__ = (
        CheckConstraint("min_lon < max_lon", name="min_lon_lt_max_lon"),
        CheckConstraint("min_lat < max_lat", name="min_lat_lt_max_lat"),
        CheckConstraint("min_lon >= -180 AND max_lon <= 180", name="lon_range"),
        CheckConstraint("min_lat >= -90 AND max_lat <= 90", name="lat_range"),
        CheckConstraint("aoi_area_m2 IS NULL OR aoi_area_m2 >= 0", name="aoi_area_non_negative"),
        CheckConstraint(
            "status IN ('queued','validating','searching_data','downloading','preprocessing',"
            "'flood_mapping','damage_analysis','connectivity_analysis','generating_report',"
            "'completed','failed','cancelled')",
            name="status_lifecycle",
        ),
        Index("ix_analyses_created_at_id", "created_at", "id"),
        Index("ix_analyses_status_created_at", "status", "created_at"),
        Index("ix_analyses_flood_date", "flood_date"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    location_label: Mapped[str | None] = mapped_column(String(500))
    flood_date: Mapped[date] = mapped_column(Date, nullable=False)

    min_lon: Mapped[float] = mapped_column(Float, nullable=False)
    min_lat: Mapped[float] = mapped_column(Float, nullable=False)
    max_lon: Mapped[float] = mapped_column(Float, nullable=False)
    max_lat: Mapped[float] = mapped_column(Float, nullable=False)
    aoi_area_m2: Mapped[float | None] = mapped_column(Float)
    aoi_geometry: Mapped[dict] = mapped_column(JSONB, nullable=False)

    request_payload: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    requested_sources: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    requested_method: Mapped[str] = mapped_column(String(40), nullable=False)

    status: Mapped[str] = mapped_column(
        String(40), nullable=False, default=AnalysisStatus.QUEUED
    )
    failure_code: Mapped[str | None] = mapped_column(String(60))
    failure_message: Mapped[str | None] = mapped_column(String(600))
    cancel_requested: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    cancel_reason: Mapped[str | None] = mapped_column(String(300))
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    artifact_directory: Mapped[str | None] = mapped_column(String(300))
    report_available: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    result: Mapped["AnalysisResult | None"] = relationship(
        back_populates="analysis", uselist=False, cascade="all, delete-orphan"
    )
    executions: Mapped[list["AnalysisExecution"]] = relationship(
        back_populates="analysis",
        cascade="all, delete-orphan",
        order_by="AnalysisExecution.started_at",
    )
    job: Mapped["AnalysisJob | None"] = relationship(
        back_populates="analysis", uselist=False, cascade="all, delete-orphan"
    )


class AnalysisResult(TimestampMixin, Base):
    __tablename__ = "analysis_results"
    __table_args__ = (
        UniqueConstraint("analysis_id", name="uq_analysis_results_analysis_id"),
        CheckConstraint(
            "flooded_area_m2 IS NULL OR flooded_area_m2 >= 0", name="flooded_area_non_negative"
        ),
        CheckConstraint(
            "flooded_percentage IS NULL OR (flooded_percentage >= 0 AND flooded_percentage <= 100)",
            name="flooded_percentage_range",
        ),
        Index("ix_analysis_results_created_at", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analyses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[str] = mapped_column(String(40), nullable=False)
    aoi_area_m2: Mapped[float | None] = mapped_column(Float)
    aoi_area_km2: Mapped[float | None] = mapped_column(Float)
    flooded_area_m2: Mapped[float | None] = mapped_column(Float)
    flooded_area_km2: Mapped[float | None] = mapped_column(Float)
    flooded_percentage: Mapped[float | None] = mapped_column(Float)
    flood_polygon_count: Mapped[int | None] = mapped_column(Integer)
    flood_geometry: Mapped[dict | None] = mapped_column(JSONB)
    geometry_summary: Mapped[dict | None] = mapped_column(JSONB)
    measurements: Mapped[dict | None] = mapped_column(JSONB)
    method: Mapped[str | None] = mapped_column(String(60))
    methodology: Mapped[str | None] = mapped_column(Text)
    algorithm: Mapped[dict | None] = mapped_column(JSONB)
    imagery: Mapped[list | None] = mapped_column(JSONB)
    provenance: Mapped[dict | None] = mapped_column(JSONB)
    confidence: Mapped[dict | None] = mapped_column(JSONB)
    infrastructure: Mapped[dict | None] = mapped_column(JSONB)
    settlements: Mapped[dict | None] = mapped_column(JSONB)
    warnings: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    limitations: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    artifacts: Mapped[dict | None] = mapped_column(JSONB)

    analysis: Mapped[Analysis] = relationship(back_populates="result")


class AnalysisExecution(Base):
    __tablename__ = "analysis_executions"
    __table_args__ = (
        UniqueConstraint(
            "analysis_id", "attempt", "stage", name="uq_analysis_executions_stage_attempt"
        ),
        Index("ix_analysis_executions_analysis_started", "analysis_id", "started_at"),
        Index("ix_analysis_executions_stage", "stage"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analyses.id", ondelete="CASCADE"),
        nullable=False,
    )
    job_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    stage: Mapped[str] = mapped_column(String(40), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    worker_id: Mapped[str | None] = mapped_column(String(120))
    message: Mapped[str | None] = mapped_column(String(600))
    error_code: Mapped[str | None] = mapped_column(String(60))
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[int | None] = mapped_column(Integer)

    analysis: Mapped[Analysis] = relationship(back_populates="executions")


class AnalysisJob(TimestampMixin, Base):
    __tablename__ = "analysis_jobs"
    __table_args__ = (
        UniqueConstraint("analysis_id", name="uq_analysis_jobs_analysis_id"),
        CheckConstraint("attempts >= 0", name="attempts_non_negative"),
        CheckConstraint("max_attempts >= 1", name="max_attempts_positive"),
        CheckConstraint(
            "status IN ('pending','running','succeeded','failed','cancelled')",
            name="job_status_lifecycle",
        ),
        Index("ix_analysis_jobs_claim", "status", "available_at", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analyses.id", ondelete="CASCADE"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=JobStatus.PENDING
    )
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=100)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    available_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    locked_by: Mapped[str | None] = mapped_column(String(120))
    last_error_code: Mapped[str | None] = mapped_column(String(60))
    last_error_message: Mapped[str | None] = mapped_column(String(600))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    analysis: Mapped[Analysis] = relationship(back_populates="job")