from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import Field, field_validator, model_validator

from app.core.config import get_settings
from app.core.enums import AnalysisStatus, DetectionMethod, ImagerySource
from app.schemas.common import ApiModel, BoundingBox

Coordinates = Annotated[list[float], Field(min_length=2, max_length=2)]

BboxPair = list[Coordinates]


def _as_finite(value: float, field_name: str) -> float:
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{field_name} must be a finite number")
    return number


class AnalysisCreateRequest(ApiModel):
    name: str | None = Field(default=None, description="Human readable analysis name")
    description: str | None = Field(default=None, description="Free form analyst notes")
    location: str | None = Field(
        default=None, description="Optional reverse geocoded location label for the area"
    )
    flood_date: date = Field(description="Date of the flood event to investigate")
    bbox: BboxPair = Field(
        description="Bounding box as [[west, south], [east, north]] in WGS84 decimal degrees"
    )
    method: DetectionMethod = Field(
        default=DetectionMethod.AUTO,
        description="Flood detection method; 'auto' selects the best available input",
    )
    sources: list[ImagerySource] | None = Field(
        default=None, description="Restrict processing to these imagery sources"
    )
    lookback_days: int | None = Field(
        default=None, ge=1, description="Days before the flood date to search for imagery"
    )
    lookahead_days: int | None = Field(
        default=None, ge=1, description="Days after the flood date to search for imagery"
    )
    min_flood_area_m2: float | None = Field(
        default=None, gt=0, description="Discard detected flood patches smaller than this area"
    )
    prediction_provider: str | None = Field(
        default=None,
        description=(
            "Override the configured prediction provider for this analysis. 'synthetic_test' "
            "produces synthetic_test_data and is only accepted in development or test."
        ),
    )

    @field_validator("bbox", mode="before")
    @classmethod
    def _validate_bbox_shape(cls, value: Any) -> Any:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("bbox must contain exactly two [longitude, latitude] pairs")
        normalised = []
        for corner in value:
            if not isinstance(corner, (list, tuple)) or len(corner) != 2:
                raise ValueError("each bbox corner must be a [longitude, latitude] pair")
            normalised.append([_as_finite(corner[0], "longitude"), _as_finite(corner[1], "latitude")])
        return normalised

    @field_validator("name", "location")
    @classmethod
    def _validate_text(cls, value: str | None) -> str | None:
        settings = get_settings()
        limit = settings.max_name_length if value is not None else 0
        if value is not None:
            trimmed = value.strip()
            if not trimmed:
                return None
            if len(trimmed) > limit:
                raise ValueError(f"value must not exceed {limit} characters")
            return trimmed
        return value

    @field_validator("description")
    @classmethod
    def _validate_description(cls, value: str | None) -> str | None:
        if value is None:
            return None
        trimmed = value.strip()
        if not trimmed:
            return None
        limit = get_settings().max_description_length
        if len(trimmed) > limit:
            raise ValueError(f"description must not exceed {limit} characters")
        return trimmed

    @model_validator(mode="after")
    def _validate_prediction_provider(self) -> "AnalysisCreateRequest":
        from app.core.exceptions import DomainValidationError
        from app.services.prediction_providers import (
            PROVIDER_SYNTHETIC,
            resolve_provider_id,
        )

        if self.prediction_provider is None:
            return self
        settings = get_settings()
        try:
            resolve_provider_id(settings, self.prediction_provider)
        except Exception as exc:  # noqa: BLE001
            raise DomainValidationError(
                getattr(exc, "public_message", None) or str(exc)
            ) from exc
        if self.prediction_provider == PROVIDER_SYNTHETIC and not (
            settings.environment in ("development", "test")
        ):
            raise DomainValidationError(
                "The synthetic test provider can only be selected outside production."
            )
        return self

    @model_validator(mode="after")
    def _validate_ranges(self) -> "AnalysisCreateRequest":
        settings = get_settings()
        for index, (lon, lat) in enumerate(self.bbox):
            if not -180.0 <= lon <= 180.0:
                raise ValueError(f"longitude {lon} is outside the valid range [-180, 180]")
            if not -90.0 <= lat <= 90.0:
                raise ValueError(f"latitude {lat} is outside the valid range [-90, 90]")
            del index
        if self.lookback_days and self.lookback_days > settings.max_flood_lookback_days:
            raise ValueError(
                f"lookback_days must not exceed {settings.max_flood_lookback_days}"
            )
        if self.lookahead_days and self.lookahead_days > settings.max_flood_lookahead_days:
            raise ValueError(
                f"lookahead_days must not exceed {settings.max_flood_lookahead_days}"
            )
        return self


class AnalysisBoundingBoxOut(ApiModel):
    model_config = {"from_attributes": True, "extra": "forbid"}

    south_west: dict[str, float] = Field(alias="southWest")
    north_east: dict[str, float] = Field(alias="northEast")

    @field_validator("south_west", "north_east", mode="before")
    @classmethod
    def _pair_to_dict(cls, value: Any) -> Any:
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return {"lon": float(value[0]), "lat": float(value[1])}
        return value


class LinksOut(ApiModel):
    self: str
    status: str
    facts: str
    map: str
    report: str


class ExecutionOut(ApiModel):
    stage: str
    status: str
    attempt: int
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: int | None = None
    message: str | None = None
    error_code: str | None = None


class AnalysisSummaryOut(ApiModel):
    model_config = {"from_attributes": True, "extra": "forbid"}

    id: UUID
    name: str
    description: str | None = None
    location: str | None = Field(default=None, validation_alias="location_label")
    flood_date: date
    status: str
    method: str
    bbox: BoundingBox
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float | None = None
    has_results: bool = False
    report_url: str | None = None
    failure_code: str | None = None
    failure_message: str | None = None
    flooded_area_km2: float | None = None
    flooded_percentage: float | None = None


class AnalysisListOut(ApiModel):
    items: list[AnalysisSummaryOut]
    total_items: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_previous: bool


class AnalysisCreateOut(ApiModel):
    id: UUID
    name: str
    status: str
    flood_date: date
    location: str | None = None
    bbox: BoundingBox
    created_at: datetime
    job_id: UUID
    queue_position: int
    links: LinksOut


class AnalysisDetailOut(ApiModel):
    model_config = {"from_attributes": True, "extra": "forbid"}

    id: UUID
    name: str
    description: str | None = None
    location: str | None = None
    flood_date: date
    bbox: BoundingBox
    aoi_area_km2: float | None = None
    aoi_geometry: dict[str, Any] | None = None
    status: str
    method: str
    requested_sources: list[str]
    parameters: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float | None = None
    attempt_count: int
    cancel_requested: bool
    failure_code: str | None = None
    failure_message: str | None = None
    has_results: bool
    report_available: bool
    report_url: str | None = None
    executions: list[ExecutionOut] = Field(default_factory=list)
    result_summary: dict[str, Any] | None = None
    links: LinksOut


class AnalysisStatusOut(ApiModel):
    id: UUID
    analysis_id: UUID
    status: str
    stage: str
    attempt: int
    max_attempts: int
    is_terminal: bool
    progress_percent: int
    next_stage: str | None = None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float | None = None
    failure_code: str | None = None
    failure_message: str | None = None
    cancel_requested: bool
    recommended_poll_seconds: int = 5
    stages: list[ExecutionOut]


StatusLiteral = Literal[
    "queued",
    "validating",
    "searching_data",
    "downloading",
    "preprocessing",
    "flood_mapping",
    "damage_analysis",
    "connectivity_analysis",
    "generating_report",
    "completed",
    "failed",
    "cancelled",
]

StatusQuery = Annotated[
    str | None,
    Field(
        default=None,
        description="Comma separated status filter, for example 'completed,failed'",
    ),
]

SortOrder = Annotated[str, Field(pattern="^(created_at|updated_at|flood_date|status)$")]

AnalysisStatusEnum = AnalysisStatus