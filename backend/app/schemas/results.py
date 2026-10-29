from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import Field

from app.schemas.common import ApiModel


class FactsOut(ApiModel):
    analysis_id: UUID = Field(description="Identifier of the analysis these facts belong to")
    name: str
    status: str
    results_ready: bool = Field(
        description="True only when persisted, validated measurements are available"
    )
    flood_date: str | None = None
    location: str | None = None

    aoi_bbox: str = Field(description="Area of interest as 'west, south, east, north'")
    aoi_area_km2: float | None = None
    aoi_area_m2: float | None = None

    flooded_area_km2: float | None = Field(
        default=None, description="Geodesic area of the detected flood extent"
    )
    flooded_area_m2: float | None = None
    flooded_percentage: float | None = Field(
        default=None, description="Share of the analysed region classified as flooded"
    )
    flood_polygon_count: int | None = None
    flood_geometry_summary: str | None = None
    flood_extent_geometry_type: str | None = None

    imagery_scenes: list[str] = Field(default_factory=list)
    imagery_acquisition_dates: list[str] = Field(default_factory=list)
    imagery_provider: str | None = None
    imagery_catalogue: str | None = None

    method: str | None = None
    algorithm: str | None = None
    algorithm_version: str | None = None
    methodology: str | None = None
    processing_resolution_m: float | None = None
    detection_threshold_db: float | None = None

    valid_pixel_count: int | None = Field(
        default=None, description="Pixels with valid model input after nodata removal"
    )
    flood_pixel_count: int | None = Field(
        default=None,
        description="Pixels the prediction provider classified as flood (raw mask)",
    )
    retained_flood_pixel_count: int | None = Field(
        default=None,
        description="Raw flood pixels contained in polygons kept after minimum-area filtering",
    )
    filtered_out_pixel_count: int | None = Field(
        default=None,
        description="Raw flood pixels dropped by minimum-area polygon filtering",
    )
    valid_coverage_fraction: float | None = None
    min_flood_polygon_area_m2: float | None = None

    prediction_provider: str | None = Field(
        default=None,
        description="Provider that produced the mask: trained_model or synthetic_test",
    )
    prediction_data_label: str | None = Field(
        default=None,
        description="'model_prediction' or 'synthetic_test_data' for every measurement below",
    )
    predictions_are_synthetic: bool = Field(
        default=False,
        description="True when the mask came from the synthetic test provider",
    )
    metrics_are_synthetic: bool = Field(
        default=False,
        description="True when flooded_area and flooded_percentage are synthetic test values",
    )

    model_name: str | None = Field(default=None, description="Model architecture used")
    model_checkpoint: str | None = Field(
        default=None, description="Opaque checkpoint identifier, never a filesystem path"
    )
    model_val_flood_iou: float | None = Field(
        default=None,
        description="Validation flood IoU recorded in the checkpoint, when the checkpoint records one",
    )
    model_input_bands: list[str] | None = None
    model_confidence_threshold: float | None = None
    model_device: str | None = None
    prediction_is_model_output_not_ground_truth: bool = Field(
        default=True,
        description="False for synthetic test data, which is not model output of any kind",
    )

    confidence_index: float | None = Field(
        default=None, description="Data derived separation index, not a model probability"
    )
    confidence_definition: str | None = None

    buildings_intersecting_flood: int | None = None
    road_segments_intersecting_flood: int | None = None
    infrastructure_summary: str | None = None
    settlements_in_flood_zone: int | None = None
    settlements_summary: str | None = None

    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    unavailable_measurements: list[str] = Field(default_factory=list)
    provenance: str | None = None
    provenance_detail: dict[str, Any] | None = None
    message: str = Field(default="", description="Empty when results are ready")

    processing_duration_seconds: float | None = None
    completed_at: datetime | None = None
    map_available: bool = False
    report_available: bool = False


class FactsPendingOut(ApiModel):
    analysis_id: UUID
    status: str
    results_ready: bool = False
    message: str = Field(
        description="Why results are not available yet; no measurements are reported"
    )
    next_poll_seconds: int = 5


class MapLayerLegendItem(ApiModel):
    id: str
    label: str
    description: str


class MapResponse(ApiModel):
    analysis_id: UUID
    type: str = "FeatureCollection"
    status: str
    crs: str = "EPSG:4326"
    bbox: list[float] | None = None
    generated_at: datetime | None = None
    prediction_data_label: str | None = None
    predictions_are_synthetic: bool = False
    features: list[dict[str, Any]] = Field(default_factory=list)
    layers: dict[str, dict[str, Any]] = Field(default_factory=dict)
    legend: list[MapLayerLegendItem] = Field(default_factory=list)
    raster_overlay: dict[str, Any] | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


class MapPendingOut(ApiModel):
    analysis_id: UUID
    status: str
    message: str
    retry_after_seconds: int = 5


class ReportDescriptorOut(ApiModel):
    analysis_id: UUID
    title: str
    description: str
    status: str
    format: str
    url: str
    pdf_url: str
    geojson_url: str | None = None
    generated_at: datetime
    analysis_completed_at: datetime | None = None
    sections: list[str] = Field(default_factory=list)