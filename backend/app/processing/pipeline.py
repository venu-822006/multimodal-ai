from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import UUID

import numpy as np
import scipy
from pyproj import Transformer
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.enums import AnalysisStatus, DetectionMethod, FailureCode
from app.core.exceptions import (
    AnalysisCancelledError,
    FloodSightError,
    PermanentProcessingError,
    RetryableProcessingError,
    sanitise_public_message,
)
from app.core.logging import analysis_id_var, get_logger
from app.models.analysis import Analysis, AnalysisResult
from app.processing.validators import (
    AreaOfInterest,
    DateWindow,
    build_date_window,
    geometry_summary,
    validate_geojson,
)
from app.repositories.analyses import AnalysisResultRepository
from app.services.damage_assessment_service import ExposureAssessment, assess_exposure
from app.services.flood_detection_service import (
    ALGORITHM_VERSION,
    FloodDetectionResult,
    detect_flood,
)
from app.services.geospatial_service import (
    TargetGrid,
    build_target_grid,
    geometry_to_feature_collection,
    measure_geometry,
)
from app.services.imagery_service import (
    PlanetaryComputerStacClient,
    SceneAsset,
    describe_sources,
    select_optical_scene,
    select_sar_pair,
)
from app.services.prediction_providers import (
    PredictionProviderError,
    get_prediction_provider,
    resolve_provider_id,
)
from app.services.preprocessing_service import PREPROCESSING_STEPS

logger = get_logger("app.pipeline")

StageHook = Callable[[str], None]
StageReporter = Callable[[str, str, str | None, str | None], None]

ARTIFACT_FILENAMES = {
    "flood_geojson": "flood_extent.geojson",
    # Raw provider output, written before any minimum-area filtering.
    "flood_mask": "flood_mask.tif",
    # Rasterised form of the filtered extent, used by the map overlay.
    "flood_extent_mask": "flood_extent_mask.tif",
    "analysis_metadata": "analysis.json",
}


@dataclass
class PipelineOutcome:
    analysis: Analysis
    result: AnalysisResult | None
    warnings: list[str] = field(default_factory=list)
    cancelled: bool = False


class AnalysisPipeline:
    """Runs the flood analysis stages for a single claimed analysis job.

    Stages mirror the persisted analysis lifecycle so that externally visible
    status transitions always correspond to real work being performed.
    """

    def __init__(
        self,
        session: Session,
        analysis: Analysis,
        *,
        settings: Settings | None = None,
        attempt: int = 1,
        worker_id: str | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> None:
        self._session = session
        self._settings = settings or get_settings()
        self._analysis = analysis
        self._attempt = attempt
        self._worker_id = worker_id
        self._cancel_check = cancel_check or (lambda: False)
        self._results = AnalysisResultRepository(session)
        self._stac = PlanetaryComputerStacClient(self._settings)
        self._grid: TargetGrid | None = None
        self._detection: FloodDetectionResult | None = None
        self._scenes: list[SceneAsset] = []
        self._exposure: ExposureAssessment | None = None
        self._warnings: list[str] = []
        self._limitations: list[str] = []
        self._skipped: dict[str, str] = {}
        self._aoi: AreaOfInterest | None = None
        self._prepared_post: Path | None = None
        self._used_scenes: list[SceneAsset] = []
        self._artifacts_extra: dict[str, str] = {}
        self._provider_descriptor = None

    def run(self, transition: StageHook) -> PipelineOutcome:
        analysis = self._analysis
        self._rewind_for_retry(analysis)
        transition(AnalysisStatus.VALIDATING)
        aoi, window = self._validate_inputs()
        self._aoi = aoi
        transition(AnalysisStatus.SEARCHING_DATA)
        self._discover_imagery(aoi, window)
        transition(AnalysisStatus.DOWNLOADING)
        self._check_cancelled()
        transition(AnalysisStatus.PREPROCESSING)
        self._preprocess(aoi)
        transition(AnalysisStatus.FLOOD_MAPPING)
        self._map_flood_extent(aoi, window)
        transition(AnalysisStatus.DAMAGE_ANALYSIS)
        self._assess_infrastructure(aoi)
        transition(AnalysisStatus.CONNECTIVITY_ANALYSIS)
        self._assess_settlements(aoi)
        transition(AnalysisStatus.GENERATING_REPORT)
        result = self._persist_result(aoi, window)
        transition(AnalysisStatus.COMPLETED)
        self._session.flush()
        return PipelineOutcome(
            analysis=analysis, result=result, warnings=list(self._warnings)
        )

    def _rewind_for_retry(self, analysis: Analysis) -> None:
        """Return a partially processed analysis to QUEUED before a new attempt.

        A retryable failure can leave the analysis parked in a mid-pipeline stage.
        Restarting the stage sequence from VALIDATING would otherwise be rejected as
        an illegal transition, so the status is rewound and a new attempt recorded.
        """
        if analysis.status == AnalysisStatus.QUEUED:
            return
        logger.info(
            "analysis_rewound_for_retry",
            extra={
                "event": "analysis_rewound_for_retry",
                "analysis_id": str(analysis.id),
                "from_status": analysis.status,
                "attempt": self._attempt,
            },
        )
        analysis.status = AnalysisStatus.QUEUED
        analysis.failure_code = None
        analysis.failure_message = None
        analysis.updated_at = datetime.now(tz=timezone.utc)
        self._session.flush()

    def _check_cancelled(self) -> None:
        if self._cancel_check():
            raise AnalysisCancelledError()

    def _validate_inputs(self) -> tuple[AreaOfInterest, DateWindow]:
        analysis = self._analysis
        from app.processing.validators import validate_bbox

        aoi = validate_bbox(
            [[analysis.min_lon, analysis.min_lat], [analysis.max_lon, analysis.max_lat]],
            self._settings,
        )
        if abs(aoi.area_m2 - (analysis.aoi_area_m2 or aoi.area_m2)) > 1.0:
            raise PermanentProcessingError(
                "The stored area of interest no longer matches the configured area limits.",
                failure_code=FailureCode.INVALID_REQUEST,
            )
        payload = analysis.request_payload or {}
        window = build_date_window(
            analysis.flood_date,
            self._settings,
            payload.get("lookback_days"),
            payload.get("lookahead_days"),
        )
        method = analysis.requested_method or DetectionMethod.AUTO.value
        sources = list(analysis.requested_sources or [])
        if not sources:
            raise PermanentProcessingError(
                "No imagery source is configured for this analysis.",
                failure_code=FailureCode.NO_IMAGERY,
            )
        analysis.request_payload = {**payload, "resolved_method": method}
        self._session.flush()
        logger.info(
            "pipeline_inputs_validated",
            extra={
                "event": "pipeline_inputs_validated",
                "method": method,
                "sources": sources,
                "aoi_area_km2": round(aoi.area_m2 / 1_000_000.0, 3),
                "window_start": window.window_start.isoformat(),
                "window_end": window.window_end.isoformat(),
            },
        )
        return aoi, window

    def _discover_imagery(self, aoi: AreaOfInterest, window: DateWindow) -> None:
        from app.core.enums import ImagerySource

        requested = [ImagerySource(value) for value in (self._analysis.requested_sources or [])]
        method = self._analysis.requested_method
        allowed: set[str] = set()
        if method in (DetectionMethod.AUTO.value, DetectionMethod.SAR_CHANGE_DETECTION.value):
            allowed.add(ImagerySource.SENTINEL1_RTC)
        if method in (DetectionMethod.AUTO.value, DetectionMethod.NDWI_OPTICAL.value):
            allowed.add(ImagerySource.SENTINEL2_L2A)
        sources = [source for source in requested if source in allowed]
        if not sources:
            raise PermanentProcessingError(
                "The requested detection method is not supported by the configured imagery "
                "sources.",
                failure_code=FailureCode.NO_IMAGERY,
            )

        discovered: dict[str, list[SceneAsset]] = {}
        unavailable: list[str] = []
        for source in sources:
            try:
                scenes = self._stac.search(source, aoi, window)
            except FloodSightError as exc:
                if exc.retryable:
                    raise
                logger.warning(
                    "imagery_source_unavailable",
                    extra={
                        "event": "imagery_source_unavailable",
                        "source": source.value,
                        "reason": sanitise_public_message(exc.public_message),
                    },
                )
                unavailable.append(source.value)
                continue
            discovered[source.value] = scenes
            self._scenes.extend(scenes)

        if not discovered:
            raise RetryableProcessingError(
                "The imagery catalogue could not be reached. Processing will retry.",
                failure_code=FailureCode.IMAGERY_UNAVAILABLE,
            )

        self._discovered = discovered
        for source_value in unavailable:
            self._warnings.append(
                f"Imagery source {source_value} could not be queried for this analysis."
            )
        if not any(discovered.values()):
            raise PermanentProcessingError(
                "No Sentinel observations overlap the requested area and date window in the "
                "configured catalogues.",
                failure_code=FailureCode.NO_IMAGERY,
            )

    def _preprocess(self, aoi: AreaOfInterest) -> None:
        self._check_cancelled()
        self._grid = build_target_grid(aoi, self._settings)
        logger.info(
            "grid_prepared",
            extra={
                "event": "grid_prepared",
                "crs": self._grid.crs.to_string(),
                "width": self._grid.width,
                "height": self._grid.height,
                "resolution_m": self._grid.pixel_size,
            },
        )

    def _map_flood_extent(self, aoi: AreaOfInterest, window: DateWindow) -> None:
        """Run the selected prediction provider over prepared Sentinel-1 input."""
        assert self._grid is not None
        from app.core.enums import ImagerySource

        provider = self._prediction_provider()
        self._provider_descriptor = provider.describe()
        try:
            descriptor = provider.ensure_available()
        except PredictionProviderError as exc:
            # The provider already phrases the cause; only the failure code is added here.
            raise PermanentProcessingError(
                exc.public_message,
                failure_code=FailureCode.DEPENDENCY_UNAVAILABLE,
            ) from exc
        self._provider_descriptor = descriptor
        logger.info(
            "prediction_provider_selected",
            extra={
                "event": "prediction_provider_selected",
                "analysis_id": str(self._analysis.id),
                "provider": descriptor.provider_id,
                "data_label": descriptor.data_label,
                "synthetic": descriptor.synthetic,
            },
        )

        discovered = getattr(self, "_discovered", {})
        sar_scenes = discovered.get(ImagerySource.SENTINEL1_RTC.value, [])
        pre, post = select_sar_pair(sar_scenes, window)
        if post is None:
            found = len(sar_scenes)
            reason = (
                "Sentinel-1 observations overlap the area of interest, but none was acquired "
                "on or after the post-flood date."
                if found
                else "No Sentinel-1 GRD observations overlap the requested area and date window."
            )
            raise PermanentProcessingError(
                f"{reason} The flood mapping stage requires Sentinel-1 VV/VH imagery.",
                failure_code=FailureCode.NO_IMAGERY if not found else FailureCode.INSUFFICIENT_COVERAGE,
            )

        self._prepared_post = self._prepare_scene(post)
        pre_prepared = None
        wants_pre = pre is not None and (
            self._settings.model_temporal_filter or self._settings.synthetic_apply_temporal_filter
        )
        if wants_pre and pre is not None:
            pre_prepared = self._prepare_scene(pre, suffix="_pre")
            self._used_scenes.append(pre)
        self._used_scenes.append(post)

        payload = self._analysis.request_payload or {}
        self._detection = detect_flood(
            provider,
            scene_path=self._prepared_post,
            pre_scene_path=pre_prepared,
            grid=self._grid,
            aoi=aoi,
            settings=self._settings,
            cancellation_check=self._check_cancelled,
            min_polygon_area_m2=payload.get("min_flood_area_m2"),
        )
        self._detection.imagery = [scene.public_record() for scene in self._used_scenes]
        self._warnings.extend(self._detection.warnings)
        self._limitations.extend(self._detection.limitations)
        self._artifacts_extra["model_input"] = str(self._prepared_post)

    def _prediction_provider(self):
        """Resolve the provider for this analysis: request override, else configuration.

        The synthetic provider is only ever reached through an explicit selection. It
        is never substituted for an unavailable trained model.
        """
        payload = self._analysis.request_payload or {}
        override = payload.get("prediction_provider")
        try:
            provider_id = resolve_provider_id(self._settings, override)
        except PredictionProviderError as exc:
            raise PermanentProcessingError(
                f"Prediction provider could not be resolved: {exc.public_message}",
                failure_code=FailureCode.INVALID_REQUEST,
            ) from exc
        return get_prediction_provider(self._settings, provider_id)

    def _prepare_scene(self, scene: SceneAsset, *, suffix: str = "") -> Path:
        from app.services.preprocessing_service import prepare_model_input

        target = Path(self._settings.artifact_root) / "_staging" / f"{scene.item_id}{suffix}.tif"
        prepared = prepare_model_input(
            scene,
            self._aoi,
            self._settings,
            grid=self._grid,
            output_path=target,
        )
        self._warnings.extend(prepared.warnings)
        return prepared.path

    def _assess_infrastructure(self, aoi: AreaOfInterest) -> None:
        geometry = self._flood_geometry()
        self._exposure = assess_exposure(aoi, geometry, settings=self._settings)
        if not self._exposure.available:
            self._skipped["damage_analysis"] = self._exposure.summary.get("status", "unavailable")
            self._warnings.extend(self._exposure.warnings)
            return
        self._warnings.extend(self._exposure.warnings)
        self._limitations.extend(self._exposure.limitations)

    def _assess_settlements(self, aoi: AreaOfInterest) -> None:
        if self._exposure is None or not self._exposure.available:
            self._skipped["connectivity_analysis"] = "unavailable"
        logger.info(
            "connectivity_assessed",
            extra={
                "event": "connectivity_assessed",
                "settlements_in_flood_zone": (
                    self._exposure.settlements_in_flood_zone if self._exposure else None
                ),
            },
        )

    def _flood_geometry(self):
        if self._detection is None:
            return None
        geometry = self._detection.geometry
        if geometry is None:
            return None
        if not geometry.is_valid:
            from shapely.validation import make_valid

            geometry = make_valid(geometry)
        return geometry

    def _persist_result(self, aoi: AreaOfInterest, window: DateWindow) -> AnalysisResult:
        detection = self._detection
        if detection is None:
            raise PermanentProcessingError(
                "Flood detection did not produce a result.",
                failure_code=FailureCode.PROCESSING_ERROR,
            )
        geometry = self._flood_geometry()
        flood_geojson = geometry_to_feature_collection(
            geometry,
            {
                "source": detection.method,
                "crs": "EPSG:4326",
                "prediction_data_label": detection.data_label,
                "predictions_are_synthetic": detection.synthetic,
            },
        )
        measurements = dict(detection.measurements)
        measurements.setdefault("aoi_area_m2", round(aoi.area_m2, 2))

        artifacts = self._write_artifacts(aoi, detection, flood_geojson)

        # Report only the scenes the model actually consumed, not every candidate.
        imagery_records = (
            detection.imagery
            if detection.imagery
            else [scene.public_record() for scene in self._used_scenes]
        )
        provenance = {
            "catalogue": self._settings.stac_endpoint,
            "catalogue_provider": "Microsoft Planetary Computer",
            "collections": sorted(
                {record.get("collection") for record in imagery_records if record.get("collection")}
            ),
            "candidate_scene_count": len(self._scenes),
            "used_scene_count": len(imagery_records),
            "retrieved_at": datetime.now(tz=timezone.utc).isoformat(),
            "software": {
                "floodsight_pipeline": ALGORITHM_VERSION,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "method": detection.method,
            "prediction_provider": {
                "provider": detection.provider_id,
                "data_label": detection.data_label,
                "synthetic": detection.synthetic,
            },
            "model": detection.model_info or {},
            "preprocessing": detection.preprocessing_steps,
            "window": {
                "start": window.window_start.isoformat(),
                "end": window.window_end.isoformat(),
                "flood_date": window.flood_date.isoformat(),
            },
        }
        if self._exposure and self._exposure.available:
            provenance["exposure"] = self._exposure.provenance

        payload = {
            "status": AnalysisStatus.COMPLETED,
            "aoi_area_m2": round(aoi.area_m2, 2),
            "aoi_area_km2": round(aoi.area_m2 / 1_000_000.0, 4),
            "flooded_area_m2": measurements.get("flooded_area_m2"),
            "flooded_area_km2": measurements.get("flooded_area_km2"),
            "flooded_percentage": measurements.get("flooded_percentage"),
            "flood_polygon_count": measurements.get("polygon_count"),
            "flood_geometry": flood_geojson,
            "geometry_summary": geometry_summary(geometry),
            "measurements": measurements,
            "method": detection.method,
            # Provider identity is persisted in measurements and provenance rather
            # than as dedicated columns; the facts and map views read it from there.
            "methodology": self._methodology(detection),
            "algorithm": detection.algorithm,
            "imagery": imagery_records,
            "provenance": provenance,
            "confidence": detection.confidence,
            "infrastructure": (self._exposure.layers if self._exposure else None),
            "settlements": (
                {"layer": self._exposure.layers.get("settlements"), **self._exposure.summary}
                if self._exposure and self._exposure.available
                else None
            ),
            "warnings": list(dict.fromkeys(self._warnings)),
            "limitations": list(dict.fromkeys(self._limitations)),
            "artifacts": artifacts,
        }
        result = self._results.upsert(self._analysis.id, payload)
        analysis = self._analysis
        analysis.artifact_directory = artifacts.get("directory")
        analysis.report_available = True
        self._session.flush()
        logger.info(
            "analysis_result_persisted",
            extra={
                "event": "analysis_result_persisted",
                "result_id": str(result.id),
                "method": detection.method,
                "flooded_area_km2": result.flooded_area_km2,
                "flooded_percentage": result.flooded_percentage,
                "flood_polygon_count": result.flood_polygon_count,
                "artifact_count": len([k for k in artifacts if k != "directory"]),
            },
        )
        return result

    def _methodology(self, detection: FloodDetectionResult) -> str:
        scenes = ", ".join(
            f"{record.get('platform') or record.get('source')} "
            f"{record.get('acquisition_date')}"
            for record in detection.imagery
        )
        model = detection.model_info or {}
        base = (
            f"{detection.algorithm['name']} v{detection.algorithm['version']}. "
            f"Preprocessing: {'; '.join(detection.preprocessing_steps) or 'not recorded'}. "
            f"Inputs: {scenes or 'none'}. "
        )
        if detection.synthetic:
            return (
                base
                + f"Prediction provider {detection.provider_id} produced synthetic_test_data "
                f"using pattern '{model.get('pattern')}' and seed {model.get('seed')}. "
                "No trained model weights were loaded and no flood detection was performed; "
                "every measurement below is a pipeline test value."
            )
        return (
            base
            + f"Architecture {model.get('architecture')} "
            f"{model.get('model_config')} on {model.get('device')} with "
            f"{model.get('patch_size')} px patches at stride {model.get('stride')} over "
            f"{model.get('tiles')} tiles. "
            f"Flood class probability threshold {model.get('confidence_threshold')}"
            f"{' with a pre-event temporal consistency filter' if model.get('temporal_filter_applied') else ''}. "
            f"Analysis resolution {detection.measurements.get('resolution_m')} m. "
            f"Reference: {detection.algorithm['reference']}."
        )

    def _write_artifacts(
        self, aoi: AreaOfInterest, detection: FloodDetectionResult, flood_geojson: dict[str, Any]
    ) -> dict[str, Any]:
        directory = self._artifact_directory(self._analysis.id)
        directory.mkdir(parents=True, exist_ok=True)
        artifacts: dict[str, Any] = {"directory": str(directory)}

        geojson_path = directory / ARTIFACT_FILENAMES["flood_geojson"]
        geojson_path.write_text(json.dumps(flood_geojson, separators=(",", ":")))
        artifacts["flood_geojson"] = str(geojson_path)

        mask_path = directory / ARTIFACT_FILENAMES["flood_mask"]
        self._write_raw_mask(mask_path, detection)
        artifacts["flood_mask"] = str(mask_path)

        extent_mask_path = directory / ARTIFACT_FILENAMES["flood_extent_mask"]
        self._write_extent_mask(extent_mask_path, detection)
        artifacts["flood_extent_mask"] = str(extent_mask_path)

        metadata_path = directory / ARTIFACT_FILENAMES["analysis_metadata"]
        metadata_path.write_text(
            json.dumps(
                {
                    "analysis_id": str(self._analysis.id),
                    "method": detection.method,
                    "prediction_provider": detection.provider_id,
                    "prediction_data_label": detection.data_label,
                    "predictions_are_synthetic": detection.synthetic,
                    "measurements": detection.measurements,
                    "confidence": detection.confidence,
                    "warnings": self._warnings,
                    "limitations": self._limitations,
                    "imagery": detection.imagery,
                    "crs": "EPSG:4326",
                },
                indent=2,
                default=str,
            )
        )
        artifacts["analysis_metadata"] = str(metadata_path)
        artifacts.update(self._artifacts_extra)
        return artifacts

    def _write_raw_mask(self, path: Path, detection: FloodDetectionResult) -> None:
        """Write the provider mask exactly as produced, before polygon filtering.

        Values are 0 background, 1 flood and 255 nodata. Minimum-area filtering is a
        vector concern, so the stored raster always shows the full prediction.
        """
        import rasterio

        grid = self._grid
        if grid is None or detection.raw_mask is None:
            return
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=grid.height,
            width=grid.width,
            count=1,
            dtype="uint8",
            crs=grid.crs,
            transform=grid.transform,
            nodata=255,
            compress="deflate",
        ) as dataset:
            dataset.write(detection.raw_mask.astype("uint8"), 1)
            dataset.update_tags(
                prediction_provider=detection.provider_id,
                prediction_data_label=detection.data_label,
                predictions_are_synthetic=str(bool(detection.synthetic)).lower(),
                method=detection.method,
            )

    def _write_extent_mask(self, path: Path, detection: FloodDetectionResult) -> None:
        """Rasterise the filtered extent geometry, which is what the map displays."""
        import rasterio

        grid = self._grid
        if grid is None:
            return
        mask = self._rasterise_mask(grid)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=grid.height,
            width=grid.width,
            count=1,
            dtype="uint8",
            crs=grid.crs,
            transform=grid.transform,
            nodata=0,
            compress="deflate",
        ) as dataset:
            dataset.write(mask.astype("uint8"), 1)
            dataset.update_tags(
                prediction_provider=detection.provider_id,
                prediction_data_label=detection.data_label,
                predictions_are_synthetic=str(bool(detection.synthetic)).lower(),
                derived_from="flood_extent.geojson",
            )

    def _rasterise_mask(self, grid: TargetGrid) -> np.ndarray:
        from rasterio.features import rasterize
        from shapely.ops import transform as shapely_transform

        geometry = self._flood_geometry()
        mask = np.zeros(grid.shape, dtype="uint8")
        if geometry is None:
            return mask
        to_grid = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
        projected = shapely_transform(to_grid.transform, geometry)
        return rasterize(
            [(projected, 1)],
            out_shape=grid.shape,
            transform=grid.transform,
            fill=0,
            dtype="uint8",
        )

    def _artifact_directory(self, analysis_id: UUID) -> Path:
        root = Path(self._settings.artifact_root).resolve()
        target = (root / str(analysis_id)).resolve()
        if not str(target).startswith(str(root) + os.sep):
            raise PermanentProcessingError(
                "Artifact path resolution failed.", failure_code=FailureCode.INTERNAL_ERROR
            )
        return target


__all__ = ["AnalysisPipeline", "PipelineOutcome", "ARTIFACT_FILENAMES", "validate_geojson"]