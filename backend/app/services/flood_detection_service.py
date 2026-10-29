from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from app.core.config import Settings, get_settings
from app.core.exceptions import PermanentProcessingError
from app.core.logging import get_logger
from app.processing.validators import AreaOfInterest
from app.services.geospatial_service import (
    TargetGrid,
    measure_geometry,
    polygonize_mask,
)
from app.services.prediction_providers import (
    FLOOD_CLASS,
    SYNTHETIC_LIMITATION,
    SYNTHETIC_WARNING,
    TRAINED_DATA_LABEL,
    PredictionProvider,
    ProviderPrediction,
    validate_prediction,
)

logger = get_logger("app.flood_detection")

ALGORITHM_VERSION = "1.0.0"


@dataclass
class FloodDetectionResult:
    geometry: Any
    method: str
    algorithm: dict[str, Any]
    measurements: dict[str, Any]
    confidence: dict[str, Any]
    raw_mask: np.ndarray | None = None
    valid_mask: np.ndarray | None = None
    provider_id: str = "trained_model"
    data_label: str = TRAINED_DATA_LABEL
    synthetic: bool = False
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    imagery: list[dict[str, Any]] = field(default_factory=list)
    grid: TargetGrid | None = None
    model_info: dict[str, Any] | None = None
    preprocessing_steps: list[str] = field(default_factory=list)

MODEL_ALGORITHM = {
    "name": "FloodSight AI custom U-Net flood segmentation",
    "version": ALGORITHM_VERSION,
    "reference": (
        "Two-band Sentinel-1 VV/VH encoder-decoder U-Net trained on preprocessed "
        "decibel rasters."
    ),
    "steps": [
        "Acquire co-registered Sentinel-1 GRD RTC observations around the flood date.",
        "Convert linear backscatter power to decibels.",
        "Apply a 3x3 Lee-sigma speckle filter and clip to the -40 dB to 0 dB training range.",
        "Reproject VV and VH onto one aligned analysis grid.",
        "Normalise with the checkpoint training statistics, clipped at three standard deviations.",
        "Run tiled sliding-window inference and average overlapping tiles in probability space.",
        "Threshold the flood class probability and restrict output to valid radar coverage.",
    ],
}

SYNTHETIC_ALGORITHM = {
    "name": "FloodSight AI synthetic test provider",
    "version": ALGORITHM_VERSION,
    "reference": (
        "Deterministic synthetic_test_data mask generated from a seeded pattern. No "
        "trained weights are loaded and no flood detection is performed."
    ),
    "steps": [
        "Acquire co-registered Sentinel-1 GRD RTC observations around the flood date.",
        "Convert linear backscatter power to decibels.",
        "Apply a 3x3 Lee-sigma speckle filter and clip to the -40 dB to 0 dB training range.",
        "Reproject VV and VH onto one aligned analysis grid.",
        "Generate a deterministic synthetic_test_data mask from the configured pattern and seed.",
        "Threshold the mask and restrict output to valid radar coverage.",
    ],
}

MODEL_LIMITATIONS = [
    "Flood extents are model predictions derived from radar imagery, not confirmed ground truth.",
    "The model was trained on Sentinel-1 VV/VH dB rasters; imagery with different "
    "calibration or processing may produce unreliable predictions.",
    "Urban layover, double-bounce returns and permanent water bodies can produce false positives.",
    "Shallow or vegetated flooding with weak radar response may be missed.",
]

#: Analysis method recorded for each provider, so results are never ambiguous.
METHOD_TRAINED_MODEL = "custom_unet"
METHOD_SYNTHETIC = "synthetic_test_data"


def detect_flood(
    provider: PredictionProvider,
    *,
    scene_path,
    pre_scene_path,
    grid: TargetGrid,
    aoi: AreaOfInterest,
    settings: Settings | None = None,
    cancellation_check=None,
    min_polygon_area_m2: float | None = None,
) -> FloodDetectionResult:
    """Run a prediction provider over a prepared two band dB raster.

    The raw provider mask is preserved untouched. Minimum-area polygonisation only
    shapes the vector geometry used for area and map output, so filtering small
    patches never erases pixels from the stored model-output raster.
    """
    from app.services.preprocessing_service import PREPROCESSING_STEPS, read_prepared_scene

    cfg = settings or get_settings()
    min_area_m2 = (
        float(min_polygon_area_m2)
        if min_polygon_area_m2 is not None and min_polygon_area_m2 > 0
        else cfg.min_flood_polygon_area_m2
    )
    image, valid_mask, transform, crs = read_prepared_scene(scene_path)
    pre_image = None
    if pre_scene_path is not None:
        pre_image, pre_valid, pre_transform, pre_crs = read_prepared_scene(pre_scene_path)
        if pre_image.shape != image.shape or pre_crs != crs or pre_transform != transform:
            raise PermanentProcessingError(
                "Pre-event and post-event rasters must share CRS, dimensions and pixel "
                "alignment for the temporal consistency filter."
            )
        pre_image = pre_image[:, :, 0]

    if cancellation_check is not None:
        cancellation_check()

    def progress(done: int, total: int) -> None:
        if cancellation_check is not None and done % max(1, total // 4) == 0:
            cancellation_check()

    prediction = provider.predict(
        image, transform, crs, valid_mask, pre_image_db=pre_image, progress=progress
    )
    prediction = validate_prediction(
        prediction,
        expected_shape=grid.shape,
        expected_transform=grid.transform,
        expected_crs=grid.crs,
    )

    raw_mask = prediction.mask
    flood = (raw_mask == FLOOD_CLASS) & valid_mask
    valid_count = int(valid_mask.sum())
    flood_count = int(flood.sum())

    geometry, poly_stats = polygonize_mask(flood, grid, min_area_m2=min_area_m2)
    retained_pixels = int(poly_stats["pixel_count"])
    # Percentage is measured against the valid analysed area so that nodata pixels
    # never inflate or deflate the flooded share.
    valid_area_m2 = valid_count * grid.pixel_size**2
    measurements = measure_geometry(geometry, aoi.area_m2, valid_area_m2=valid_area_m2)
    measurements.update(
        {
            "valid_pixel_count": valid_count,
            # The raw provider mask is authoritative for pixel counts; the filtered
            # geometry only shapes the reported area.
            "flood_pixel_count": flood_count,
            "retained_flood_pixel_count": retained_pixels,
            "filtered_out_pixel_count": max(0, flood_count - retained_pixels),
            "resolution_m": round(grid.pixel_size, 3),
            "valid_coverage_fraction": (
                round(valid_count / float(valid_mask.size), 6) if valid_mask.size else None
            ),
            "retained_polygons": int(poly_stats["retained_polygons"]),
            "dropped_polygons": int(poly_stats["dropped_polygons"]),
            "min_flood_polygon_area_m2": min_area_m2,
            "coverage_fraction_of_valid": (
                round(flood_count / valid_count, 6) if valid_count else None
            ),
            "model_tiles": prediction.inference.get("tiles"),
            "prediction_provider": prediction.provenance.get("provider"),
            "prediction_data_label": prediction.data_label,
            "predictions_are_synthetic": bool(prediction.synthetic),
        }
    )

    warnings: list[str] = []
    if prediction.synthetic:
        warnings.append(SYNTHETIC_WARNING)
    if flood_count == 0:
        warnings.append(
            "The model classified no pixel as flooded in the covered portion of this area."
        )
    if valid_count and (valid_count / float(valid_mask.size)) < 0.5:
        warnings.append(
            "Less than half of the analysis grid had valid radar coverage; the reported "
            "extent describes only the covered area."
        )

    if prediction.synthetic:
        confidence = {
            "definition": (
                "Synthetic test data. No probability of flooding is expressed because no "
                "trained model was run."
            ),
            "confidence_threshold": None,
            "validation_flood_iou": None,
            "validation_iou_available": False,
            "mean_flood_probability_in_mask": None,
        }
    else:
        confidence = {
            "definition": (
                "This is a model segmentation output. FloodSight reports the configured "
                "probability threshold, not a calibrated per-pixel probability of flooding."
            ),
            "confidence_threshold": prediction.inference.get("confidence_threshold"),
            "validation_flood_iou": prediction.provenance.get("val_flood_iou"),
            "validation_iou_available": prediction.provenance.get("val_flood_iou") is not None,
            "mean_flood_probability_in_mask": (
                round(
                    float(prediction.probabilities[FLOOD_CLASS][flood].mean()),
                    4,
                )
                if flood_count and prediction.probabilities is not None
                else None
            ),
        }

    return FloodDetectionResult(
        geometry=geometry,
        method=METHOD_SYNTHETIC if prediction.synthetic else METHOD_TRAINED_MODEL,
        algorithm=SYNTHETIC_ALGORITHM if prediction.synthetic else MODEL_ALGORITHM,
        measurements=measurements,
        confidence=confidence,
        raw_mask=raw_mask,
        valid_mask=valid_mask,
        provider_id=str(prediction.provenance.get("provider") or provider.provider_id),
        data_label=prediction.data_label,
        synthetic=bool(prediction.synthetic),
        warnings=warnings,
        limitations=[SYNTHETIC_LIMITATION] if prediction.synthetic else list(MODEL_LIMITATIONS),
        imagery=[],
        grid=grid,
        model_info=prediction.provenance,
        preprocessing_steps=list(PREPROCESSING_STEPS),
    )


def detect_flood_with_model(
    model,
    *,
    scene_path,
    pre_scene_path,
    grid: TargetGrid,
    aoi: AreaOfInterest,
    settings: Settings | None = None,
    cancellation_check=None,
    min_polygon_area_m2: float | None = None,
) -> FloodDetectionResult:
    """Backwards compatible entry point for callers holding a loaded model."""
    from app.services.prediction_providers import TrainedModelProvider

    class _Loaded(TrainedModelProvider):
        def __init__(self, loaded) -> None:
            self._settings = settings or get_settings()
            self._model = loaded

    return detect_flood(
        _Loaded(model),
        scene_path=scene_path,
        pre_scene_path=pre_scene_path,
        grid=grid,
        aoi=aoi,
        settings=settings,
        cancellation_check=cancellation_check,
        min_polygon_area_m2=min_polygon_area_m2,
    )


__all__ = [
    "ALGORITHM_VERSION",
    "FloodDetectionResult",
    "METHOD_SYNTHETIC",
    "METHOD_TRAINED_MODEL",
    "MODEL_ALGORITHM",
    "MODEL_LIMITATIONS",
    "ProviderPrediction",
    "SYNTHETIC_ALGORITHM",
    "detect_flood",
    "detect_flood_with_model",
]
