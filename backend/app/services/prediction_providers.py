"""Prediction providers for the flood-mapping stage.

The pipeline needs a binary flood mask from a prepared two band Sentinel-1 VV/VH
raster. *How* that mask is produced is deliberately isolated behind
:class:`PredictionProvider` so the pipeline never depends on the trained weights
being present.

Two providers are implemented:

``trained_model``
    The real FloodSight U-Net, loaded from a checkpoint via
    :mod:`app.services.model_service`. Fails loudly when the checkpoint is missing
    or incompatible.

``synthetic_test``
    A deterministic, explicitly labelled stand-in that exercises the identical
    downstream geospatial path without any trained weights. Every result it produces
    is tagged ``synthetic_test_data``; it is never presented as a flood prediction.

Selection is always explicit. The synthetic provider is never used as a fallback
when the trained model is unavailable, so an absent checkpoint surfaces as an error
instead of silently producing meaningless extents.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np

from app.core.config import Settings, get_settings
from app.core.exceptions import PermanentProcessingError
from app.core.logging import get_logger

logger = get_logger("app.prediction_provider")

PROVIDER_TRAINED_MODEL = "trained_model"
PROVIDER_SYNTHETIC = "synthetic_test"
PROVIDER_CHOICES: frozenset[str] = frozenset({PROVIDER_TRAINED_MODEL, PROVIDER_SYNTHETIC})

SYNTHETIC_PATTERN_CHOICES: frozenset[str] = frozenset(
    {"blocky", "blobs", "gradient", "checkerboard"}
)

#: Value stamped onto every synthetic result, artifact and API payload.
SYNTHETIC_DATA_LABEL = "synthetic_test_data"
#: Value stamped onto genuine model inference.
TRAINED_DATA_LABEL = "model_prediction"

NODATA_OUT = 255
FLOOD_CLASS = 1

SYNTHETIC_WARNING = (
    "This analysis was produced by the synthetic test provider. The mask is a "
    "deterministic test pattern, not a flood prediction, and none of the reported "
    "areas, counts or extents describe real flooding."
)
SYNTHETIC_LIMITATION = (
    "Every measurement in this analysis is derived from synthetic_test_data. No "
    "trained model weights were used and no flood extent was detected."
)


class PredictionProviderError(PermanentProcessingError):
    """Raised when a provider cannot produce a usable prediction."""

    failure_code = "dependency_unavailable"


@dataclass(frozen=True)
class ProviderDescriptor:
    """Readiness and provenance of a provider, surfaced by ``/health/model``."""

    provider_id: str
    data_label: str
    synthetic: bool
    available: bool
    reason: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "provider": self.provider_id,
            "data_label": self.data_label,
            "synthetic": self.synthetic,
            "status": "ok" if self.available else "unavailable",
        }
        if not self.available and self.reason:
            payload["reason"] = self.reason
        payload.update(self.detail)
        return payload


@dataclass
class ProviderPrediction:
    """A binary prediction plus everything needed to audit and georeference it."""

    mask: np.ndarray
    transform: Any
    crs: Any
    valid_mask: np.ndarray
    inference: dict[str, Any]
    provenance: dict[str, Any]
    synthetic: bool = False
    probabilities: np.ndarray | None = None

    @property
    def data_label(self) -> str:
        return SYNTHETIC_DATA_LABEL if self.synthetic else TRAINED_DATA_LABEL


@runtime_checkable
class PredictionProvider(Protocol):
    """The contract the flood-mapping stage depends on."""

    provider_id: str
    synthetic: bool

    def describe(self) -> ProviderDescriptor:
        """Return readiness without raising, for health reporting."""
        ...

    def ensure_available(self) -> ProviderDescriptor:
        """Load or validate the provider, raising if it cannot produce predictions."""
        ...

    def predict(
        self,
        image_db: np.ndarray,
        transform: Any,
        crs: Any,
        valid_mask: np.ndarray,
        *,
        pre_image_db: np.ndarray | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> ProviderPrediction:
        """Return a binary mask aligned to the prepared raster."""
        ...


def validate_prediction(
    prediction: ProviderPrediction,
    *,
    expected_shape: tuple[int, int],
    expected_transform: Any,
    expected_crs: Any,
) -> ProviderPrediction:
    """Fail loudly when a provider returns something the pipeline cannot georeference.

    A mask that drifts from its grid silently corrupts every downstream area, so the
    contract is checked rather than assumed.
    """
    mask = prediction.mask
    if mask.ndim != 2:
        raise PredictionProviderError(
            f"The prediction provider returned a {mask.ndim}-dimensional mask; a single "
            "two-dimensional band is required."
        )
    if tuple(mask.shape) != tuple(expected_shape):
        raise PredictionProviderError(
            "The prediction provider returned a mask whose shape does not match the "
            f"analysis grid ({tuple(mask.shape)} against {tuple(expected_shape)})."
        )
    if tuple(prediction.valid_mask.shape) != tuple(expected_shape):
        raise PredictionProviderError(
            "The prediction provider returned a valid-data mask that does not match the "
            "analysis grid."
        )
    if prediction.transform is None or prediction.crs is None:
        raise PredictionProviderError(
            "The prediction provider returned a mask without georeferencing information."
        )
    if tuple(np.asarray(prediction.transform).ravel()[:6]) != tuple(
        np.asarray(expected_transform).ravel()[:6]
    ):
        raise PredictionProviderError(
            "The prediction provider returned a mask whose affine transform does not match "
            "the analysis grid."
        )
    if str(prediction.crs) != str(expected_crs):
        raise PredictionProviderError(
            "The prediction provider returned a mask in a different coordinate reference "
            f"system ({prediction.crs} against {expected_crs})."
        )
    unexpected = set(np.unique(mask).tolist()) - {0, 1, NODATA_OUT}
    if unexpected:
        raise PredictionProviderError(
            "The prediction provider returned values outside the expected class set: "
            f"{sorted(unexpected)}."
        )
    if not prediction.valid_mask[mask == FLOOD_CLASS].all():
        raise PredictionProviderError(
            "The prediction provider marked flooded pixels outside the valid data mask."
        )
    return prediction


class TrainedModelProvider:
    """Provider backed by the real trained FloodSight U-Net checkpoint."""

    provider_id = PROVIDER_TRAINED_MODEL
    synthetic = False

    def __init__(self, settings: Settings | None = None) -> None:
        from app.services.model_service import get_segmentation_model

        self._settings = settings or get_settings()
        self._model = get_segmentation_model(self._settings)

    def describe(self) -> ProviderDescriptor:
        from app.services.model_service import ModelUnavailableError

        try:
            descriptor = self._model.describe()
        except Exception as exc:  # noqa: BLE001
            return ProviderDescriptor(
                provider_id=self.provider_id,
                data_label=TRAINED_DATA_LABEL,
                synthetic=False,
                available=False,
                reason=str(getattr(exc, "public_message", exc)),
            )
        if not descriptor.available:
            return ProviderDescriptor(
                provider_id=self.provider_id,
                data_label=TRAINED_DATA_LABEL,
                synthetic=False,
                available=False,
                reason=descriptor.reason,
            )
        detail = {
            "architecture": descriptor.architecture,
            "model_config": descriptor.model_config,
            "checkpoint": descriptor.checkpoint_identifier,
            "checkpoint_name": descriptor.checkpoint_name,
            "device": descriptor.device,
            "patch_size": descriptor.patch_size,
            "stride": descriptor.stride,
            "input_bands": list(descriptor.input_bands),
            "input_units": descriptor.input_units,
            "normalization": descriptor.normalization,
            "val_flood_iou": descriptor.val_flood_iou,
            "epoch": descriptor.epoch,
            "confidence_threshold": descriptor.confidence_threshold,
        }
        if descriptor.val_flood_iou is None:
            detail["note"] = (
                "The loaded checkpoint does not record a validation IoU, so the accuracy of "
                "these weights cannot be verified from the checkpoint."
            )
        del ModelUnavailableError
        return ProviderDescriptor(
            provider_id=self.provider_id,
            data_label=TRAINED_DATA_LABEL,
            synthetic=False,
            available=True,
            detail=detail,
        )

    def ensure_available(self) -> ProviderDescriptor:
        descriptor = self.describe()
        if not descriptor.available:
            raise PredictionProviderError(
                "Flood extent mapping is unavailable: "
                f"{descriptor.reason or 'the trained model could not be loaded.'}"
            )
        return descriptor

    def predict(
        self,
        image_db: np.ndarray,
        transform: Any,
        crs: Any,
        valid_mask: np.ndarray,
        *,
        pre_image_db: np.ndarray | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> ProviderPrediction:
        self.ensure_available()
        result = self._model.predict(
            image_db,
            transform,
            crs,
            valid_mask,
            pre_image_db=pre_image_db,
            progress=progress,
        )
        descriptor = self._model.describe()
        return ProviderPrediction(
            mask=result.mask,
            probabilities=result.probabilities,
            transform=result.transform,
            crs=result.crs,
            valid_mask=result.valid_mask,
            inference=dict(result.inference),
            provenance={
                "provider": self.provider_id,
                "data_label": TRAINED_DATA_LABEL,
                "synthetic": False,
                "architecture": descriptor.architecture,
                "model_config": descriptor.model_config,
                "checkpoint": descriptor.checkpoint_identifier,
                "checkpoint_name": descriptor.checkpoint_name,
                "val_flood_iou": descriptor.val_flood_iou,
                "epoch": descriptor.epoch,
                "device": descriptor.device,
                "patch_size": descriptor.patch_size,
                "stride": descriptor.stride,
                "temperature": result.inference.get("temperature"),
                "confidence_threshold": result.inference.get("confidence_threshold"),
                "min_cluster_px": result.inference.get("min_cluster_px"),
                "temporal_filter_applied": result.inference.get("temporal_filter_applied"),
                "input_bands": list(descriptor.input_bands),
                "input_units": descriptor.input_units,
                "tiles": result.inference.get("tiles"),
                "normalization": descriptor.normalization,
            },
            synthetic=False,
        )


class SyntheticTestProvider:
    """Deterministic stand-in that exercises the pipeline without trained weights.

    The mask is derived from the prepared raster's own geometry and a fixed seed, so
    repeated runs over the same scene are byte-identical. Output is never a claim
    about flooding: it is labelled ``synthetic_test_data`` everywhere.
    """

    provider_id = PROVIDER_SYNTHETIC
    synthetic = True

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def describe(self) -> ProviderDescriptor:
        return ProviderDescriptor(
            provider_id=self.provider_id,
            data_label=SYNTHETIC_DATA_LABEL,
            synthetic=True,
            available=True,
            detail={
                "pattern": self._settings.synthetic_pattern,
                "seed": self._settings.synthetic_seed,
                "target_coverage": self._settings.synthetic_coverage,
                "min_blob_px": self._settings.synthetic_min_blob_px,
                "temporal_filter_applied": self._settings.synthetic_apply_temporal_filter,
                "warning": SYNTHETIC_WARNING,
            },
        )

    def ensure_available(self) -> ProviderDescriptor:
        return self.describe()

    def predict(
        self,
        image_db: np.ndarray,
        transform: Any,
        crs: Any,
        valid_mask: np.ndarray,
        *,
        pre_image_db: np.ndarray | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> ProviderPrediction:
        if image_db.ndim != 3 or image_db.shape[2] != 2:
            raise PredictionProviderError(
                "The synthetic provider requires the prepared two band (VV, VH) raster."
            )
        if pre_image_db is not None and pre_image_db.ndim == 3:
            pre_image_db = pre_image_db[:, :, 0]
        if (
            pre_image_db is not None
            and tuple(pre_image_db.shape) != tuple(image_db.shape[:2])
        ):
            raise PredictionProviderError(
                "Pre-event and post-event rasters must share dimensions and pixel "
                "alignment for the temporal consistency filter."
            )

        height, width = image_db.shape[:2]
        if progress is not None:
            progress(0, 1)

        field = self._pattern_field(height, width)
        candidate = field >= np.quantile(field, 1.0 - self._settings.synthetic_coverage)
        if self._settings.synthetic_min_blob_px > 0:
            candidate = _keep_large_regions(
                candidate, self._settings.synthetic_min_blob_px
            )

        if pre_image_db is not None and self._settings.synthetic_apply_temporal_filter:
            # Mirror the trained provider so downstream behaviour stays comparable.
            drop_db = pre_image_db - image_db[:, :, 0]
            candidate = candidate & (drop_db >= 1.5)

        mask = candidate.astype(np.uint8)
        mask[~valid_mask] = NODATA_OUT

        if progress is not None:
            progress(1, 1)

        probabilities = np.zeros((2, height, width), dtype=np.float32)
        probabilities[FLOOD_CLASS] = candidate.astype(np.float32)
        probabilities[0] = 1.0 - probabilities[FLOOD_CLASS]

        return ProviderPrediction(
            mask=mask,
            probabilities=probabilities,
            transform=transform,
            crs=crs,
            valid_mask=valid_mask,
            inference={
                "provider": self.provider_id,
                "pattern": self._settings.synthetic_pattern,
                "seed": self._settings.synthetic_seed,
                "target_coverage": self._settings.synthetic_coverage,
                "tiles": 1,
                "temperature": None,
                "confidence_threshold": None,
                "min_cluster_px": self._settings.synthetic_min_blob_px,
                "temporal_filter_applied": (
                    pre_image_db is not None
                    and self._settings.synthetic_apply_temporal_filter
                ),
                "device": "cpu",
                "synthetic": True,
            },
            provenance={
                "provider": self.provider_id,
                "data_label": SYNTHETIC_DATA_LABEL,
                "synthetic": True,
                "pattern": self._settings.synthetic_pattern,
                "seed": self._settings.synthetic_seed,
                "target_coverage": self._settings.synthetic_coverage,
                "min_blob_px": self._settings.synthetic_min_blob_px,
                "tiles": 1,
                "architecture": None,
                "checkpoint": None,
                "checkpoint_name": None,
                "val_flood_iou": None,
                "temporal_filter_applied": (
                    pre_image_db is not None
                    and self._settings.synthetic_apply_temporal_filter
                ),
                "device": "cpu",
                "warning": SYNTHETIC_WARNING,
                "note": (
                    "No trained checkpoint was loaded. This mask is a deterministic test "
                    "pattern used to exercise the processing pipeline."
                ),
            },
            synthetic=True,
        )

    def _pattern_field(self, height: int, width: int) -> np.ndarray:
        """Build the deterministic field the mask is thresholded from."""
        pattern = self._settings.synthetic_pattern
        seed = self._settings.synthetic_seed
        rng = np.random.default_rng(seed)
        rows = np.arange(height, dtype=np.float32)[:, None]
        cols = np.arange(width, dtype=np.float32)[None, :]

        if pattern == "gradient":
            span = max(height + width, 1)
            return ((rows + cols) / span).astype(np.float32)
        if pattern == "checkerboard":
            block = max(4, int(self._settings.synthetic_min_blob_px**0.5))
            return (
                ((rows.astype(np.int64) // block + cols.astype(np.int64) // block) % 2)
                .astype(np.float32)
            )
        if pattern == "blobs":
            return self._blob_field(height, width, rng)
        return self._blocky_field(height, width, rng)

    def _blocky_field(self, height: int, width: int, rng: np.random.Generator) -> np.ndarray:
        """Low frequency noise, so the mask forms connected regions rather than speckle."""
        from scipy.ndimage import zoom

        scale = 12.0
        coarse_h = max(2, int(height / scale))
        coarse_w = max(2, int(width / scale))
        coarse = rng.random((coarse_h, coarse_w)).astype(np.float32)
        row_zoom = height / coarse_h
        col_zoom = width / coarse_w
        return zoom(coarse, (row_zoom, col_zoom), order=1, mode="nearest").astype(
            np.float32
        )[:height, :width]

    def _blob_field(self, height: int, width: int, rng: np.random.Generator) -> np.ndarray:
        """Explicit circular blobs at seeded positions."""
        field = np.zeros((height, width), dtype=np.float32)
        centre_x, centre_y = width / 2.0, height / 2.0
        count = 12
        base = max(self._settings.synthetic_min_blob_px, 8)
        for index in range(count):
            angle = 2.0 * np.pi * index / count
            offset = 0.18 + 0.22 * rng.random()
            radius = base * (1.5 + 3.5 * rng.random())
            cx = centre_x + np.cos(angle) * centre_x * offset
            cy = centre_y + np.sin(angle) * centre_y * offset
            yy = (np.arange(height, dtype=np.float32)[:, None] - cy) ** 2
            xx = (np.arange(width, dtype=np.float32)[None, :] - cx) ** 2
            field = np.maximum(field, 1.0 - (yy + xx) / (radius * radius))
        return field


def _keep_large_regions(candidate: np.ndarray, min_px: int) -> np.ndarray:
    """Drop connected regions smaller than ``min_px`` so polygonisation is exercised."""
    from skimage.measure import label as label_regions

    if not candidate.any():
        return candidate
    # connectivity=2 is 8-connectivity for a 2D array; 4 is not a valid value.
    labels = label_regions(candidate, connectivity=2)
    if labels.max() == 0:
        return candidate
    counts = np.bincount(labels.ravel())
    keep = counts >= min_px
    keep[0] = False
    return keep[labels]


def resolve_provider_id(settings: Settings, override: str | None = None) -> str:
    """Resolve which provider an analysis must use.

    An override is only honoured outside production, and an unknown value is an
    error rather than a silent fallback to the configured provider.
    """
    if override is None:
        return settings.prediction_provider
    normalised = str(override).strip().lower()
    if normalised not in PROVIDER_CHOICES:
        raise PredictionProviderError(
            f"Unknown prediction provider '{override}'. Expected one of "
            f"{sorted(PROVIDER_CHOICES)}."
        )
    if normalised == PROVIDER_SYNTHETIC and not settings.allows_per_request_prediction_provider:
        raise PredictionProviderError(
            "The synthetic test provider cannot be selected by an analysis request in "
            f"the '{settings.environment}' environment. Select it in local configuration "
            "instead."
        )
    if (
        normalised == PROVIDER_SYNTHETIC
        and settings.is_production
        and not settings.allow_synthetic_provider_in_production
    ):
        raise PredictionProviderError(
            "The synthetic test provider emits synthetic_test_data and cannot run in "
            "production."
        )
    return normalised


_registry_lock = threading.Lock()
_registry: dict[tuple[str, str, str], PredictionProvider] = {}


def get_prediction_provider(
    settings: Settings | None = None, override: str | None = None
) -> PredictionProvider:
    """Return a process-wide provider instance for the resolved provider id."""
    cfg = settings or get_settings()
    provider_id = resolve_provider_id(cfg, override)
    key = (provider_id, str(cfg.model_path or ""), cfg.synthetic_pattern)
    with _registry_lock:
        provider = _registry.get(key)
        if provider is None:
            provider = (
                SyntheticTestProvider(cfg)
                if provider_id == PROVIDER_SYNTHETIC
                else TrainedModelProvider(cfg)
            )
            _registry[key] = provider
            logger.info(
                "prediction_provider_resolved",
                extra={
                    "event": "prediction_provider_resolved",
                    "provider": provider_id,
                    "synthetic": provider.synthetic,
                },
            )
        return provider


def reset_prediction_provider_registry() -> None:
    with _registry_lock:
        _registry.clear()


__all__ = [
    "FLOOD_CLASS",
    "NODATA_OUT",
    "PROVIDER_CHOICES",
    "PROVIDER_SYNTHETIC",
    "PROVIDER_TRAINED_MODEL",
    "ProviderDescriptor",
    "ProviderPrediction",
    "PredictionProvider",
    "PredictionProviderError",
    "SYNTHETIC_DATA_LABEL",
    "SYNTHETIC_LIMITATION",
    "SYNTHETIC_PATTERN_CHOICES",
    "SYNTHETIC_WARNING",
    "SyntheticTestProvider",
    "TRAINED_DATA_LABEL",
    "TrainedModelProvider",
    "get_prediction_provider",
    "reset_prediction_provider_registry",
    "resolve_provider_id",
    "validate_prediction",
]