from __future__ import annotations

import importlib.util
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import Settings, get_settings
from app.core.enums import FailureCode
from app.core.exceptions import (
    DependencyError,
    PermanentProcessingError,
    RetryableProcessingError,
)
from app.core.logging import get_logger

logger = get_logger("app.model")

MODEL_MODULE_NAME = "floodsight_aimodel_model"
NODATA_OUT = 255
FLOOD_CLASS = 1


@dataclass(frozen=True)
class ModelDescriptor:
    available: bool
    architecture: str = "FloodUNet"
    model_config: dict[str, Any] = field(default_factory=dict)
    checkpoint_name: str | None = None
    checkpoint_identifier: str | None = None
    patch_size: int | None = None
    stride: int | None = None
    device: str | None = None
    normalization: dict[str, Any] = field(default_factory=dict)
    val_flood_iou: float | None = None
    epoch: int | None = None
    temperature: float | None = None
    confidence_threshold: float | None = None
    input_bands: tuple[str, ...] = ("VV", "VH")
    input_units: str = "decibels (dB)"
    inference_mode: str = "single_scene"
    reason: str | None = None


@dataclass
class ModelPrediction:
    mask: np.ndarray
    probabilities: np.ndarray
    transform: Any
    crs: Any
    valid_mask: np.ndarray
    inference: dict[str, Any]


class ModelUnavailableError(DependencyError):
    code = "model_unavailable"
    status_code = 503

    def __init__(self, message: str) -> None:
        super().__init__(message, failure_code=FailureCode.DEPENDENCY_UNAVAILABLE)


def _load_model_module(repo_path: Path):
    """Import the trained model definition from the model repository.

    The repository is a flat script layout rather than an installable package, so
    ``model.py`` is loaded by path. The architecture itself is never redefined
    here; the repository module is the single source of truth.
    """
    module_file = repo_path / "model.py"
    if not module_file.is_file():
        raise ModelUnavailableError(
            "The FloodSight model repository does not expose model.py at the configured "
            "location. Set MODEL_REPO_PATH to a checkout of the aimodel branch."
        )
    cached = sys.modules.get(MODEL_MODULE_NAME)
    if cached is not None and getattr(cached, "__floodsight_source__", None) == str(module_file):
        return cached
    spec = importlib.util.spec_from_file_location(MODEL_MODULE_NAME, module_file)
    if spec is None or spec.loader is None:
        raise ModelUnavailableError("The model definition could not be loaded.")
    module = importlib.util.module_from_spec(spec)
    module.__floodsight_source__ = str(module_file)
    sys.modules[MODEL_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001
        sys.modules.pop(MODEL_MODULE_NAME, None)
        raise ModelUnavailableError(
            f"The model definition could not be imported ({type(exc).__name__})."
        ) from exc
    if not hasattr(module, "create_unet_model"):
        raise ModelUnavailableError(
            "The configured model repository does not define create_unet_model()."
        )
    return module


class FloodSegmentationModel:
    """Loads the custom-trained U-Net checkpoint and runs tiled inference.

    One instance is reused per process so weights are loaded once per worker
    rather than once per analysis request.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._lock = threading.Lock()
        self._model: Any = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._patch_size: int | None = None
        self._stride: int | None = None
        self._device: Any = None
        self._checkpoint: dict[str, Any] | None = None
        self._checkpoint_path: Path | None = None
        self._torch: Any = None

    def describe(self) -> ModelDescriptor:
        try:
            self.load()
        except Exception as exc:  # noqa: BLE001
            return self._unavailable_descriptor(str(getattr(exc, "public_message", exc)))
        assert self._checkpoint is not None
        return ModelDescriptor(
            available=True,
            model_config=dict(self._checkpoint.get("model_config", {})),
            checkpoint_name=self._checkpoint_path.name if self._checkpoint_path else None,
            checkpoint_identifier=self._checkpoint_identifier(),
            patch_size=self._patch_size,
            stride=self._stride,
            device=str(self._device),
            normalization={
                "mean_db": self._mean.tolist(),
                "std_db": self._std.tolist(),
            },
            val_flood_iou=self._checkpoint.get("val_flood_iou"),
            epoch=self._checkpoint.get("epoch"),
            temperature=self._settings.model_temperature,
            confidence_threshold=self._settings.model_confidence_threshold,
        )

    def _unavailable_descriptor(self, reason: str) -> ModelDescriptor:
        return ModelDescriptor(available=False, reason=reason)

    def _checkpoint_identifier(self) -> str | None:
        """Return a stable, non-path-revealing identifier for the checkpoint."""
        if self._checkpoint_path is None:
            return None
        import hashlib

        digest = hashlib.sha256(self._checkpoint_path.name.encode("utf-8")).hexdigest()
        return f"{self._checkpoint_path.stem}:{digest[:12]}"

    def is_available(self) -> bool:
        try:
            self.load()
        except Exception:  # noqa: BLE001
            return False
        return True

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            self._model, self._mean, self._std, self._device, self._checkpoint = (
                self._initialise()
            )
            patch = self._settings.model_patch_size or self._checkpoint.get("patch_size") or 256
            self._patch_size = int(patch)
            self._stride = int(self._settings.model_stride or max(1, self._patch_size // 2))
            logger.info(
                "model_loaded",
                extra={
                    "event": "model_loaded",
                    "architecture": "FloodUNet",
                    "checkpoint": self._checkpoint_path.name if self._checkpoint_path else None,
                    "checkpoint_id": self._checkpoint_identifier(),
                    "patch_size": self._patch_size,
                    "stride": self._stride,
                    "device": str(self._device),
                    "model_config": dict(self._checkpoint.get("model_config", {})),
                    "val_flood_iou": self._checkpoint.get("val_flood_iou"),
                },
            )

    def _initialise(self):
        checkpoint_path = self._settings.model_path
        if checkpoint_path is None:
            raise ModelUnavailableError(
                "No trained checkpoint is configured. Set MODEL_PATH to the FloodSight "
                "checkpoint (checkpoints/best.pt produced by train.py)."
            )
        if not checkpoint_path.is_file():
            raise ModelUnavailableError(
                "The configured model checkpoint does not exist. Train the model with "
                "train.py or point MODEL_PATH at a valid checkpoint."
            )
        try:
            import torch
        except ImportError as exc:
            raise ModelUnavailableError(
                "PyTorch is not installed, so model inference cannot run."
            ) from exc
        self._torch = torch

        module = _load_model_module(self._settings.model_repo_path)
        device = self._select_device(torch)

        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except Exception as exc:  # noqa: BLE001
            raise ModelUnavailableError(
                f"The model checkpoint could not be read ({type(exc).__name__})."
            ) from exc
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ModelUnavailableError(
                "The checkpoint does not contain a model_state_dict and is not a "
                "FloodSight training checkpoint."
            )

        model_config = checkpoint.get(
            "model_config", {"in_channels": 2, "classes": 2, "base_channels": 24}
        )
        in_channels = int(model_config.get("in_channels", 2))
        if in_channels != 2:
            raise ModelUnavailableError(
                "The checkpoint expects "
                f"{in_channels} input channels, but this application supplies two "
                "Sentinel-1 VV/VH bands. The checkpoint does not match the supported contract."
            )

        try:
            model = module.create_unet_model(**model_config).to(device)
        except Exception as exc:  # noqa: BLE001
            raise ModelUnavailableError(
                f"The model architecture could not be instantiated ({type(exc).__name__})."
            ) from exc
        try:
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        except Exception as exc:  # noqa: BLE001
            raise ModelUnavailableError(
                "The checkpoint weights are incompatible with the model architecture. "
                f"({type(exc).__name__})"
            ) from exc
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        norm = checkpoint.get("normalization", {})
        mean = np.array(norm.get("mean_db", [-12.5, -18.0]), dtype=np.float32)
        std = np.array(norm.get("std_db", [4.5, 4.0]), dtype=np.float32)
        if mean.shape != (2,) or std.shape != (2,) or np.any(std <= 0):
            raise ModelUnavailableError(
                "The checkpoint normalisation statistics are malformed; two positive standard "
                "deviations are required."
            )
        self._checkpoint_path = checkpoint_path
        return model, mean, std, device, checkpoint

    def _select_device(self, torch):
        requested = self._settings.model_device
        if requested == "cpu":
            return torch.device("cpu")
        if requested == "cuda":
            if not torch.cuda.is_available():
                raise ModelUnavailableError(
                    "MODEL_DEVICE requests cuda but no CUDA device is available."
                )
            return torch.device("cuda")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def normalise(self, image: np.ndarray) -> np.ndarray:
        """Z-score normalise a (H, W, 2) dB image exactly as training does.

        Values are clipped at three standard deviations before normalising to
        suppress radar outliers, matching dataset.normalise_sar and
        predict.normalize_scene.
        """
        mean = self._mean
        std = self._std
        assert mean is not None and std is not None
        low = mean - 3.0 * std
        high = mean + 3.0 * std
        clipped = np.clip(image, low, high)
        return ((clipped - mean) / (std + 1e-6)).astype(np.float32)

    def predict(
        self,
        image_db: np.ndarray,
        transform: Any,
        crs: Any,
        valid_mask: np.ndarray,
        *,
        pre_image_db: np.ndarray | None = None,
        progress: Any = None,
    ) -> ModelPrediction:
        """Run tiled inference over a (H, W, 2) dB raster.

        Overlapping tiles are averaged in probability space and the last row
        and column of tiles are aligned to the raster edge so no valid pixel is
        dropped, preserving the sliding-window contract in predict.py.
        """
        self.load()
        torch = self._torch
        assert torch is not None and self._model is not None
        if image_db.ndim != 3 or image_db.shape[2] != 2:
            raise PermanentProcessingError(
                "Model inference requires a two band (VV, VH) array; the prepared raster "
                f"provided {image_db.shape[2] if image_db.ndim == 3 else 1} band(s)."
            )
        height, width, _ = image_db.shape
        patch = int(self._patch_size or 256)
        stride = int(self._stride or max(1, patch // 2))

        pad_h = max(0, patch - height)
        pad_w = max(0, patch - width)
        if pad_h or pad_w:
            image_db = np.pad(
                image_db, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge"
            )

        normalised = self.normalise(image_db)
        if pre_image_db is not None:
            # Accept a single VV band as (H, W), (H, W, 1) or a full (H, W, 2) stack;
            # the temporal filter only needs the VV channel.
            if pre_image_db.ndim == 3:
                pre_image_db = pre_image_db[:, :, 0]
            if pre_image_db.shape != image_db.shape[:2]:
                raise PermanentProcessingError(
                    "Pre-event and post-event rasters must share dimensions and pixel "
                    "alignment for the temporal consistency filter."
                )
            pre_image_db = np.pad(pre_image_db, ((0, pad_h), (0, pad_w)), mode="edge")

        prob_map = np.zeros((2, normalised.shape[0], normalised.shape[1]), dtype=np.float32)
        count_map = np.zeros(normalised.shape[:2], dtype=np.float32)

        ys = _window_starts(normalised.shape[0], patch, stride)
        xs = _window_starts(normalised.shape[1], patch, stride)
        total = len(ys) * len(xs)
        completed = 0
        temperature = float(self._settings.model_temperature)
        batch_size = max(1, int(self._settings.model_batch_size))
        patches = [
            (y, x) for y in ys for x in xs
        ]

        with torch.inference_mode():
            for start in range(0, len(patches), batch_size):
                chunk = patches[start : start + batch_size]
                tensors = torch.from_numpy(
                    np.stack(
                        [
                            np.ascontiguousarray(
                                np.transpose(normalised[y : y + patch, x : x + patch, :], (2, 0, 1))
                            )
                            for y, x in chunk
                        ]
                    )
                ).to(self._device)
                logits = self._model(tensors)
                probabilities = torch.softmax(logits / temperature, dim=1).cpu().numpy()
                for index, (y, x) in enumerate(chunk):
                    prob_map[:, y : y + patch, x : x + patch] += probabilities[index]
                    count_map[y : y + patch, x : x + patch] += 1.0
                completed += len(chunk)
                if progress is not None and total:
                    progress(completed, total)

        count_map = np.maximum(count_map, 1.0)
        prob_map /= count_map[None, :, :]
        prob_map = prob_map[:, :height, :width]

        flood_probability = prob_map[FLOOD_CLASS]
        threshold = float(self._settings.model_confidence_threshold)
        candidate = flood_probability >= threshold
        if pre_image_db is not None:
            # A genuine flood signature is a large drop in VV backscatter.
            drop_db = pre_image_db - image_db[:, :, 0]
            candidate = candidate & (drop_db >= 1.5)
        mask = self._post_process(candidate)
        mask[~valid_mask] = NODATA_OUT

        inference = {
            "patch_size": patch,
            "stride": stride,
            "tiles": total,
            "temperature": temperature,
            "confidence_threshold": threshold,
            "min_cluster_px": int(self._settings.model_min_cluster_px),
            "temporal_filter_applied": pre_image_db is not None,
            "device": str(self._device),
        }
        return ModelPrediction(
            mask=mask,
            probabilities=prob_map,
            transform=transform,
            crs=crs,
            valid_mask=valid_mask,
            inference=inference,
        )

    def _post_process(self, candidate: np.ndarray) -> np.ndarray:
        min_cluster = int(self._settings.model_min_cluster_px)
        cleaned = candidate
        if min_cluster > 0:
            from skimage.morphology import remove_small_holes, remove_small_objects

            cleaned = remove_small_objects(cleaned, max_size=min_cluster)
            cleaned = remove_small_holes(cleaned, max_size=min_cluster)
        return cleaned.astype(np.uint8)


def _window_starts(extent: int, patch: int, stride: int) -> list[int]:
    if extent <= patch:
        return [0]
    starts = list(range(0, extent - patch + 1, stride))
    if starts[-1] != extent - patch:
        starts.append(extent - patch)
    return starts


_registry_lock = threading.Lock()
_registry: dict[str, FloodSegmentationModel] = {}


def get_segmentation_model(settings: Settings | None = None) -> FloodSegmentationModel:
    """Return the process-wide model adapter, loading weights at most once."""
    cfg = settings or get_settings()
    key = f"{cfg.model_path}|{cfg.model_repo_path}|{cfg.model_device}"
    with _registry_lock:
        model = _registry.get(key)
        if model is None:
            model = FloodSegmentationModel(cfg)
            _registry[key] = model
        return model


def reset_model_registry() -> None:
    with _registry_lock:
        _registry.clear()


__all__ = [
    "FloodSegmentationModel",
    "ModelDescriptor",
    "ModelPrediction",
    "ModelUnavailableError",
    "get_segmentation_model",
    "reset_model_registry",
    "RetryableProcessingError",
]