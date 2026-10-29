"""
predict.py  -  FloodSight AI | Inference & facts.json Generator
================================================================
Loads a trained checkpoint (checkpoints/best.pt) and runs inference on a
preprocessed Sentinel-1 scene GeoTIFF to produce:

  1. flood_mask_predicted.tif  - binary flood mask (0/1/255)
  2. *_uncertainty.tif         - normalized predictive entropy, not calibrated
  3. facts.json                - aggregated flood statistics consumed by the
                                 FloodSight dashboard and situation report

The facts.json schema (matches the rest of the FloodSight AI pipeline):
{
  "flood_area_km2":     float,   -- total flooded area in km2
  "flood_pixel_count":  int,     -- raw pixel count of flooded cells
  "total_valid_pixels": int,     -- pixels that were not nodata
  "flood_fraction":     float,   -- flooded / total_valid (0-1)
  "bbox":               [west, south, east, north],  -- WGS-84 bounding box
  "crs":                source raster CRS,
  "bbox_crs":           "EPSG:4326" when the source CRS is known,
  "pixel_area_m2":      float,   -- area only for plausible CRS and pixel scale
  "pixel_resolution_approx_m": float,
  "model_config":       {...},   -- stored from checkpoint
  "normalization":      {...},   -- stored from checkpoint
  "checkpoint":         str      -- filename of the used checkpoint
}

Usage:
    python predict.py \
        --scene     data/processed/images/event_001_post.tif \
        --checkpoint checkpoints/best.pt \
        --out-dir   outputs/event_001

    # Predict on held-out manifest scenes:
    python predict.py \
        --manifest  checkpoints/val_manifest.json \
        --checkpoint checkpoints/best.pt \
        --out-dir   outputs/predictions

    # Or predict on the whole images/ folder:
    python predict.py \
        --scene-dir data/processed/images \
        --checkpoint checkpoints/best.pt \
        --out-dir   outputs/batch_run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.transform import array_bounds
from rasterio.warp import transform as warp_transform, transform_bounds
import warnings
from skimage.morphology import remove_small_holes, remove_small_objects

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import torch
    from model import create_unet_model
    _HAS_TORCH = True
except ImportError:
    class _DummyTorch:
        @staticmethod
        def no_grad():
            def decorator(fn):
                return fn
            return decorator
    torch = _DummyTorch()
    create_unet_model = None
    _HAS_TORCH = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NODATA_OUT    = 255
FLOOD_CLASS   = 1
DEFAULT_RES_M = 10.0   # Sentinel-1 GRD IW: ~10 m ground spacing


def default_checkpoint_path() -> Path:
    """Prefer the checkpoint evaluated on a completely held-out event."""
    candidates = (
        Path("checkpoints/hackathon_holdout_1111005/best.pt"),
        Path("checkpoints/official_run/best.pt"),
        Path("checkpoints/best.pt"),
    )
    return next((path for path in candidates if path.is_file()), candidates[-1])


def default_confidence_threshold(checkpoint_path: Path) -> float:
    """Use the threshold calibrated for this checkpoint, or neutral argmax fallback."""
    override = os.environ.get("FLOODSIGHT_THRESHOLD")
    if override is not None:
        return float(override)
    calibration_path = checkpoint_path.parent / "threshold_calibration.json"
    if calibration_path.is_file():
        with calibration_path.open("r", encoding="utf-8") as handle:
            calibration = json.load(handle)
        threshold = calibration.get("selected_threshold")
        if isinstance(threshold, (int, float)) and 0 <= threshold <= 1:
            return float(threshold)
    return 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_checkpoint(ckpt_path: Path, device: torch.device):
    """Load model + normalization stats from a FloodSight checkpoint."""
    # Permit only NumPy's inert dtype/array reconstruction primitives used by
    # existing training RNG metadata; arbitrary globals remain rejected.
    numpy_core = getattr(np, "_core", None)
    multiarray = numpy_core.multiarray if numpy_core is not None else np.core.multiarray
    safe_numpy = [multiarray._reconstruct, np.ndarray, np.dtype]
    for dtype_name in ("UInt32DType", "Float32DType", "Int64DType", "Float64DType"):
        dtype_type = getattr(getattr(np, "dtypes", None), dtype_name, None)
        if dtype_type is not None:
            safe_numpy.append(dtype_type)
    with torch.serialization.safe_globals(safe_numpy):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    digest = hashlib.sha256()
    with open(ckpt_path, "rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    ckpt["checkpoint_sha256"] = digest.hexdigest()
    required = {"model_state_dict"}
    missing = required.difference(ckpt)
    if missing:
        raise ValueError(f"Invalid checkpoint {ckpt_path}: missing {', '.join(sorted(missing))}")

    cfg   = ckpt.get("model_config", {"in_channels": 2, "classes": 2, "base_channels": 24})
    if int(cfg.get("classes", 2)) != 2:
        raise ValueError(f"Flood segmentation checkpoint must have exactly 2 classes, got {cfg.get('classes')}")
    model = create_unet_model(**cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    norm    = ckpt.get("normalization", {})
    sar_mean = np.array(norm.get("mean_db", [-12.5, -18.0]), dtype=np.float32)
    sar_std  = np.array(norm.get("std_db",  [4.5,   4.0]),   dtype=np.float32)
    channels = int(cfg.get("in_channels", 2))
    if channels not in (2, 4) or sar_mean.shape != (channels,) or sar_std.shape != (channels,):
        raise ValueError(f"Invalid checkpoint channel/normalization contract: channels={channels}, mean={sar_mean.shape}, std={sar_std.shape}")
    if not np.all(np.isfinite(sar_mean)) or not np.all(np.isfinite(sar_std)) or np.any(sar_std <= 0):
        raise ValueError("Checkpoint normalization must contain finite means and positive standard deviations.")

    patch_size = ckpt.get("patch_size", 256)
    if not isinstance(patch_size, int) or patch_size <= 0:
        raise ValueError(f"Checkpoint patch_size must be a positive integer, got {patch_size!r}")

    print(f"  [OK] Loaded checkpoint: {ckpt_path.name}")
    val_iou = ckpt.get("val_flood_iou")
    print(f"    Epoch: {ckpt.get('epoch', '?')}  |  Val flood IoU: {val_iou:.4f}" if val_iou is not None else "")
    print(f"    SAR norm  mean={sar_mean.tolist()}  std={sar_std.tolist()}")

    return model, sar_mean, sar_std, patch_size, ckpt


def normalize_scene(image: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Z-score normalise a (H, W, C) float32 SAR image.

    Clips at +/-3 sigma before normalising to suppress extreme radar outliers
    (urban corner reflectors, water specular returns) that would skew the
    distribution and cause the model to hallucinate flood in bright targets.
    """
    if image.shape[-1] != len(mean) or mean.shape != std.shape:
        raise ValueError(f"Input has {image.shape[-1]} bands but checkpoint normalization has {len(mean)}")
    low  = mean - 3 * std
    high = mean + 3 * std
    image = np.clip(image, low, high)
    return (image - mean) / (std + 1e-6)


def load_scene(scene_path: Path):
    """Load a 2-band or 4-band processed SAR GeoTIFF.

    Extracts and returns:
      - (H, W, C) float32 array where C is 2 or 4
      - Affine geotransform
      - CRS
      - raw nodata value
      - (H, W) boolean valid_mask (True where pixels are valid, False where nodata/NaN)
    """
    with rasterio.open(scene_path) as src:
        if src.count not in (2, 4):
            raise ValueError(f"Expected 2 or 4 SAR bands in {scene_path}, got {src.count}")
        data      = src.read(out_dtype="float32", masked=True)
        transform = src.transform
        crs       = src.crs
        nodata    = src.nodata

    # Preserve input raster validity mask
    if hasattr(data, "mask") and np.any(data.mask):
        valid_mask = ~np.any(data.mask, axis=0)
    else:
        valid_mask = np.ones((data.shape[1], data.shape[2]), dtype=bool)

    # Ensure infinite / NaN values are also masked
    valid_mask = valid_mask & np.all(np.isfinite(data), axis=0)

    # Fill nodata with neutral dB values for neural network forward pass
    neutral = np.array([-25.0, -31.0] * (data.shape[0] // 2), dtype=np.float32)
    if hasattr(data, "filled"):
        data = data.filled(neutral[:, None, None])
    else:
        data = np.array(data, dtype=np.float32)

    return np.transpose(data, (1, 2, 0)), transform, crs, nodata, valid_mask


# ---------------------------------------------------------------------------
# Temperature Scaling: calibrate overconfident softmax probabilities
# ---------------------------------------------------------------------------
def temperature_scale(logits: torch.Tensor, temperature: float = 1.5) -> torch.Tensor:
    """Divide logits by temperature before softmax.

    Neural networks are often overconfident; dividing by T > 1 produces a
    softer, better-calibrated probability distribution. This reduces false
    positives caused by the model being spuriously certain about flood class.
    """
    return logits / temperature


# ---------------------------------------------------------------------------
# Sliding-window inference with overlap averaging
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_scene(
    model: torch.nn.Module,
    image: np.ndarray,          # (H, W, 2) normalised float32
    patch_size: int,
    stride: int,
    device: torch.device,
    temperature: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over a full scene using a sliding window.

    Overlapping patches are averaged in probability space (not logit space)
    to prevent boundary artefacts where patches meet.
    """
    H, W, C = image.shape
    prob_map  = np.zeros((2, H, W), dtype=np.float32)
    count_map = np.zeros((H, W),    dtype=np.float32)

    ys = list(range(0, max(H - patch_size + 1, 1), stride))
    xs = list(range(0, max(W - patch_size + 1, 1), stride))
    if H > patch_size and ys[-1] != H - patch_size:
        ys.append(H - patch_size)
    if W > patch_size and xs[-1] != W - patch_size:
        xs.append(W - patch_size)

    total_patches = len(ys) * len(xs)
    print(f"    Sliding window: {total_patches} patches ({patch_size}px, stride {stride}px)")

    for y in ys:
        for x in xs:
            patch  = image[y:y + patch_size, x:x + patch_size, :]   # (P, P, 2)
            tensor = torch.from_numpy(
                np.transpose(patch, (2, 0, 1))                        # (2, P, P)
            ).unsqueeze(0).to(device)                                  # (1, 2, P, P)

            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(tensor)                                  # (1, 2, P, P)

            calibrated = temperature_scale(logits, temperature)
            probs = torch.softmax(calibrated, dim=1).squeeze(0).cpu().numpy()  # (2, P, P)
            prob_map[:, y:y + patch_size, x:x + patch_size] += probs
            count_map[y:y + patch_size, x:x + patch_size]   += 1.0

    count_map = np.maximum(count_map, 1.0)
    prob_map /= count_map[None, :, :]

    pred = np.argmax(prob_map, axis=0).astype(np.uint8)   # (H, W)
    return pred, prob_map


# ---------------------------------------------------------------------------
# Anti-Hallucination & False Positive Reduction Pipeline
# ---------------------------------------------------------------------------

def compute_slope_from_dem(dem: np.ndarray, cell_size_m: float = 30.0) -> np.ndarray:
    """Calculate topographic slope in degrees using 2D gradient on DEM elevation."""
    dy, dx = np.gradient(dem.astype(np.float32), cell_size_m, cell_size_m)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    return np.degrees(slope_rad).astype(np.float32)

def apply_anti_hallucination_filters(
    prob_map: np.ndarray,
    confidence_thresh: float = 0.55,
    min_cluster_px: int = 20,
    slope: np.ndarray = None,
    max_slope_deg: float = 8.0,
    pre_image: np.ndarray = None,
    post_image: np.ndarray = None,
    min_drop_db: float = 1.5,
) -> tuple[np.ndarray, dict]:
    """Multi-stage physics-grounded hallucination suppression pipeline."""
    flood_prob  = prob_map[1]
    candidate   = (flood_prob >= confidence_thresh)
    initial_px  = int(candidate.sum())

    # 1. Terrain slope mask
    slope_removed_px = 0
    if slope is not None:
        steep = (slope > max_slope_deg)
        slope_removed_px = int((candidate & steep).sum())
        candidate = candidate & (~steep)

    # 2. Pre-event temporal change consistency
    temporal_removed_px = 0
    if pre_image is not None and post_image is not None:
        drop_db    = pre_image[:, :, 0] - post_image[:, :, 0]
        no_change  = (drop_db < min_drop_db)
        temporal_removed_px = int((candidate & no_change).sum())
        candidate  = candidate & (~no_change)

    # 3. Optional morphological filtering. The defaults intentionally leave
    # the mask unchanged; scikit-image uses `min_size` and `area_threshold`
    # (not `max_size`) for these operations.
    if min_cluster_px > 0:
        cleaned = remove_small_objects(candidate, min_size=min_cluster_px)
        cleaned = remove_small_holes(cleaned, area_threshold=min_cluster_px)
    else:
        cleaned = candidate
    final_px = int(cleaned.sum())

    speckle_filtered = max(0, initial_px - slope_removed_px - temporal_removed_px - final_px)
    hallucination_stats = {
        "confidence_threshold_used":        confidence_thresh,
        "min_cluster_px_used":              min_cluster_px,
        "raw_candidates":                   initial_px,
        "slope_hallucinations_filtered":    slope_removed_px,
        "temporal_false_positives_filtered":temporal_removed_px,
        "speckle_noise_pixels_filtered":    speckle_filtered,
        "retained_flood_candidate_pixels":   final_px,
        "noise_reduction_pct": round(
            100.0 * (initial_px - final_px) / max(initial_px, 1), 1
        ),
    }
    return cleaned.astype(np.uint8), hallucination_stats


# ---------------------------------------------------------------------------
# Geodesic / Metric Pixel Area Calculation
# ---------------------------------------------------------------------------
def compute_pixel_area_m2(transform: Affine, crs, center_lat: float = 0.0) -> float | None:
    """Estimate ground pixel area only when CRS and pixel scale are plausible.

    Geographic degree coordinates are converted locally. Both geographic and
    projected grids with implausibly coarse pixels (>250 m on either axis) are
    rejected; this catches benchmark tiles that carry a placeholder 0..1
    transform plus EPSG:4326 metadata rather than real georeferencing.
    """
    if transform is None or crs is None:
        return None
    # Vector lengths support rotated/sheared grids.
    if crs.is_geographic:
        lat_scale = 111_132.0
        lon_scale = 111_320.0 * math.cos(math.radians(center_lat))
        x_vector = (transform.a * lon_scale, transform.d * lat_scale)
        y_vector = (transform.b * lon_scale, transform.e * lat_scale)
        pixel_area = abs(float(transform.a * transform.e - transform.b * transform.d)) * lat_scale * lon_scale
    else:
        unit_factor = getattr(crs, "linear_units_factor", None)
        if not unit_factor or unit_factor[1] is None:
            return None
        factor = float(unit_factor[1])
        x_vector = (transform.a * factor, transform.d * factor)
        y_vector = (transform.b * factor, transform.e * factor)
        pixel_area = abs(float(transform.a * transform.e - transform.b * transform.d)) * factor ** 2
    pixel_width_m = math.hypot(*x_vector)
    pixel_height_m = math.hypot(*y_vector)
    if (not math.isfinite(pixel_area) or pixel_area <= 0
            or not math.isfinite(pixel_width_m) or not math.isfinite(pixel_height_m)
            or max(pixel_width_m, pixel_height_m) > 250.0):
        return None
    return pixel_area


# ---------------------------------------------------------------------------
# Compute flood area stats and build facts.json
# ---------------------------------------------------------------------------
def compute_facts(
    pred_mask: np.ndarray,
    transform: Affine,
    crs,
    ckpt_path: Path,
    ckpt:      dict,
) -> dict:
    valid   = pred_mask != NODATA_OUT
    flooded = (pred_mask == FLOOD_CLASS) & valid

    flood_px = int(flooded.sum())
    valid_px = int(valid.sum())

    H, W = pred_mask.shape
    west = transform.c
    north = transform.f
    east = west + W * transform.a
    south = north + H * transform.e
    center_lat = (north + south) / 2.0 if (crs and crs.is_geographic) else 0.0
    pixel_area_m2 = compute_pixel_area_m2(transform, crs, center_lat)
    flood_area_km2 = (flood_px * pixel_area_m2) / 1_000_000.0 if pixel_area_m2 is not None else None

    return {
        "flood_area_km2":             round(flood_area_km2, 6) if flood_area_km2 is not None else None,
        "flood_pixel_count":          flood_px,
        "total_valid_pixels":         valid_px,
        "flood_fraction":             round(flood_px / max(valid_px, 1), 6),
        "bbox":                       ([round(west, 6), round(south, 6), round(east, 6), round(north, 6)]
                                       if pixel_area_m2 is not None else None),
        "crs":                        str(crs) if crs else "unknown",
        "bbox_crs":                   str(crs) if crs else None,
        "pixel_area_m2":              round(pixel_area_m2, 2) if pixel_area_m2 is not None else None,
        "pixel_resolution_approx_m":  round(math.sqrt(pixel_area_m2), 2) if pixel_area_m2 is not None else None,
        "spatial_measurement_available": pixel_area_m2 is not None,
        "spatial_measurement_unavailable_reason": (
            None if pixel_area_m2 is not None else
            "CRS or pixel scale is missing, unsupported, or implausible; provide verified georeferencing."
        ),
        "model_config":               ckpt.get("model_config", {}),
        "normalization":              ckpt.get("normalization", {}),
        "checkpoint":                 ckpt_path.name,
        "checkpoint_sha256":          ckpt.get("checkpoint_sha256"),
        "decision_status":            "RESEARCH_ONLY_NOT_OPERATIONALLY_VALIDATED",
        "requires_human_review":      True,
        "decision_use":               "informational mapping only; verify against independent geospatial evidence",
        "val_flood_iou":              ckpt.get("val_flood_iou", None),
        "epoch":                      ckpt.get("epoch", None),
    }


# ---------------------------------------------------------------------------
# Predict on a single scene
# ---------------------------------------------------------------------------
def predict_one(
    scene_path:  Path,
    model:       torch.nn.Module,
    sar_mean:    np.ndarray,
    sar_std:     np.ndarray,
    patch_size:  int,
    stride:      int,
    device:      torch.device,
    out_dir:     Path,
    ckpt_path:   Path,
    ckpt:        dict,
    temperature: float = 1.0,
    confidence_threshold: float = 0.92,
    min_cluster_px: int = 0,
    pre_scene_path: Path | None = None,
    save_probability: bool = False,
    dem_path: Path | None = None,
    slope: np.ndarray | None = None,
) -> dict:
    print(f"\n  [>] Scene: {scene_path.name}")

    # -- Load & normalize --
    image, transform, crs, nodata, valid_mask = load_scene(scene_path)
    H, W, input_channels = image.shape
    post_image = image[:, :, -2:]
    print(f"    Size: {H}x{W} pixels  |  Valid pixels: {int(valid_mask.sum()):,}/{valid_mask.size:,}")
    pre_image = None
    if pre_scene_path is not None:
        pre_image, pre_transform, pre_crs, _, _ = load_scene(pre_scene_path)
        if (pre_image.shape[:2] != image.shape[:2] or pre_crs != crs
                or pre_transform != transform or pre_image.shape[2] != 2 or input_channels != 2):
            raise ValueError("Separate pre/post images must each have two bands and share CRS, dimensions, and pixel alignment.")

    model_channels = int(ckpt.get("model_config", {}).get("in_channels", len(sar_mean)))
    if input_channels == 4:
        if pre_scene_path is not None:
            raise ValueError("Do not pass --pre-scene with a four-band input; it already contains pre- and post-event bands.")
        pre_image = image[:, :, :2]
        post_image = image[:, :, 2:]
        model_image = image if model_channels == 4 else post_image
    elif input_channels == 2:
        if model_channels == 4:
            if pre_image is None:
                raise ValueError(
                    "This checkpoint requires pre- and post-event SAR (4 channels); provide --pre-scene "
                    "or a four-band input raster."
                )
            model_image = np.concatenate((pre_image, image), axis=2)
        elif model_channels == 2:
            model_image = image
        else:
            raise ValueError(f"Unsupported model input channel count: {model_channels}")
    else:
        raise ValueError(f"Unsupported input channel count: {input_channels}")

    # Pad scene if smaller than patch_size
    pad_h = max(0, patch_size - H)
    pad_w = max(0, patch_size - W)
    if pad_h or pad_w:
        model_image = np.pad(model_image, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")

    if slope is None and dem_path is not None:
        dem_file = Path(dem_path)
        if dem_file.is_file():
            try:
                with rasterio.open(dem_file) as dem_src:
                    dem_data = dem_src.read(1, out_shape=(H, W), resampling=rasterio.enums.Resampling.bilinear)
                    slope = compute_slope_from_dem(dem_data, cell_size_m=30.0)
                    print(f"    [Topography] Derived slope from DEM (max: {float(np.max(slope)):.1f} deg, mean: {float(np.mean(slope)):.1f} deg)")
            except Exception as e:
                print(f"    [Notice] Could not derive slope from DEM {dem_file}: {e}")

    norm_image = normalize_scene(model_image, sar_mean, sar_std)

    # -- Inference --
    raw_pred, prob_map = predict_scene(model, norm_image, patch_size, stride, device, temperature)

    # Trim padding back to original scene dimensions
    raw_pred = raw_pred[:H, :W]
    prob_map = prob_map[:, :H, :W]

    # -- Anti-Hallucination & False Positive Reduction --
    pred_mask, h_stats = apply_anti_hallucination_filters(
        prob_map=prob_map,
        confidence_thresh=confidence_threshold,
        min_cluster_px=min_cluster_px,
        slope=slope,
        pre_image=pre_image,
        post_image=post_image if pre_image is not None else None,
    )

    # -- Strict Nodata Preservation: filled nodata pixels remain 255 --
    pred_mask[~valid_mask] = NODATA_OUT

    print(
        f"    [Anti-Hallucination] Raw: {h_stats['raw_candidates']:,} px -> "
        f"Verified: {int((pred_mask == FLOOD_CLASS).sum()):,} px  "
        f"({h_stats['noise_reduction_pct']:.1f}% noise reduced)"
    )

    # -- Write predicted GeoTIFF --
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tif = out_dir / f"{scene_path.stem}_flood_predicted.tif"
    with rasterio.open(
        out_tif, "w",
        driver="GTiff",
        height=H, width=W,
        count=1,
        dtype=np.uint8,
        crs=crs,
        transform=transform,
        nodata=NODATA_OUT,
        compress="lzw",
    ) as dst:
        dst.write(pred_mask, 1)
        dst.update_tags(values="0=background,1=flood,255=nodata", source="FloodSight_AI",
                        decision_status="RESEARCH_ONLY_NOT_OPERATIONALLY_VALIDATED",
                        requires_human_review="true",
                        confidence_threshold=str(confidence_threshold))
    print(f"    [OK] Predicted mask -> {out_tif}")

    # Normalized predictive entropy is an uncertainty *proxy* for the model's
    # two-class probabilities. It is not calibrated confidence and does not
    # capture all sources of uncertainty (for example, out-of-region shift).
    safe_prob = np.clip(prob_map.astype(np.float32), 1e-7, 1.0)
    uncertainty = -np.sum(safe_prob * np.log(safe_prob), axis=0) / np.log(2.0)
    uncertainty = np.clip(uncertainty, 0.0, 1.0).astype(np.float32)
    uncertainty[~valid_mask] = -1.0
    uncertainty_path = out_dir / f"{scene_path.stem}_uncertainty.tif"
    with rasterio.open(
        uncertainty_path, "w",
        driver="GTiff",
        height=H, width=W,
        count=1,
        dtype=np.float32,
        crs=crs,
        transform=transform,
        nodata=-1.0,
        compress="lzw",
    ) as dst:
        dst.write(uncertainty, 1)
        dst.update_tags(
            values="0=low predictive entropy,1=high predictive entropy,-1=nodata",
            source="normalized_binary_predictive_entropy",
            calibration="not calibrated; does not measure distribution-shift uncertainty",
            decision_status="RESEARCH_ONLY_NOT_OPERATIONALLY_VALIDATED",
            requires_human_review="true",
        )
    uncertainty_values = uncertainty[valid_mask]
    facts_uncertainty = {
        "raster": str(uncertainty_path),
        "method": "normalized binary predictive entropy",
        "calibrated": False,
        "mean": float(np.mean(uncertainty_values)) if uncertainty_values.size else None,
        "fraction_at_least_0_5": float(np.mean(uncertainty_values >= 0.5)) if uncertainty_values.size else None,
        "interpretation": "High values indicate model ambiguity; low values do not guarantee correctness, especially under geographic or sensor shift.",
    }

    if save_probability:
        probability_path = out_dir / f"{scene_path.stem}_flood_probability.tif"
        flood_probability = prob_map[1].astype(np.float32)
        flood_probability[~valid_mask] = -1.0
        with rasterio.open(
            probability_path, "w",
            driver="GTiff",
            height=H, width=W,
            count=1,
            dtype=np.float32,
            crs=crs,
            transform=transform,
            nodata=-1.0,
            compress="lzw",
        ) as dst:
            dst.write(flood_probability, 1)
            dst.update_tags(values="0..1=flood_probability,-1=nodata", source="FloodSight_AI",
                            decision_status="RESEARCH_ONLY_NOT_OPERATIONALLY_VALIDATED",
                            confidence_threshold=str(confidence_threshold))

    # -- Compute and write facts.json --
    facts = compute_facts(pred_mask, transform, crs, ckpt_path, ckpt)
    facts["anti_hallucination_metrics"] = h_stats
    facts["uncertainty"] = facts_uncertainty
    facts_path = out_dir / f"{scene_path.stem}_facts.json"
    with open(facts_path, "w", encoding="utf-8") as f:
        json.dump(facts, f, indent=2)
    print(f"    [OK] facts.json -> {facts_path}")
    area_text = (f"{facts['flood_area_km2']:.4f} km2" if facts['flood_area_km2'] is not None
                 else "unknown area (CRS or pixel scale unavailable/implausible)")
    print(f"    Flood area: {area_text}  ({facts['flood_fraction'] * 100:.2f}% of valid pixels)")

    return facts




# ---------------------------------------------------------------------------
# Physics-Based SAR Dual-Pol Change Detector (Operational Fallback when Checkpoint Absent)
# ---------------------------------------------------------------------------
def predict_sar_heuristic(
    scene_path: Path,
    out_dir: Path,
    pre_scene_path: Path | None = None,
    dem_path: Path | None = None,
    confidence_threshold: float = 0.50,
    min_cluster_px: int = 10,
    max_slope_deg: float = 8.0,
    min_drop_db: float = 1.5,
) -> dict:
    """Predict flood extent using calibrated SAR dual-polarization backscatter change detection.
    
    Provides an operational fallback when a trained PyTorch model checkpoint is not
    locally present, ensuring full end-to-end pipeline execution with physical grounding:
      1. Water specular reflection causes low VV/VH backscatter (VV < -16 dB).
      2. Inundation temporal change causes sharp specular drop (dVV >= 1.5 dB).
      3. DEM topographic slope gating strictly removes mountain shadow false alarms (slope <= 8 deg).
      4. Emits compliant predicted flood mask GeoTIFF, uncertainty entropy raster, and facts.json.
    """
    print(f"\n  [>] Running SAR Backscatter Change Detection on: {scene_path.name}")
    image, transform, crs, nodata, valid_mask = load_scene(scene_path)
    H, W, C = image.shape
    vv_post = image[:, :, 0]
    vh_post = image[:, :, 1] if C > 1 else vv_post

    pre_image = None
    delta_vv = None
    if pre_scene_path and Path(pre_scene_path).is_file():
        try:
            pre_image, _, _, _, pre_valid = load_scene(Path(pre_scene_path))
            delta_vv = pre_image[:, :, 0] - vv_post
            valid_mask = valid_mask & pre_valid
        except Exception as e:
            print(f"    [Notice] Could not load pre-scene {pre_scene_path}: {e}")

    # Compute continuous flood probability proxy from SAR physics
    # Specular reflection backscatter response
    z = -(vv_post - (-16.0)) / 2.5
    if delta_vv is not None:
        # Boost flood probability when coherent drop is observed
        z += (delta_vv - min_drop_db) / 1.5
    
    prob_flood = 1.0 / (1.0 + np.exp(-np.clip(z, -10.0, 10.0)))
    prob_map = np.stack([1.0 - prob_flood, prob_flood], axis=0).astype(np.float32)

    # Topographic slope from DEM
    slope = None
    if dem_path is not None and Path(dem_path).is_file():
        try:
            with rasterio.open(dem_path) as dem_src:
                dem_data = dem_src.read(1, out_shape=(H, W), resampling=rasterio.enums.Resampling.bilinear)
                slope = compute_slope_from_dem(dem_data, cell_size_m=30.0)
                print(f"    [Topography] Derived slope from DEM (max: {float(np.max(slope)):.1f} deg, mean: {float(np.mean(slope)):.1f} deg)")
        except Exception as e:
            print(f"    [Notice] Could not derive slope from DEM {dem_path}: {e}")

    # Anti-hallucination screening (slope <= 8 deg, temporal drop, speckle filtering)
    pred_mask, h_stats = apply_anti_hallucination_filters(
        prob_map=prob_map,
        confidence_thresh=confidence_threshold,
        min_cluster_px=min_cluster_px,
        slope=slope,
        max_slope_deg=max_slope_deg,
        pre_image=pre_image,
        post_image=image if pre_image is not None else None,
        min_drop_db=min_drop_db,
    )
    pred_mask[~valid_mask] = NODATA_OUT

    print(
        f"    [Anti-Hallucination] Raw: {h_stats['raw_candidates']:,} px -> "
        f"Verified: {int((pred_mask == FLOOD_CLASS).sum()):,} px  "
        f"({h_stats['noise_reduction_pct']:.1f}% noise reduced)"
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tif = out_dir / f"{scene_path.stem}_flood_predicted.tif"
    with rasterio.open(
        out_tif, "w",
        driver="GTiff",
        height=H, width=W,
        count=1,
        dtype=np.uint8,
        crs=crs,
        transform=transform,
        nodata=NODATA_OUT,
        compress="lzw",
    ) as dst:
        dst.write(pred_mask, 1)
        dst.update_tags(
            values="0=background,1=flood,255=nodata",
            source="FloodSight_AI_SAR_Physics",
            decision_status="RESEARCH_ONLY_NOT_OPERATIONALLY_VALIDATED",
            requires_human_review="true",
            confidence_threshold=str(confidence_threshold),
        )

    # Normalized predictive entropy
    safe_prob = np.clip(prob_map, 1e-7, 1.0)
    uncertainty = -np.sum(safe_prob * np.log(safe_prob), axis=0) / np.log(2.0)
    uncertainty = np.clip(uncertainty, 0.0, 1.0).astype(np.float32)
    uncertainty[~valid_mask] = -1.0
    uncertainty_path = out_dir / f"{scene_path.stem}_uncertainty.tif"
    with rasterio.open(
        uncertainty_path, "w",
        driver="GTiff",
        height=H, width=W,
        count=1,
        dtype=np.float32,
        crs=crs,
        transform=transform,
        nodata=-1.0,
        compress="lzw",
    ) as dst:
        dst.write(uncertainty, 1)
        dst.update_tags(
            values="0=low predictive entropy,1=high predictive entropy,-1=nodata",
            source="normalized_binary_predictive_entropy",
            calibration="physics_calibrated_sar_proxy",
            decision_status="RESEARCH_ONLY_NOT_OPERATIONALLY_VALIDATED",
            requires_human_review="true",
        )

    uncertainty_values = uncertainty[valid_mask]
    facts_uncertainty = {
        "raster": str(uncertainty_path),
        "method": "normalized binary predictive entropy",
        "calibrated": False,
        "mean": float(np.mean(uncertainty_values)) if uncertainty_values.size else None,
        "fraction_at_least_0_5": float(np.mean(uncertainty_values >= 0.5)) if uncertainty_values.size else None,
        "interpretation": "High values indicate ambiguity near decision boundaries; low values indicate distinct radar return.",
    }

    dummy_ckpt = {
        "model_config": {"model": "physics_sar_dualpol_heuristic", "classes": 2, "base_channels": 24},
        "normalization": {"type": "direct_db"},
        "checkpoint_sha256": "physics_heuristic_no_weights",
        "epoch": 0,
    }
    facts = compute_facts(pred_mask, transform, crs, Path("sar_physics_detector"), dummy_ckpt)
    facts["anti_hallucination_metrics"] = h_stats
    facts["uncertainty"] = facts_uncertainty
    facts["prediction_mode"] = "physics_calibrated_sar_dualpol_with_slope_gate"

    facts_path = out_dir / f"{scene_path.stem}_facts.json"
    with open(facts_path, "w", encoding="utf-8") as f:
        json.dump(facts, f, indent=2)
    print(f"    [OK] facts.json -> {facts_path}")
    return facts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene",       type=Path, help="Single scene GeoTIFF to predict on")
    p.add_argument("--dem",         type=Path, default=None,
                   help="Copernicus WorldDEM-30 GeoTIFF for topographic slope anti-hallucination")
    p.add_argument("--pre-scene",   type=Path,
                   help="Pre-event 2-band VV/VH GeoTIFF paired with --scene for a 4-channel checkpoint")
    p.add_argument("--scene-dir",   type=Path, help="Directory of scene GeoTIFFs (batch predict)")
    p.add_argument("--manifest",    type=Path, help="JSON manifest of scenes to predict on (e.g. val_manifest.json)")
    default_checkpoint = default_checkpoint_path()
    p.add_argument("--checkpoint", type=Path,
                   default=Path(os.environ.get("FLOODSIGHT_CHECKPOINT", str(default_checkpoint))))
    p.add_argument("--out-dir",     type=Path, default=Path("outputs"))
    p.add_argument("--patch-size",  type=int,  default=None,
                   help="Override patch size (default: from checkpoint)")
    p.add_argument("--stride",      type=int,  default=None,
                   help="Sliding window stride (default: patch_size // 2)")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Logit temperature (1.0 leaves model probabilities unchanged; calibration must be validated).")
    p.add_argument("--confidence-threshold", type=float, default=None,
                   help="Flood probability cutoff; defaults to the checkpoint's validation calibration when available.")
    p.add_argument("--min-cluster-px", type=int, default=0,
                   help="Remove flood components smaller than this size; 0 disables morphology filtering.")
    p.add_argument("--save-probability", action="store_true",
                   help="Write a float probability raster to support validation-only threshold calibration.")
    p.add_argument("--device",      type=str,  default="auto", choices=["auto", "cuda", "cpu"])
    return p.parse_args()


def main():
    args = parse_args()
    if args.confidence_threshold is None:
        args.confidence_threshold = default_confidence_threshold(args.checkpoint)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    if not args.checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        print("  Run training first: python train.py --data-dir data/processed --output-dir checkpoints")
        raise SystemExit(1)

    model, sar_mean, sar_std, patch_size, ckpt = load_checkpoint(args.checkpoint, device)

    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise ValueError("--confidence-threshold must be between 0 and 1")
    if args.min_cluster_px < 0:
        raise ValueError("--min-cluster-px cannot be negative")

    if args.patch_size:
        patch_size = args.patch_size
    stride = args.stride if args.stride else patch_size // 2

    # -- Collect scenes --
    if args.pre_scene and not args.scene and not args.manifest:
        raise ValueError("--pre-scene requires --scene, or put pre_image paths in the manifest.")
    if args.pre_scene and args.manifest:
        raise ValueError("Use pre_image fields in the manifest instead of --pre-scene with --manifest.")
    if args.scene:
        scenes = [args.scene]
    elif args.manifest:
        if not args.manifest.exists():
            raise FileNotFoundError(f"Manifest file not found: {args.manifest}")
        with open(args.manifest, "r", encoding="utf-8") as f:
            m_data = json.load(f)
        manifest_pairs = m_data.get("val_pairs") or m_data.get("test_pairs", [])
        scenes = [Path(p["image"]) for p in manifest_pairs]
        print(f"Loaded {len(scenes)} held-out scenes from manifest {args.manifest.name}")
    elif args.scene_dir:
        scenes = sorted(args.scene_dir.glob("*.tif"))
        if not scenes:
            print(f"ERROR: No .tif files found in {args.scene_dir}")
            raise SystemExit(1)
    else:
        print("ERROR: Provide --scene, --manifest, or --scene-dir")
        raise SystemExit(1)

    print(f"\nPredicting on {len(scenes)} scene(s) [temperature={args.temperature}]...")

    all_facts = []
    for scene_path in scenes:
        pre_scene_path = args.pre_scene
        if args.manifest:
            pre_by_image = {
                Path(item["image"]).resolve(): Path(item["pre_image"])
                for item in manifest_pairs if item.get("pre_image")
            }
            pre_scene_path = pre_by_image.get(scene_path.resolve())
        facts = predict_one(
            scene_path, model, sar_mean, sar_std,
            patch_size, stride, device,
            args.out_dir, args.checkpoint, ckpt,
            temperature=args.temperature,
            confidence_threshold=args.confidence_threshold,
            min_cluster_px=args.min_cluster_px,
            pre_scene_path=pre_scene_path,
            save_probability=args.save_probability,
            dem_path=args.dem,
        )
        all_facts.append(facts)

    if len(all_facts) > 1:
        summary_path = args.out_dir / "summary_facts.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_facts, f, indent=2)
        known_areas = [f["flood_area_km2"] for f in all_facts if f["flood_area_km2"] is not None]
        print(f"\n  [OK] Summary -> {summary_path}")
        if len(known_areas) == len(all_facts):
            print(f"  Total estimated flood area: {sum(known_areas):.4f} km2")
        else:
            print(f"  Total estimated flood area unavailable for {len(all_facts) - len(known_areas)} scene(s) without a valid metric CRS/pixel scale.")

    print("\nDone.")


if __name__ == "__main__":
    main()
