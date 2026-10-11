"""
predict.py  -  FloodSight AI | Inference & facts.json Generator
================================================================
Loads a trained checkpoint (checkpoints/best.pt) and runs inference on a
preprocessed Sentinel-1 scene GeoTIFF to produce:

  1. flood_mask_predicted.tif  - georeferenced binary flood mask (0/1/255)
  2. facts.json                - aggregated flood statistics consumed by the
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
  "pixel_area_m2":      float,   -- geodesic area estimate at scene center
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

import argparse
import json
import math
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS
from rasterio.transform import array_bounds
from rasterio.warp import transform as warp_transform, transform_bounds
import torch
import warnings
from skimage.morphology import remove_small_holes, remove_small_objects

warnings.filterwarnings("ignore", category=FutureWarning)

from model import create_unet_model

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NODATA_OUT    = 255
FLOOD_CLASS   = 1
DEFAULT_RES_M = 10.0   # Sentinel-1 GRD IW: ~10 m ground spacing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_checkpoint(ckpt_path: Path, device: torch.device):
    """Load model + normalization stats from a FloodSight checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg   = ckpt.get("model_config", {"in_channels": 2, "classes": 2, "base_channels": 24})
    model = create_unet_model(**cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    norm    = ckpt.get("normalization", {})
    sar_mean = np.array(norm.get("mean_db", [-12.5, -18.0]), dtype=np.float32)
    sar_std  = np.array(norm.get("std_db",  [4.5,   4.0]),   dtype=np.float32)

    patch_size = ckpt.get("patch_size", 256)

    print(f"  [OK] Loaded checkpoint: {ckpt_path.name}")
    val_iou = ckpt.get("val_flood_iou")
    print(f"    Epoch: {ckpt.get('epoch', '?')}  |  Val flood IoU: {val_iou:.4f}" if val_iou is not None else "")
    print(f"    SAR norm  mean={sar_mean.tolist()}  std={sar_std.tolist()}")

    return model, sar_mean, sar_std, patch_size, ckpt


def normalize_scene(image: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Z-score normalise a (H, W, 2) float32 SAR image.

    Clips at +/-3 sigma before normalising to suppress extreme radar outliers
    (urban corner reflectors, water specular returns) that would skew the
    distribution and cause the model to hallucinate flood in bright targets.
    """
    low  = mean - 3 * std
    high = mean + 3 * std
    image = np.clip(image, low, high)
    return (image - mean) / (std + 1e-6)


def load_scene(scene_path: Path):
    """Load a 2-band (VV, VH) processed GeoTIFF.

    Extracts and returns:
      - (H, W, 2) float32 array
      - Affine geotransform
      - CRS
      - raw nodata value
      - (H, W) boolean valid_mask (True where pixels are valid, False where nodata/NaN)
    """
    with rasterio.open(scene_path) as src:
        if src.count < 2:
            raise ValueError(f"Expected 2 bands (VV, VH) in {scene_path}, got {src.count}")
        data      = src.read(out_dtype="float32", masked=True)   # (2, H, W)
        transform = src.transform
        crs       = src.crs
        nodata    = src.nodata

    # Preserve input raster validity mask
    if hasattr(data, "mask") and np.any(data.mask):
        valid_mask = ~(data.mask[0] | data.mask[1])
    else:
        valid_mask = np.ones((data.shape[1], data.shape[2]), dtype=bool)

    # Ensure infinite / NaN values are also masked
    valid_mask = valid_mask & np.isfinite(data[0]) & np.isfinite(data[1])

    # Fill nodata with neutral dB values for neural network forward pass
    neutral = np.array([-25.0, -31.0], dtype=np.float32)
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

    # 3. Morphological speckle removal + hole filling
    cleaned  = remove_small_objects(candidate, max_size=min_cluster_px)
    cleaned  = remove_small_holes(cleaned, max_size=min_cluster_px)
    final_px = int(cleaned.sum())

    speckle_filtered = max(0, initial_px - slope_removed_px - temporal_removed_px - final_px)
    hallucination_stats = {
        "confidence_threshold_used":        confidence_thresh,
        "min_cluster_px_used":              min_cluster_px,
        "raw_candidates":                   initial_px,
        "slope_hallucinations_filtered":    slope_removed_px,
        "temporal_false_positives_filtered":temporal_removed_px,
        "speckle_noise_pixels_filtered":    speckle_filtered,
        "verified_flood_pixels":            final_px,
        "noise_reduction_pct": round(
            100.0 * (initial_px - final_px) / max(initial_px, 1), 1
        ),
    }
    return cleaned.astype(np.uint8), hallucination_stats


# ---------------------------------------------------------------------------
# Geodesic / Metric Pixel Area Calculation
# ---------------------------------------------------------------------------
def compute_pixel_area_m2(transform: Affine, crs, center_lat: float = 0.0) -> float:
    """Calculate exact ground area per pixel in m^2.

    For geographic CRS (EPSG:4326 / degrees):
      Uses ellipsoid geodesic scaling at the scene's centroid latitude:
        dy = |transform.e| * 111,139.0 m (meridional distance per degree)
        dx = |transform.a| * 111,320.0 * cos(radians(center_lat)) m (parallel distance)
        area = dx * dy
    For projected CRS (e.g. UTM):
      Units are already metres: area = |transform.a * transform.e|
    """
    if transform is None:
        return 100.0   # Default 10 m x 10 m
    dx = abs(transform.a)
    dy = abs(transform.e)

    if crs is not None and crs.is_geographic:
        dy_m = dy * 111_139.0
        dx_m = dx * 111_320.0 * math.cos(math.radians(center_lat))
        return max(float(dx_m * dy_m), 1.0)
    return max(float(dx * dy), 1.0)


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
    west  = transform.c
    north = transform.f
    east  = west  + W * transform.a
    south = north + H * transform.e
    center_lat = (north + south) / 2.0 if (crs and crs.is_geographic) else 0.0

    pixel_area_m2 = compute_pixel_area_m2(transform, crs, center_lat)
    flood_area_km2 = (flood_px * pixel_area_m2) / 1_000_000.0

    return {
        "flood_area_km2":             round(flood_area_km2, 6),
        "flood_pixel_count":          flood_px,
        "total_valid_pixels":         valid_px,
        "flood_fraction":             round(flood_px / max(valid_px, 1), 6),
        "bbox":                       [round(west, 6), round(south, 6), round(east, 6), round(north, 6)],
        "crs":                        str(crs) if crs else "unknown",
        "pixel_area_m2":              round(pixel_area_m2, 2),
        "pixel_resolution_approx_m":  round(math.sqrt(pixel_area_m2), 2),
        "model_config":               ckpt.get("model_config", {}),
        "normalization":              ckpt.get("normalization", {}),
        "checkpoint":                 ckpt_path.name,
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
    confidence_threshold: float = 0.5,
    min_cluster_px: int = 0,
    pre_scene_path: Path | None = None,
) -> dict:
    print(f"\n  [>] Scene: {scene_path.name}")

    # -- Load & normalize --
    image, transform, crs, nodata, valid_mask = load_scene(scene_path)
    H, W, _ = image.shape
    post_image = image
    print(f"    Size: {H}x{W} pixels  |  Valid pixels: {int(valid_mask.sum()):,}/{valid_mask.size:,}")
    pre_image = None
    if pre_scene_path is not None:
        pre_image, pre_transform, pre_crs, _, _ = load_scene(pre_scene_path)
        if pre_image.shape != image.shape or pre_crs != crs or pre_transform != transform:
            raise ValueError("Pre/post images must share CRS, dimensions, and pixel alignment for temporal filtering.")

    # Pad scene if smaller than patch_size
    pad_h = max(0, patch_size - H)
    pad_w = max(0, patch_size - W)
    if pad_h or pad_w:
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")

    norm_image = normalize_scene(image, sar_mean, sar_std)

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
        crs=crs if crs else CRS.from_epsg(4326),
        transform=transform,
        nodata=NODATA_OUT,
        compress="lzw",
    ) as dst:
        dst.write(pred_mask, 1)
        dst.update_tags(values="0=background,1=flood,255=nodata", source="FloodSight_AI")
    print(f"    [OK] Predicted mask -> {out_tif}")

    # -- Compute and write facts.json --
    facts = compute_facts(pred_mask, transform, crs, ckpt_path, ckpt)
    facts["anti_hallucination_metrics"] = h_stats
    facts_path = out_dir / f"{scene_path.stem}_facts.json"
    with open(facts_path, "w", encoding="utf-8") as f:
        json.dump(facts, f, indent=2)
    print(f"    [OK] facts.json -> {facts_path}")
    print(
        f"    Flood area: {facts['flood_area_km2']:.4f} km2  "
        f"({facts['flood_fraction'] * 100:.2f}% of valid pixels)"
    )

    return facts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene",       type=Path, help="Single scene GeoTIFF to predict on")
    p.add_argument("--scene-dir",   type=Path, help="Directory of scene GeoTIFFs (batch predict)")
    p.add_argument("--manifest",    type=Path, help="JSON manifest of scenes to predict on (e.g. val_manifest.json)")
    p.add_argument("--checkpoint",  type=Path, default=Path("checkpoints/best.pt"))
    p.add_argument("--out-dir",     type=Path, default=Path("outputs"))
    p.add_argument("--patch-size",  type=int,  default=None,
                   help="Override patch size (default: from checkpoint)")
    p.add_argument("--stride",      type=int,  default=None,
                   help="Sliding window stride (default: patch_size // 2)")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Logit temperature (1.0 leaves model probabilities unchanged; calibration must be validated).")
    p.add_argument("--confidence-threshold", type=float, default=0.5,
                   help="Flood probability cutoff; default 0.5 matches binary argmax.")
    p.add_argument("--min-cluster-px", type=int, default=0,
                   help="Remove flood components smaller than this size; 0 disables morphology filtering.")
    p.add_argument("--device",      type=str,  default="auto", choices=["auto", "cuda", "cpu"])
    return p.parse_args()


def main():
    args = parse_args()

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
    if args.scene:
        scenes = [args.scene]
    elif args.manifest:
        if not args.manifest.exists():
            raise FileNotFoundError(f"Manifest file not found: {args.manifest}")
        with open(args.manifest, "r", encoding="utf-8") as f:
            m_data = json.load(f)
        scenes = [Path(p["image"]) for p in m_data.get("val_pairs", [])]
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
        facts = predict_one(
            scene_path, model, sar_mean, sar_std,
            patch_size, stride, device,
            args.out_dir, args.checkpoint, ckpt,
            temperature=args.temperature,
            confidence_threshold=args.confidence_threshold,
            min_cluster_px=args.min_cluster_px,
        )
        all_facts.append(facts)

    if len(all_facts) > 1:
        summary_path = args.out_dir / "summary_facts.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_facts, f, indent=2)
        total_flood = sum(f["flood_area_km2"] for f in all_facts)
        print(f"\n  [OK] Summary -> {summary_path}")
        print(f"  Total estimated flood area: {total_flood:.4f} km2")

    print("\nDone.")


if __name__ == "__main__":
    main()
