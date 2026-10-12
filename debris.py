"""
debris.py  —  FloodSight AI | Debris Candidate Detection Module (Class B)
========================================================================
Implements Gap 1: Debris / surface change detection as required by the
FloodSight problem statement:

  - Class A (Inundation / Core): Flood candidate mapped by U-Net / Baseline.
  - Class B (Debris / Experimental): Surface change in valley corridors
    not explained by open water:
      * Backscatter increase (radar roughness increase from gravel/mud/debris deposits).
      * Local radar texture entropy increase.
      * Constrained by valley corridor / terrain masking.
  - Honest Labelling:
      * Output values:
          0   = Background
          1   = Class A (predicted flood candidate; requires review)
          2   = Class B (Possible debris / surface change, experimental)
          255 = Nodata
      * Uncalibrated heuristic score; it is not a probability or validated confidence.

Usage:
    python debris.py \
        --pre-image   data/processed/images/event_001_pre.tif \
        --post-image  data/processed/images/event_001_post.tif \
        --flood-mask  outputs/event_001_flood_mask.tif \
        --out-mask    outputs/event_001_multiclass_mask.tif \
        [--slope-tif  data/terrain/slope.tif]
"""

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import rasterio
from scipy.ndimage import generic_filter, uniform_filter
from skimage.morphology import remove_small_holes, remove_small_objects


def compute_local_variance(image: np.ndarray, size: int = 5) -> np.ndarray:
    """Calculate local spatial texture variance for radar roughness."""
    mean = uniform_filter(image, size=size)
    mean_sq = uniform_filter(image ** 2, size=size)
    var = np.maximum(mean_sq - mean ** 2, 0)
    return var


def detect_debris_candidates(
    pre_vv: np.ndarray,
    post_vv: np.ndarray,
    pre_vh: np.ndarray,
    post_vh: np.ndarray,
    flood_mask: np.ndarray,
    slope: np.ndarray = None,
    min_backscatter_increase_db: float = 2.0,
    max_slope_deg: float = 25.0,
    min_slope_deg: float = 1.0,
    min_cluster_px: int = 30,
) -> tuple[np.ndarray, float]:
    """
    Detect Class B debris / surface change candidates.

    Returns:
        (debris_binary_mask, uncalibrated_heuristic_score)
    """
    # Exclude open water (Class A) and nodata
    valid = (flood_mask != 255) & (flood_mask != 1) & np.isfinite(post_vv) & np.isfinite(pre_vv)

    # 1. Backscatter increase: post_dB - pre_dB > threshold
    # Debris / boulder deposits / wet rough sediment increase radar diffuse backscatter
    diff_vv = post_vv - pre_vv
    diff_vh = post_vh - pre_vh
    combined_increase = 0.5 * diff_vv + 0.5 * diff_vh

    # 2. Local texture roughness change
    pre_var = compute_local_variance(pre_vh, size=5)
    post_var = compute_local_variance(post_vh, size=5)
    roughness_increase = post_var - pre_var

    # Candidate condition:
    # Noticeable increase in backscatter + texture change
    candidate = (
        (combined_increase >= min_backscatter_increase_db)
        & (post_vh > -22.0)  # Debris is radar bright, not deep dark water
        & (roughness_increase > 0.0)
        & valid
    )

    # 3. Terrain filter (valley corridor / runout zone)
    # Debris deposits settle in valley floors or moderate slope runouts (< 25 deg, > 1 deg)
    if slope is not None:
        slope_valid = (slope >= min_slope_deg) & (slope <= max_slope_deg)
        candidate = candidate & slope_valid

    # 4. Clean morphological noise
    if min_cluster_px > 0:
        cleaned = remove_small_objects(candidate, min_size=min_cluster_px)
        cleaned = remove_small_holes(cleaned, area_threshold=min_cluster_px)
    else:
        cleaned = candidate

    # Heuristic score: proportional to mean backscatter change; not calibrated.
    if cleaned.sum() > 0:
        mean_sig = float(np.mean(combined_increase[cleaned]))
        heuristic_score = round(min(0.75, max(0.40, 0.40 + (mean_sig / 10.0))), 2)
    else:
        heuristic_score = 0.0

    return cleaned.astype(np.uint8), heuristic_score


def generate_multiclass_map(
    flood_mask: np.ndarray,
    debris_mask: np.ndarray,
) -> np.ndarray:
    """
    Combine Class A (Flood) and Class B (Debris) into a single labeled raster:
        0   = Background
        1   = Class A: Core Flood / Inundation
        2   = Class B: Possible Debris / Surface Change (Experimental)
        255 = Nodata
    """
    output = np.zeros_like(flood_mask, dtype=np.uint8)
    output[flood_mask == 255] = 255
    output[flood_mask == 1] = 1
    # Only assign Class B where not already marked as Class A
    output[(debris_mask == 1) & (output != 1) & (output != 255)] = 2
    return output


def run_debris_module(
    pre_path: Path,
    post_path: Path,
    flood_mask_path: Path,
    out_mask_path: Path,
    slope_path: Path = None,
) -> dict:
    if pre_path.resolve() == post_path.resolve():
        raise ValueError(
            f"Debris detection requires distinct pre- and post-flood images to measure temporal change. "
            f"Received identical path for both: '{pre_path}'. "
            f"Passing identical images produces identically zero dB change and invalidates Class B detection."
        )

    with rasterio.open(flood_mask_path) as src_f:
        flood_mask = src_f.read(1)
        transform = src_f.transform
        crs = src_f.crs

    with rasterio.open(pre_path) as src_pre:
        if src_pre.shape != flood_mask.shape:
            raise ValueError(
                f"Pre-image shape {src_pre.shape} does not match flood mask shape {flood_mask.shape}."
            )
        pre_vv = src_pre.read(1).astype(np.float32)
        pre_vh = src_pre.read(2).astype(np.float32) if src_pre.count >= 2 else pre_vv.copy()

    with rasterio.open(post_path) as src_post:
        if src_post.shape != flood_mask.shape:
            raise ValueError(
                f"Post-image shape {src_post.shape} does not match flood mask shape {flood_mask.shape}."
            )
        post_vv = src_post.read(1).astype(np.float32)
        post_vh = src_post.read(2).astype(np.float32) if src_post.count >= 2 else post_vv.copy()

    slope = None
    if slope_path and slope_path.exists():
        with rasterio.open(slope_path) as src_s:
            slope = src_s.read(1).astype(np.float32)

    debris_mask, heuristic_score = detect_debris_candidates(
        pre_vv, post_vv, pre_vh, post_vh, flood_mask, slope=slope
    )

    multi_mask = generate_multiclass_map(flood_mask, debris_mask)

    out_mask_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_mask_path, "w",
        driver="GTiff",
        height=multi_mask.shape[0],
        width=multi_mask.shape[1],
        count=1,
        dtype=np.uint8,
        crs=crs,
        transform=transform,
        nodata=255,
        compress="lzw",
    ) as dst:
        dst.write(multi_mask, 1)
        dst.update_tags(
            class_0="Background",
            class_1="Class_A_Core_Flood",
            class_2="Class_B_Possible_Debris_Experimental",
            debris_heuristic_score=str(heuristic_score),
            score_calibration="uncalibrated; not a probability",
        )

    flood_px = (multi_mask == 1).sum()
    debris_px = (multi_mask == 2).sum()
    valid_px = (multi_mask != 255).sum()

    print(f"  [OK] Multi-class map saved -> {out_mask_path}")
    print(f"       Class A (Inundation): {flood_px:,} px")
    print(f"       Class B candidates:   {debris_px:,} px (uncalibrated heuristic score: {heuristic_score})")

    return {
        "flood_pixel_count": int(flood_px),
        "debris_pixel_count": int(debris_px),
        "total_valid_pixels": int(valid_px),
        "debris_heuristic_score": float(heuristic_score),
        "method": "uncalibrated VV/VH backscatter increase plus positive local VH variance change",
        "minimum_backscatter_increase_db": float(min_backscatter_increase_db),
        "classes": {
            "0": "Background",
            "1": "Class A: Flood/Inundation (core)",
            "2": "Class B: Possible debris/surface change (experimental)"
        }
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pre-image",   type=Path, required=True, help="Pre-event 2-band SAR GeoTIFF (VV, VH in dB)")
    p.add_argument("--post-image",  type=Path, required=True, help="Post-event 2-band SAR GeoTIFF (VV, VH in dB)")
    p.add_argument("--flood-mask",  type=Path, required=True, help="Binary flood mask from U-Net or baseline")
    p.add_argument("--out-mask",    type=Path, required=True, help="Output multi-class GeoTIFF (0=bg, 1=flood, 2=debris)")
    p.add_argument("--slope-tif",   type=Path, default=None,  help="(Optional) Slope GeoTIFF in degrees")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_debris_module(
        pre_path=args.pre_image,
        post_path=args.post_image,
        flood_mask_path=args.flood_mask,
        out_mask_path=args.out_mask,
        slope_path=args.slope_tif,
    )
