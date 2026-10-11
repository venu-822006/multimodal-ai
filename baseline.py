"""
baseline.py  —  FloodSight AI | SAR Change Detection Baseline
==============================================================
Implements the SAR log-ratio change detection baseline according to the
FloodSight architecture (Section 5: Flood mapping details):

  1. Log-ratio change: Difference between pre-flood and post-flood SAR (dB scale).
     Smooth water causes specular reflection away from radar -> sharp drop in backscatter.
  2. Automated or calibrated thresholding:
     - Uses Otsu thresholding or histogram bimodal analysis on backscatter decrease.
  3. Terrain filtering (slope mask):
     - Excludes steep slopes (> 10-15 degrees) where standing floodwater is physical nonsense.
  4. Speckle / morphological cleaning:
     - Median filtering / small island removal to eliminate false SAR speckle detections.

Usage:
    python baseline.py \
        --pre-image  data/processed/images/event_001_pre.tif \
        --post-image data/processed/images/event_001_post.tif \
        --out-mask   outputs/event_001_baseline_mask.tif
"""

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import rasterio
from scipy.ndimage import median_filter
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_holes, remove_small_objects


def load_sar_db(tif_path: Path) -> tuple[np.ndarray, np.ndarray, object, object]:
    """
    Load a 2-band (VV, VH) GeoTIFF in dB scale.
    Returns: (vv_db, vh_db, transform, crs)
    """
    with rasterio.open(tif_path) as src:
        vv = src.read(1).astype(np.float32)
        vh = src.read(2).astype(np.float32) if src.count >= 2 else vv.copy()
        transform = src.transform
        crs = src.crs
    return vv, vh, transform, crs


def compute_log_ratio_flood(
    pre_vv: np.ndarray,
    post_vv: np.ndarray,
    pre_vh: np.ndarray = None,
    post_vh: np.ndarray = None,
    threshold_db: float = None,
    min_size_px: int = 50,
) -> tuple[np.ndarray, float]:
    """
    Compute flood mask using SAR backscatter drop (pre_dB - post_dB).

    Args:
        pre_vv: Pre-event VV band in dB.
        post_vv: Post-event VV band in dB.
        pre_vh: Pre-event VH band in dB (optional).
        post_vh: Post-event VH band in dB (optional).
        threshold_db: Threshold in dB. If None, automatically determined using Otsu.
        min_size_px: Minimum connected flood patch size to keep.

    Returns:
        (flood_mask, chosen_threshold_db)
    """
    # Difference in dB: pre_dB - post_dB. High positive values mean significant backscatter decrease
    diff_vv = pre_vv - post_vv

    if pre_vh is not None and post_vh is not None:
        diff_vh = pre_vh - post_vh
        # Combine dual-pol difference (VV is dominant for open water, VH helps for rough water)
        diff = 0.6 * diff_vv + 0.4 * diff_vh
    else:
        diff = diff_vv

    # Filter out nodata and extreme values
    valid_mask = np.isfinite(diff) & (post_vv > -40.0) & (post_vv < 0.0) & (pre_vv > -40.0)

    valid_diff = diff[valid_mask]
    if len(valid_diff) == 0:
        return np.zeros_like(diff, dtype=np.uint8), 0.0

    # Determine threshold
    if threshold_db is None:
        # Otsu on the positive side of differences (water inundation)
        positive_diff = valid_diff[valid_diff > 0]
        if len(positive_diff) > 100:
            threshold_db = float(threshold_otsu(positive_diff))
            # Bound threshold to reasonable physical limits (between 2.5 dB and 8.0 dB drop)
            threshold_db = max(2.5, min(8.0, threshold_db))
        else:
            threshold_db = 3.5  # standard SAR flood drop default
    else:
        threshold_db = float(threshold_db)

    # Initial candidate mask: backscatter drop > threshold AND post-event is dark enough (< -12 dB)
    raw_flood = (diff >= threshold_db) & (post_vv < -12.0) & valid_mask

    # Morphological cleaning to remove radar speckle
    cleaned = median_filter(raw_flood.astype(np.uint8), size=3).astype(bool)
    cleaned = remove_small_objects(cleaned, max_size=min_size_px)
    cleaned = remove_small_holes(cleaned, max_size=min_size_px)

    flood_mask = np.zeros_like(raw_flood, dtype=np.uint8)
    flood_mask[~valid_mask] = 255  # Nodata
    flood_mask[cleaned] = 1        # Flood

    return flood_mask, threshold_db


def apply_slope_filter(
    flood_mask: np.ndarray,
    slope_deg: np.ndarray,
    max_slope_deg: float = 8.0,
) -> np.ndarray:
    """
    Zero-out flood detections on steep terrain where standing floodwater is physical nonsense.
    """
    steep = (slope_deg > max_slope_deg) & (flood_mask == 1)
    flood_mask[steep] = 0
    return flood_mask


def run_baseline(
    pre_path: Path,
    post_path: Path,
    out_path: Path,
    threshold_db: float = None,
    slope_path: Path = None,
    max_slope: float = 8.0,
) -> Path:
    """
    Full pipeline to generate baseline flood mask from pre/post GeoTIFFs.
    """
    pre_vv, pre_vh, transform, crs = load_sar_db(pre_path)
    post_vv, post_vh, _, _ = load_sar_db(post_path)

    mask, chosen_thresh = compute_log_ratio_flood(
        pre_vv, post_vv, pre_vh, post_vh, threshold_db=threshold_db
    )

    if slope_path and slope_path.exists():
        with rasterio.open(slope_path) as src:
            slope = src.read(1).astype(np.float32)
            mask = apply_slope_filter(mask, slope, max_slope_deg=max_slope)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path, "w",
        driver="GTiff",
        height=mask.shape[0],
        width=mask.shape[1],
        count=1,
        dtype=np.uint8,
        crs=crs,
        transform=transform,
        nodata=255,
        compress="lzw",
    ) as dst:
        dst.write(mask, 1)
        dst.update_tags(
            method="SAR_log_ratio_baseline",
            threshold_db=f"{chosen_thresh:.2f}",
            values="0=background,1=flood,255=nodata"
        )

    flood_px = (mask == 1).sum()
    valid_px = (mask != 255).sum()
    print(f"  [OK] Baseline mask saved -> {out_path}")
    print(f"       Threshold used: {chosen_thresh:.2f} dB")
    print(f"       Flood pixels: {flood_px:,} / {valid_px:,} ({100*flood_px/max(valid_px, 1):.2f}%)")
    return out_path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pre-image",  type=Path, required=True, help="Pre-event 2-band SAR GeoTIFF (VV, VH in dB)")
    p.add_argument("--post-image", type=Path, required=True, help="Post-event 2-band SAR GeoTIFF (VV, VH in dB)")
    p.add_argument("--out-mask",   type=Path, required=True, help="Output baseline flood mask GeoTIFF")
    p.add_argument("--threshold",  type=float, default=None, help="Backscatter drop threshold in dB (default: auto Otsu)")
    p.add_argument("--slope-tif",  type=Path, default=None, help="(Optional) Slope GeoTIFF in degrees for terrain filtering")
    p.add_argument("--max-slope",  type=float, default=8.0,  help="Max slope angle in degrees for standing water (default: 8.0)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_baseline(
        pre_path=args.pre_image,
        post_path=args.post_image,
        out_path=args.out_mask,
        threshold_db=args.threshold,
        slope_path=args.slope_tif,
        max_slope=args.max_slope,
    )
