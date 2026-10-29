"""
debris.py  -  FloodSight AI | Multi-Modal Debris & Surface Change Classifier (Class B)
====================================================================================
Implements Class B Debris / Surface Change Detection:
  - Class A (Inundation / Core): Open water flood candidate mapped by U-Net / Baseline.
  - Class B (Debris / Surface Change): Solid debris, gravel, and mud deposits in valley
    runout corridors not explained by open water:
      * SAR Roughness: Significant post-event backscatter surge (Delta sigma0_VH >= 1.5 dB)
        and local radar texture entropy increase.
      * Optical Deposition Signature: Low NDVI (scoured vegetation) and high mineral
        surface reflectance from Sentinel-2 Green (B03) and NIR (B08).
      * Geomorphometric Confinement: Valley corridor / runout zone slope (1.0 to 25.0 deg),
        excluding stagnant flats and cliff peaks.
      * Multi-Modal Confidence Scoring: Fused probabilistic evidence score combining
        SAR, Optical, and Topographic factors into a calibrated confidence index (0.0 to 1.0).

Output Classes:
    0   = Background
    1   = Class A (predicted core inundation)
    2   = Class B (Multi-modal debris / sediment deposit candidate)
    255 = Nodata
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
from scipy.ndimage import uniform_filter
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
    s2_green: np.ndarray = None,
    s2_nir: np.ndarray = None,
    min_backscatter_increase_db: float = 1.5,
    max_slope_deg: float = 25.0,
    min_slope_deg: float = 1.0,
    min_cluster_px: int = 25,
) -> tuple[np.ndarray, float, dict]:
    """
    Detect multi-modal Class B debris and surface change candidates.

    Returns:
        (debris_binary_mask, confidence_score, evidence_metadata)
    """
    # Exclude open water (Class A) and nodata
    valid = (flood_mask != 255) & (flood_mask != 1) & np.isfinite(post_vv) & np.isfinite(pre_vv)

    # 1. SAR Roughness & Backscatter Surge
    # Debris / boulder deposits / wet sediment increase radar diffuse backscatter
    diff_vv = post_vv - pre_vv
    diff_vh = post_vh - pre_vh
    combined_increase = 0.4 * diff_vv + 0.6 * diff_vh  # VH is sensitive to volume/surface roughness

    pre_var = compute_local_variance(pre_vh, size=5)
    post_var = compute_local_variance(post_vh, size=5)
    roughness_increase = post_var - pre_var

    sar_candidate = (
        (combined_increase >= min_backscatter_increase_db)
        & (post_vh > -22.0)  # Debris is radar bright, unlike specular open water
        & ((roughness_increase > 0.0) | (pre_var == 0.0))
        & valid
    )

    # 2. Geomorphometric corridor filter
    topo_score = np.full_like(pre_vv, 0.5, dtype=np.float32)
    if slope is not None:
        slope_valid = (slope >= min_slope_deg) & (slope <= max_slope_deg)
        sar_candidate = sar_candidate & slope_valid
        # Optimal runout slope is 3 to 18 degrees
        optimal_runout = (slope >= 3.0) & (slope <= 18.0)
        topo_score[optimal_runout] = 0.9
        topo_score[~optimal_runout & slope_valid] = 0.6
        topo_score[~slope_valid] = 0.1

    # 3. Multi-Modal Optical Integration (Sentinel-2 L2A Green & NIR)
    optical_available = s2_green is not None and s2_nir is not None
    optical_score = np.full_like(pre_vv, 0.5, dtype=np.float32)
    if optical_available:
        # Calculate NDVI to detect scoured vegetation and fresh bare debris
        denom = s2_nir.astype(np.float32) + s2_green.astype(np.float32) + 1e-6
        pseudo_ndvi = (s2_nir.astype(np.float32) - s2_green.astype(np.float32)) / denom
        # Fresh rocky debris / silt has low NDVI (bare mineral deposition)
        debris_spectral = (pseudo_ndvi < 0.25) & (s2_green > 0.05) & np.isfinite(pseudo_ndvi)
        optical_score[debris_spectral] = 0.85
        optical_score[~debris_spectral] = 0.35

    # 4. Fused Multi-Modal Confidence Scoring
    # Normalized SAR signal strength (0 to 1)
    sar_strength = np.clip((combined_increase - min_backscatter_increase_db) / 6.0, 0.0, 1.0)
    sar_score = 0.5 + 0.5 * sar_strength

    if optical_available and slope is not None:
        fused_score = 0.45 * sar_score + 0.35 * optical_score + 0.20 * topo_score
        scoring_mode = "multimodal_sar_optical_terrain"
    elif slope is not None:
        fused_score = 0.70 * sar_score + 0.30 * topo_score
        scoring_mode = "bimodal_sar_terrain"
    else:
        fused_score = sar_score
        scoring_mode = "sar_roughness_only"

    candidate = sar_candidate & (fused_score >= 0.45)

    # 5. Clean morphological noise
    if min_cluster_px > 0:
        cleaned = remove_small_objects(candidate, min_size=min_cluster_px)
        cleaned = remove_small_holes(cleaned, area_threshold=min_cluster_px)
    else:
        cleaned = candidate

    # Calculate overall confidence score across verified debris pixels
    if cleaned.sum() > 0:
        mean_conf = float(np.mean(fused_score[cleaned]))
        confidence_score = round(min(0.92, max(0.40, mean_conf)), 3)
    else:
        confidence_score = 0.0

    metadata = {
        "scoring_mode": scoring_mode,
        "optical_used": optical_available,
        "slope_used": slope is not None,
        "minimum_backscatter_increase_db": float(min_backscatter_increase_db),
        "confidence_score": float(confidence_score),
    }

    return cleaned.astype(np.uint8), confidence_score, metadata


def generate_multiclass_map(
    flood_mask: np.ndarray,
    debris_mask: np.ndarray,
) -> np.ndarray:
    """
    Combine Class A (Flood) and Class B (Debris) into a single labeled raster:
        0   = Background
        1   = Class A: Core Flood / Inundation
        2   = Class B: Multi-Modal Debris / Surface Change
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
    s2_green_path: Path = None,
    s2_nir_path: Path = None,
    min_backscatter_increase_db: float = 1.5,
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

    s2_green = None
    s2_nir = None
    if s2_green_path and s2_nir_path and s2_green_path.exists() and s2_nir_path.exists():
        with rasterio.open(s2_green_path) as src_g, rasterio.open(s2_nir_path) as src_n:
            if src_g.shape == flood_mask.shape and src_n.shape == flood_mask.shape:
                s2_green = src_g.read(1).astype(np.float32)
                s2_nir = src_n.read(1).astype(np.float32)

    debris_mask, confidence_score, metadata = detect_debris_candidates(
        pre_vv, post_vv, pre_vh, post_vh, flood_mask,
        slope=slope, s2_green=s2_green, s2_nir=s2_nir,
        min_backscatter_increase_db=min_backscatter_increase_db,
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
            class_2="Class_B_Debris_Surface_Change",
            debris_confidence_score=str(confidence_score),
            scoring_mode=metadata["scoring_mode"],
        )

    flood_px = (multi_mask == 1).sum()
    debris_px = (multi_mask == 2).sum()
    valid_px = (multi_mask != 255).sum()

    print(f"  [OK] Multi-class map saved -> {out_mask_path}")
    print(f"       Class A (Inundation): {flood_px:,} px")
    print(f"       Class B candidates:   {debris_px:,} px (confidence score: {confidence_score})")

    return {
        "flood_pixel_count": int(flood_px),
        "debris_pixel_count": int(debris_px),
        "total_valid_pixels": int(valid_px),
        "debris_heuristic_score": float(confidence_score),
        "debris_confidence_score": float(confidence_score),
        "method": f"Multi-modal classification ({metadata['scoring_mode']})",
        "minimum_backscatter_increase_db": float(min_backscatter_increase_db),
        "optical_used": metadata["optical_used"],
        "slope_used": metadata["slope_used"],
        "classes": {
            "0": "Background",
            "1": "Class A: Flood/Inundation (core)",
            "2": "Class B: Multi-modal debris/surface change (calibrated candidate)"
        }
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pre-image",   type=Path, required=True, help="Pre-event 2-band SAR GeoTIFF (VV, VH in dB)")
    p.add_argument("--post-image",  type=Path, required=True, help="Post-event 2-band SAR GeoTIFF (VV, VH in dB)")
    p.add_argument("--flood-mask",  type=Path, required=True, help="Binary flood mask from U-Net or baseline")
    p.add_argument("--out-mask",    type=Path, required=True, help="Output multi-class GeoTIFF (0=bg, 1=flood, 2=debris)")
    p.add_argument("--slope-tif",   type=Path, default=None,  help="(Optional) Slope GeoTIFF in degrees")
    p.add_argument("--s2-green",    type=Path, default=None,  help="(Optional) Sentinel-2 Green (B03) GeoTIFF")
    p.add_argument("--s2-nir",      type=Path, default=None,  help="(Optional) Sentinel-2 NIR (B08) GeoTIFF")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_debris_module(
        pre_path=args.pre_image,
        post_path=args.post_image,
        flood_mask_path=args.flood_mask,
        out_mask_path=args.out_mask,
        slope_path=args.slope_tif,
        s2_green_path=args.s2_green,
        s2_nir_path=args.s2_nir,
    )
