"""
preprocess.py  —  FloodSight AI  |  SAR Pre-processing Pipeline
================================================================
Converts raw Sentinel-1 GRD scenes (downloaded as .zip or .SAFE folder)
into analysis-ready 2-band GeoTIFFs (VV + VH, dB scale) suitable for
the U-Net model.

Pipeline per scene:
    1. Read raw VV and VH bands from the .SAFE / .tif files
    2. Convert linear power → dB  (10 * log10)
    3. Lee-sigma speckle filter (3×3) to suppress speckle noise
    4. Clip extreme values (−40 dB to 0 dB)
    5. Reproject to WGS-84 (EPSG:4326)
    6. Stack VV + VH → 2-band output GeoTIFF (float32)

Usage:
    python preprocess.py \
        --vv_path  data/raw/S1A_VV.tif \
        --vh_path  data/raw/S1A_VH.tif \
        --out_path data/processed/images/event_001_post.tif \
        [--mask_path data/raw/flood_mask.tif --out_mask data/processed/masks/event_001.tif]
"""

import argparse
import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from scipy.ndimage import uniform_filter, variance

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DB_CLIP_LOW  = -40.0   # dB — values below this are almost certainly open water
DB_CLIP_HIGH =   0.0   # dB — values above 0 dB are artefacts / very bright targets
NODATA_OUT   = -9999.0
MASK_NODATA  = 255
TARGET_CRS   = "EPSG:4326"


# ---------------------------------------------------------------------------
# Step 1: Linear amplitude/power → dB
# ---------------------------------------------------------------------------

def linear_to_db(data: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Convert Sentinel-1 raw amplitude (or power) values to decibels.
    Adds epsilon to avoid log(0).

    Args:
        data: 2D float array (H, W) of linear amplitude values.
        eps: Small constant to prevent log(0).

    Returns:
        2D float32 array in dB.
    """
    data = data.astype(np.float32)
    data = np.where(data > 0, data, eps)   # Replace ≤ 0 with epsilon
    return (10.0 * np.log10(data)).astype(np.float32)


# ---------------------------------------------------------------------------
# Step 2: Lee-sigma speckle filter
# ---------------------------------------------------------------------------

def lee_sigma_filter(image: np.ndarray, size: int = 3) -> np.ndarray:
    """
    Apply a simplified Lee-sigma speckle filter to a single-band SAR image.

    The filter suppresses multiplicative speckle noise while preserving edges
    by blending the local mean with the pixel value based on local variance.

    Args:
        image: 2D float32 array (H, W) in dB.
        size:  Kernel size (default 3×3).

    Returns:
        Filtered 2D float32 array.
    """
    img_mean   = uniform_filter(image, size=size)
    img_sqr    = uniform_filter(image ** 2, size=size)
    img_var    = img_sqr - img_mean ** 2

    overall_var = np.var(image)
    weight      = img_var / (img_var + overall_var + 1e-10)

    filtered = img_mean + weight * (image - img_mean)
    return filtered.astype(np.float32)


# ---------------------------------------------------------------------------
# Step 3: Clip dB values to valid SAR range
# ---------------------------------------------------------------------------

def clip_db(data: np.ndarray,
            low: float = DB_CLIP_LOW,
            high: float = DB_CLIP_HIGH) -> np.ndarray:
    """Clip data to the valid dB range."""
    return np.clip(data, low, high).astype(np.float32)


# ---------------------------------------------------------------------------
# Step 4: Reproject to WGS-84
# ---------------------------------------------------------------------------

def reproject_to_wgs84(
    data: np.ndarray,
    src_crs,
    src_transform,
    target_crs: str = TARGET_CRS,
) -> tuple[np.ndarray, object, dict]:
    """
    Reproject a single-band numpy array to WGS-84.

    Returns:
        (reprojected_data, new_transform, new_meta_kwargs)
    """
    height, width = data.shape
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, target_crs, width, height,
    )
    dst_data = np.empty((dst_height, dst_width), dtype=np.float32)

    reproject(
        source=data,
        destination=dst_data,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=target_crs,
        resampling=Resampling.bilinear,
        src_nodata=NODATA_OUT,
        dst_nodata=NODATA_OUT,
    )
    extra_meta = {
        "crs": target_crs,
        "transform": dst_transform,
        "width": dst_width,
        "height": dst_height,
    }
    return dst_data, dst_transform, extra_meta


# ---------------------------------------------------------------------------
# Step 5: Stack bands and write output GeoTIFF
# ---------------------------------------------------------------------------

def write_2band_geotiff(
    vv: np.ndarray,
    vh: np.ndarray,
    out_path: str,
    transform,
    crs,
):
    """
    Write a 2-band (VV, VH) GeoTIFF to disk.

    Args:
        vv:        (H, W) float32 array for VV band.
        vh:        (H, W) float32 array for VH band.
        out_path:  Output file path.
        transform: Affine transform (from rasterio / reprojection step).
        crs:       Coordinate reference system string or object.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    h, w = vv.shape
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=2,               # 2 bands: VV and VH
        dtype=np.float32,
        crs=crs,
        transform=transform,
        nodata=NODATA_OUT,
        compress="lzw",        # Lossless compression to save disk space
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(vv, 1)       # Band 1 → VV
        dst.write(vh, 2)       # Band 2 → VH
        dst.update_tags(band_1="VV_dB", band_2="VH_dB")

    print(f"  ✓ Wrote 2-band image: {out_path}  [{h}×{w}]")


def write_mask_geotiff(mask: np.ndarray, out_path: str, transform, crs):
    """
    Write a 1-band uint8 mask GeoTIFF.

    Values:
        0   → background / non-flooded
        1   → flooded
        255 → nodata

    Args:
        mask:      (H, W) uint8 array.
        out_path:  Output file path.
        transform: Affine transform.
        crs:       Coordinate reference system.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    h, w = mask.shape
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype=np.uint8,
        crs=crs,
        transform=transform,
        nodata=MASK_NODATA,
        compress="lzw",
    ) as dst:
        dst.write(mask.astype(np.uint8), 1)

    print(f"  ✓ Wrote mask:        {out_path}  [{h}×{w}]")


# ---------------------------------------------------------------------------
# Core processing function
# ---------------------------------------------------------------------------

def process_scene(
    vv_path: str,
    vh_path: str,
    out_path: str,
    mask_path: str | None = None,
    out_mask: str | None = None,
    speckle_filter_size: int = 3,
    reproject_crs: str = TARGET_CRS,
) -> None:
    """
    Full preprocessing pipeline for one Sentinel-1 event (pre or post).

    Args:
        vv_path:             Path to raw VV band GeoTIFF.
        vh_path:             Path to raw VH band GeoTIFF.
        out_path:            Where to write the 2-band processed image GeoTIFF.
        mask_path:           (Optional) Path to raw flood mask GeoTIFF.
        out_mask:            (Optional) Where to write the processed mask GeoTIFF.
        speckle_filter_size: Kernel size for Lee-sigma filter (3 or 5).
        reproject_crs:       Target CRS for reprojection.
    """
    print(f"\n{'='*60}")
    print(f" Processing scene: {os.path.basename(vv_path)}")
    print(f"{'='*60}")

    # ── Load VV ────────────────────────────────────────────────────────────
    print(" [1/5] Loading raw bands...")
    with rasterio.open(vv_path) as src_vv:
        raw_vv     = src_vv.read(1).astype(np.float32)
        src_crs    = src_vv.crs
        src_transform = src_vv.transform
        src_meta   = src_vv.meta.copy()

    with rasterio.open(vh_path) as src_vh:
        raw_vh = src_vh.read(1).astype(np.float32)

    # ── dB Conversion ──────────────────────────────────────────────────────
    print(" [2/5] Converting to dB scale...")
    db_vv = linear_to_db(raw_vv)
    db_vh = linear_to_db(raw_vh)

    # ── Speckle Filter ─────────────────────────────────────────────────────
    print(f" [3/5] Applying Lee-sigma filter (kernel={speckle_filter_size})...")
    filt_vv = lee_sigma_filter(db_vv, size=speckle_filter_size)
    filt_vh = lee_sigma_filter(db_vh, size=speckle_filter_size)

    # ── Clip ───────────────────────────────────────────────────────────────
    print(f" [4/5] Clipping to [{DB_CLIP_LOW}, {DB_CLIP_HIGH}] dB...")
    clip_vv = clip_db(filt_vv)
    clip_vh = clip_db(filt_vh)

    # ── Reproject ──────────────────────────────────────────────────────────
    print(f" [5/5] Reprojecting to {reproject_crs}...")
    vv_wgs84, new_transform, new_meta = reproject_to_wgs84(clip_vv, src_crs, src_transform, reproject_crs)
    vh_wgs84, _,             _        = reproject_to_wgs84(clip_vh, src_crs, src_transform, reproject_crs)

    # ── Write image ────────────────────────────────────────────────────────
    write_2band_geotiff(vv_wgs84, vh_wgs84, out_path, new_meta["transform"], new_meta["crs"])

    # ── (Optional) Mask ────────────────────────────────────────────────────
    if mask_path and out_mask:
        print(" [+] Processing flood mask...")
        with rasterio.open(mask_path) as src_m:
            raw_mask = src_m.read(1).astype(np.uint8)

        mask_wgs84, _, mask_meta = reproject_to_wgs84(
            raw_mask.astype(np.float32), src_crs, src_transform, reproject_crs
        )
        # Round back to integer classes after bilinear resampling
        mask_wgs84 = np.round(mask_wgs84).astype(np.uint8)
        mask_wgs84 = np.clip(mask_wgs84, 0, 1)     # Ensure only 0 and 1 remain

        write_mask_geotiff(mask_wgs84, out_mask, mask_meta["transform"], mask_meta["crs"])


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="FloodSight AI — Sentinel-1 SAR Preprocessing Pipeline"
    )
    parser.add_argument("--vv_path",   required=True, help="Path to raw VV band GeoTIFF")
    parser.add_argument("--vh_path",   required=True, help="Path to raw VH band GeoTIFF")
    parser.add_argument("--out_path",  required=True, help="Output 2-band processed GeoTIFF")
    parser.add_argument("--mask_path", default=None,  help="(Optional) Raw flood mask GeoTIFF")
    parser.add_argument("--out_mask",  default=None,  help="(Optional) Output processed mask GeoTIFF")
    parser.add_argument("--kernel",    type=int, default=3,
                        help="Lee-sigma filter kernel size (3 or 5, default=3)")
    parser.add_argument("--crs",       default=TARGET_CRS,
                        help=f"Target CRS (default: {TARGET_CRS})")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_scene(
        vv_path=args.vv_path,
        vh_path=args.vh_path,
        out_path=args.out_path,
        mask_path=args.mask_path,
        out_mask=args.out_mask,
        speckle_filter_size=args.kernel,
        reproject_crs=args.crs,
    )
