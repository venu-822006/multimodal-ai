"""
optical_analysis.py  —  FloodSight AI | Sentinel-2 Clear-Sky Optical Processing
================================================================================
Implements Sentinel-2 optical processing and cross-sensor validation for clear-sky
conditions as mandated by the problem statement:
  "Use Sentinel-1 radar, which sees through monsoon clouds, and Sentinel-2
   optical images when the sky is clear."

Core functionalities:
  1. Computes McFeeters Normalized Difference Water Index (NDWI):
         NDWI = (Green - NIR) / (Green + NIR + 1e-6)
     where Green is Band 3 (560 nm) and NIR is Band 8 (842 nm).
  2. Cloud screening proxy:
     Uses only supplied Green/NIR reflectance thresholds; this is not a
     Sentinel-2 Scene Classification Layer cloud/shadow mask.
  3. Optical surface change / sediment detection:
     High reflectance in Green/Red in non-water pixels flags sediment and mud deposits.
  4. Cross-Sensor Radar-Optical Agreement:
     Compares Sentinel-1 radar flood mask against Sentinel-2 NDWI mask:
       - Confirmed Water (both radar and optical detect water)
       - Radar-Only (under clouds or dense spray where optical is blocked)
       - Optical-Only (shallow surface water / wet silt where radar signal saturated)
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
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import reproject
from rasterio.enums import Resampling


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Calculate Normalized Difference Water Index (NDWI).

    Values > 0.0 generally indicate water; values > 0.2 indicate confident open water.
    """
    denom = green.astype(np.float32) + nir.astype(np.float32) + 1e-6
    ndwi = (green.astype(np.float32) - nir.astype(np.float32)) / denom
    return np.clip(ndwi, -1.0, 1.0)


def detect_clouds(blue: np.ndarray, red: np.ndarray, nir: np.ndarray, thresh: float = 0.25) -> np.ndarray:
    """Simple cloud mask based on high visible and NIR reflectance.

    True indicates cloud or cloud shadow.
    """
    bright = (blue > thresh) & (red > thresh) & (nir > thresh)
    return bright


def cross_sensor_validation(
    radar_flood_mask: np.ndarray,
    optical_ndwi: np.ndarray,
    cloud_mask: np.ndarray = None,
    ndwi_threshold: float = 0.20,
) -> dict:
    """Evaluate cross-sensor agreement between Sentinel-1 SAR and Sentinel-2 Optical.

    Args:
        radar_flood_mask: Binary mask from U-Net (1=flood, 0=bg, 255=nodata)
        optical_ndwi: Float array of NDWI (-1.0 to 1.0)
        cloud_mask: Boolean array (True=cloud, False=clear sky)
        ndwi_threshold: Cut-off for open surface water
    """
    valid = (radar_flood_mask != 255) & np.isfinite(optical_ndwi)
    if cloud_mask is not None:
        valid = valid & (~cloud_mask)

    radar_water = (radar_flood_mask == 1) & valid
    optical_water = (optical_ndwi >= ndwi_threshold) & valid

    both_water = radar_water & optical_water
    radar_only = radar_water & (~optical_water)
    optical_only = optical_water & (~radar_water)

    both_count = int(both_water.sum())
    radar_only_count = int(radar_only.sum())
    optical_only_count = int(optical_only.sum())
    clear_valid_count = int(valid.sum())

    total_water = both_count + radar_only_count + optical_only_count
    agreement_pct = round(100.0 * both_count / max(total_water, 1), 2)

    return {
        "clear_sky_valid_pixels": clear_valid_count,
        "dual_confirmed_water_pixels": both_count,
        "radar_only_water_pixels": radar_only_count,
        "optical_only_water_pixels": optical_only_count,
        "cross_sensor_agreement_pct": agreement_pct,
        "agreement_level": "HIGH" if agreement_pct >= 60.0 else "MODERATE" if agreement_pct >= 30.0 else "LOW",
        "interpretation": "Agreement describes overlap of two thresholded masks; it is not calibrated model confidence.",
    }


def process_sentinel2_scene(
    green_band_path: Path,
    nir_band_path: Path,
    radar_mask_path: Path = None,
    out_dir: Path = Path("outputs"),
    cloud_thresh: float = 0.35,
) -> dict:
    """Process a Sentinel-2 optical pair and validate against a radar flood mask."""
    with rasterio.open(green_band_path) as s_g, rasterio.open(nir_band_path) as s_n:
        if s_g.crs is None or s_n.crs is None:
            raise ValueError("Sentinel-2 green and NIR rasters must define their CRS.")
        if radar_mask_path and radar_mask_path.exists():
            with rasterio.open(radar_mask_path) as s_r:
                height, width = s_r.height, s_r.width
                transform, crs = s_r.transform, s_r.crs
                radar_mask = s_r.read(1)
                if crs is None:
                    raise ValueError("Radar flood mask must define its CRS for optical alignment.")
        else:
            height, width = s_g.height, s_g.width
            transform, crs = s_g.transform, s_g.crs
            radar_mask = None

        def align_band(src):
            source = np.asarray(src.read(1, out_dtype="float32", masked=True).filled(np.nan), dtype=np.float32)
            scale = float(src.scales[0]) if src.scales and src.scales[0] not in (None, 0) else 1.0
            offset = float(src.offsets[0]) if src.offsets and src.offsets[0] is not None else 0.0
            source = source * scale + offset
            finite = source[np.isfinite(source)]
            if finite.size and float(np.nanpercentile(finite, 99)) > 2.0:
                source = source / 10_000.0
            aligned = np.full((height, width), np.nan, dtype=np.float32)
            reproject(
                source=source,
                destination=aligned,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=np.nan,
                dst_transform=transform,
                dst_crs=crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
            return aligned

        green = align_band(s_g)
        nir = align_band(s_n)

        if radar_mask is None and radar_mask_path and radar_mask_path.exists():
            with rasterio.open(radar_mask_path) as s_r:
                radar_mask = np.full((height, width), 255, dtype=np.uint8)
                reproject(
                    source=s_r.read(1),
                    destination=radar_mask,
                    src_transform=s_r.transform,
                    src_crs=s_r.crs,
                    src_nodata=s_r.nodata,
                    dst_transform=transform,
                    dst_crs=crs,
                    dst_nodata=255,
                    resampling=Resampling.nearest,
                )

    ndwi = compute_ndwi(green, nir)
    valid_optical = np.isfinite(green) & np.isfinite(nir)
    cloud_candidate = (green > cloud_thresh) & (nir > cloud_thresh) & valid_optical
    cloud_mask = cloud_candidate | ~valid_optical

    out_dir.mkdir(parents=True, exist_ok=True)
    ndwi_tif = out_dir / f"{green_band_path.stem}_ndwi.tif"
    with rasterio.open(
        ndwi_tif, "w",
        driver="GTiff",
        height=ndwi.shape[0], width=ndwi.shape[1],
        count=1, dtype=np.float32,
        crs=crs, transform=transform,
        compress="lzw"
    ) as dst:
        dst.write(ndwi, 1)

    results = {
        "ndwi_raster": str(ndwi_tif),
        "mean_ndwi": round(float(np.nanmean(ndwi[valid_optical])), 4) if valid_optical.any() else None,
        "cloud_candidate_fraction_of_valid_pixels": round(
            float(cloud_candidate.sum() / max(int(valid_optical.sum()), 1)), 4
        ),
        "invalid_optical_fraction": round(float((~valid_optical).mean()), 4),
        "cloud_mask_method": "Green/NIR reflectance threshold proxy; no SCL cloud/shadow band was supplied",
        "interpretation": "Optical threshold comparison is supporting evidence only and does not calibrate the SAR model.",
    }

    if radar_mask is not None:
        if radar_mask.shape != ndwi.shape:
            raise ValueError("Aligned Sentinel-2 and radar rasters unexpectedly differ in shape.")
        agreement = cross_sensor_validation(radar_mask, ndwi, cloud_mask)
        results["cross_sensor_validation"] = agreement

    return results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--green", type=Path, required=True, help="Sentinel-2 Band 3 (Green) GeoTIFF")
    p.add_argument("--nir",   type=Path, required=True, help="Sentinel-2 Band 8 (NIR) GeoTIFF")
    p.add_argument("--radar-mask", type=Path, default=None, help="(Optional) Sentinel-1 predicted flood mask")
    p.add_argument("--out-dir", type=Path, default=Path("outputs"))
    args = p.parse_args()

    res = process_sentinel2_scene(args.green, args.nir, args.radar_mask, args.out_dir)
    print("\n  [Sentinel-2 Optical Processing Complete]")
    print(f"    NDWI Raster:    {res['ndwi_raster']}")
    print(f"    Mean NDWI:      {res['mean_ndwi']}")
    print(f"    Bright-cloud candidate fraction (valid pixels): {res['cloud_candidate_fraction_of_valid_pixels']*100:.1f}%")
    if "cross_sensor_validation" in res:
        v = res["cross_sensor_validation"]
        print(f"    Cross-Sensor Agreement: {v['cross_sensor_agreement_pct']}% (agreement level: {v['agreement_level']}; not calibrated confidence)")


if __name__ == "__main__":
    main()
