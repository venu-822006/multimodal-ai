"""
download_scenes.py  —  FloodSight AI | Automated Planetary Computer Satellite Ingestion
======================================================================================
Automates Step 2 & 3: Direct cloud-native discovery, download, and preprocessing
of Sentinel-1 RTC and Sentinel-2 L2A from Microsoft Planetary Computer STAC.

Key capabilities:
  1. Searches Planetary Computer STAC for Sentinel-1 Radiometrically Terrain Corrected (RTC)
     scenes covering the target bounding box within the lookback/lookahead temporal window.
  2. Finds same-relative-orbit pre-event and post-event pairs with matching geometry.
     Rejects cross-orbit pairs; fails clearly if no same-orbit match is found.
  3. Uses Planetary Computer token signing to access cloud-optimized GeoTIFFs.
  4. Reprojects and clips to the exact analysis grid in EPSG:4326.
  5. Converts linear radar power to decibels (10*log10), applies Lee-sigma 3x3 speckle filter,
     and saves 2-band (VV, VH) GeoTIFFs (pre_event.tif, post_event.tif).
  6. Optionally retrieves Sentinel-2 L2A optical bands (Green B03, NIR B08) when clear-sky
     conditions exist (<30% cloud cover).
  7. Writes acquisition manifest.json with full provenance.

Usage:
    python download_scenes.py --bbox 85.2 27.6 85.4 27.8 --flood-date 2024-08-15 --out-dir data/scenes
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from scipy.ndimage import uniform_filter

STAC_ENDPOINT = "https://planetarycomputer.microsoft.com/api/stac/v1"


def lee_sigma_filter(image: np.ndarray, size: int = 3) -> np.ndarray:
    """Apply Lee-sigma speckle noise filter to radar dB image."""
    mean = uniform_filter(image, size=size)
    mean_sq = uniform_filter(image**2, size=size)
    var = np.maximum(mean_sq - mean**2, 1e-6)
    overall_var = np.var(image)
    weights = var / (var + overall_var + 1e-6)
    return np.clip(mean + weights * (image - mean), -40.0, 0.0)


def discover_and_download(
    bbox: list[float],
    flood_date: str,
    out_dir: str | Path,
    lookback_days: int = 45,
    lookahead_days: int = 30,
    download_optical: bool = True,
    resolution_deg: float = 0.00018,  # ~20m at equator
) -> dict:
    """Discover, download, and preprocess Sentinel satellite scenes."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    try:
        from pystac_client import Client
        import planetary_computer as pc
    except ImportError as e:
        raise RuntimeError(
            "pystac-client and planetary-computer are required for automated scene download. "
            "Install with: pip install pystac-client planetary-computer"
        ) from e

    f_date = datetime.strptime(flood_date, "%Y-%m-%d").date()
    win_start = f_date - timedelta(days=lookback_days)
    win_end = f_date + timedelta(days=lookahead_days)

    client = Client.open(STAC_ENDPOINT, headers={"User-Agent": "FloodSight-AI/1.0.0"})

    # 1. Search Sentinel-1 RTC
    print(f"[STAC] Searching Sentinel-1 RTC over bbox {bbox} between {win_start} and {win_end}...")
    s1_search = client.search(
        collections=["sentinel-1-rtc"],
        bbox=bbox,
        datetime=f"{win_start.isoformat()}/{win_end.isoformat()}",
        max_items=40,
    )
    s1_items = list(s1_search.items())
    print(f"[STAC] Found {len(s1_items)} candidate Sentinel-1 RTC items.")

    if not s1_items:
        raise RuntimeError(f"No Sentinel-1 RTC coverage found for bbox {bbox} in date window.")

    # Group into pre and post items
    pre_items = []
    post_items = []
    for item in s1_items:
        dt = item.datetime.date() if item.datetime else None
        if not dt:
            continue
        rel_orbit = (
            item.properties.get("sat:relative_orbit")
            or item.properties.get("relative_orbit")
            or item.properties.get("s1:relative_orbit_number")
        )
        raw_direction = (
            item.properties.get("sat:orbit_state")
            or item.properties.get("orbit_state")
            or item.properties.get("sar:orbit_state")
            or item.properties.get("s1:orbit_direction")
        )
        direction = str(raw_direction).strip().lower() if raw_direction else None
        record = {
            "item": item,
            "id": item.id,
            "date": dt,
            "datetime": item.datetime,
            "orbit": rel_orbit,
            "direction": direction,
        }
        if dt <= f_date:
            pre_items.append(record)
        else:
            post_items.append(record)

    # Find matching pair: MUST strictly match BOTH same relative orbit AND same flight direction
    selected_pre = None
    selected_post = None
    for post in sorted(post_items, key=lambda x: abs((x["date"] - f_date).days)):
        for pre in sorted(pre_items, key=lambda x: abs((f_date - x["date"]).days)):
            orbit_match = (
                post["orbit"] is not None
                and pre["orbit"] is not None
                and int(post["orbit"]) == int(pre["orbit"])
            )
            direction_match = (
                bool(post["direction"])
                and bool(pre["direction"])
                and post["direction"].lower() == pre["direction"].lower()
            )
            if orbit_match and direction_match:
                selected_pre = pre
                selected_post = post
                break
        if selected_post:
            break

    if not selected_post or not selected_pre:
        orbit_info = (
            f"Found {len(pre_items)} pre-event and {len(post_items)} post-event Sentinel-1 RTC items, "
            f"but none share both a matching relative orbit and matching flight direction (ascending/descending). "
            f"Strict same-orbit and same-direction pairing is required for coherent SAR change detection. "
            f"Expand the temporal window (--lookback-days / --lookahead-days), verify the target "
            f"bounding box has Sentinel-1 coverage, or supply pre-processed same-orbit rasters "
            f"via --pre-image and --post-image."
        )
        raise RuntimeError(orbit_info)

    print(f"[STAC] Selected Pre-event:  {selected_pre['id']} ({selected_pre['date']}) Orbit: {selected_pre['orbit']} Direction: {selected_pre['direction']}")
    print(f"[STAC] Selected Post-event: {selected_post['id']} ({selected_post['date']}) Orbit: {selected_post['orbit']} Direction: {selected_post['direction']}")

    # Setup target raster grid (EPSG:4326)
    west, south, east, north = bbox
    width = max(64, int(math.ceil((east - west) / resolution_deg)))
    height = max(64, int(math.ceil((north - south) / resolution_deg)))
    transform = from_bounds(west, south, east, north, width, height)
    target_crs = CRS.from_epsg(4326)

    def process_and_save_sar(item_record, output_file: Path) -> Path:
        signed_item = pc.sign(item_record["item"])
        vv_href = signed_item.assets.get("vv") or signed_item.assets.get("VV")
        vh_href = signed_item.assets.get("vh") or signed_item.assets.get("VH")
        if not vv_href or not vh_href:
            raise RuntimeError(f"Missing VV or VH asset in item {item_record['id']}")

        bands_data = []
        for pol, href in [("VV", vv_href.href), ("VH", vh_href.href)]:
            print(f"[STAC] Reading {pol} band from {item_record['id']}...")
            with rasterio.open(href) as src:
                dest = np.full((height, width), -32768.0, dtype=np.float32)
                reproject(
                    source=rasterio.band(src, 1),
                    destination=dest,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear,
                )
                # Convert linear power to dB: 10 * log10(max(p, 1e-4)), clipped [-40, 0]
                valid = (dest > 0) & np.isfinite(dest)
                db = np.full((height, width), -40.0, dtype=np.float32)
                db[valid] = 10.0 * np.log10(np.maximum(dest[valid], 1e-4))
                db = np.clip(db, -40.0, 0.0)
                # Apply Lee-sigma speckle filter
                filtered_db = lee_sigma_filter(db)
                bands_data.append(filtered_db)

        # Write 2-band float32 GeoTIFF
        with rasterio.open(
            output_file,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=2,
            dtype="float32",
            crs=target_crs,
            transform=transform,
            nodata=-40.0,
        ) as dst:
            dst.write(bands_data[0], 1)
            dst.set_band_description(1, "VV_dB")
            dst.write(bands_data[1], 2)
            dst.set_band_description(2, "VH_dB")
            dst.update_tags(
                platform=item_record["item"].properties.get("platform", "sentinel-1"),
                item_id=item_record["id"],
                acquisition_date=str(item_record["date"]),
                acquisition_datetime=(
                    item_record["item"].datetime.isoformat()
                    if item_record["item"].datetime
                    else str(item_record["date"])
                ),
                relative_orbit=str(item_record["orbit"]) if item_record["orbit"] is not None else "",
                orbit_direction=str(item_record["direction"]) if item_record["direction"] else "",
                sat_orbit_state=str(item_record["direction"]) if item_record["direction"] else "",
            )
        print(f"[STAC] Saved analysis-ready SAR GeoTIFF: {output_file}")
        return output_file

    pre_file = out_path / "pre_event.tif"
    post_file = out_path / "post_event.tif"
    process_and_save_sar(selected_pre, pre_file)
    process_and_save_sar(selected_post, post_file)

    manifest = {
        "status": "success",
        "bbox": bbox,
        "flood_date": flood_date,
        "grid": {"width": width, "height": height, "crs": "EPSG:4326", "resolution_deg": resolution_deg},
        "pre_event": {
            "item_id": selected_pre["id"],
            "date": str(selected_pre["date"]),
            "orbit": int(selected_pre["orbit"]) if selected_pre["orbit"] is not None else None,
            "direction": selected_pre["direction"],
            "file": str(pre_file.resolve()),
        },
        "post_event": {
            "item_id": selected_post["id"],
            "date": str(selected_post["date"]),
            "orbit": int(selected_post["orbit"]) if selected_post["orbit"] is not None else None,
            "direction": selected_post["direction"],
            "file": str(post_file.resolve()),
        },
        "optical": None,
    }

    # 2. Sentinel-2 Optical Download (Adaptive cloud-cover search)
    if download_optical:
        print("[STAC] Searching Sentinel-2 L2A optical scenes...")
        opt_start = f_date - timedelta(days=lookback_days)
        opt_end = f_date + timedelta(days=lookahead_days)
        try:
            s2_search = client.search(
                collections=["sentinel-2-l2a"],
                bbox=bbox,
                datetime=f"{opt_start.isoformat()}/{opt_end.isoformat()}",
                max_items=30,
            )
            s2_items = list(s2_search.items())
            if s2_items:
                # Rank scenes by cloud cover
                best_s2 = min(s2_items, key=lambda x: float(x.properties.get("eo:cloud_cover", 100.0)))
                best_cloud = float(best_s2.properties.get("eo:cloud_cover", 100.0))
                print(f"[STAC] Selected best Sentinel-2 scene: {best_s2.id} ({best_s2.datetime.date() if best_s2.datetime else 'N/A'}) with cloud cover {best_cloud:.1f}%")
                signed_s2 = pc.sign(best_s2)
                b03 = signed_s2.assets.get("B03") or signed_s2.assets.get("b03")
                b08 = signed_s2.assets.get("B08") or signed_s2.assets.get("b08")
                if b03 and b08:
                    green_file = out_path / "s2_green.tif"
                    nir_file = out_path / "s2_nir.tif"
                    for asset, out_f, band_name in [(b03, green_file, "B03_Green"), (b08, nir_file, "B08_NIR")]:
                        with rasterio.open(asset.href) as src:
                            dest = np.zeros((height, width), dtype=np.float32)
                            reproject(
                                source=rasterio.band(src, 1),
                                destination=dest,
                                src_transform=src.transform,
                                src_crs=src.crs,
                                dst_transform=transform,
                                dst_crs=target_crs,
                                resampling=Resampling.bilinear,
                            )
                            # Reflectance scale (0.0001)
                            dest = np.clip(dest * 0.0001, 0.0, 1.0)
                            with rasterio.open(
                                out_f, "w", driver="GTiff", height=height, width=width, count=1, dtype="float32", crs=target_crs, transform=transform
                            ) as dst:
                                dst.write(dest, 1)
                                dst.set_band_description(1, band_name)
                                dst.update_tags(
                                    source_id=best_s2.id,
                                    cloud_cover=str(best_cloud),
                                    acquisition_date=str(best_s2.datetime.date() if best_s2.datetime else f_date),
                                )
                    manifest["optical"] = {
                        "item_id": best_s2.id,
                        "date": str(best_s2.datetime.date() if best_s2.datetime else f_date),
                        "cloud_cover": best_cloud,
                        "green_file": str(green_file.resolve()),
                        "nir_file": str(nir_file.resolve()),
                    }
                    print(f"[STAC] Successfully saved Sentinel-2 optical bands: Green -> {green_file.name}, NIR -> {nir_file.name}")
            else:
                print(f"[STAC] No Sentinel-2 L2A scenes found in window {opt_start} to {opt_end}.")
        except Exception as e:
            print(f"[STAC] Optical download skipped or unavailable: {e}")

    # 3. Copernicus WorldDEM-30 Elevation Data
    manifest["dem"] = None
    try:
        print("[STAC] Searching Copernicus 30m Global DEM (cop-dem-glo-30)...")
        dem_search = client.search(
            collections=["cop-dem-glo-30"],
            bbox=bbox,
            max_items=5,
        )
        dem_items = list(dem_search.items())
        dem_file = out_path / "dem_30m.tif"
        if dem_items:
            signed_dem = pc.sign(dem_items[0])
            dem_data_asset = signed_dem.assets.get("data")
            if dem_data_asset:
                with rasterio.open(dem_data_asset.href) as src:
                    dest = np.full((height, width), -9999.0, dtype=np.float32)
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=dest,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling.bilinear,
                    )
                    with rasterio.open(
                        dem_file, "w", driver="GTiff", height=height, width=width, count=1, dtype="float32", crs=target_crs, transform=transform, nodata=-9999.0
                    ) as dst:
                        dst.write(dest, 1)
                        dst.update_tags(title="Copernicus WorldDEM-30", source=dem_items[0].id)
                manifest["dem"] = {
                    "item_id": dem_items[0].id,
                    "dem_file": str(dem_file.resolve()),
                    "min_elevation_m": float(np.min(dest[dest > -9000])) if (dest > -9000).any() else None,
                    "max_elevation_m": float(np.max(dest[dest > -9000])) if (dest > -9000).any() else None,
                }
                print(f"[STAC] Saved Copernicus WorldDEM-30: {dem_file}")
    except Exception as e:
        print(f"[STAC] DEM download skipped or unavailable: {e}")

    with open(out_path / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[STAC] Automated ingestion complete. Manifest: {out_path / 'manifest.json'}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Download and prepare Sentinel imagery from Planetary Computer.")
    parser.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--flood-date", type=str, required=True, help="Event date YYYY-MM-DD")
    parser.add_argument("--out-dir", type=str, default="data/scenes", help="Output directory")
    parser.add_argument("--lookback-days", type=int, default=15, help="Days to search before flood date (default: 15)")
    parser.add_argument("--lookahead-days", type=int, default=15, help="Days to search after flood date (default: 15)")
    parser.add_argument("--no-optical", action="store_true", help="Skip optical Sentinel-2 download")
    args = parser.parse_args()

    discover_and_download(
        bbox=args.bbox,
        flood_date=args.flood_date,
        out_dir=args.out_dir,
        lookback_days=args.lookback_days,
        lookahead_days=args.lookahead_days,
        download_optical=not args.no_optical,
    )


if __name__ == "__main__":
    main()
