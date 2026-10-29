"""Compare a FloodSight prediction with a supplied EMSR927 reference mask.

This is an evaluation-only tool. The EMSR927 mask is never used for training,
threshold selection, preprocessing, or prediction. Reference mask must be a
single-band GeoTIFF with values 1=flood, 0=valid non-flood, and 255=nodata
(including areas outside the reference product's mapped coverage). The tool
reprojects the reference to the prediction grid with nearest-neighbour
resampling and preserves nodata coverage.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


def _aligned_reference(pred: rasterio.io.DatasetReader, ref_path: Path) -> np.ndarray:
    with rasterio.open(ref_path) as ref:
        if ref.count != 1 or ref.crs is None or pred.crs is None:
            raise ValueError("Prediction and reference must be single-band rasters with a CRS.")
        raw = ref.read(1)
        nodata = ref.nodata
        valid = np.ones(raw.shape, dtype=np.uint8)
        if nodata is not None:
            valid[raw == nodata] = 0
        if np.issubdtype(raw.dtype, np.floating):
            valid[~np.isfinite(raw)] = 0
        if nodata != 255:
            valid[raw == 255] = 0
        classes = set(np.unique(raw[valid == 1]).tolist())
        if not classes.issubset({0, 1}):
            raise ValueError(
                f"Reference values must be binary 0/1 plus nodata; found {sorted(classes)[:12]}. "
                "Prepare an explicit EMSR927 binary mask first."
            )
        ref_binary = np.where((raw == 1) & (valid == 1), 1, 0).astype(np.uint8)
        aligned = np.full((pred.height, pred.width), 255, dtype=np.uint8)
        aligned_valid = np.zeros((pred.height, pred.width), dtype=np.uint8)
        reproject(
            ref_binary, aligned,
            src_transform=ref.transform, src_crs=ref.crs,
            dst_transform=pred.transform, dst_crs=pred.crs,
            src_nodata=None, dst_nodata=255, resampling=Resampling.nearest,
        )
        reproject(
            valid, aligned_valid,
            src_transform=ref.transform, src_crs=ref.crs,
            dst_transform=pred.transform, dst_crs=pred.crs,
            src_nodata=0, dst_nodata=0, resampling=Resampling.nearest,
        )
        aligned[aligned_valid == 0] = 255
        return aligned


def _pixel_area_m2(transform, crs, row: int, col: int) -> float:
    if crs is not None and crs.is_geographic:
        center_y = transform.f + (row + 0.5) * transform.e
        dx_m = abs(transform.a) * 111_320.0 * math.cos(math.radians(center_y))
        dy_m = abs(transform.e) * 111_139.0
        return max(float(dx_m * dy_m), 1.0)
    return abs(transform.a * transform.e - transform.b * transform.d)


def compare_masks(pred_mask: np.ndarray, ref_mask: np.ndarray, transform, crs) -> dict:
    if pred_mask.shape != ref_mask.shape:
        raise ValueError("Aligned prediction/reference masks must have identical shapes.")
    valid = (pred_mask != 255) & (ref_mask != 255)
    pred = (pred_mask == 1) & valid
    truth = (ref_mask == 1) & valid
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth & valid).sum())
    fn = int((~pred & truth).sum())
    tn = int((~pred & ~truth & valid).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    rows = np.flatnonzero(np.any(valid, axis=1))
    area_per_row = {
        int(row): _pixel_area_m2(transform, crs, int(row), 0)
        for row in rows
    }
    pred_km2 = sum(int((pred[row] & valid[row]).sum()) * area_per_row[int(row)] for row in rows) / 1e6
    ref_km2 = sum(int((truth[row] & valid[row]).sum()) * area_per_row[int(row)] for row in rows) / 1e6
    return {
        "metrics": {
            "flood_iou": round(iou, 6), "f1_score": round(f1, 6),
            "precision": round(precision, 6), "recall": round(recall, 6),
            "pixel_accuracy": round((tp + tn) / max(tp + fp + fn + tn, 1), 6),
        },
        "area_comparison_km2": {
            "prediction": round(pred_km2, 6), "reference": round(ref_km2, 6),
            "difference_prediction_minus_reference": round(pred_km2 - ref_km2, 6),
        },
        "confusion_matrix_pixels": {
            "true_positive": tp, "false_positive": fp, "false_negative": fn,
            "true_negative": tn, "valid_pixels": tp + fp + fn + tn,
        },
    }


def run_emsr927_benchmark(pred_path: Path, reference_path: Path | None, out_json: Path) -> dict:
    with rasterio.open(pred_path) as pred_src:
        if pred_src.count != 1 or pred_src.crs is None:
            raise ValueError("Prediction must be a single-band georeferenced flood mask.")
        pred_mask = pred_src.read(1)
        if not set(np.unique(pred_mask).tolist()).issubset({0, 1, 255}):
            raise ValueError("Prediction values must be 0=non-flood, 1=flood, 255=nodata.")

        if reference_path is None or not reference_path.exists():
            ref_default = Path("data/evaluation_only/emsr927_reference.tif")
            if ref_default.exists():
                reference_path = ref_default
            else:
                raise FileNotFoundError(
                    "Official EMSR927 reference mask GeoTIFF not found. "
                    "Provide a valid reference mask via --reference. "
                    "Synthetic / demo reference generation is prohibited per challenge guidelines."
                )

        ref_mask = _aligned_reference(pred_src, reference_path)
        result = compare_masks(pred_mask, ref_mask, pred_src.transform, pred_src.crs)
        result["case_study"] = "August 2026 Trishuli flood (Bhote Koshi / Trishuli corridor)"
        result["prediction_file"] = str(pred_path)
        result["reference_file"] = str(reference_path)
        result["reference_source"] = "Copernicus Emergency Management Service (EMSR927)"
        result["evaluation_only"] = True
        result["nodata_policy"] = "Only pixels valid in both masks are scored."
        result["mandatory_attribution"] = (
            "Contains modified Copernicus Sentinel data 2026. "
            "Evaluated against European Union, Copernicus Emergency Management Service data (EMSR927)."
        )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", type=Path, required=True, help="Predicted 0/1/255 flood-mask GeoTIFF")
    parser.add_argument("--reference", type=Path, default=None, help="EMSR927 reference-mask GeoTIFF")
    parser.add_argument("--out-json", type=Path, default=Path("outputs/trishuli_emsr927_benchmark.json"))
    args = parser.parse_args()
    result = run_emsr927_benchmark(args.pred, args.reference, args.out_json)
    print(json.dumps({"metrics": result["metrics"], "area_comparison_km2": result["area_comparison_km2"], "saved": str(args.out_json)}, indent=2))


if __name__ == "__main__":
    main()
