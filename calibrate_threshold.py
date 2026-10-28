"""Choose a flood-probability threshold using validation data only."""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio


def read_aligned_pair(
    probability_path: Path, truth_path: Path, image_path: Path, min_drop_db: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with (rasterio.open(probability_path) as prob_src, rasterio.open(truth_path) as truth_src,
          rasterio.open(image_path) as image_src):
        if (prob_src.shape != truth_src.shape or prob_src.transform != truth_src.transform
                or prob_src.crs != truth_src.crs or image_src.shape != truth_src.shape
                or image_src.transform != truth_src.transform or image_src.crs != truth_src.crs):
            raise ValueError(f"Probability and truth rasters are not aligned: {probability_path.name}")
        probability = prob_src.read(1).astype(np.float32)
        truth = truth_src.read(1)
        valid = (truth != 255) & np.isfinite(probability) & (probability >= 0.0) & (probability <= 1.0)
        temporal_eligible = np.ones(truth.shape, dtype=bool)
        if image_src.count == 4:
            pre_vv = image_src.read(1, out_dtype="float32")
            post_vv = image_src.read(3, out_dtype="float32")
            temporal_eligible = np.isfinite(pre_vv) & np.isfinite(post_vv) & ((pre_vv - post_vv) >= min_drop_db)
    return probability, truth, valid, temporal_eligible


def metric_counts(
    probability: np.ndarray, truth: np.ndarray, valid: np.ndarray,
    temporal_eligible: np.ndarray, threshold: float,
) -> dict:
    pred = (probability >= threshold) & temporal_eligible
    actual = truth == 1
    tp = int(np.count_nonzero(valid & pred & actual))
    fp = int(np.count_nonzero(valid & pred & ~actual))
    fn = int(np.count_nonzero(valid & ~pred & actual))
    tn = int(np.count_nonzero(valid & ~pred & ~actual))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "iou": tp / max(tp + fp + fn, 1),
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "precision": precision,
        "recall": recall,
        "valid_pixels": int(valid.sum()),
    }


def calibrate(
    manifest_path: Path, probability_dir: Path, output_path: Path,
    min_drop_db: float = 1.5,
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pairs = manifest.get("val_pairs", [])
    if not pairs:
        raise ValueError(f"No val_pairs found in {manifest_path}")

    arrays = []
    missing = []
    for item in pairs:
        stem = Path(item["image_name"]).stem
        probability_path = probability_dir / f"{stem}_flood_probability.tif"
        truth_path = Path(item["mask"])
        if not probability_path.is_file():
            missing.append(str(probability_path))
            continue
        image_path = Path(item["image"])
        arrays.append(read_aligned_pair(probability_path, truth_path, image_path, min_drop_db))
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} validation probability maps; first: {missing[0]}")
    if len(arrays) != len(pairs):
        raise RuntimeError("Validation probability coverage is incomplete")

    candidates = np.arange(0.05, 0.951, 0.01)
    rows = []
    for threshold in candidates:
        counts = [
            metric_counts(prob, truth, valid, temporal_eligible, float(threshold))
            for prob, truth, valid, temporal_eligible in arrays
        ]
        total = {key: sum(row[key] for row in counts) for key in ("tp", "fp", "fn", "tn", "valid_pixels")}
        tp, fp, fn = total["tp"], total["fp"], total["fn"]
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        rows.append({
            "threshold": float(threshold),
            "iou": tp / max(tp + fp + fn, 1),
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "precision": precision,
            "recall": recall,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": total["tn"],
            "valid_pixels": total["valid_pixels"],
        })

    # Maximize pooled validation F1; ties choose the higher threshold to reduce
    # false positives. The independent test manifest must not be used here.
    best = max(rows, key=lambda row: (row["f1"], row["threshold"]))
    report = {
        "selection_metric": "pooled_validation_f1",
        "selected_threshold": best["threshold"],
        "validation_event_ids": manifest.get("val_event_ids", []),
        "validation_scenes": len(arrays),
        "postprocessing": {"temporal_min_vv_drop_db": min_drop_db, "min_cluster_px": 0},
        "selected_metrics": best,
        "threshold_sweep": rows,
        "warning": "Use this threshold only for the matching checkpoint; evaluate once on the separate untouched test set.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("selected_threshold", "validation_event_ids", "validation_scenes", "selected_metrics")}, indent=2))
    print(f"Saved threshold calibration -> {output_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Validation manifest written by train.py")
    parser.add_argument("--prob-dir", type=Path, required=True, help="Folder of *_flood_probability.tif files")
    parser.add_argument("--out-json", type=Path, default=Path("outputs/threshold_calibration.json"))
    parser.add_argument("--min-drop-db", type=float, default=1.5,
                        help="Match predict.py temporal filter for 4-band inputs; 2-band inputs ignore this setting")
    args = parser.parse_args()
    if args.min_drop_db < 0:
        raise ValueError("--min-drop-db cannot be negative")
    calibrate(args.manifest, args.prob_dir, args.out_json, args.min_drop_db)


if __name__ == "__main__":
    main()
