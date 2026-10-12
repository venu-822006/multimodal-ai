"""
evaluate.py  —  FloodSight AI | Evaluation & Model Comparison
=============================================================
Calculates rigorous segmentation metrics comparing predicted flood masks
against ground truth masks (e.g. KuroSiwo held-out scenes or EMSR927).

Metrics reported:
  - Flood Intersection-over-Union (IoU / Jaccard Index)
  - Precision, Recall, F1-Score
  - Overall Pixel Accuracy
  - Confusion Matrix (TP, FP, TN, FN)

Guarantees:
  - Exact filename pairing (no loose prefix guessing)
  - Strict shape, CRS, and geotransform equality (no silent cropping)
  - Direct evaluation on held-out manifests (--manifest)

Usage:
    # Single scene evaluation:
    python evaluate.py \
        --pred  outputs/event_001_post_flood_predicted.tif \
        --truth data/processed/masks/event_001.tif

    # Batch comparison of methods across test directory:
    python evaluate.py \
        --pred-dir  outputs/predictions \
        --truth-dir data/processed/masks \
        --out-json  outputs/metrics.json

    # Strictly evaluate a held-out manifest:
    python evaluate.py \
        --manifest  checkpoints/val_manifest.json \
        --pred-dir  outputs/predictions \
        --out-json  outputs/metrics.json
"""

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import rasterio


def compute_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    """Compute binary classification metrics ignoring nodata pixels (255).

    Class 1 = Flood, Class 0 = Background.
    """
    valid = truth != 255
    bad_truth = set(np.unique(truth[valid]).tolist()) - {0, 1}
    bad_pred = set(np.unique(pred).tolist()) - {0, 1, 255}
    if bad_truth:
        raise ValueError(f"Truth mask contains labels other than 0, 1, 255: {sorted(bad_truth)}")
    if bad_pred:
        raise ValueError(f"Prediction contains labels other than 0, 1, 255: {sorted(bad_pred)}")
    p = (pred == 1) & valid
    t = (truth == 1) & valid

    tp = int((p & t).sum())
    fp = int((p & ~t).sum())
    fn = int((~p & t).sum())
    tn = int((~p & ~t).sum())

    total_valid = tp + fp + fn + tn
    prediction_coverage = float(((pred != 255) & valid).sum() / max(total_valid, 1))
    if total_valid == 0:
        return {
            "iou": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0,
            "specificity": 0.0, "balanced_accuracy": 0.0, "accuracy": 0.0,
            "prediction_coverage": 0.0, "nodata_predictions": 0,
            "tp": 0, "fp": 0, "fn": 0, "tn": 0, "total_valid": 0,
        }

    intersection = tp
    union = tp + fp + fn

    if union == 0:
        # Both ground truth and prediction contain 0 flood pixels: perfect agreement on negative scene
        iou = 1.0
        precision = 1.0
        recall = 1.0
        f1 = 1.0
    else:
        iou = intersection / max(union, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-7)
    specificity = tn / max(tn + fp, 1)
    accuracy = (tp + tn) / total_valid

    return {
        "iou": round(float(iou), 4),
        "f1": round(float(f1), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "specificity": round(float(specificity), 4),
        "balanced_accuracy": round(float((recall + specificity) / 2), 4),
        "accuracy": round(float(accuracy), 4),
        "prediction_coverage": round(prediction_coverage, 4),
        "nodata_predictions": int(((pred == 255) & valid).sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "total_valid": total_valid,
    }


def evaluate_pair(pred_path: Path, truth_path: Path) -> dict:
    """Evaluate a single predicted mask against ground truth.

    Fails with ValueError if dimensions, declared CRSs, or affine transforms mismatch.
    Missing CRS metadata is permitted only when both rasters lack it and their
    dimensions and affine transforms still match exactly.
    Silent cropping or resizing is strictly prohibited to guarantee data integrity.
    """
    with rasterio.open(pred_path) as src_p:
        pred = src_p.read(1)
        shape_p = pred.shape
        crs_p = src_p.crs
        trans_p = src_p.transform

    with rasterio.open(truth_path) as src_t:
        truth = src_t.read(1)
        shape_t = truth.shape
        crs_t = src_t.crs
        trans_t = src_t.transform

    if shape_p != shape_t:
        raise ValueError(
            f"Spatial dimension mismatch: prediction {pred_path.name} is {shape_p}, "
            f"ground truth {truth_path.name} is {shape_t}. "
            f"Silent cropping is prohibited to ensure evaluation integrity."
        )

    if (crs_p is None) != (crs_t is None):
        raise ValueError(
            f"CRS is present on only one raster: {pred_path.name}, {truth_path.name}."
        )
    if crs_p is None and crs_t is None:
        # Some training tiles carry a local affine transform without a declared
        # CRS. They can still be compared when both rasters share that transform.
        pass
    if crs_p != crs_t:
        raise ValueError(
            f"CRS mismatch: prediction {pred_path.name} is {crs_p}, "
            f"ground truth {truth_path.name} is {crs_t}."
        )

    if not np.allclose(list(trans_p)[:6], list(trans_t)[:6], atol=1e-9, rtol=0):
        raise ValueError(
            f"Affine geotransform mismatch between prediction {pred_path.name} "
            f"and ground truth {truth_path.name}."
        )

    metrics = compute_metrics(pred, truth)
    metrics["scene"] = pred_path.stem
    metrics["truth"] = truth_path.stem
    return metrics


def match_truth_file(pred_file: Path, truth_dir: Path) -> Path | None:
    """Find the exact ground truth mask file corresponding to a predicted mask.

    Removes inference suffixes (_flood_predicted, _post, _after) to reconstruct
    the exact mask stem (<act>_<aoi>_<patch_hash>).
    """
    stem = pred_file.stem
    for sfx in ("_flood_predicted", "_predicted"):
        if stem.endswith(sfx):
            stem = stem[: -len(sfx)]
            break
    for sfx in ("_post", "_after"):
        if stem.endswith(sfx):
            stem = stem[: -len(sfx)]
            break

    candidate = truth_dir / f"{stem}.tif"
    if candidate.exists():
        return candidate
    candidate = truth_dir / f"{stem}.tiff"
    if candidate.exists():
        return candidate
    return None


def summarize_scenes(results: list[dict]) -> dict:
    """Report both scene-macro and pixel-pooled scores so large scenes don't hide weak ones."""
    if not results:
        raise ValueError("Cannot summarize an empty evaluation result set.")
    macro = {
        f"mean_{key}": float(np.mean([row[key] for row in results]))
        for key in ("iou", "f1", "precision", "recall", "specificity", "balanced_accuracy", "accuracy", "prediction_coverage")
    }
    # Separate breakdown: positive-flood scenes vs zero-flood scenes
    pos_results = [r for r in results if (r["tp"] + r["fn"]) > 0]
    zero_results = [r for r in results if (r["tp"] + r["fn"]) == 0]

    pos_summary = {}
    if pos_results:
        pos_summary = {
            "scene_count": len(pos_results),
            "mean_iou": float(np.mean([r["iou"] for r in pos_results])),
            "mean_f1": float(np.mean([r["f1"] for r in pos_results])),
            "mean_precision": float(np.mean([r["precision"] for r in pos_results])),
            "mean_recall": float(np.mean([r["recall"] for r in pos_results])),
        }

    zero_summary = {}
    if zero_results:
        clean_count = sum(1 for r in zero_results if r["fp"] == 0)
        zero_summary = {
            "scene_count": len(zero_results),
            "clean_count": clean_count,
            "clean_pct": round(100.0 * clean_count / len(zero_results), 2),
            "mean_fp_pixels": float(np.mean([r["fp"] for r in zero_results])),
        }

    tp = sum(row["tp"] for row in results)
    fp = sum(row["fp"] for row in results)
    fn = sum(row["fn"] for row in results)
    tn = sum(row["tn"] for row in results)
    valid = tp + fp + fn + tn
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        **macro,
        "positive_scenes_eval": pos_summary,
        "zero_flood_scenes_eval": zero_summary,
        "pixel_weighted": {
            "iou": tp / max(tp + fp + fn, 1),
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "balanced_accuracy": (recall + specificity) / 2,
            "accuracy": (tp + tn) / max(valid, 1),
            "prediction_coverage": (valid - sum(row["nodata_predictions"] for row in results)) / max(valid, 1),
            "valid_pixels": valid,
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred",      type=Path, help="Single predicted mask GeoTIFF")
    p.add_argument("--truth",     type=Path, help="Single ground truth mask GeoTIFF")
    p.add_argument("--pred-dir",  type=Path, help="Directory containing predicted GeoTIFFs")
    p.add_argument("--truth-dir", type=Path, help="Directory containing ground truth GeoTIFFs")
    p.add_argument("--manifest",  type=Path, help="JSON manifest of held-out pairs (e.g. checkpoints/val_manifest.json)")
    p.add_argument("--out-json",  type=Path, default=Path("outputs/eval_metrics.json"))
    args = p.parse_args()

    # ── Single pair ──────────────────────────────────────────────────────────
    if args.pred and args.truth:
        metrics = evaluate_pair(args.pred, args.truth)
        print("=" * 55)
        print(f" Evaluation: {metrics['scene']}")
        print("=" * 55)
        print(f"  Flood IoU:         {metrics['iou']*100:.2f}%")
        print(f"  F1-Score:          {metrics['f1']:.4f}")
        print(f"  Precision:         {metrics['precision']*100:.2f}%")
        print(f"  Recall:            {metrics['recall']*100:.2f}%")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']*100:.2f}%")
        print(f"  Pixel Accuracy:    {metrics['accuracy']*100:.2f}%")
        print(f"  TP: {metrics['tp']:,} | FP: {metrics['fp']:,} | FN: {metrics['fn']:,}")
        print("=" * 55)
        if args.out_json:
            args.out_json.parent.mkdir(parents=True, exist_ok=True)
            with open(args.out_json, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

    # ── Manifest-based evaluation (Clean Held-Out Split) ──────────────────────
    elif args.manifest and args.pred_dir:
        if not args.manifest.exists():
            raise FileNotFoundError(f"Manifest not found: {args.manifest}")
        with open(args.manifest, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        held_out_pairs = manifest_data.get("val_pairs") or manifest_data.get("test_pairs", [])
        held_out_events = manifest_data.get("val_event_ids") or manifest_data.get("test_event_ids", [])
        if not held_out_pairs:
            raise ValueError(f"No val_pairs or test_pairs found in manifest {args.manifest}")

        results = []
        missing = []
        for item in held_out_pairs:
            img_stem = Path(item["image_name"]).stem
            pred_file = args.pred_dir / f"{img_stem}_flood_predicted.tif"
            truth_file = Path(item["mask"])

            if not pred_file.exists():
                missing.append(pred_file.name)
                continue
            if not truth_file.exists():
                raise FileNotFoundError(f"Ground truth mask referenced in manifest does not exist: {truth_file}")

            m = evaluate_pair(pred_file, truth_file)
            results.append(m)

        if missing:
            raise RuntimeError(
                f"Held-out evaluation is incomplete: {len(missing)} of {len(held_out_pairs)} "
                f"predictions are missing (first: {', '.join(missing[:5])})."
            )
        if not results:
            raise RuntimeError(
                f"No predicted files matched manifest pairs in {args.pred_dir}"
            )

        summary = summarize_scenes(results)
        mean_iou = summary["mean_iou"]
        mean_f1 = summary["mean_f1"]
        mean_prec = summary["mean_precision"]
        mean_rec = summary["mean_recall"]
        mean_bal_acc = summary["mean_balanced_accuracy"]
        mean_acc = summary["mean_accuracy"]

        print("=" * 65)
        print(f" Held-Out Manifest Evaluation ({len(results)} scenes)")
        print(f" Held-out Events: {held_out_events or 'N/A'}")
        print("=" * 65)
        print(f"  Mean Flood IoU:         {mean_iou*100:.2f}%")
        print(f"  Mean F1-Score:          {mean_f1:.4f}")
        print(f"  Mean Precision:         {mean_prec*100:.2f}%")
        print(f"  Mean Recall:            {mean_rec*100:.2f}%")
        print(f"  Mean Balanced Accuracy: {mean_bal_acc*100:.2f}%")
        print(f"  Mean Pixel Accuracy:    {mean_acc*100:.2f}%")
        print("=" * 65)

        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({
                "summary": {
                    **summary,
                    "scenes_evaluated": len(results),
                    "held_out_events": held_out_events,
                },
                "per_scene": results,
            }, f, indent=2)
        print(f"  [OK] Saved report -> {args.out_json}")

    # ── Directory-based exact matching ───────────────────────────────────────
    elif args.pred_dir and args.truth_dir:
        pred_files = sorted(args.pred_dir.glob("*.tif"))
        results = []
        for pf in pred_files:
            tf = match_truth_file(pf, args.truth_dir)
            if tf is None:
                continue
            m = evaluate_pair(pf, tf)
            results.append(m)

        if not results:
            raise RuntimeError(
                f"No exact prediction/truth matches found between {args.pred_dir} and {args.truth_dir}"
            )

        summary = summarize_scenes(results)
        mean_iou = summary["mean_iou"]
        mean_f1 = summary["mean_f1"]
        mean_prec = summary["mean_precision"]
        mean_rec = summary["mean_recall"]

        print("=" * 60)
        print(f" Batch Evaluation ({len(results)} scenes)")
        print("=" * 60)
        print(f"  Mean Flood IoU:   {mean_iou*100:.2f}%")
        print(f"  Mean F1-Score:    {mean_f1:.4f}")
        print(f"  Mean Precision:   {mean_prec*100:.2f}%")
        print(f"  Mean Recall:      {mean_rec*100:.2f}%")
        print("=" * 60)

        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({
                "summary": {
                    **summary,
                    "scenes_evaluated": len(results),
                },
                "per_scene": results,
            }, f, indent=2)
        print(f"  [OK] Saved report -> {args.out_json}")
    else:
        p.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
