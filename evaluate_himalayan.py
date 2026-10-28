"""
evaluate_himalayan.py  —  FloodSight AI | Unseen Himalayan Scenes Benchmark
============================================================================
Evaluates FloodUNet specifically on unseen Himalayan test events that were
held out from both training and model selection:
  "Show how well it works on Himalayan scenes it has never seen."

Guarantees:
  1. Reads only test pairs from a dedicated test manifest.
  2. Rejects event IDs used for training or validation/model selection.
  3. Measures generalization to steep topography and deep gorge terrain.
  4. Reports Flood IoU, F1-Score, Precision, Recall, and Accuracy.
  5. Exports structured JSON benchmark report and terminal summary table.
"""

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch

from evaluate import evaluate_pair, summarize_scenes
from predict import load_checkpoint, predict_one


def evaluate_unseen_himalayan_scenes(
    manifest_path: Path,
    checkpoint_path: Path,
    out_dir: Path = Path("outputs/himalayan_eval"),
    max_scenes: int = 0,
) -> dict:
    """Run end-to-end inference and evaluation strictly on held-out Himalayan events."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std, p_size, ckpt = load_checkpoint(checkpoint_path, device)
    test_pairs = manifest_data.get("test_pairs", [])
    if not test_pairs:
        raise ValueError("A dedicated test manifest with test_pairs is required; validation pairs are not an unseen test.")
    train_events = {str(value) for value in ckpt.get("train_events", [])}
    val_events = {str(value) for value in ckpt.get("val_events", [])}
    declared_test_events = {str(value) for value in ckpt.get("test_events", [])}
    manifest_test_events = {str(value) for value in manifest_data.get("test_event_ids", [])}
    if not declared_test_events or not manifest_test_events:
        raise ValueError("Checkpoint and test manifest must record event split metadata to verify an unseen test.")
    if manifest_test_events != declared_test_events:
        raise ValueError("The manifest test events do not match the checkpoint's recorded test events.")
    if manifest_test_events & (train_events | val_events):
        raise ValueError("Test event leakage detected: an evaluation event was used in training or model selection.")
    selected_pairs = [p for p in test_pairs if str(p.get("event_id")) in manifest_test_events]
    if not selected_pairs:
        raise ValueError("The test manifest contains no scenes for its declared held-out test events.")
    if max_scenes < 0:
        raise ValueError("max_scenes must be zero (all scenes) or a positive scene limit.")
    if max_scenes > 0:
        selected_pairs = selected_pairs[:max_scenes]
    print(f"\n  [Himalayan Benchmark] Evaluating {len(selected_pairs)} test scenes from unseen events {sorted(manifest_test_events)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for item in selected_pairs:
        img_path = Path(item["image"])
        mask_path = Path(item["mask"])

        if not img_path.exists() or not mask_path.exists():
            raise FileNotFoundError(f"Manifest scene or label is missing: {img_path} / {mask_path}")

        # Inference
        predict_one(
            scene_path=img_path,
            model=model,
            sar_mean=mean,
            sar_std=std,
            patch_size=p_size,
            stride=p_size // 2,
            device=device,
            out_dir=out_dir,
            ckpt_path=checkpoint_path,
            ckpt=ckpt,
        )

        pred_tif = out_dir / f"{img_path.stem}_flood_predicted.tif"
        metrics = evaluate_pair(pred_tif, mask_path)
        metrics["event_id"] = item.get("event_id")
        results.append(metrics)

    if not results:
        raise RuntimeError("No scenes were successfully evaluated.")

    summary = summarize_scenes(results)
    mean_iou = summary["mean_iou"]
    mean_f1 = summary["mean_f1"]
    mean_prec = summary["mean_precision"]
    mean_rec = summary["mean_recall"]
    mean_bal_acc = summary["mean_balanced_accuracy"]
    mean_acc = summary["mean_accuracy"]

    report = {
        "benchmark": "Unseen Himalayan Flood Generalization Benchmark",
        "held_out_events": list(set(r["event_id"] for r in results)),
        "training_events": sorted(train_events),
        "validation_events": sorted(val_events),
        "scenes_evaluated": len(results),
        "checkpoint_used": checkpoint_path.name,
        "summary_metrics": {
            **summary,
            "mean_flood_iou": round(mean_iou, 4),
            "mean_f1_score": round(mean_f1, 4),
            "mean_precision": round(mean_prec, 4),
            "mean_recall": round(mean_rec, 4),
            "mean_balanced_accuracy": round(mean_bal_acc, 4),
            "mean_pixel_accuracy": round(mean_acc, 4),
        },
        "per_scene_results": results,
    }

    out_json = out_dir / "himalayan_evaluation_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 65)
    print(" UNSEEN HIMALAYAN SCENES BENCHMARK REPORT")
    print("=" * 65)
    print(f"  Scenes Evaluated:       {len(results):,} (Held-out Events: {report['held_out_events']})")
    print(f"  Mean Flood IoU:         {mean_iou*100:.2f}%")
    print(f"  Mean F1-Score:          {mean_f1:.4f}")
    print(f"  Mean Precision:         {mean_prec*100:.2f}%")
    print(f"  Mean Recall:            {mean_rec*100:.2f}%")
    print(f"  Mean Balanced Accuracy: {mean_bal_acc*100:.2f}%")
    print(f"  Mean Pixel Accuracy:    {mean_acc*100:.2f}%")
    print("=" * 65)
    print(f"  [OK] Saved report -> {out_json}\n")

    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=Path("checkpoints/hackathon_holdout_1111005/test_manifest.json"))
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/hackathon_holdout_1111005/best.pt"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/himalayan_eval"))
    p.add_argument("--max-scenes", type=int, default=0, help="Maximum test scenes; 0 evaluates the complete test manifest")
    args = p.parse_args()

    evaluate_unseen_himalayan_scenes(
        manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        out_dir=args.out_dir,
        max_scenes=args.max_scenes,
    )


if __name__ == "__main__":
    main()
