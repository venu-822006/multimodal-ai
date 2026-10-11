"""
evaluate_himalayan.py  —  FloodSight AI | Unseen Himalayan Scenes Benchmark
============================================================================
Evaluates FloodUNet specifically on unseen Himalayan flood events that were
held out during training, satisfying Page 3 of the problem statement:
  "Show how well it works on Himalayan scenes it has never seen."

Guarantees:
  1. Only evaluates scenes from the held-out Himalayan event (e.g. Event 1111005).
  2. Measures generalization to steep topography and deep gorge terrain.
  3. Reports Flood IoU, F1-Score, Precision, Recall, and Accuracy.
  4. Exports structured JSON benchmark report and terminal summary table.
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
    max_scenes: int = 50,
) -> dict:
    """Run end-to-end inference and evaluation strictly on held-out Himalayan events."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    # Filter strictly to the explicitly identified Himalayan validation events.
    val_pairs = manifest_data.get("val_pairs", [])
    himalayan_pairs = [p for p in val_pairs if p.get("event_id") in {"1111005", "1111003"}]
    if not himalayan_pairs:
        raise ValueError(
            "The validation manifest has no held-out Himalayan events 1111005 or 1111003. "
            "Refusing to label unrelated scenes as a Himalayan benchmark."
        )
    if max_scenes < 1:
        raise ValueError("max_scenes must be at least 1.")

    selected_pairs = himalayan_pairs[:max_scenes]
    print(f"\n  [Himalayan Benchmark] Selected {len(selected_pairs)} unseen Himalayan scenes from events {manifest_data.get('val_event_ids')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std, p_size, ckpt = load_checkpoint(checkpoint_path, device)

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
    p.add_argument("--manifest", type=Path, default=Path("checkpoints/val_manifest.json"))
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/himalayan_eval"))
    p.add_argument("--max-scenes", type=int, default=30)
    args = p.parse_args()

    evaluate_unseen_himalayan_scenes(
        manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        out_dir=args.out_dir,
        max_scenes=args.max_scenes,
    )


if __name__ == "__main__":
    main()
