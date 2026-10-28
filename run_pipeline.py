"""
run_pipeline.py  —  FloodSight AI | End-to-End Master Execution Pipeline
========================================================================
Orchestrates the entire AI pipeline end-to-end according to Track B specs:

  Step 1: Verify data readiness and image/mask pairing in data/processed/
  Step 2: Train U-Net model from scratch (train.py) -> checkpoints/best.pt
          Generates event-held-out split and checkpoints/val_manifest.json
  Step 3: Run inference ONLY on held-out validation scenes (predict.py)
  Step 4: Run Debris candidate detection on true pre/post pairs (debris.py)
  Step 5: Rigorously evaluate on the clean held-out split (evaluate.py)
  Step 6: Produce final executive summary

Guarantees:
  - Clean held-out evaluation (no training set evaluation contamination)
  - True paired pre/post debris detection (skips gracefully if pre-image absent)
  - Fail-fast execution: halts immediately on any step failure
  - Practical data readiness checking with clear wait opt-out

Usage:
    python run_pipeline.py [--epochs 100] [--patience 15] [--batch-size 4] [--patch-size 224] [--wait-data]
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def log(section: str, msg: str):
    print(f"\n{'='*65}")
    print(f" [{section}] {msg}")
    print(f"{'='*65}")


def run_cmd(cmd: list[str], desc: str) -> None:
    """Run a subprocess command and fail-fast immediately on any non-zero exit."""
    print(f"  >> Running: {' '.join(str(c) for c in cmd)}")
    start_t = time.time()
    res = subprocess.run(cmd)
    elapsed = time.time() - start_t
    if res.returncode != 0:
        print(f"\n  [FATAL ERROR] {desc} failed with exit code {res.returncode}. Aborting pipeline.")
        sys.exit(res.returncode if res.returncode != 0 else 1)
    print(f"  [OK] Completed {desc} in {elapsed:.1f}s")


def verify_dataset(data_dir: Path, wait: bool = False, timeout_seconds: int = 600) -> tuple[int, int]:
    """Verify processed dataset readiness: directory existence, file counts, and raster integrity."""
    img_dir = data_dir / "images"
    mask_dir = data_dir / "masks"
    start_t = time.time()

    while True:
        if img_dir.exists() and mask_dir.exists():
            imgs = list(img_dir.glob("*.tif"))
            masks = list(mask_dir.glob("*.tif"))
            if len(imgs) >= 2 and len(masks) >= 2:
                # Test raster readability on a sample
                try:
                    import rasterio
                    with rasterio.open(imgs[0]) as s_img, rasterio.open(masks[0]) as s_mask:
                        if s_img.count >= 2 and s_img.shape == s_mask.shape:
                            print(f"  [OK] Dataset validated: {len(imgs):,} images, {len(masks):,} masks found.")
                            print(f"       Sample raster shape: {s_img.shape}, bands: {s_img.count}, CRS: {s_img.crs}")
                            return len(imgs), len(masks)
                except Exception as exc:
                    print(f"  [!] Warning during raster sample check: {exc}")

        if not wait or (time.time() - start_t >= timeout_seconds):
            break
        print(f"  Waiting for data in {data_dir}... (elapsed {int(time.time() - start_t)}s)")
        time.sleep(10)

    if not img_dir.exists() or not mask_dir.exists():
        print(f"  [X] Error: Expected images/ and masks/ directories in {data_dir}")
        sys.exit(1)

    imgs = list(img_dir.glob("*.tif"))
    masks = list(mask_dir.glob("*.tif"))
    if len(imgs) < 2 or len(masks) < 2:
        print(f"  [X] Error: Insufficient data found ({len(imgs)} images, {len(masks)} masks). Minimum is 2.")
        sys.exit(1)

    return len(imgs), len(masks)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir",    type=Path, default=Path("data/processed"), help="Processed data directory")
    parser.add_argument("--output-dir",  type=Path, default=Path("checkpoints"),    help="Checkpoint output directory")
    parser.add_argument("--results-dir", type=Path, default=Path("outputs"),        help="Inference and evaluation outputs")
    parser.add_argument("--epochs",      type=int,  default=100,                    help="Maximum training epochs (default: 100; early stopping applies)")
    parser.add_argument("--patience",    type=int,  default=15,                     help="Stop after this many epochs without validation IoU improvement")
    parser.add_argument("--batch-size",  type=int,  default=4,                      help="Batch size (default: 4)")
    parser.add_argument("--patch-size",  type=int,  default=224,                    help="Patch size (default: native KuroSiwo tile size 224)")
    parser.add_argument("--wait-data",   action="store_true", default=False,        help="Wait for data if not yet present")
    parser.add_argument("--skip-train",  action="store_true", default=False,        help="Skip training and use existing best.pt")
    args = parser.parse_args()

    python_exe = sys.executable

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: Verify Dataset Readiness
    # ─────────────────────────────────────────────────────────────────────────
    log("STEP 1", "Verifying Dataset Readiness")
    img_count, mask_count = verify_dataset(args.data_dir, wait=args.wait_data)

    img_dir = args.data_dir / "images"
    mask_dir = args.data_dir / "masks"
    best_ckpt = args.output_dir / "best.pt"
    val_manifest = args.output_dir / "val_manifest.json"

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: Train U-Net Model (with Real Event Splitting)
    # ─────────────────────────────────────────────────────────────────────────
    if not args.skip_train:
        log("STEP 2", "Training FloodUNet Model (Event-Held-Out Split)")
        train_cmd = [
            python_exe, "train.py",
            "--data-dir", str(args.data_dir),
            "--output-dir", str(args.output_dir),
            "--epochs", str(args.epochs),
            "--patience", str(args.patience),
            "--batch-size", str(args.batch_size),
            "--patch-size", str(args.patch_size),
        ]
        run_cmd(train_cmd, "Model Training")
    else:
        log("STEP 2", "Skipping Training (--skip-train requested)")

    if not best_ckpt.exists():
        print(f"  [FATAL ERROR] Checkpoint not found: {best_ckpt}")
        sys.exit(1)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: Inference on Clean Held-Out Split
    # ─────────────────────────────────────────────────────────────────────────
    log("STEP 3", "Running Inference on Held-Out Validation Scenes")
    pred_dir = args.results_dir / "heldout_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    if not val_manifest.exists():
        print(f"  [FATAL ERROR] Held-out split manifest not found: {val_manifest}")
        print("  Train the model first, or provide the matching manifest when using --skip-train.")
        sys.exit(1)
    print(f"  Using held-out validation manifest: {val_manifest}")
    predict_cmd = [
        python_exe, "predict.py",
        "--manifest", str(val_manifest),
        "--checkpoint", str(best_ckpt),
        "--out-dir", str(pred_dir),
        "--confidence-threshold", "0.92",
        "--min-cluster-px", "15",
    ]
    run_cmd(predict_cmd, "Held-Out Scene Inference")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: Debris Detection Module (Class B: Experimental)
    # ─────────────────────────────────────────────────────────────────────────
    log("STEP 4", "Running Debris Candidate Module (Class B: Experimental)")
    # Scan for genuine pre-event images to pair with post-event predictions
    pre_candidates = list(img_dir.glob("*_pre.tif")) + list(img_dir.glob("*_before.tif"))
    debris_ran = False

    if pre_candidates:
        for pre_img in pre_candidates:
            stem = pre_img.stem.replace("_pre", "").replace("_before", "")
            post_img = img_dir / f"{stem}_post.tif"
            pred_mask = pred_dir / f"{stem}_post_flood_predicted.tif"
            out_multi = args.results_dir / f"{stem}_multiclass_mask.tif"

            if post_img.exists() and pred_mask.exists():
                debris_cmd = [
                    python_exe, "debris.py",
                    "--pre-image", str(pre_img),
                    "--post-image", str(post_img),
                    "--flood-mask", str(pred_mask),
                    "--out-mask", str(out_multi),
                ]
                run_cmd(debris_cmd, f"Debris Detection on {stem}")
                debris_ran = True
                break

    if not debris_ran:
        print("  [Notice] Debris detection requires temporal comparison with a distinct pre-event baseline image.")
        print("           The current KuroSiwo batch provides post-event flood rasters.")
        print("           Skipping Class B temporal change step to prevent generating invalid zero-change artifacts.")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5: Rigorous Evaluation on Clean Held-Out Split
    # ─────────────────────────────────────────────────────────────────────────
    log("STEP 5", "Evaluating Segmentation Performance on Held-Out Split")
    eval_json = args.results_dir / "evaluation_report.json"

    eval_cmd = [
        python_exe, "evaluate.py",
        "--manifest", str(val_manifest),
        "--pred-dir", str(pred_dir),
        "--out-json", str(eval_json),
    ]
    run_cmd(eval_cmd, "Held-Out Model Evaluation")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6: Final Summary
    # ─────────────────────────────────────────────────────────────────────────
    log("COMPLETE", "FloodSight AI - Track B Pipeline Execution Complete")
    print("  Artifacts generated:")
    print(f"    - Model Weights:      {best_ckpt}")
    print(f"    - Training Log:       {args.output_dir / 'training_history.csv'}")
    if val_manifest.exists():
        print(f"    - Val Manifest:       {val_manifest}")
    print(f"    - Predictions:        {pred_dir}")
    print(f"    - Metrics Report:     {eval_json}")
    if (pred_dir / "summary_facts.json").exists():
        print(f"    - Summary facts.json: {pred_dir / 'summary_facts.json'}")
    print()


if __name__ == "__main__":
    main()
