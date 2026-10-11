"""Train FloodSight's SAR flood segmentation model from random initialization."""

import argparse
import csv
import hashlib
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import KuroSiwoDataset, estimate_sar_stats, get_train_transforms, get_val_transforms, set_sar_stats
from model import create_unet_model


def _dataset_fingerprint(pairs: list[tuple[Path, Path]], data_dir: Path) -> str:
    """Fingerprint the exact input files so a resumed run cannot silently change data."""
    rows = []
    root = data_dir.resolve()
    for image, mask in pairs:
        for path in (image, mask):
            stat = path.stat()
            rows.append((path.resolve().relative_to(root).as_posix(), stat.st_size, stat.st_mtime_ns))
    payload = json.dumps(sorted(rows), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_save_checkpoint(checkpoint: dict, path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


def _tile_key(path: Path) -> str:
    """Extract tile key to pair images and masks (e.g. 1111002_00_4b_..._post -> 1111002_00_4b_...)."""
    stem = path.stem.lower()
    for suffix in ("_post", "_pre", "_before", "_after"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def _event_id(path: Path) -> str:
    """Extract real disaster event ID from filename (first segment before underscore).

    KuroSiwo filenames follow: <event_id>_<aoi_id>_<patch_hash>_post.tif
    e.g. '1111004_01_4b_..._post.tif' -> event '1111004'
    """
    return path.stem.split("_")[0]


def discover_pairs(data_dir: Path) -> list[tuple[Path, Path]]:
    """Match processed 2-band images to masks by tile key.

    Layout: data_dir/images/*.tif and data_dir/masks/*.tif. If both pre- and
    post-event images exist, post is selected for supervised flood mapping.
    """
    image_dir, mask_dir = data_dir / "images", data_dir / "masks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Expected images/ and masks/ folders under {data_dir}")
    masks = {p.stem.lower(): p for p in mask_dir.glob("*") if p.suffix.lower() in {".tif", ".tiff"}}
    images_by_tile: dict[str, list[Path]] = {}
    for image in image_dir.glob("*"):
        if image.suffix.lower() in {".tif", ".tiff"}:
            images_by_tile.setdefault(_tile_key(image), []).append(image)

    pairs = []
    missing_masks = []
    for key, images in sorted(images_by_tile.items()):
        mask = masks.get(key)
        if mask is None:
            missing_masks.append(key)
            continue
        images.sort(key=lambda p: (0 if p.stem.lower().endswith(("_post", "_after")) else 1, p.name.lower()))
        pairs.append((images[0], mask))
    if not pairs:
        suffix = f"; images without matching masks: {', '.join(missing_masks[:5])}" if missing_masks else ""
        raise ValueError(f"No image/mask pairs found in {data_dir}{suffix}")
    if missing_masks:
        print(f"Skipping {len(missing_masks)} image tile(s) without matching masks.")
    return pairs


def split_by_event(pairs: list[tuple[Path, Path]], val_fraction: float, seed: int):
    """Partition scenes by REAL disaster event ID to prevent spatial data leakage.

    All tiles belonging to a given event are placed exclusively into either
    the training split or the validation split.
    """
    groups: dict[str, list[tuple[Path, Path]]] = {}
    for pair in pairs:
        eid = _event_id(pair[0])
        groups.setdefault(eid, []).append(pair)

    keys = sorted(groups.keys())
    if len(keys) < 2:
        raise ValueError(
            f"At least two distinct flood events are needed for an event-held-out split; found {len(keys)}: {keys}"
        )

    rng = random.Random(seed)
    rng.shuffle(keys)

    total_tiles = len(pairs)
    target_val_tiles = int(total_tiles * val_fraction)

    # Greedily allocate full events to validation until target fraction is met
    val_keys: list[str] = []
    current_val_tiles = 0
    for k in keys:
        if current_val_tiles + len(groups[k]) <= target_val_tiles or not val_keys:
            val_keys.append(k)
            current_val_tiles += len(groups[k])
            if current_val_tiles >= target_val_tiles:
                break

    # Guard: never put all events in validation
    if len(val_keys) == len(keys):
        val_keys.pop()

    val_keys_set = set(val_keys)
    train_keys = [k for k in keys if k not in val_keys_set]

    train_pairs = [pair for k in train_keys for pair in groups[k]]
    val_pairs   = [pair for k in val_keys for pair in groups[k]]

    print("\n  [Split by Real Flood Event]")
    print(f"    Train Events ({len(train_keys)}): {train_keys} -> {len(train_pairs):,} tiles")
    print(f"    Val Events   ({len(val_keys)}): {val_keys} -> {len(val_pairs):,} tiles "
          f"({len(val_pairs)/total_tiles*100:.1f}% held-out)\n")

    return train_pairs, val_pairs, train_keys, val_keys


def make_loader(pairs, patch_size, batch_size, training, workers, device, samples_per_image: int = 1):
    images = [str(pair[0]) for pair in pairs]
    masks = [str(pair[1]) for pair in pairs]
    transform = get_train_transforms(patch_size) if training else get_val_transforms(patch_size)
    dataset = KuroSiwoDataset(
        images,
        masks,
        patch_size=patch_size,
        transforms=transform,
        use_rasterio=True,
        samples_per_image=samples_per_image if training else 1,
        grid_crops=not training,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def flood_iou(predictions: torch.Tensor, targets: torch.Tensor) -> tuple[int, int, int]:
    valid = targets != 255
    pred_flood = (predictions == 1) & valid
    true_flood = (targets == 1) & valid
    intersection = int((pred_flood & true_flood).sum().item())
    union = int((pred_flood | true_flood).sum().item())
    correct = int(((predictions == targets) & valid).sum().item())
    return intersection, union, correct


# ---------------------------------------------------------------------------
# Composite Loss: Weighted CE + Focal + Dice
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Focal loss (Lin et al. 2017) for hard-example mining on imbalanced SAR data."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None, ignore_index: int = 255):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Derive p_t from unweighted CE. Applying class weights before exp()
        # makes p_t cease to be a probability and distorts focal modulation.
        ce = nn.functional.cross_entropy(
            logits, targets, ignore_index=self.ignore_index, reduction="none"
        )
        pt = torch.exp(-ce)
        if self.weight is not None:
            safe_targets = targets.clamp(0, len(self.weight) - 1)
            ce = ce * self.weight[safe_targets]
        loss = ((1 - pt) ** self.gamma) * ce
        valid_mask = targets != self.ignore_index
        return loss[valid_mask].mean()


class DiceLoss(nn.Module):
    """Soft Dice loss over the flood class (class=1), ignoring nodata=255."""

    def __init__(self, smooth: float = 1.0, ignore_index: int = 255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        flood_prob = torch.softmax(logits, dim=1)[:, 1]   # P(flood) -- (N, H, W)
        valid      = (targets != self.ignore_index).float()
        target_f   = (targets == 1).float() * valid
        pred_f     = flood_prob * valid
        intersection = (pred_f * target_f).sum()
        denom = pred_f.sum() + target_f.sum() + self.smooth
        return 1.0 - (2.0 * intersection + self.smooth) / denom


class ComboLoss(nn.Module):
    """Weighted CE (35%) + Focal (35%) + Dice (30%) for KuroSiwo class imbalance (~2% flood)."""

    def __init__(self, class_weights: torch.Tensor, ignore_index: int = 255):
        super().__init__()
        self.ce    = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.focal = FocalLoss(gamma=2.0, weight=class_weights, ignore_index=ignore_index)
        self.dice  = DiceLoss(ignore_index=ignore_index)
        self.w_ce    = 0.35
        self.w_focal = 0.35
        self.w_dice  = 0.30

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.w_ce    * self.ce(logits, targets)
            + self.w_focal * self.focal(logits, targets)
            + self.w_dice  * self.dice(logits, targets)
        )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, max_grad_norm: float = 1.0):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    total_intersection = total_union = total_correct = total_valid = 0
    total_tp = total_fp = total_fn = total_tn = 0
    amp_enabled = device.type == "cuda"
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for images, masks in tqdm(loader, leave=False):
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True)
            if not torch.any(masks != 255):
                continue
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss   = criterion(logits, masks)
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            preds = logits.argmax(dim=1)
            inter, union, correct = flood_iou(preds, masks)
            valid_mask = masks != 255
            pred_flood = (preds == 1) & valid_mask
            true_flood = (masks == 1) & valid_mask
            total_tp += int((pred_flood & true_flood).sum().item())
            total_fp += int((pred_flood & ~true_flood).sum().item())
            total_fn += int((~pred_flood & true_flood).sum().item())
            total_tn += int((~pred_flood & ~true_flood & valid_mask).sum().item())
            valid = int((masks != 255).sum().item())
            total_loss     += loss.item() * images.size(0)
            total_samples  += images.size(0)
            total_intersection += inter
            total_union        += union
            total_correct      += correct
            total_valid        += valid
    if total_valid == 0:
        raise ValueError("The epoch contains no labelled pixels (all mask values are nodata=255).")
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    specificity = total_tn / max(total_tn + total_fp, 1)
    return {
        "loss":           total_loss / max(total_samples, 1),
        "flood_iou":      total_intersection / max(total_union, 1),
        "pixel_accuracy": total_correct / total_valid,
        "pixel_error_rate": 1.0 - total_correct / total_valid,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",      type=Path,  required=True,
                        help="Folder containing images/ and masks/ GeoTIFFs")
    parser.add_argument("--output-dir",    type=Path,  default=Path("checkpoints"))
    parser.add_argument("--epochs",        type=int,   default=100,
                        help="Total target epoch (including epochs already completed when resuming)")
    parser.add_argument("--patience",      type=int,   default=15,
                        help="Stop after this many epochs without validation IoU improvement")
    parser.add_argument("--batch-size",    type=int,   default=4,
                        help="Conservative default for a 6 GB GPU")
    parser.add_argument("--patch-size",    type=int,   default=224,
                        help="Training crop size (default 224 matches native KuroSiwo tiles)")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--val-fraction",  type=float, default=0.2)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--workers",       type=int,   default=2)
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Path to a checkpoint to resume from (e.g. checkpoints/latest.pt). "
             "Optimizer, scheduler state, and epoch counter are all restored.",
    )
    parser.add_argument(
        "--flood-weight", type=float, default=49.5,
        help="Class weight for the flood pixel class (bg=1.0). "
             "Default 49.5 matches ~2%% flood pixel ratio in KuroSiwo Batch 0.",
    )
    parser.add_argument(
        "--samples-per-image", type=int, default=1,
        help="Number of patches sampled per tile per epoch (default 1 matches native 224x224 KuroSiwo tiles)",
    )
    return parser.parse_args()


def train():
    args = parse_args()
    if (args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.patch_size < 32
            or not 0.0 < args.val_fraction < 1.0 or args.learning_rate <= 0 or args.flood_weight <= 0
            or args.samples_per_image < 1):
        raise ValueError(
            "epochs, patience, batch-size, learning-rate, flood-weight, and samples-per-image must be positive; "
            "patch-size must be at least 32 and val-fraction must be between 0 and 1."
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    pairs = discover_pairs(args.data_dir)
    train_pairs, val_pairs, train_events, val_events = split_by_event(pairs, args.val_fraction, args.seed)
    dataset_fingerprint = _dataset_fingerprint(pairs, args.data_dir)
    print(f"Paired scenes: {len(pairs)}; train: {len(train_pairs)}; validation: {len(val_pairs)}")

    # ── Save held-out validation manifest for trustworthy downstream evaluation ───
    args.output_dir.mkdir(parents=True, exist_ok=True)
    val_manifest = {
        "val_event_ids": val_events,
        "train_event_ids": train_events,
        "val_scene_count": len(val_pairs),
        "train_scene_count": len(train_pairs),
        "val_pairs": [
            {
                "image": str(pair[0].resolve()),
                "mask": str(pair[1].resolve()),
                "image_name": pair[0].name,
                "mask_name": pair[1].name,
                "event_id": _event_id(pair[0]),
            }
            for pair in val_pairs
        ],
    }
    manifest_path = args.output_dir / "val_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(val_manifest, f, indent=2)
    print(f"  [OK] Saved held-out validation manifest -> {manifest_path}")

    sar_mean, sar_std = estimate_sar_stats([str(pair[0]) for pair in train_pairs])
    set_sar_stats(sar_mean, sar_std)
    print(f"Training SAR stats (VV, VH): mean={sar_mean.tolist()}, std={sar_std.tolist()}")

    train_loader = make_loader(train_pairs, args.patch_size, args.batch_size, True,  args.workers, device, args.samples_per_image)
    val_loader   = make_loader(val_pairs,   args.patch_size, args.batch_size, False, args.workers, device, 1)

    model = create_unet_model(in_channels=2, classes=2, base_channels=24).to(device)

    # ── Combo Loss: Weighted CE + Focal + Dice ─────────────────────────────
    class_weights = torch.tensor([1.0, args.flood_weight], device=device)
    criterion = ComboLoss(class_weights=class_weights, ignore_index=255)
    print(f"Loss: ComboLoss(CE+Focal+Dice)  |  bg=1.0, flood={args.flood_weight:.1f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    T0 = max(5, args.epochs // 4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T0, T_mult=2, eta_min=1e-6
    )
    print(f"Scheduler: CosineAnnealingWarmRestarts  T_0={T0}, T_mult=2, eta_min=1e-6")

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # ── Resume from checkpoint ────────────────────────────────────────────
    start_epoch = 1
    best_iou    = -1.0
    epochs_without_improvement = 0
    if args.resume and args.resume.exists():
        saved_train = ckpt.get("train_events")
        saved_val = ckpt.get("val_events")
        if not saved_train or not saved_val:
            raise ValueError(
                "Resume checkpoint lacks event split metadata (train_events/val_events) "
                "needed to safely resume. Start a fresh run without --resume."
            )
        if sorted(saved_train) != sorted(train_events) or sorted(saved_val) != sorted(val_events):
            raise ValueError(
                "Resume checkpoint was trained with a different event split. "
                "Use the original --seed/--val-fraction and dataset, or start a new run."
            )
        if ckpt.get("dataset_fingerprint") and ckpt["dataset_fingerprint"] != dataset_fingerprint:
            raise ValueError("Resume checkpoint dataset files changed; refusing a mixed-data training run.")
        saved_config = ckpt.get("training_config", {})
        for key, current_value in (
            ("learning_rate", args.learning_rate),
            ("flood_weight", args.flood_weight),
            ("batch_size", args.batch_size),
            ("val_fraction", args.val_fraction),
        ):
            if key in saved_config and saved_config[key] != current_value:
                raise ValueError(
                    f"Resume checkpoint used {key}={saved_config[key]}, but the current run uses "
                    f"{current_value}. Keep the original training settings or start a new run."
                )
        if ckpt.get("patch_size") is not None and ckpt["patch_size"] != args.patch_size:
            raise ValueError(
                f"Resume checkpoint uses patch size {ckpt['patch_size']}; got {args.patch_size}. "
                "Pass the checkpoint's patch size explicitly to continue the same run."
            )
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_iou = ckpt.get("best_val_flood_iou")
        if best_iou is None:
            best_iou = ckpt.get("val_flood_iou", -1.0)
            best_checkpoint = args.output_dir / "best.pt"
            if args.resume.name == "latest.pt" and best_checkpoint.exists():
                best_ckpt = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
                best_iou = best_ckpt.get("val_flood_iou", best_iou)
        epochs_without_improvement = int(ckpt.get("epochs_without_improvement", 0))
        norm = ckpt.get("normalization", {})
        if norm.get("mean_db") and norm.get("std_db"):
            saved_mean = np.array(norm["mean_db"], dtype=np.float32)
            saved_std = np.array(norm["std_db"], dtype=np.float32)
            if not (np.allclose(saved_mean, sar_mean, rtol=0, atol=1e-5)
                    and np.allclose(saved_std, sar_std, rtol=0, atol=1e-5)):
                raise ValueError("Training normalization changed since the checkpoint was created.")
            set_sar_stats(saved_mean, saved_std)
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        rng = ckpt.get("rng_state", {})
        if rng:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"].cpu())
            if device.type == "cuda" and rng.get("cuda") is not None:
                torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])
        print(f"Resumed from {args.resume.name}  (epoch {ckpt.get('epoch','?')}, best IoU={best_iou:.4f})")
    elif args.resume:
        raise FileNotFoundError(f"Resume checkpoint does not exist: {args.resume}")

    if start_epoch > args.epochs:
        print(f"Checkpoint already reached target epoch {args.epochs}; nothing to train.")
        return

    history_path = args.output_dir / "training_history.csv"
    fieldnames = [
        "epoch", "train_loss", "train_flood_iou", "train_pixel_accuracy",
        "train_pixel_error_rate", "train_precision", "train_recall", "train_specificity",
        "train_balanced_accuracy", "train_f1",
        "val_loss", "val_flood_iou", "val_pixel_accuracy", "val_pixel_error_rate",
        "val_precision", "val_recall", "val_specificity", "val_balanced_accuracy", "val_f1", "lr",
    ]
    append_history = start_epoch > 1 and history_path.exists()
    if append_history:
        try:
            with history_path.open("r", newline="", encoding="utf-8") as existing:
                rows = csv.reader(existing)
                old_header = next(rows, [])
                history_rows = list(rows)
                history_epochs = [int(row[0]) for row in history_rows if row]
                schema_ok = (
                    old_header == fieldnames
                    and all(len(row) == len(fieldnames) for row in history_rows)
                    and history_epochs == list(range(1, start_epoch))
                )
        except (OSError, csv.Error):
            schema_ok = False
        if not schema_ok:
            archive = history_path.with_name(
                f"training_history_legacy_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.csv"
            )
            history_path.replace(archive)
            print(f"Archived incompatible training log as {archive.name}; starting a clean log.")
            append_history = False
    elif history_path.exists():
        archive = history_path.with_name(
            f"training_history_previous_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.csv"
        )
        history_path.replace(archive)
        print(f"Preserved prior training log as {archive.name}.")
    file_mode = "a" if append_history else "w"

    with history_path.open(file_mode, newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=fieldnames)
        if file_mode == "w":
            writer.writeheader()

        for epoch in range(start_epoch, args.epochs + 1):
            train_metrics = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
            val_metrics   = run_epoch(model, val_loader,   criterion, device)
            scheduler.step(epoch)
            current_lr = optimizer.param_groups[0]["lr"]

            row = {
                "epoch":                epoch,
                "train_loss":           f"{train_metrics['loss']:.6f}",
                "train_flood_iou":      f"{train_metrics['flood_iou']:.6f}",
                "train_pixel_accuracy": f"{train_metrics['pixel_accuracy']:.6f}",
                "train_pixel_error_rate": f"{train_metrics['pixel_error_rate']:.6f}",
                "train_precision":      f"{train_metrics['precision']:.6f}",
                "train_recall":         f"{train_metrics['recall']:.6f}",
                "train_specificity":    f"{train_metrics['specificity']:.6f}",
                "train_balanced_accuracy": f"{train_metrics['balanced_accuracy']:.6f}",
                "train_f1":             f"{train_metrics['f1']:.6f}",
                "val_loss":             f"{val_metrics['loss']:.6f}",
                "val_flood_iou":        f"{val_metrics['flood_iou']:.6f}",
                "val_pixel_accuracy":   f"{val_metrics['pixel_accuracy']:.6f}",
                "val_pixel_error_rate": f"{val_metrics['pixel_error_rate']:.6f}",
                "val_precision":        f"{val_metrics['precision']:.6f}",
                "val_recall":           f"{val_metrics['recall']:.6f}",
                "val_specificity":      f"{val_metrics['specificity']:.6f}",
                "val_balanced_accuracy": f"{val_metrics['balanced_accuracy']:.6f}",
                "val_f1":               f"{val_metrics['f1']:.6f}",
                "lr":                   f"{current_lr:.2e}",
            }
            writer.writerow(row)
            history_file.flush()
            print(
                f"Epoch {epoch:03d}/{args.epochs}: "
                f"train loss {train_metrics['loss']:.4f}, flood IoU {train_metrics['flood_iou']:.4f} | "
                f"val loss {val_metrics['loss']:.4f}, flood IoU {val_metrics['flood_iou']:.4f}, "
                f"F1 {val_metrics['f1']:.4f}, balanced accuracy {val_metrics['balanced_accuracy']:.4f}, "
                f"pixel error {val_metrics['pixel_error_rate']:.4f}  "
                f"lr={current_lr:.2e}"
            )
            improved = val_metrics["flood_iou"] > best_iou
            if improved:
                best_iou = val_metrics["flood_iou"]
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            checkpoint = {
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict":    scaler.state_dict(),
                "model_config":         {"in_channels": 2, "classes": 2, "base_channels": 24},
                "normalization":        {"mean_db": sar_mean.tolist(), "std_db": sar_std.tolist()},
                "patch_size":           args.patch_size,
                "seed":                 args.seed,
                "training_config":      {
                    "learning_rate": args.learning_rate,
                    "flood_weight": args.flood_weight,
                    "batch_size": args.batch_size,
                    "val_fraction": args.val_fraction,
                    "patience": args.patience,
                },
                "dataset_fingerprint":  dataset_fingerprint,
                "val_flood_iou":        val_metrics["flood_iou"],
                "best_val_flood_iou":   best_iou,
                "epochs_without_improvement": epochs_without_improvement,
                "train_events":         train_events,
                "val_events":           val_events,
                "rng_state": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                },
            }
            _atomic_save_checkpoint(checkpoint, args.output_dir / "latest.pt")
            if improved:
                _atomic_save_checkpoint(checkpoint, args.output_dir / "best.pt")
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after {args.patience} epochs without validation flood-IoU improvement.")
                break

    print(f"Training finished. Best flood IoU: {best_iou:.4f}")
    print(f"Best checkpoint: {args.output_dir / 'best.pt'}")
    print(f"Metrics history: {history_path}")


if __name__ == "__main__":
    train()
