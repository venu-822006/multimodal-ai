import torch
from torch.utils.data import Dataset
import numpy as np
import rasterio
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ---------------------------------------------------------------------------
# Normalization defaults; train.py replaces these with statistics estimated
# from its training split before constructing either dataset.
# ---------------------------------------------------------------------------
SAR_MEAN = np.array([-12.5, -18.0], dtype=np.float32)   # [VV_mean, VH_mean] in dB
SAR_STD  = np.array([4.5,   4.0],   dtype=np.float32)   # [VV_std,  VH_std]  in dB


def set_sar_stats(mean: np.ndarray, std: np.ndarray) -> None:
    """Set per-channel training statistics used by normalization."""
    global SAR_MEAN, SAR_STD
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if mean.shape != (2,) or std.shape != (2,) or np.any(std <= 0):
        raise ValueError("SAR mean/std must each contain two values and std must be positive")
    SAR_MEAN, SAR_STD = mean, std


def estimate_sar_stats(image_paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Estimate per-band dB mean/std over valid pixels in training GeoTIFFs."""
    sums = np.zeros(2, dtype=np.float64)
    sums_sq = np.zeros(2, dtype=np.float64)
    counts = np.zeros(2, dtype=np.int64)
    for path in image_paths:
        with rasterio.open(path) as src:
            if src.count != 2:
                raise ValueError(f"Expected two VV/VH bands in {path}; found {src.count}")
            bands = src.read((1, 2), masked=True, out_dtype="float32")
        for channel in range(2):
            values = np.asarray(bands[channel].compressed(), dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size:
                sums[channel] += values.sum()
                sums_sq[channel] += np.square(values).sum()
                counts[channel] += values.size
    if np.any(counts == 0):
        raise ValueError("Could not estimate SAR statistics: a band has no valid finite pixels")
    mean = sums / counts
    variance = np.maximum(sums_sq / counts - np.square(mean), 1e-6)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def get_train_transforms(patch_size: int = 256) -> A.Compose:
    """
    Augmentation pipeline for training.
    Designed for SAR data — avoids colour-only transforms (no hue/saturation).
    All transforms preserve the corresponding flood mask.
    """
    return A.Compose([
        # ── Geometry ──────────────────────────────────────────────────────────
        A.RandomCrop(height=patch_size, width=patch_size, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=0,   # cv2.BORDER_CONSTANT — pad with 0 (clean for SAR)
            mask_value=255,  # Compatible with the project's Albumentations 1.3+ range.
            p=0.4,
        ),
        # ── Radiometric / noise (SAR-safe) ────────────────────────────────────
        A.GaussNoise(var_limit=(0.001, 0.01), p=0.3),   # Simulate speckle variation
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.3,
        ),
        # ── Occlusion ─────────────────────────────────────────────────────────
        A.CoarseDropout(
            max_holes=4,
            max_height=32,
            max_width=32,
            fill_value=0,
            p=0.15,
        ),
        # ── Convert to tensor (does NOT normalise — we handle that separately) ─
        ToTensorV2(),
    ])


def get_val_transforms(patch_size: int = 256) -> A.Compose:
    """
    Center-crop and tensorise for validation callers that do not use grid crops.
    The training pipeline's grid-cropped patches are already patch_size square.
    """
    return A.Compose([
        A.CenterCrop(height=patch_size, width=patch_size, p=1.0),
        ToTensorV2(),
    ])


def normalise_sar(
    image: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> np.ndarray:
    """
    Z-score normalise a (H, W, 2) SAR image using pre-computed global stats.
    Clips extreme dB values first to reduce the effect of outliers.
    """
    mean = SAR_MEAN if mean is None else np.asarray(mean, dtype=np.float32)
    std = SAR_STD if std is None else np.asarray(std, dtype=np.float32)
    if mean.shape != (2,) or std.shape != (2,) or np.any(std <= 0):
        raise ValueError("SAR mean/std must each contain two values and std must be positive")
    # Clip to ±3σ before normalising.
    low = mean - 3 * std
    high = mean + 3 * std
    image = np.clip(image, low, high)
    return (image - mean) / (std + 1e-6)


class KuroSiwoDataset(Dataset):
    """
    PyTorch Dataset for Kuro Siwo Sentinel-1 flood mapping.

    Expected on-disk format (produced by preprocess.py):
        <root>/
            images/
                event_001_pre.tif   # 2-band float32 GeoTIFF (VV, VH) in dB
                event_001_post.tif
                ...
            masks/
                event_001.tif       # 1-band uint8 GeoTIFF  (0=bg, 1=flood, 255=nodata)

    Args:
        image_paths  (list[str]): Paths to pre-processed image GeoTIFFs.
        mask_paths   (list[str] | None): Paths to matching mask GeoTIFFs.
        patch_size   (int): Spatial size of cropped patches.
        transforms   (A.Compose | None): Albumentations pipeline.
        use_rasterio (bool): If False, fall back to numpy random data (unit testing).
    """

    NODATA_VALUE = 255   # Pixels to ignore during loss computation

    def __init__(
        self,
        image_paths: list,
        mask_paths: list | None = None,
        patch_size: int = 256,
        transforms=None,
        use_rasterio: bool = True,
        samples_per_image: int = 1,
        grid_crops: bool = False,
        sar_mean: np.ndarray | None = None,
        sar_std: np.ndarray | None = None,
    ):
        if not image_paths:
            raise ValueError("image_paths must not be empty")
        if mask_paths is not None:
            assert len(image_paths) == len(mask_paths), (
                f"Mismatch: {len(image_paths)} images vs {len(mask_paths)} masks"
            )

        self.image_paths  = image_paths
        self.mask_paths   = mask_paths
        self.patch_size   = patch_size
        self.transforms   = transforms
        self.use_rasterio = use_rasterio
        self.sar_mean = np.asarray(SAR_MEAN if sar_mean is None else sar_mean, dtype=np.float32)
        self.sar_std = np.asarray(SAR_STD if sar_std is None else sar_std, dtype=np.float32)
        if (self.sar_mean.shape != (2,) or self.sar_std.shape != (2,)
                or np.any(self.sar_std <= 0)):
            raise ValueError("SAR mean/std must each contain two values and std must be positive")
        if samples_per_image < 1:
            raise ValueError("samples_per_image must be positive")
        self.samples_per_image = samples_per_image
        self.grid_crops = grid_crops
        self.grid = []
        if grid_crops:
            if not use_rasterio:
                raise ValueError("grid_crops requires real raster files")
            for image_index, path in enumerate(self.image_paths):
                with rasterio.open(path) as src:
                    height, width = src.height, src.width
                ys = list(range(0, max(height - patch_size + 1, 1), patch_size))
                xs = list(range(0, max(width - patch_size + 1, 1), patch_size))
                if height > patch_size and ys[-1] != height - patch_size:
                    ys.append(height - patch_size)
                if width > patch_size and xs[-1] != width - patch_size:
                    xs.append(width - patch_size)
                self.grid.extend((image_index, y, x) for y in ys for x in xs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_image(self, path: str) -> np.ndarray:
        """
        Load a 2-band SAR GeoTIFF and return a (H, W, 2) float32 array in dB.
        Nodata pixels are filled with SAR_MEAN (neutral fill before normalisation).
        """
        with rasterio.open(path) as src:
            # shape: (bands, H, W)
            data = src.read(
                out_dtype="float32",
                masked=True,          # Returns a masked array honouring nodata
            )
            nodata = src.nodata

        # Fill masked / nodata areas with band mean so augmentation doesn't blow up
        if hasattr(data, "filled"):
            fill_vals = self.sar_mean[:, None, None]   # broadcast to (C, 1, 1)
            data = data.filled(fill_vals)
        else:
            data = np.array(data, dtype=np.float32)
            if nodata is not None:
                for c in range(data.shape[0]):
                    data[c][data[c] == nodata] = self.sar_mean[c]

        # Transpose to (H, W, C) for Albumentations
        return np.transpose(data, (1, 2, 0))    # → (H, W, 2)

    def _load_mask(self, path: str) -> np.ndarray:
        """
        Load a 1-band uint8 mask GeoTIFF and return a (H, W) uint8 array.
        Values: 0 = background, 1 = flood, 255 = nodata (ignored during loss).
        """
        with rasterio.open(path) as src:
            band = src.read(1, masked=True)
            data = np.asarray(band.filled(self.NODATA_VALUE), dtype=np.uint8)
            if src.nodata is not None:
                data[np.asarray(band.data) == src.nodata] = self.NODATA_VALUE
        unexpected = set(np.unique(data).tolist()) - {0, 1, self.NODATA_VALUE}
        if unexpected:
            raise ValueError(f"Mask {path} has unsupported labels {sorted(unexpected)}; expected 0, 1, or 255")
        return data

    def __len__(self):
        if self.grid_crops:
            return len(self.grid)
        return len(self.image_paths) * self.samples_per_image

    def __getitem__(self, idx):
        if self.grid_crops:
            image_index, crop_y, crop_x = self.grid[idx]
        else:
            image_index = idx % len(self.image_paths)

        # ── Load ──────────────────────────────────────────────────────────────
        image = self._load_image(self.image_paths[image_index])  # (H, W, 2)
        mask = (
            self._load_mask(self.mask_paths[image_index])
            if self.mask_paths is not None
            else np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        )

        # ── Normalise ─────────────────────────────────────────────────────────
        if image.shape[:2] != mask.shape:
            raise ValueError(
                f"Image and mask dimensions differ for {self.image_paths[image_index]}: "
                f"{image.shape[:2]} vs {mask.shape}"
            )
        image = normalise_sar(image, self.sar_mean, self.sar_std)      # (H, W, 2)

        # Keep tiny scenes usable by crop-based transforms without inventing
        # valid labels: padded mask pixels are ignored by the loss.
        pad_h = max(0, self.patch_size - image.shape[0])
        pad_w = max(0, self.patch_size - image.shape[1])
        if pad_h or pad_w:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=self.NODATA_VALUE)
        if self.grid_crops:
            image = image[crop_y:crop_y + self.patch_size, crop_x:crop_x + self.patch_size]
            mask = mask[crop_y:crop_y + self.patch_size, crop_x:crop_x + self.patch_size]

        # ── Augment ───────────────────────────────────────────────────────────
        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]   # ToTensorV2 → (C, H, W) float32 tensor
            mask  = augmented["mask"]    # ToTensorV2 → (H, W)   uint8  tensor
        else:
            # Manual conversion when no transform pipeline is given
            image = torch.from_numpy(np.transpose(image, (2, 0, 1)))
            mask  = torch.from_numpy(mask)

        # ── Cast mask for CrossEntropyLoss ────────────────────────────────────
        # CE loss expects LongTensor; nodata pixels (255) are handled via
        # ignore_index=255 in the loss function.
        mask = mask.long()

        return image, mask
