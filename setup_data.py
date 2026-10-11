"""
setup_data.py  —  FloodSight AI | KuroSiwo Data Download & Preparation
=======================================================================
Downloads KuroSiwo Batch 0 from Dropbox, extracts it, and converts
the raw 224x224 patch folders into analysis-ready 2-band GeoTIFFs
(VV + VH in dB, float32) and matching flood mask GeoTIFFs (uint8).

Raw patch folder layout (from KuroSiwo GRD dataset):
    <batch>/
        <act_id>/<aoi_id>/<patch_id>/
            MS1_IVV*.tif   <-- post-flood VV amplitude (uint16, linear)
            MS1_IVH*.tif   <-- post-flood VH amplitude (uint16, linear)
            MK0_MLU*.tif   <-- flood label  (0=bg/perm-water, 1=flood)
            MK0_MNA*.tif   <-- valid pixel mask (1=valid, 0=nodata)

Output layout (consumed by train.py):
    data/processed/
        images/<act>_<aoi>_<patch>_post.tif   # 2-band float32 (VV dB, VH dB)
        masks/<act>_<aoi>_<patch>.tif         # 1-band uint8  (0=bg, 1=flood, 255=nodata)

Usage:
    python setup_data.py                            # downloads + prepares Batch 0
    python setup_data.py --batch 1                  # downloads Batch 1 instead
    python setup_data.py --skip-download            # only converts already-extracted data
    python setup_data.py --raw-dir data/raw/batch0  # point at existing extracted folder
"""

import argparse
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Dropbox URLs for all 11 KuroSiwo GRD batches (from download_kuro_siwo.sh)
# ---------------------------------------------------------------------------
BATCH_URLS = {
    0:  "https://www.dropbox.com/scl/fi/19mm9v5pnd5yor8b15alj/00.tar.gz?rlkey=f7qrqgv7h7z9j6r595xz1720e&st=k4ptwanm&dl=1",
    1:  "https://www.dropbox.com/scl/fi/pjlgcqc3fm8lx97vdfjyn/01.tar.gz?rlkey=bvue0u3jgovc3qewkh464uyog&st=qfsa601d&dl=1",
    2:  "https://www.dropbox.com/scl/fi/7tt843025s8hqi00xpofx/02.tar.gz?rlkey=8vbfh3qc2h7zmzjem0c14lljh&st=xda2zf5j&dl=1",
    3:  "https://www.dropbox.com/scl/fi/v7997kv11cc8ptj1vstim/03.tar.gz?rlkey=yj0qspgmlmohaf728a94curqf&st=6qmn63y7&dl=1",
    4:  "https://www.dropbox.com/scl/fi/2o3cxwxu79phijxylm99h/04.tar.gz?rlkey=3mcmjrzbf8vmxwl8aacou7vvr&st=yh0e4p72&dl=1",
    5:  "https://www.dropbox.com/scl/fi/q1jy4ep4j6bj38dva2es3/05.tar.gz?rlkey=g46498pgox03it080p83xhx8g&st=haxheepl&dl=1",
    6:  "https://www.dropbox.com/scl/fi/843fba3poe6nu67og5n6x/06.tar.gz?rlkey=bnxhf9zj0y8mtvjv7k683ywaa&st=p4pg93ko&dl=1",
    7:  "https://www.dropbox.com/scl/fi/89kogi7nisinbfa3z2wze/07.tar.gz?rlkey=byduc9q5cndwjphq4r1727xrl&st=a2sgt9dp&dl=1",
    8:  "https://www.dropbox.com/scl/fi/sctg1ybqxtzz23018htwx/08.tar.gz?rlkey=6h887jxezjq5gibqz4zhh7gd&st=9gcgiwdu&dl=1",
    9:  "https://www.dropbox.com/scl/fi/425wrfqqc7sy8pbdnkgzw/09.tar.gz?rlkey=ziejzpo2ddgnygpbd1emn1nir&st=irjm2j98&dl=1",
    10: "https://www.dropbox.com/scl/fi/rwc2yvd0g070qicg300de/10.tar.gz?rlkey=12i17si2kzjpey4vgig9pjnv7&st=p2v45gue&dl=1",
}

CATALOGUE_URL = (
    "https://www.dropbox.com/scl/fi/wu6nvj73cz4h7k3gxpzx6/catalogue.gpkg"
    "?rlkey=hsij2o0k60r2n0z6z4d2ngww9&st=0zjqhzgx&dl=1"
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DB_CLIP_LOW  = -40.0
DB_CLIP_HIGH =   0.0
NODATA_MASK  = 255        # value written to mask pixels that are invalid
EPS          = 1e-10      # avoid log(0)

# A dummy WGS-84 transform for patches that have no georeferencing.
# train.py does not use CRS/transform for pixel-level training.
DUMMY_CRS       = CRS.from_epsg(4326)
DUMMY_TRANSFORM = from_bounds(0, 0, 1, 1, 224, 224)


# ---------------------------------------------------------------------------
# Helper: download with progress bar
# ---------------------------------------------------------------------------
class _ProgressHook:
    def __init__(self, desc: str):
        self._pbar = None
        self._desc = desc

    def __call__(self, block_num, block_size, total_size):
        if self._pbar is None:
            self._pbar = tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=self._desc,
            )
        downloaded = block_num * block_size
        self._pbar.update(min(block_size, total_size - self._pbar.n) if total_size > 0 else block_size)
        if total_size > 0 and downloaded >= total_size:
            self._pbar.close()


def download_file(url: str, dest: Path, desc: str = "Downloading") -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  ✓ Already downloaded: {dest.name}")
        return
    print(f"  >> Downloading -> {dest}")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_ProgressHook(desc))
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed: {exc}") from exc
    print(f"  ✓ Saved: {dest}  ({dest.stat().st_size / 1e9:.2f} GB)")


def extract_tar(tar_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"  >> Extracting {tar_path.name} -> {dest_dir} …")
    with tarfile.open(tar_path, "r:gz") as tf:
        members = tf.getmembers()
        for member in tqdm(members, desc="Extracting", unit="file"):
            tf.extract(member, path=dest_dir)
    print(f"  ✓ Extraction complete.")


# ---------------------------------------------------------------------------
# SAR helpers
# ---------------------------------------------------------------------------
def linear_to_db(arr: np.ndarray) -> np.ndarray:
    """Convert uint16 linear amplitude to dB, clip to valid SAR range."""
    arr = arr.astype(np.float32)
    arr = np.where(arr > 0, arr, EPS)
    db = 10.0 * np.log10(arr)
    return np.clip(db, DB_CLIP_LOW, DB_CLIP_HIGH).astype(np.float32)


# ---------------------------------------------------------------------------
# Core converter: one raw patch folder -> one image GeoTIFF + one mask GeoTIFF
# ---------------------------------------------------------------------------
def convert_patch(patch_dir: Path, out_image: Path, out_mask: Path) -> bool:
    """
    Read MS1_IVV, MS1_IVH (post-flood) and MK0_MLU (label), MK0_MNA (valid mask)
    from a KuroSiwo patch directory and write analysis-ready GeoTIFFs.

    Returns True on success, False if required files are missing.
    """
    vv_file = vh_file = lbl_file = valid_file = None

    for f in patch_dir.iterdir():
        name = f.name
        if name.endswith(".xml"):
            continue
        if name.startswith("MS1_IVV"):
            vv_file = f
        elif name.startswith("MS1_IVH"):
            vh_file = f
        elif name.startswith("MK0_MLU"):
            lbl_file = f
        elif name.startswith("MK0_MNA"):
            valid_file = f

    if vv_file is None or vh_file is None or lbl_file is None or valid_file is None:
        return False  # Incomplete SAR/label/validity tuple is not a training example.

    # ── Read raw VV / VH (uint16 linear amplitude) ──────────────────────────
    raw_vv = cv2.imread(str(vv_file), cv2.IMREAD_ANYDEPTH)
    raw_vh = cv2.imread(str(vh_file), cv2.IMREAD_ANYDEPTH)
    if raw_vv is None or raw_vh is None:
        return False
    if raw_vv.shape != raw_vh.shape:
        raise ValueError(f"VV/VH dimension mismatch in {patch_dir}: {raw_vv.shape} vs {raw_vh.shape}")

    db_vv = linear_to_db(raw_vv)
    db_vh = linear_to_db(raw_vh)
    h, w = db_vv.shape

    # ── Read valid-pixel mask ────────────────────────────────────────────────
    valid = np.ones((h, w), dtype=bool)
    raw_valid = cv2.imread(str(valid_file), cv2.IMREAD_ANYDEPTH)
    if raw_valid is None or raw_valid.shape != (h, w):
        raise ValueError(f"Missing, unreadable, or misaligned valid-pixel mask in {patch_dir}")
    valid = raw_valid == 1
    if not valid.any():
        return False

    # ── Apply nodata fill for invalid pixels (neutral dB value) ─────────────
    db_vv[~valid] = -25.0   # roughly SAR_MEAN for VV
    db_vh[~valid] = -31.0   # roughly SAR_MEAN for VH

    # ── Write 2-band image GeoTIFF ───────────────────────────────────────────
    out_image.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_image, "w",
        driver="GTiff",
        height=h, width=w,
        count=2,
        dtype=np.float32,
        crs=DUMMY_CRS,
        transform=DUMMY_TRANSFORM,
        nodata=-9999.0,
        compress="lzw",
    ) as dst:
        dst.write(db_vv, 1)   # Band 1 = VV dB
        dst.write(db_vh, 2)   # Band 2 = VH dB
        dst.update_tags(band_1="VV_dB", band_2="VH_dB", source="KuroSiwo_GRD")

    # ── Build flood mask ─────────────────────────────────────────────────────
    # MK0_MLU: 0=background/permanent water, 1=flood
    # We map invalid pixels -> 255 (ignored by loss)
    raw_lbl = cv2.imread(str(lbl_file), cv2.IMREAD_ANYDEPTH)
    if raw_lbl is None or raw_lbl.shape != (h, w):
        raise ValueError(f"Missing, unreadable, or misaligned flood label in {patch_dir}")
    observed_labels = set(np.unique(raw_lbl[valid]).tolist())
    if not observed_labels.issubset({0, 1}):
        raise ValueError(
            f"Unexpected KuroSiwo mask labels in {patch_dir}: {sorted(observed_labels)}; "
            "expected 0=background, 1=flood on valid pixels."
        )
    mask_arr = np.where(raw_lbl == 1, 1, 0).astype(np.uint8)
    mask_arr[~valid] = NODATA_MASK   # mark invalid pixels as nodata

    # ── Write 1-band mask GeoTIFF ────────────────────────────────────────────
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_mask, "w",
        driver="GTiff",
        height=h, width=w,
        count=1,
        dtype=np.uint8,
        crs=DUMMY_CRS,
        transform=DUMMY_TRANSFORM,
        nodata=NODATA_MASK,
        compress="lzw",
    ) as dst:
        dst.write(mask_arr, 1)
        dst.update_tags(values="0=background,1=flood,255=nodata")

    return True


# ---------------------------------------------------------------------------
# Walk extracted batch and convert all patches
# ---------------------------------------------------------------------------
def convert_batch(raw_dir: Path, out_dir: Path) -> None:
    """
    Recursively find all KuroSiwo patch directories (containing MS1_IVV*.tif)
    inside raw_dir and convert each to analysis-ready GeoTIFFs in out_dir.
    """
    img_dir  = out_dir / "images"
    mask_dir = out_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    # Collect all patch dirs: a dir is a patch dir if it contains MS1_IVV*.tif
    patch_dirs = []
    for root, dirs, files in os.walk(raw_dir):
        if any(f.startswith("MS1_IVV") and f.endswith(".tif") for f in files):
            patch_dirs.append(Path(root))

    if not patch_dirs:
        print(f"  [!] No patch directories found under {raw_dir}")
        print("    Make sure you have extracted the .tar.gz correctly.")
        return

    print(f"\n  Found {len(patch_dirs)} patch directories. Converting …")

    ok = skipped = 0
    for patch_dir in tqdm(patch_dirs, desc="Converting patches", unit="patch"):
        # Build a unique name from the directory hierarchy relative to raw_dir
        rel_parts = patch_dir.relative_to(raw_dir).parts
        event_key = "_".join(rel_parts)  # e.g. "470_01_00003"

        out_image = img_dir  / f"{event_key}_post.tif"
        out_mask  = mask_dir / f"{event_key}.tif"

        if out_image.exists() and out_mask.exists():
            ok += 1
            continue

        success = convert_patch(patch_dir, out_image, out_mask)
        if success:
            ok += 1
        else:
            skipped += 1

    total = ok + skipped
    print(f"\n  ✓ Conversion done: {ok}/{total} patches converted, {skipped} skipped.")
    print(f"  Images -> {img_dir}")
    print(f"  Masks  -> {mask_dir}")

    # ── Summary stats for flood pixel ratio ─────────────────────────────────
    print("\n  Computing flood pixel statistics (sample of first 50 masks)…")
    mask_files = list(mask_dir.glob("*.tif"))[:50]
    total_px = flood_px = 0
    for mf in mask_files:
        with rasterio.open(mf) as src:
            arr = src.read(1)
        valid = arr != NODATA_MASK
        total_px += valid.sum()
        flood_px += (arr == 1).sum()

    if total_px > 0:
        ratio = 100 * flood_px / total_px
        print(f"  Flood pixels: {flood_px:,} / {total_px:,} valid ({ratio:.2f}%)")
        if ratio < 1.0:
            print("  [!] Very low flood ratio — consider oversampling or adjusting loss weights.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch",         type=int,  default=0,                       help="Batch index to download (0–10). Default: 0 (~8.9 GB)")
    p.add_argument("--raw-dir",       type=Path, default=Path("data/raw"),         help="Directory where raw .tar.gz is saved and extracted")
    p.add_argument("--out-dir",       type=Path, default=Path("data/processed"),   help="Output directory for converted GeoTIFFs")
    p.add_argument("--skip-download", action="store_true",                          help="Skip download; use already-extracted data in --raw-dir")
    p.add_argument("--skip-extract",  action="store_true",                          help="Skip extraction; raw_dir already contains extracted folders")
    p.add_argument("--keep-tar",      action="store_true",                          help="Do not delete the .tar.gz after extraction")
    return p.parse_args()


def main():
    args = parse_args()
    batch_idx = args.batch

    if batch_idx not in BATCH_URLS:
        print(f"ERROR: Batch {batch_idx} is not valid. Choose 0–10.")
        sys.exit(1)

    raw_dir   = args.raw_dir
    out_dir   = args.out_dir
    tar_file  = raw_dir / f"{batch_idx:02d}.tar.gz"
    batch_dir = raw_dir / f"batch{batch_idx:02d}"

    print("=" * 65)
    print(f"  FloodSight AI — KuroSiwo Batch {batch_idx} Setup")
    print("=" * 65)

    # ── Step 1: Download ────────────────────────────────────────────────────
    if not args.skip_download and not args.skip_extract:
        print(f"\n[1/3] Downloading Batch {batch_idx} (~8–9 GB)…")
        download_file(BATCH_URLS[batch_idx], tar_file, desc=f"Batch {batch_idx:02d}")
    else:
        print(f"\n[1/3] Skipping download.")

    # ── Step 2: Extract ─────────────────────────────────────────────────────
    if not args.skip_extract:
        if not tar_file.exists():
            print(f"ERROR: Tar file not found: {tar_file}")
            sys.exit(1)
        print(f"\n[2/3] Extracting to {batch_dir} …")
        extract_tar(tar_file, batch_dir)
        if not args.keep_tar:
            tar_file.unlink()
            print(f"  🗑  Deleted {tar_file.name} to recover disk space.")
    else:
        batch_dir = raw_dir   # user already has extracted data in --raw-dir
        print(f"\n[2/3] Skipping extraction. Using raw dir: {batch_dir}")

    # ── Step 3: Convert to analysis-ready GeoTIFFs ──────────────────────────
    print(f"\n[3/3] Converting patches to GeoTIFFs in {out_dir} …")
    convert_batch(batch_dir, out_dir)

    # ── Done ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Setup complete! Now run training:")
    print()
    print("  python train.py --data-dir data/processed --output-dir checkpoints")
    print("=" * 65)


if __name__ == "__main__":
    main()
