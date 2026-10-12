"""Convert official KuroSiwo GRD WebDataset shards to FloodSight GeoTIFFs.

The source backscatter is linear power; each output converts it to dB and
contains four float32 bands in this order:
pre-event VV, pre-event VH, post-event VV, post-event VH. The label raster maps
KuroSiwo class 2 (flood) to 1, classes 0/1 to 0, and invalid pixels to 255.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


IMAGE_FIELDS = ("sec1_vv", "sec1_vh", "flood_vv", "flood_vh")
REQUIRED_FIELDS = (*IMAGE_FIELDS, "mask", "valid_mask", "info.json")


def _write_sample(sample_id: str, sample: dict[str, object], out_dir: Path) -> str:
    info = sample["info.json"]
    if not isinstance(info, dict):
        raise ValueError(f"Sample {sample_id} has invalid info.json")
    actid = str(info.get("actid", "unknown"))
    grid_id = re.sub(r"[^A-Za-z0-9-]", "", str(info.get("grid_id", sample_id)))
    tile = f"{actid}_{grid_id or sample_id}"
    image_path = out_dir / "images" / f"{tile}.tif"
    mask_path = out_dir / "masks" / f"{tile}.tif"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)

    arrays = [np.squeeze(np.asarray(sample[name], dtype=np.float32)) for name in IMAGE_FIELDS]
    mask = np.squeeze(np.asarray(sample["mask"]))
    valid = np.squeeze(np.asarray(sample["valid_mask"])).astype(bool)
    shape = arrays[0].shape
    if len(shape) != 2 or any(a.shape != shape for a in arrays) or mask.shape != shape or valid.shape != shape:
        raise ValueError(f"Sample {sample_id} has inconsistent array shapes")
    for band in arrays:
        positive = np.isfinite(band) & (band > 0)
        band[~positive] = np.nan
        band[positive] = 10.0 * np.log10(band[positive])
    valid &= np.logical_and.reduce([np.isfinite(band) for band in arrays])

    # The official info geometry is a projected tile footprint. Preserve its
    # affine bounds but leave CRS unset because the WebDataset info does not
    # declare a coordinate reference system.
    bounds = None
    geom = info.get("geom")
    if isinstance(geom, str):
        coords = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", geom)]
        if len(coords) >= 8 and len(coords) % 2 == 0:
            xs, ys = coords[0::2], coords[1::2]
            bounds = (min(xs), min(ys), max(xs), max(ys))
    height, width = shape
    if bounds and bounds[2] > bounds[0] and bounds[3] > bounds[1]:
        transform = from_origin(bounds[0], bounds[3], (bounds[2] - bounds[0]) / width,
                                (bounds[3] - bounds[1]) / height)
    else:
        transform = from_origin(0, height, 1, 1)

    stacked = np.stack(arrays)
    stacked[:, ~valid] = np.nan
    with rasterio.open(
        image_path, "w", driver="GTiff", height=height, width=width,
        count=4, dtype="float32", transform=transform, crs=None,
        nodata=np.nan, compress="deflate", predictor=3,
    ) as dst:
        dst.write(stacked)
        for index, name in enumerate(IMAGE_FIELDS, 1):
            dst.set_band_description(index, name)
        dst.update_tags(event_id=actid, grid_id=grid_id, source="KuroSiwo WebDataset",
                        backscatter_scale="dB (10*log10 of source linear power)")

    output_mask = np.where(mask == 2, 1, 0).astype(np.uint8)
    output_mask[~valid] = 255
    with rasterio.open(
        mask_path, "w", driver="GTiff", height=height, width=width,
        count=1, dtype="uint8", transform=transform, crs=None,
        nodata=255, compress="lzw",
    ) as dst:
        dst.write(output_mask, 1)
        dst.update_tags(values="0=background_or_permanent_water,1=flood,255=invalid",
                        event_id=actid, grid_id=grid_id, source="KuroSiwo WebDataset")
    return actid


def import_shard(
    shard: Path, out_dir: Path, limit: int | None = None, allow_truncated: bool = False,
) -> dict:
    if not shard.is_file():
        raise FileNotFoundError(shard)
    event_counts: Counter[str] = Counter()
    imported = 0
    source_complete = True
    truncation_note = None
    current_id: str | None = None
    sample: dict[str, object] = {}
    with tarfile.open(shard, mode="r|") as archive:
      try:
        for member in archive:
            if not member.isfile():
                continue
            match = re.fullmatch(r"([^/.]+)\.(.+)", member.name)
            if not match:
                continue
            sample_id, field = match.groups()
            if field.endswith(".npy"):
                field = field[:-4]
            if current_id is not None and sample_id != current_id:
                if all(name in sample for name in REQUIRED_FIELDS):
                    event = _write_sample(current_id, sample, out_dir)
                    event_counts[event] += 1
                    imported += 1
                    if limit is not None and imported >= limit:
                        break
                current_id, sample = sample_id, {}
            elif current_id is None:
                current_id = sample_id

            if field not in REQUIRED_FIELDS:
                continue
            stream = archive.extractfile(member)
            if stream is None:
                raise OSError(f"Could not read {member.name} from {shard}")
            payload = stream.read()
            if field == "info.json":
                sample[field] = json.loads(payload.decode("utf-8"))
            else:
                sample[field] = np.load(io.BytesIO(payload), allow_pickle=False)

      except (EOFError, tarfile.ReadError) as exc:
        if not allow_truncated:
            raise
        source_complete = False
        truncation_note = f"{type(exc).__name__}: {exc}"

    if ((limit is None or imported < limit) and current_id is not None
            and all(name in sample for name in REQUIRED_FIELDS)):
        event = _write_sample(current_id, sample, out_dir)
        event_counts[event] += 1
        imported += 1

    result = {
        "source_shard": str(shard.resolve()),
        "source_complete": source_complete,
        "truncation_note": truncation_note,
        "imported_scenes": imported,
        "event_counts": dict(sorted(event_counts.items())),
        "band_order": list(IMAGE_FIELDS),
        "label_mapping": {"source_flood_class": 2, "output_flood_class": 1,
                          "invalid_output_class": 255},
        "crs": "not supplied by source info.json; raster CRS left unset",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{shard.parent.name}_import_report.json"
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard", type=Path, help="Path to a train_GRD or test_GRD tar shard")
    parser.add_argument("--out-dir", type=Path, default=Path("data/official_kuro/processed"))
    parser.add_argument("--limit", type=int, default=None, help="Import only the first N scenes")
    parser.add_argument("--allow-truncated", action="store_true",
                        help="Import only complete samples from a truncated shard and mark the report incomplete")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    import_shard(args.shard, args.out_dir, args.limit, args.allow_truncated)


if __name__ == "__main__":
    main()
