from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import Resampling as WarpResampling
from rasterio.warp import reproject
from scipy.ndimage import uniform_filter

from app.core.config import Settings, get_settings
from app.core.exceptions import PermanentProcessingError, UpstreamError
from app.core.logging import get_logger
from app.processing.validators import AreaOfInterest
from app.services.geospatial_service import TargetGrid, build_target_grid, gdal_env
from app.services.imagery_service import SceneAsset

logger = get_logger("app.preprocessing")

DB_CLIP_LOW = -40.0
DB_CLIP_HIGH = 0.0
NODATA_OUT = -9999.0

PREPROCESSING_STEPS = (
    "Read Sentinel-1 GRD RTC linear-power VV and VH backscatter.",
    "Convert linear power to decibels with 10*log10.",
    "Apply a 3x3 Lee-sigma speckle filter to both polarisations.",
    "Clip backscatter to the -40 dB to 0 dB range used during training.",
    "Reproject and resample both polarisations onto one aligned analysis grid.",
    "Write a two band (VV, VH) float32 dB GeoTIFF as the model input.",
)


@dataclass(frozen=True)
class PreparedScene:
    path: Path
    grid: TargetGrid
    shape: tuple[int, int]
    scenes: tuple[SceneAsset, ...]
    records: tuple[dict[str, Any], ...]
    warnings: tuple[str, ...] = ()


def linear_to_db(data: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    values = np.where(data > 0, data, eps).astype(np.float32)
    return (10.0 * np.log10(values)).astype(np.float32)


def lee_sigma_filter(image: np.ndarray, size: int = 3) -> np.ndarray:
    """Lee-sigma speckle filter, matching preprocess.lee_sigma_filter."""
    mean = uniform_filter(image, size=size)
    square = uniform_filter(image**2, size=size)
    variance = square - mean**2
    overall = np.var(image)
    weight = variance / (variance + overall + 1e-10)
    return (mean + weight * (image - mean)).astype(np.float32)


def clip_db(data: np.ndarray) -> np.ndarray:
    return np.clip(data, DB_CLIP_LOW, DB_CLIP_HIGH).astype(np.float32)


def read_polarisation(href: str, grid: TargetGrid, settings: Settings) -> np.ndarray:
    destination = np.full(grid.shape, np.nan, dtype=np.float32)
    with gdal_env(settings):
        with rasterio.open(href) as dataset:
            if dataset.crs is None:
                raise PermanentProcessingError(
                    "The Sentinel-1 asset is missing coordinate reference information."
                )
            from rasterio.windows import Window
            from pyproj import Transformer

            bounds = grid.bounds()
            transformer = Transformer.from_crs(grid.crs, dataset.crs, always_xy=True)
            xs, ys = transformer.transform(
                [bounds[0], bounds[0], bounds[2], bounds[2]],
                [bounds[1], bounds[3], bounds[1], bounds[3]],
            )
            col_off, row_off = ~dataset.transform @ (min(xs), max(ys))
            col_end, row_end = ~dataset.transform @ (max(xs), min(ys))
            col_off = max(0, math.floor(col_off))
            row_off = max(0, math.floor(row_off))
            col_end = min(dataset.width, math.ceil(col_end))
            row_end = min(dataset.height, math.ceil(row_end))
            width = int(col_end - col_off)
            height = int(row_end - row_off)
            if width <= 0 or height <= 0:
                return destination
            window = Window(col_off, row_off, width, height)
            scale = min(1.0, math.sqrt(12_000_000.0 / max(width * height, 1)))
            out_width = max(1, int(width * scale))
            out_height = max(1, int(height * scale))
            raw = dataset.read(
                1,
                window=window,
                out_shape=(out_height, out_width),
                resampling=Resampling.average if scale < 1.0 else Resampling.bilinear,
                masked=False,
                out_dtype="float32",
            )
            source_transform = dataset.transform @ Affine.translation(col_off, row_off)
            if (out_width, out_height) != (width, height):
                source_transform = source_transform @ Affine.scale(
                    width / out_width, height / out_height
                )
            reproject(
                source=_mask_nodata(raw, dataset.nodata),
                destination=destination,
                src_transform=source_transform,
                src_crs=dataset.crs,
                dst_transform=grid.transform,
                dst_crs=grid.crs,
                resampling=WarpResampling.bilinear,
                src_nodata=np.nan,
                dst_nodata=np.nan,
            )
    return destination


def _mask_nodata(raw: np.ndarray, nodata: float | None) -> np.ndarray:
    values = raw.astype(np.float32)
    if nodata is not None:
        values = np.where(values == np.float32(nodata), np.nan, values)
    values[~np.isfinite(values)] = np.nan
    return values


def write_model_input(
    path: Path, vv_db: np.ndarray, vh_db: np.ndarray, grid: TargetGrid
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=grid.height,
        width=grid.width,
        count=2,
        dtype="float32",
        crs=grid.crs,
        transform=grid.transform,
        nodata=NODATA_OUT,
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dataset:
        dataset.write(vv_db, 1)
        dataset.write(vh_db, 2)
        dataset.update_tags(1, band="VV_dB")
        dataset.update_tags(2, band="VH_dB")
        dataset.update_tags(source="FloodSight-AI", stages="dB;lee_sigma_3x3;clip_-40_0")


def prepare_model_input(
    scene: SceneAsset,
    aoi: AreaOfInterest,
    settings: Settings | None = None,
    *,
    grid: TargetGrid | None = None,
    output_path: Path | None = None,
) -> PreparedScene:
    """Produce the two band dB GeoTIFF the trained model expects."""
    cfg = settings or get_settings()
    if "vv" not in scene.hrefs or "vh" not in scene.hrefs:
        raise PermanentProcessingError(
            "The selected Sentinel-1 asset does not expose both VV and VH polarisations."
        )
    target = grid or build_target_grid(aoi, cfg)
    warnings: list[str] = []

    try:
        vv_linear = read_polarisation(scene.hrefs["vv"], target, cfg)
        vh_linear = read_polarisation(scene.hrefs["vh"], target, cfg)
    except UpstreamError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise UpstreamError(
            f"The Sentinel-1 asset could not be read ({type(exc).__name__})."
        ) from exc

    valid = np.isfinite(vv_linear) & np.isfinite(vh_linear)
    if not valid.any():
        raise PermanentProcessingError(
            "The selected Sentinel-1 acquisition contains no valid radar data over the "
            "requested area of interest."
        )

    vv_db = np.full(target.shape, np.nan, dtype=np.float32)
    vh_db = np.full(target.shape, np.nan, dtype=np.float32)
    vv_db[valid] = clip_db(lee_sigma_filter(linear_to_db(vv_linear[valid])))
    vh_db[valid] = clip_db(lee_sigma_filter(linear_to_db(vh_linear[valid])))

    coverage = float(valid.sum()) / float(valid.size)
    if coverage < 0.2:
        warnings.append(
            f"Only {coverage * 100:.1f}% of the analysis grid has valid Sentinel-1 "
            "coverage; results describe the covered portion only."
        )

    path = output_path or Path(cfg.artifact_root) / "_staging" / f"{scene.item_id}.tif"
    write_model_input(path, np.nan_to_num(vv_db, nan=NODATA_OUT), np.nan_to_num(vh_db, nan=NODATA_OUT), target)
    logger.info(
        "model_input_prepared",
        extra={
            "event": "model_input_prepared",
            "item_id": scene.item_id,
            "grid_shape": list(target.shape),
            "resolution_m": round(target.pixel_size, 3),
            "valid_fraction": round(coverage, 4),
        },
    )
    return PreparedScene(
        path=path,
        grid=target,
        shape=target.shape,
        scenes=(scene,),
        records=(scene.public_record(),),
        warnings=tuple(warnings),
    )


def read_prepared_scene(path: Path) -> tuple[np.ndarray, np.ndarray, Any, Any]:
    """Load a prepared two band dB raster plus its validity mask and georeferencing."""
    with rasterio.open(path) as dataset:
        if dataset.count < 2:
            raise PermanentProcessingError(
                "A prepared model input must contain both VV and VH bands."
            )
        data = dataset.read(out_dtype="float32", masked=True)
        transform = dataset.transform
        crs = dataset.crs
        nodata = dataset.nodata
    masked = np.ma.masked_invalid(data)
    valid = ~(masked.mask[0] | masked.mask[1])
    if nodata is not None:
        valid &= (data[0] != np.float32(nodata)) & (data[1] != np.float32(nodata))
    valid &= np.isfinite(data[0]) & np.isfinite(data[1])
    neutral = np.array([-25.0, -31.0], dtype=np.float32)
    filled = data.filled(neutral[:, None, None]) if hasattr(data, "filled") else np.asarray(data)
    return np.transpose(filled, (1, 2, 0)).astype(np.float32), valid, transform, crs


__all__ = [
    "PREPROCESSING_STEPS",
    "PreparedScene",
    "clip_db",
    "lee_sigma_filter",
    "linear_to_db",
    "prepare_model_input",
    "read_polarisation",
    "read_prepared_scene",
    "write_model_input",
]