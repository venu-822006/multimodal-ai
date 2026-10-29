from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.features import shapes as rasterio_shapes
from rasterio.transform import array_bounds
from rasterio.warp import Resampling as WarpResampling
from rasterio.warp import reproject
from rasterio.windows import Window
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform
from shapely.validation import make_valid

from app.core.config import Settings, get_settings
from app.core.exceptions import DomainValidationError
from app.core.logging import get_logger
from app.processing.validators import AreaOfInterest, geodesic_area_m2

logger = get_logger("app.geospatial")


@dataclass(frozen=True)
class TargetGrid:
    crs: CRS
    transform: Affine
    width: int
    height: int
    pixel_size: float

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def bounds(self) -> tuple[float, float, float, float]:
        return array_bounds(self.height, self.width, self.transform)

    def geographic_bounds(self) -> list[float]:
        transformer = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        left, bottom, right, top = self.bounds()
        xs: list[float] = []
        ys: list[float] = []
        for x, y in ((left, bottom), (right, bottom), (right, top), (left, top)):
            lon, lat = transformer.transform(x, y)
            xs.append(lon)
            ys.append(lat)
        return [
            round(min(xs), 6),
            round(min(ys), 6),
            round(max(xs), 6),
            round(max(ys), 6),
        ]

    def geographic_transformer(self) -> Transformer:
        return Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)

    def to_wgs84(self, geometry: BaseGeometry) -> BaseGeometry:
        transformer = self.geographic_transformer()
        return shapely_transform(transformer.transform, geometry)


def utm_crs_for(aoi: AreaOfInterest) -> CRS:
    centre_lon = (aoi.min_lon + aoi.max_lon) / 2.0
    zone = int((centre_lon + 180.0) // 6.0) + 1
    zone = max(1, min(60, zone))
    return CRS.from_epsg((32600 if centre_lon >= 0 else 32700) + zone)


def build_target_grid(
    aoi: AreaOfInterest, settings: Settings | None = None, *, resolution_m: float | None = None
) -> TargetGrid:
    cfg = settings or get_settings()
    crs = utm_crs_for(aoi)
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = transformer.transform(
        [aoi.min_lon, aoi.min_lon, aoi.max_lon, aoi.max_lon],
        [aoi.min_lat, aoi.max_lat, aoi.min_lat, aoi.max_lat],
    )
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    pixel = resolution_m or cfg.raster_target_resolution_m
    width = max(1, math.ceil((max_x - min_x) / pixel))
    height = max(1, math.ceil((max_y - min_y) / pixel))
    if width * height > cfg.raster_max_pixels:
        scale = math.sqrt(cfg.raster_max_pixels / float(width * height))
        pixel = pixel * scale
        width = max(1, math.ceil((max_x - min_x) / pixel))
        height = max(1, math.ceil((max_y - min_y) / pixel))
    transform = Affine(pixel, 0.0, min_x, 0.0, -pixel, max_y)
    return TargetGrid(crs=crs, transform=transform, width=width, height=height, pixel_size=pixel)


def gdal_env(settings: Settings | None = None):
    """GDAL options tuned for range-read access to signed remote COG assets.

    ``CPL_VSIL_CURL_ALLOWED_EXTENSIONS`` is deliberately not set: signed asset URLs
    carry a SAS query string after the file extension, which the extension filter
    rejects outright. Restricting reads is instead handled by the signed host and
    by the read window computed in :func:`read_window_for_grid`.
    """
    cfg = settings or get_settings()
    return rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_HTTP_MULTIRANGE="YES",
        GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
        GDAL_HTTP_TIMEOUT=int(cfg.imagery_read_timeout_seconds),
        GDAL_HTTP_CONNECTTIMEOUT=int(cfg.imagery_connect_timeout_seconds),
        VSI_CACHE="TRUE",
        VSI_CACHE_SIZE=64 * 1024 * 1024,
        GDAL_NUM_THREADS="2",
    )


def open_dataset(href: str, settings: Settings | None = None):
    return rasterio.open(href)


def read_into_grid(
    href: str,
    grid: TargetGrid,
    *,
    settings: Settings | None = None,
    resampling: Resampling = Resampling.bilinear,
    nodata_values: Sequence[float] | None = None,
) -> np.ndarray:
    cfg = settings or get_settings()
    destination = np.full(grid.shape, np.nan, dtype=np.float32)
    with gdal_env(cfg):
        with rasterio.open(href) as dataset:
            if dataset.crs is None:
                raise DomainValidationError(
                    "Input imagery is missing coordinate reference information."
                )
            source, source_transform = read_window_for_grid(dataset, grid)
            if source is None:
                return destination
            reproject(
                source=source,
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


def read_window_for_grid(dataset, grid: TargetGrid) -> tuple[np.ndarray | None, Affine]:
    transformer = Transformer.from_crs(grid.crs, dataset.crs, always_xy=True)
    geo_bounds = grid.bounds()
    corners = [
        (geo_bounds[0], geo_bounds[1]),
        (geo_bounds[2], geo_bounds[1]),
        (geo_bounds[2], geo_bounds[3]),
        (geo_bounds[0], geo_bounds[3]),
    ]
    xs: list[float] = []
    ys: list[float] = []
    for x, y in corners:
        lon, lat = transformer.transform(x, y)
        xs.append(lon)
        ys.append(lat)
    requested = (min(xs), min(ys), max(xs), max(ys))

    dataset_transform = dataset.transform
    col_off, row_off = ~dataset_transform * (requested[0], requested[3])
    col_end, row_end = ~dataset_transform * (requested[2], requested[1])
    col_off = max(0, math.floor(col_off))
    row_off = max(0, math.floor(row_off))
    col_end = min(dataset.width, math.ceil(col_end))
    row_end = min(dataset.height, math.ceil(row_end))
    width = int(col_end - col_off)
    height = int(row_end - row_off)
    if width <= 0 or height <= 0:
        return None, dataset_transform

    out_height = height
    out_width = width
    max_window_pixels = 12_000_000
    if width * height > max_window_pixels:
        factor = math.sqrt(width * height / max_window_pixels)
        out_width = max(1, int(width / factor))
        out_height = max(1, int(height / factor))

    window = Window(col_off, row_off, width, height)
    raw = dataset.read(
        1,
        window=window,
        out_shape=(out_height, out_width),
        resampling=resampling_of(dataset, out_width, width),
        masked=False,
        out_dtype="float32",
    )
    source_transform = dataset_transform * Affine.translation(col_off, row_off)
    if (out_width, out_height) != (width, height):
        source_transform = source_transform * Affine.scale(width / out_width, height / out_height)
    return _to_float(raw, dataset.nodata), source_transform


def _to_float(raw: np.ndarray, nodata: float | None) -> np.ndarray:
    """Convert raw samples to float32, mapping nodata and non-finite values to NaN."""
    values = np.asarray(raw, dtype=np.float32)
    if nodata is not None and np.isfinite(nodata):
        values = np.where(values == np.float32(nodata), np.nan, values)
    values[~np.isfinite(values)] = np.nan
    return values


def resampling_of(dataset, out_width: int, native_width: int) -> Resampling:
    if out_width < native_width:
        return Resampling.average if dataset.dtypes and dataset.dtypes[0].startswith("float") else Resampling.nearest
    return Resampling.bilinear


def to_decibels(power: np.ndarray, *, floor_db: float = -25.0) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10.0 * np.log10(np.maximum(power, 1e-12))
    db[~np.isfinite(db)] = np.nan
    return np.clip(db, floor_db, 5.0)


def polygonize_mask(
    mask: np.ndarray, grid: TargetGrid, *, min_area_m2: float
) -> tuple[BaseGeometry | None, dict[str, float]]:
    """Vectorise a boolean mask, dropping parts below a minimum ground area.

    Areas are compared in square metres. When the grid CRS is projected with
    metre units the planar area is already a ground area; otherwise the part is
    converted to WGS-84 and measured geodesically.
    """
    geometries: list[Polygon] = []
    metric = bool(grid.crs and grid.crs.is_projected and grid.crs.axis_info[0].unit_name == "metre")
    pixel_area = grid.pixel_size**2
    raw_pixels = 0
    dropped = 0
    for geometry_mapping, value in rasterio_shapes(
        mask.astype(np.uint8), mask=mask, transform=grid.transform, connectivity=4
    ):
        if int(value) != 1:
            continue
        polygon = shape(geometry_mapping)
        if polygon.is_empty:
            continue
        if not polygon.is_valid:
            polygon = make_valid(polygon)
        parts = [part for part in _flatten(polygon) if isinstance(part, Polygon)]
        for part in parts:
            if metric:
                area_m2 = part.area
            else:
                area_m2 = geodesic_area_m2(grid.to_wgs84(part))
            if area_m2 < min_area_m2:
                dropped += 1
                continue
            geometries.append(part)
            raw_pixels += max(1, int(round(area_m2 / pixel_area))) if pixel_area else 1
    if not geometries:
        return None, {
            "pixel_count": 0.0,
            "retained_polygons": 0.0,
            "dropped_polygons": float(dropped),
        }
    merged = geometries[0] if len(geometries) == 1 else MultiPolygon(geometries)
    wgs84 = grid.to_wgs84(merged)
    if not wgs84.is_valid:
        wgs84 = make_valid(wgs84)
    return wgs84, {
        "pixel_count": float(raw_pixels),
        "retained_polygons": float(len(geometries)),
        "dropped_polygons": float(dropped),
    }


def _flatten(geometry: BaseGeometry) -> Iterable[BaseGeometry]:
    geoms = getattr(geometry, "geoms", None)
    if geoms is None:
        return [geometry]
    output: list[BaseGeometry] = []
    for part in geoms:
        output.extend(_flatten(part))
    return output


def geometry_to_feature_collection(
    geometry: BaseGeometry | None, properties: dict[str, object] | None = None
) -> dict[str, object]:
    if geometry is None or geometry.is_empty:
        return {"type": "FeatureCollection", "features": []}
    features = []
    for part in _flatten(geometry):
        if part.is_empty:
            continue
        features.append(
            {
                "type": "Feature",
                "properties": dict(properties or {}),
                "geometry": mapping(part),
            }
        )
    return {"type": "FeatureCollection", "features": features}


def measure_geometry(
    geometry: BaseGeometry | None,
    aoi_area_m2: float,
    *,
    valid_area_m2: float | None = None,
) -> dict[str, float | None]:
    """Geodesic area measurements for a predicted flood geometry.

    ``flooded_percentage`` is expressed against the valid analysed area when it
    is supplied, because pixels without valid radar coverage cannot be either
    flooded or dry. It falls back to the whole area of interest otherwise.
    """
    if geometry is None or geometry.is_empty:
        return {
            "flooded_area_m2": None,
            "flooded_area_km2": None,
            "flooded_percentage": None,
            "polygon_count": 0,
        }
    area_m2 = geodesic_area_m2(geometry)
    denominator = valid_area_m2 if valid_area_m2 and valid_area_m2 > 0 else aoi_area_m2
    percentage = (area_m2 / denominator * 100.0) if denominator > 0 else None
    return {
        "flooded_area_m2": round(area_m2, 2),
        "flooded_area_km2": round(area_m2 / 1_000_000.0, 4),
        "flooded_percentage": round(percentage, 4) if percentage is not None else None,
        "polygon_count": len([p for p in _flatten(geometry) if isinstance(p, Polygon)]),
        "percentage_denominator": "valid_analysis_area" if valid_area_m2 else "area_of_interest",
    }
