from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from pyproj import Geod
from shapely.geometry import Polygon, box, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.validation import explain_validity, make_valid

from app.core.config import Settings
from app.core.exceptions import DomainValidationError

_GEOD = Geod(ellps="WGS84")


@dataclass(frozen=True)
class AreaOfInterest:
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    area_m2: float

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)

    @property
    def bbox_list(self) -> list[float]:
        return [self.min_lon, self.min_lat, self.max_lon, self.max_lat]

    def to_geojson_polygon(self) -> dict[str, Any]:
        return {
            "type": "Feature",
            "properties": {"kind": "area_of_interest", "crs": "EPSG:4326"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [self.min_lon, self.min_lat],
                        [self.max_lon, self.min_lat],
                        [self.max_lon, self.max_lat],
                        [self.min_lon, self.max_lat],
                        [self.min_lon, self.min_lat],
                    ]
                ],
            },
        }

    @property
    def geometry(self) -> Polygon:
        return box(self.min_lon, self.min_lat, self.max_lon, self.max_lat)


@dataclass(frozen=True)
class DateWindow:
    flood_date: date
    window_start: date
    window_end: date
    pre_flood_date: date
    post_flood_date: date


def geodesic_area_m2(geometry: BaseGeometry) -> float:
    if geometry is None or geometry.is_empty:
        return 0.0
    if not geometry.is_valid:
        geometry = make_valid(geometry)
    area = 0.0
    geoms = getattr(geometry, "geoms", [geometry])
    for part in geoms:
        if part.is_empty:
            continue
        if part.geom_type == "Polygon":
            area += abs(_GEOD.geometry_area_perimeter(part)[0])
        elif part.geom_type in ("MultiPolygon", "GeometryCollection"):
            area += geodesic_area_m2(part)
    return area


def validate_bbox(
    bbox: list[list[float]], settings: Settings, *, field: str = "bbox"
) -> AreaOfInterest:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 2:
        raise DomainValidationError(
            f"{field} must contain exactly two [longitude, latitude] corners."
        )
    corners: list[tuple[float, float]] = []
    for corner in bbox:
        if not isinstance(corner, (list, tuple)) or len(corner) != 2:
            raise DomainValidationError(
                f"{field} corners must each be a [longitude, latitude] pair."
            )
        try:
            lon, lat = float(corner[0]), float(corner[1])
        except (TypeError, ValueError) as exc:
            raise DomainValidationError(
                f"{field} coordinates must be numbers."
            ) from exc
        if not math.isfinite(lon) or not math.isfinite(lat):
            raise DomainValidationError(f"{field} coordinates must be finite numbers.")
        if not -180.0 <= lon <= 180.0:
            raise DomainValidationError(
                f"{field} longitude {lon} is outside the valid WGS84 range [-180, 180]."
            )
        if not -90.0 <= lat <= 90.0:
            raise DomainValidationError(
                f"{field} latitude {lat} is outside the valid WGS84 range [-90, 90]."
            )
        corners.append((lon, lat))

    (lon_a, lat_a), (lon_b, lat_b) = corners
    if lon_b < lon_a or lat_b < lat_a:
        raise DomainValidationError(
            f"{field} corners must be ordered south-west first, that is "
            "[[west, south], [east, north]], where west <= east and south <= north."
        )
    min_lon, max_lon = lon_a, lon_b
    min_lat, max_lat = lat_a, lat_b

    if max_lon == min_lon or max_lat == min_lat:
        raise DomainValidationError(
            f"{field} describes a zero-area region; a bounding box must span a real area."
        )

    width_deg = max_lon - min_lon
    height_deg = max_lat - min_lat
    if width_deg > settings.max_bbox_dimension_deg or height_deg > settings.max_bbox_dimension_deg:
        raise DomainValidationError(
            "Area of interest is too large; the maximum supported bounding box dimension is "
            f"{settings.max_bbox_dimension_deg} degrees."
        )

    polygon = box(min_lon, min_lat, max_lon, max_lat)
    area_m2 = geodesic_area_m2(polygon)
    area_km2 = area_m2 / 1_000_000.0
    if area_km2 > settings.max_bbox_area_km2:
        raise DomainValidationError(
            f"Area of interest covers {area_km2:.1f} km² which exceeds the configured "
            f"limit of {settings.max_bbox_area_km2:.1f} km²."
        )
    if area_km2 < settings.min_bbox_area_km2:
        raise DomainValidationError(
            f"Area of interest covers {area_km2:.4f} km² which is below the minimum "
            f"supported size of {settings.min_bbox_area_km2:.4f} km²."
        )
    return AreaOfInterest(
        min_lon=min_lon,
        min_lat=min_lat,
        max_lon=max_lon,
        max_lat=max_lat,
        area_m2=area_m2,
    )


def validate_flood_date(value: date, settings: Settings, *, today: date | None = None) -> date:
    reference = today or date.today()
    earliest = reference - timedelta(days=settings.max_flood_lookback_days)
    if value > reference:
        raise DomainValidationError(
            "flood_date must not be in the future; only historical events can be analysed."
        )
    if value < earliest:
        raise DomainValidationError(
            "flood_date is outside the supported archive window of "
            f"{settings.max_flood_lookback_days} days."
        )
    return value


def build_date_window(
    flood_date: date, settings: Settings, lookback_days: int | None, lookahead_days: int | None
) -> DateWindow:
    lookback = min(
        lookback_days or settings.default_lookback_days, settings.max_flood_lookback_days
    )
    lookahead = min(
        lookahead_days or settings.default_lookahead_days, settings.max_flood_lookahead_days
    )
    start = flood_date - timedelta(days=lookback)
    end = flood_date + timedelta(days=lookahead)
    pre_cutoff = flood_date - timedelta(days=1)
    post_cutoff = flood_date + timedelta(days=1)
    return DateWindow(
        flood_date=flood_date,
        window_start=start,
        window_end=end,
        pre_flood_date=pre_cutoff,
        post_flood_date=post_cutoff,
    )


def validate_geojson(geojson: dict[str, Any] | None) -> dict[str, Any] | None:
    if geojson is None:
        return None
    if not isinstance(geojson, dict):
        raise DomainValidationError("Stored geometry is not a GeoJSON object.")
    try:
        geometry = shape(geojson)
    except Exception as exc:  # noqa: BLE001
        raise DomainValidationError("Stored geometry could not be parsed.") from exc
    if geometry.is_empty:
        return None
    if not geometry.is_valid:
        reason = explain_validity(geometry)
        geometry = make_valid(geometry)
        if geometry.is_empty:
            raise DomainValidationError(f"Stored geometry is unusable: {reason}.")
    return mapping(geometry)


def geometry_summary(geometry: BaseGeometry | None) -> dict[str, Any]:
    if geometry is None or geometry.is_empty:
        return {"geometry_type": None, "polygon_count": 0, "part_count": 0}
    polygons = _polygon_parts(geometry)
    total_perimeter = 0.0
    for polygon in polygons:
        _, perimeter = _GEOD.geometry_area_perimeter(polygon)
        total_perimeter += abs(perimeter)
    return {
        "geometry_type": geometry.geom_type,
        "polygon_count": len(polygons),
        "part_count": len(getattr(geometry, "geoms", [geometry])),
        "perimeter_m": round(total_perimeter, 2),
        "crs": "EPSG:4326",
        "valid": bool(geometry.is_valid),
    }


def _polygon_parts(geometry: BaseGeometry) -> list[Polygon]:
    parts: list[Polygon] = []
    geoms = getattr(geometry, "geoms", None)
    if geoms is None:
        parts = [geometry] if isinstance(geometry, Polygon) else []
    else:
        for part in geoms:
            if isinstance(part, Polygon):
                parts.append(part)
            elif part.geom_type == "MultiPolygon":
                parts.extend(_polygon_parts(part))
            elif part.geom_type == "GeometryCollection":
                parts.extend(_polygon_parts(part))
    return parts