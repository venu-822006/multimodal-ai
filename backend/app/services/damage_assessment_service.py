from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import httpx
from pyproj import Transformer
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon, box, mapping
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.processing.validators import AreaOfInterest

logger = get_logger("app.damage")

ROAD_CLASSES = (
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    "motorway_link",
    "trunk_link",
    "primary_link",
    "secondary_link",
    "tertiary_link",
)


@dataclass
class ExposureAssessment:
    buildings_intersecting: int | None = None
    road_segments_intersecting: int | None = None
    roads_length_km_flooded: float | None = None
    settlements_total: int | None = None
    settlements_in_flood_zone: int | None = None
    settlements_near_flood: int | None = None
    buildings_total: int | None = None
    layers: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    available: bool = False


class OverpassUnavailableError(RuntimeError):
    """Every configured Overpass endpoint failed or returned an unusable response."""


class OverpassClient:
    """Thin Overpass API client with ordered endpoint fallback.

    Endpoint order matters. ``overpass-api.de`` is listed first because it is the
    most reliable of the public mirrors. Note that the configured User-Agent is a
    plain application identifier on purpose: several Overpass mirrors return HTTP 406
    to browser-like agents, so impersonating a browser makes the lookup *less* likely
    to succeed.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._endpoints = [
            endpoint
            for endpoint in [self._settings.osm_endpoint, *self._settings.osm_fallback_endpoints]
            if endpoint
        ]

    @property
    def endpoints(self) -> list[str]:
        return list(self._endpoints)

    def query(self, ql: str) -> dict[str, Any]:
        if not self._settings.osm_enabled:
            raise OverpassUnavailableError("OpenStreetMap lookups are disabled by configuration.")
        last_error: Exception | None = None
        for endpoint in self._endpoints:
            try:
                with httpx.Client(
                    timeout=httpx.Timeout(self._settings.osm_timeout_seconds),
                    follow_redirects=True,
                    headers={"User-Agent": self._settings.imagery_user_agent},
                ) as client:
                    response = client.post(
                        endpoint,
                        data={"data": ql},
                        headers={"Content-Type": "application/x-www-form-urlencoded"},
                    )
                    response.raise_for_status()
                    payload = response.json()
                if not isinstance(payload, dict) or "elements" not in payload:
                    # A mirror that answers with an HTML error page can still return 200.
                    raise OverpassUnavailableError(
                        f"{endpoint} returned a response that is not an Overpass result."
                    )
                return payload
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "overpass_endpoint_failed",
                    extra={
                        "event": "overpass_endpoint_failed",
                        "endpoint": endpoint,
                        "reason": type(exc).__name__,
                    },
                )
        raise OverpassUnavailableError(
            f"All configured OpenStreetMap endpoints failed: {last_error}"
        )


def _bbox_clause(aoi: AreaOfInterest) -> str:
    return f"{aoi.min_lat},{aoi.min_lon},{aoi.max_lat},{aoi.max_lon}"


def _overpass_query(cfg: Settings, aoi: AreaOfInterest, selector: str, limit: int) -> str:
    """One Overpass query.

    Categories are queried separately on purpose. A single combined query hits the
    element limit on whichever category is densest, and every later category then
    looks like it returned zero features when it was actually never returned.
    """
    return (
        "[out:json][timeout:{timeout}];{selector}({bbox});out geom {limit};"
    ).format(
        timeout=int(cfg.osm_timeout_seconds),
        selector=selector,
        bbox=_bbox_clause(aoi),
        limit=limit,
    )


@dataclass
class _CategoryResult:
    """Outcome of one category query.

    ``truncated`` means the element limit was reached, so a count of zero would be
    a false claim of absence rather than an observed zero.
    """

    elements: list[dict[str, Any]] = field(default_factory=list)
    available: bool = False
    truncated: bool = False
    reason: str | None = None

    @property
    def usable(self) -> bool:
        return self.available and not self.truncated


def _run_category(
    client: OverpassClient, cfg: Settings, aoi: AreaOfInterest, selector: str
) -> _CategoryResult:
    try:
        payload = client.query(_overpass_query(cfg, aoi, selector, cfg.osm_max_elements))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "overpass_query_failed",
            extra={"event": "overpass_query_failed", "reason": type(exc).__name__},
        )
        return _CategoryResult(available=False, reason=str(exc))
    elements = list(payload.get("elements", []))
    return _CategoryResult(
        elements=elements,
        available=True,
        # Overpass reports `remark: runtime error` when it truncates; the element
        # count hitting the limit is the reliable signal.
        truncated=len(elements) >= cfg.osm_max_elements,
    )


def assess_exposure(
    aoi: AreaOfInterest,
    flood_geometry: BaseGeometry | None,
    *,
    settings: Settings | None = None,
) -> ExposureAssessment:
    """Intersect OpenStreetMap features with the mapped flood extent.

    Statistics that could not be retrieved are reported as ``None``, never as zero,
    so a failed lookup is distinguishable from a genuinely empty area.
    """
    cfg = settings or get_settings()
    assessment = ExposureAssessment(
        limitations=[
            "OpenStreetMap coverage is incomplete and community maintained. Absence of a "
            "feature on the map is not evidence that it is unaffected.",
            "Intersections are geometric only. They do not model inundation depth, flow "
            "velocity or structural damage.",
        ]
    )
    if not cfg.osm_enabled:
        assessment.warnings.append(
            "OpenStreetMap exposure assessment is disabled by configuration, so building, "
            "road and settlement statistics are unavailable."
        )
        assessment.summary = {
            "status": "disabled",
            "message": "OpenStreetMap integration is disabled by server configuration.",
            "buildings_intersecting_flood": None,
            "road_segments_intersecting_flood": None,
            "settlements_inside_flood_extent": None,
            "settlements_within_2km_of_flood_extent": None,
        }
        assessment.provenance = {"source": "OpenStreetMap", "status": "disabled"}
        return assessment

    clip = box(*aoi.bounds)
    if flood_geometry is not None and not flood_geometry.is_valid:
        flood_geometry = flood_geometry.buffer(0)
    if flood_geometry is None:
        flood_geometry = Polygon()

    client = OverpassClient(cfg)
    buildings_q = _run_category(client, cfg, aoi, 'way["building"]')
    roads_q = _run_category(client, cfg, aoi, f'way["highway"~"^{"|".join(ROAD_CLASSES)}$"]')
    places_q = _run_category(
        client,
        cfg,
        aoi,
        'node["place"~"^(city|town|village|hamlet|suburb|neighbourhood|locality)$"]',
    )
    categories = {"buildings": buildings_q, "roads": roads_q, "settlements": places_q}

    failed = [name for name, result in categories.items() if not result.available]
    if len(failed) == len(categories):
        assessment.warnings.append(
            "The OpenStreetMap exposure lookup did not complete, so building, road and "
            "settlement statistics are unavailable for this analysis."
        )
        assessment.summary = {
            "status": "unavailable",
            "buildings_intersecting_flood": None,
            "road_segments_intersecting_flood": None,
            "settlements_inside_flood_extent": None,
            "settlements_within_2km_of_flood_extent": None,
        }
        assessment.provenance = {
            "source": "OpenStreetMap",
            "provider": "Overpass API",
            "status": "unavailable",
            "failed_categories": failed,
        }
        return assessment

    for name, result in categories.items():
        if not result.available:
            assessment.warnings.append(
                f"The OpenStreetMap lookup for {name} failed, so its statistics are "
                "unavailable rather than zero."
            )
        elif result.truncated:
            assessment.warnings.append(
                f"The area of interest exceeds the configured OpenStreetMap element limit of "
                f"{cfg.osm_max_elements} for {name}, so its statistics are incomplete and are "
                "reported as unavailable rather than zero."
            )

    buildings = _parse_buildings(buildings_q.elements, clip)
    roads = _parse_roads(roads_q.elements, clip)
    settlements = _parse_settlements(places_q.elements, clip)

    has_extent = not flood_geometry.is_empty

    touched_buildings = [
        building
        for building in buildings
        if has_extent and building.intersects(flood_geometry)
    ]
    flooded_roads = [
        road for road in roads if has_extent and road.intersects(flood_geometry)
    ]
    flooded_road_length = sum(_linear_length_km(road) for road in flooded_roads)

    near_flood = 0
    in_flood = 0
    settlement_features: list[dict[str, Any]] = []
    for tags, point, osm_id in settlements:
        feature: dict[str, Any] = {
            "type": "Feature",
            "properties": {
                "name": tags.get("name"),
                "place": tags.get("place"),
                "osm_id": f"node/{osm_id}" if osm_id else None,
            },
            "geometry": mapping(point),
        }
        if has_extent:
            if flood_geometry.contains(point):
                in_flood += 1
                feature["properties"]["flood_relation"] = "inside_mapped_flood_extent"
            elif point.distance(flood_geometry) <= 0.02:
                near_flood += 1
                feature["properties"]["flood_relation"] = (
                    "within_2km_of_mapped_flood_extent"
                )
            else:
                feature["properties"]["flood_relation"] = "outside_mapped_flood_extent"
        settlement_features.append(feature)

    buildings_usable = buildings_q.usable
    roads_usable = roads_q.usable
    places_usable = places_q.usable

    assessment.available = True
    assessment.buildings_total = len(buildings) if buildings_usable else None
    assessment.buildings_intersecting = len(touched_buildings) if buildings_usable else None
    assessment.road_segments_intersecting = (
        len(flooded_roads) if roads_usable else None
    )
    assessment.roads_length_km_flooded = (
        round(flooded_road_length, 4) if roads_usable else None
    )
    assessment.settlements_total = len(settlements) if places_usable else None
    assessment.settlements_in_flood_zone = in_flood if places_usable else None
    assessment.settlements_near_flood = near_flood if places_usable else None

    assessment.layers = {
        "infrastructure": _feature_collection(
            [
                fragment
                for building in touched_buildings
                for fragment in (
                    _feature(part, {"kind": "building", "intersects_flood": True})
                    for part in _polygonal_fragments(building)
                )
            ]
            + [
                _feature(road, {"kind": "road", "intersects_flood": True})
                for road in flooded_roads
            ]
        ),
        "settlements": _feature_collection(settlement_features),
    }
    assessment.summary = {
        "status": "ok" if not failed and not any(r.truncated for r in categories.values()) else "partial",
        "buildings_total": assessment.buildings_total,
        "buildings_intersecting_flood": assessment.buildings_intersecting,
        "road_segments_total": len(roads) if roads_usable else None,
        "road_segments_intersecting_flood": assessment.road_segments_intersecting,
        "roads_length_km_flooded": assessment.roads_length_km_flooded,
        "settlements_total": assessment.settlements_total,
        "settlements_inside_flood_extent": assessment.settlements_in_flood_zone,
        "settlements_within_2km_of_flood_extent": assessment.settlements_near_flood,
        "unavailable_categories": [
            name
            for name, result in categories.items()
            if not result.usable
        ],
    }
    assessment.provenance = {
        "source": "OpenStreetMap",
        "provider": "Overpass API",
        "endpoint": cfg.osm_endpoint,
        "endpoints_tried": client.endpoints,
        "license": "ODbL 1.0",
        "attribution": "(c) OpenStreetMap contributors",
        "query_scope": "buildings, primary road network and populated places within the AOI",
        "element_limit": cfg.osm_max_elements,
        "note": "Intersection statistics are geometric only and are not a damage estimate.",
    }
    assessment.warnings.append(
        "Building and road counts are geometric intersections with the mapped flood extent. "
        "They are not verified damage assessments."
    )
    return assessment


def _parse_buildings(elements: Sequence[dict[str, Any]], clip: BaseGeometry) -> list[BaseGeometry]:
    out: list[BaseGeometry] = []
    for element in elements:
        if element.get("type") != "way":
            continue
        geometry = element.get("geometry")
        if not geometry or "building" not in (element.get("tags") or {}):
            continue
        coordinates = [(point["lon"], point["lat"]) for point in geometry]
        if len(coordinates) < 3:
            continue
        candidate = Polygon(coordinates).buffer(0)
        if candidate.is_empty or not clip.intersects(candidate):
            continue
        out.append(candidate.intersection(clip))
    return out


def _parse_roads(elements: Sequence[dict[str, Any]], clip: BaseGeometry) -> list[BaseGeometry]:
    out: list[BaseGeometry] = []
    for element in elements:
        if element.get("type") != "way" or "highway" not in (element.get("tags") or {}):
            continue
        geometry = element.get("geometry")
        if not geometry:
            continue
        coordinates = [(point["lon"], point["lat"]) for point in geometry]
        if len(coordinates) < 2:
            continue
        candidate = LineString(coordinates)
        if not clip.intersects(candidate):
            continue
        clipped = candidate.intersection(clip)
        if clipped.is_empty:
            continue
        out.append(clipped)
    return out


def _parse_settlements(
    elements: Sequence[dict[str, Any]], clip: BaseGeometry
) -> list[tuple[dict[str, Any], Point, str]]:
    out: list[tuple[dict[str, Any], Point, str]] = []
    for element in elements:
        if element.get("type") != "node":
            continue
        tags = element.get("tags") or {}
        geometry = element.get("geometry")
        if "place" not in tags or not geometry:
            continue
        point = Point(geometry["lon"], geometry["lat"])
        if not clip.contains(point):
            continue
        out.append((tags, point, str(element.get("id", ""))))
    return out


def _feature(geometry: BaseGeometry, properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "Feature",
        "properties": properties,
        "geometry": mapping(geometry) if geometry is not None and not geometry.is_empty else None,
    }


def _feature_collection(features: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "FeatureCollection", "features": list(features)}


def _flatten(geometry: BaseGeometry) -> list[BaseGeometry]:
    geoms = getattr(geometry, "geoms", None)
    if geoms is None:
        return [geometry]
    parts: list[BaseGeometry] = []
    for part in geoms:
        parts.extend(_flatten(part))
    return parts


def _polygonal_fragments(geometry: BaseGeometry) -> list[Polygon]:
    if geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, (MultiPolygon, LineString, MultiLineString)):
        return [part for part in _flatten(geometry) if isinstance(part, Polygon)]
    return []


def _linear_length_km(geometry: BaseGeometry) -> float:
    if geometry.is_empty:
        return 0.0
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)
    total = 0.0
    for part in _flatten(geometry):
        if isinstance(part, (LineString, MultiLineString)):
            total += shapely_transform(transformer.transform, part).length
    return total / 1000.0