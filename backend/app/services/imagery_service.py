from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Iterable, Sequence
from urllib.parse import urlsplit, urlunsplit

from app.core.config import Settings, get_settings
from app.core.enums import ImagerySource
from app.core.exceptions import UpstreamError
from app.core.logging import get_logger
from app.processing.validators import AreaOfInterest, DateWindow

logger = get_logger("app.imagery")

SAR_COLLECTIONS = {
    ImagerySource.SENTINEL1_RTC: {
        "collection": "sentinel-1-rtc",
        "label": "Sentinel-1 GRD Radiometrically Terrain Corrected",
        "bands": ("vv", "vh"),
        "resolution_m": 10.0,
        "nodata": (-32768.0,),
    }
}

OPTICAL_COLLECTIONS = {
    ImagerySource.SENTINEL2_L2A: {
        "collection": "sentinel-2-l2a",
        "label": "Sentinel-2 Level-2A surface reflectance",
        "bands": ("B03", "B08"),
        "resolution_m": 10.0,
        "nodata": (0.0,),
    }
}


@dataclass(frozen=True)
class SceneAsset:
    source: str
    collection: str
    item_id: str
    platform: str | None
    acquisition: datetime
    bbox: list[float]
    hrefs: dict[str, str]
    resolution_m: float
    nodata_values: tuple[float, ...]
    properties: dict[str, Any] = field(default_factory=dict)

    def public_record(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "collection": self.collection,
            "item_id": self.item_id,
            "platform": self.platform,
            "acquisition_datetime": self.acquisition.isoformat(),
            "acquisition_date": self.acquisition.date().isoformat(),
            "bbox": [round(value, 6) for value in self.bbox],
            "bands": sorted(self.hrefs),
            "resolution_m": self.resolution_m,
            "properties": {
                key: self.properties.get(key)
                for key in (
                    "sar:instrument_mode",
                    "sar:polarizations",
                    "sat:relative_orbit",
                    "platform",
                    "constellation",
                    "eo:cloud_cover",
                    "s2:mgrs_tile",
                )
                if self.properties.get(key) is not None
            },
        }


def strip_sas_token(href: str) -> str:
    parts = urlsplit(href)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


class PlanetaryComputerStacClient:
    """Thin wrapper around the Microsoft Planetary Computer STAC API.

    The endpoint serves Copernicus Sentinel data as cloud optimised GeoTIFFs.
    Asset access uses short lived signed URLs which are never persisted.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def _client(self):
        try:
            from pystac_client import Client
        except ImportError as exc:  # pragma: no cover
            raise UpstreamError(
                "STAC client support is not installed on this deployment.",
                retryable=False,
            ) from exc
        return Client.open(
            self._settings.stac_endpoint,
            headers={"User-Agent": self._settings.imagery_user_agent},
        )

    def search(
        self,
        source: ImagerySource,
        aoi: AreaOfInterest,
        window: DateWindow,
        *,
        max_cloud_cover: float = 100.0,
        limit: int | None = None,
    ) -> list[SceneAsset]:
        catalogue = (
            SAR_COLLECTIONS.get(source) or OPTICAL_COLLECTIONS.get(source)
        )
        if catalogue is None:
            raise UpstreamError(f"Imagery source {source} is not supported.", retryable=False)
        client = self._client()
        query: dict[str, Any] = {
            "collections": [catalogue["collection"]],
            "bbox": aoi.bbox_list,
            "datetime": f"{window.window_start.isoformat()}/{window.window_end.isoformat()}",
            "max_items": limit or self._settings.imagery_max_candidates,
        }
        if source is ImagerySource.SENTINEL2_L2A:
            query["query"] = {"eo:cloud_cover": {"lt": max_cloud_cover}}
        started = time.perf_counter()
        try:
            search = client.search(**query)
            items = list(search.items())
        except Exception as exc:  # noqa: BLE001
            raise UpstreamError(
                f"The imagery catalogue could not be queried ({type(exc).__name__}).",
                retryable=True,
            ) from exc
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        scenes: list[SceneAsset] = []
        for item in items:
            scene = self._to_scene(item, source, catalogue)
            if scene is not None:
                scenes.append(scene)
        scenes.sort(key=lambda item: item.acquisition)
        logger.info(
            "imagery_search_completed",
            extra={
                "event": "imagery_search_completed",
                "source": str(source),
                "collection": catalogue["collection"],
                "candidates": len(scenes),
                "duration_ms": elapsed_ms,
                "window_start": window.window_start.isoformat(),
                "window_end": window.window_end.isoformat(),
            },
        )
        return scenes

    def _to_scene(self, item, source: ImagerySource, catalogue: dict[str, Any]):
        try:
            hrefs = {}
            for band in catalogue["bands"]:
                asset = item.assets.get(band)
                if asset is None:
                    return None
                hrefs[band] = asset.href
            signed = _sign_item(item)
            signed_hrefs = {}
            for band, href in hrefs.items():
                signed_asset = signed.assets.get(band)
                signed_hrefs[band] = signed_asset.href if signed_asset else href
            acquisition = item.datetime or item.properties.get("start_datetime")
            if acquisition is None:
                return None
            if acquisition.tzinfo is None:
                acquisition = acquisition.replace(tzinfo=timezone.utc)
            return SceneAsset(
                source=str(source),
                collection=catalogue["collection"],
                item_id=item.id,
                platform=item.properties.get("platform"),
                acquisition=acquisition,
                bbox=list(item.bbox),
                hrefs=signed_hrefs,
                resolution_m=float(catalogue["resolution_m"]),
                nodata_values=tuple(catalogue["nodata"]),
                properties=dict(item.properties),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "imagery_item_skipped",
                extra={
                    "event": "imagery_item_skipped",
                    "source": str(source),
                    "item_id": getattr(item, "id", "unknown"),
                    "reason": type(exc).__name__,
                },
            )
            return None


def _sign_item(item):
    try:
        from planetary_computer import sign
    except ImportError:  # pragma: no cover
        return item
    return sign(item)


def select_sar_pair(
    scenes: Sequence[SceneAsset], window: DateWindow
) -> tuple[SceneAsset | None, SceneAsset | None]:
    pre = [scene for scene in scenes if scene.acquisition.date() <= window.pre_flood_date]
    post = [scene for scene in scenes if scene.acquisition.date() >= window.post_flood_date]
    if not pre or not post:
        return None, None
    pre_scene = max(pre, key=lambda scene: scene.acquisition)
    post_scene = min(post, key=lambda scene: scene.acquisition)
    if pre_scene.item_id == post_scene.item_id:
        return pre_scene, None
    return pre_scene, post_scene


def select_optical_scene(
    scenes: Sequence[SceneAsset], window: DateWindow, *, max_cloud_cover: float
) -> SceneAsset | None:
    candidates = [
        scene
        for scene in scenes
        if window.pre_flood_date <= scene.acquisition.date() <= window.window_end
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda scene: (
            float(scene.properties.get("eo:cloud_cover", 100.0) or 100.0),
            abs((scene.acquisition.date() - window.flood_date).days),
        )
    )
    return candidates[0]


def describe_sources(sources: Iterable[ImagerySource]) -> str:
    labels = []
    for source in sources:
        catalogue = SAR_COLLECTIONS.get(source) or OPTICAL_COLLECTIONS.get(source)
        if catalogue:
            labels.append(catalogue["collection"])
    return ", ".join(labels) if labels else "none configured"


def scene_dates(scenes: Iterable[SceneAsset]) -> list[date]:
    return sorted({scene.acquisition.date() for scene in scenes})