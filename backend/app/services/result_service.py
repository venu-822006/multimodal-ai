from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.core.enums import AnalysisStatus
from app.models.analysis import Analysis, AnalysisResult
from app.repositories.analyses import AnalysisView

NOT_COMPLETED_MESSAGE = (
    "This analysis has not finished processing, so no measurements are available yet. "
    "Poll the status endpoint until it reaches a terminal state."
)
FAILED_MESSAGE = (
    "This analysis failed before results were produced, so no flood extent or statistics "
    "exist for it."
)


def results_pending(analysis: Analysis) -> tuple[bool, str]:
    if analysis.status == AnalysisStatus.COMPLETED:
        return False, ""
    if analysis.status == AnalysisStatus.FAILED:
        return True, FAILED_MESSAGE
    if analysis.status == AnalysisStatus.CANCELLED:
        return True, "This analysis was cancelled before results were produced."
    return True, NOT_COMPLETED_MESSAGE


def build_facts(analysis: Analysis, result: AnalysisResult | None) -> dict[str, Any]:
    """Flatten persisted results into the facts contract the frontend renders."""
    pending, message = results_pending(analysis)
    payload: dict[str, Any] = {
        "analysis_id": str(analysis.id),
        "name": analysis.name,
        "status": analysis.status,
        "results_ready": False,
        "message": message,
        "flood_date": analysis.flood_date.isoformat(),
        "location": analysis.location_label,
        "aoi_bbox": ", ".join(
            f"{value:.6f}"
            for value in (analysis.min_lon, analysis.min_lat, analysis.max_lon, analysis.max_lat)
        ),
        "processing_duration_seconds": AnalysisView.duration_seconds(analysis),
        "completed_at": analysis.completed_at.isoformat() if analysis.completed_at else None,
    }
    if pending or result is None:
        payload["aoi_area_km2"] = (
            round(analysis.aoi_area_m2 / 1_000_000.0, 4) if analysis.aoi_area_m2 else None
        )
        payload["flooded_area_km2"] = None
        payload["flooded_percentage"] = None
        payload["warnings"] = (
            [analysis.failure_message] if analysis.failure_message else []
        )
        payload["limitations"] = []
        payload["unavailable_measurements"] = [
            "All measurements are unavailable until the analysis completes successfully."
        ]
        payload["map_available"] = False
        payload["report_available"] = False
        return payload

    measurements = result.measurements or {}
    provenance = result.provenance or {}
    model_info = provenance.get("model", {})
    provider_info = provenance.get("prediction_provider") or {}
    synthetic = bool(measurements.get("predictions_are_synthetic"))
    warnings = list(result.warnings or [])
    limitations = list(result.limitations or [])
    unavailable = list(measurements.get("unavailable", []))

    payload.update(
        {
            "results_ready": True,
            "message": "",
            "aoi_area_km2": result.aoi_area_km2,
            "aoi_area_m2": result.aoi_area_m2,
            "flooded_area_km2": result.flooded_area_km2,
            "flooded_area_m2": result.flooded_area_m2,
            "flooded_percentage": result.flooded_percentage,
            "flood_polygon_count": result.flood_polygon_count,
            "flood_geometry_summary": _geometry_summary_text(result),
            "valid_pixel_count": measurements.get("valid_pixel_count"),
            "flood_pixel_count": measurements.get("flood_pixel_count"),
            "retained_flood_pixel_count": measurements.get("retained_flood_pixel_count"),
            "filtered_out_pixel_count": measurements.get("filtered_out_pixel_count"),
            "valid_coverage_fraction": measurements.get("valid_coverage_fraction"),
            "min_flood_polygon_area_m2": measurements.get("min_flood_polygon_area_m2"),
            "processing_resolution_m": measurements.get("resolution_m"),
            "method": result.method,
            "methodology": result.methodology,
            "algorithm": (result.algorithm or {}).get("name"),
            "algorithm_version": (result.algorithm or {}).get("version"),
            "model_name": model_info.get("architecture"),
            "model_checkpoint": model_info.get("checkpoint"),
            "model_val_flood_iou": model_info.get("val_flood_iou"),
            "model_input_bands": model_info.get("input_bands"),
            "model_confidence_threshold": model_info.get("confidence_threshold"),
            "model_device": model_info.get("device"),
            "prediction_is_model_output_not_ground_truth": not synthetic,
            "prediction_provider": provider_info.get("provider")
            or measurements.get("prediction_provider"),
            "prediction_data_label": measurements.get("prediction_data_label"),
            "predictions_are_synthetic": synthetic,
            "metrics_are_synthetic": synthetic,
            "imagery_provider": provenance.get("catalogue_provider"),
            "imagery_catalogue": provenance.get("catalogue"),
            "imagery_scenes": [
                record.get("item_id") for record in (result.imagery or []) if record.get("item_id")
            ],
            "imagery_acquisition_dates": [
                record.get("acquisition_date")
                for record in (result.imagery or [])
                if record.get("acquisition_date")
            ],
            "warnings": warnings,
            "limitations": limitations,
            "unavailable_measurements": unavailable,
            "provenance": _provenance_text(provenance),
            "provenance_detail": {
                "prediction_provider": provider_info,
                "model": model_info,
                "imagery": result.imagery or [],
                "software": provenance.get("software"),
                "settlement_counts": (result.settlements or {}).get("settlement_counts")
                or measurements.get("settlement_counts"),
                "settlements": (result.settlements or {}).get("settlements")
                or measurements.get("settlements"),
                "debris_detection": measurements.get("debris_detection"),
                "cross_sensor_optical": measurements.get("cross_sensor_optical"),
                "dem_flow_analysis": measurements.get("dem_flow_analysis"),
                "anti_hallucination_metrics": measurements.get("anti_hallucination_metrics"),
                "uncertainty": measurements.get("uncertainty"),
                "infrastructure_detail": result.infrastructure or {},
                "buildings_affected": (result.infrastructure or {}).get("buildings_intersecting_flood")
                or (result.infrastructure or {}).get("buildings_intersecting")
                or measurements.get("buildings_affected"),
                "road_length_affected_km": (result.infrastructure or {}).get("roads_length_km_flooded")
                or measurements.get("road_length_affected_km"),
                "bridges_affected": measurements.get("bridges_affected")
                or (result.infrastructure or {}).get("bridges_affected"),
            },
            "buildings_intersecting_flood": (result.infrastructure or {}).get(
                "buildings_intersecting_flood"
            )
            or (result.infrastructure or {}).get("buildings_intersecting")
            or measurements.get("buildings_affected"),
            "road_segments_intersecting_flood": (result.infrastructure or {}).get(
                "road_segments_intersecting_flood"
            )
            or (result.infrastructure or {}).get("road_segments_intersecting")
            or measurements.get("road_segments_intersecting_flood"),
            "settlements_in_flood_zone": (result.settlements or {}).get(
                "settlements_in_flood_zone"
            )
            or (result.settlements or {}).get("settlements_inside_flood_extent")
            or measurements.get("settlements_in_flood_zone"),
            "map_available": True,
            "report_available": bool(analysis.report_available),
        }
    )
    return payload


def _geometry_summary_text(result: AnalysisResult) -> str | None:
    summary = result.geometry_summary or {}
    if not summary:
        return None
    parts = []
    if summary.get("polygon_count") is not None:
        parts.append(f"{summary['polygon_count']} polygon(s)")
    if summary.get("total_area_km2") is not None:
        parts.append(f"{summary['total_area_km2']} km²")
    if summary.get("geometry_type"):
        parts.append(str(summary["geometry_type"]))
    return ", ".join(parts) or None


def _provenance_text(provenance: dict[str, Any]) -> str | None:
    parts = []
    provider = provenance.get("prediction_provider") or {}
    if provider.get("provider"):
        parts.append(f"provider {provider['provider']}")
        if provider.get("data_label"):
            parts.append(str(provider["data_label"]))
    model = provenance.get("model", {})
    if model.get("architecture"):
        parts.append(str(model["architecture"]))
    if model.get("checkpoint"):
        parts.append(f"checkpoint {model['checkpoint']}")
    if provenance.get("catalogue_provider"):
        parts.append(str(provenance["catalogue_provider"]))
    if provenance.get("retrieved_at"):
        parts.append(f"retrieved {provenance['retrieved_at']}")
    return "; ".join(parts) or None


def build_map_payload(analysis: Analysis, result: AnalysisResult | None) -> dict[str, Any]:
    pending, message = results_pending(analysis)
    if pending or result is None:
        return {"status": analysis.status, "message": message, "layers": {}}

    flood = result.flood_geometry or {"type": "FeatureCollection", "features": []}
    infrastructure = (result.infrastructure or {}).get("layer") or {
        "type": "FeatureCollection",
        "features": [],
    }
    settlements = (result.settlements or {}).get("layer") or {
        "type": "FeatureCollection",
        "features": [],
    }
    features = flood.get("features", []) if isinstance(flood, dict) else []
    synthetic = bool((result.measurements or {}).get("predictions_are_synthetic"))
    extent_label = (
        "Synthetic test extent (synthetic_test_data). This is not a flood prediction."
        if synthetic
        else (
            "Model-predicted flood extent. Predictions are derived from radar "
            "imagery and are not confirmed ground truth."
        )
    )
    return {
        "analysis_id": str(analysis.id),
        "status": analysis.status,
        "type": "FeatureCollection",
        "crs": "EPSG:4326",
        "bbox": _feature_collection_bbox(flood),
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "features": features,
        "prediction_data_label": (result.measurements or {}).get("prediction_data_label"),
        "predictions_are_synthetic": synthetic,
        "layers": {
            "flood_extent": flood,
            "infrastructure": infrastructure,
            "settlements": settlements,
        },
        "legend": [
            {
                "id": "flood_extent",
                "label": "Flood extent" if not synthetic else "Synthetic test extent",
                "description": extent_label,
            },
            {
                "id": "infrastructure",
                "label": "Infrastructure intersecting flood extent",
                "description": (
                    "OpenStreetMap features intersecting the predicted extent. Intersection "
                    "does not establish damage."
                ),
            },
            {
                "id": "settlements",
                "label": "Settlements in the flood zone",
                "description": "Settlement locations falling within the predicted extent.",
            },
        ],
        "raster_overlay": _raster_overlay(result),
        "metrics": {
            "flooded_area_km2": result.flooded_area_km2,
            "flooded_percentage": result.flooded_percentage,
            "flood_polygon_count": result.flood_polygon_count,
            "flood_pixel_count": (result.measurements or {}).get("flood_pixel_count"),
            "predictions_are_synthetic": synthetic,
        },
    }


def _raster_overlay(result: AnalysisResult) -> dict[str, Any] | None:
    """Describe the stored rasters.

    ``flood_mask`` is the raw provider output; ``flood_extent_mask`` is the filtered
    extent the vector layer is derived from.
    """
    artifacts = result.artifacts or {}
    overlay: dict[str, Any] | None = None
    if artifacts.get("flood_mask"):
        overlay = {
            "kind": "GeoTIFF",
            "values": {"background": 0, "flood": 1, "nodata": 255},
            "artifact_key": "flood_mask",
            "description": (
                "Raw prediction mask as produced by the provider, before minimum-area "
                "polygon filtering."
            ),
        }
    if artifacts.get("flood_extent_mask"):
        overlay = overlay or {"kind": "GeoTIFF"}
        overlay["filtered_artifact_key"] = "flood_extent_mask"
        overlay["filtered_description"] = (
            "Minimum-area filtered extent used for the displayed geometry."
        )
    if overlay is not None:
        measurements = result.measurements or {}
        overlay["predictions_are_synthetic"] = bool(
            measurements.get("predictions_are_synthetic")
        )
        overlay["prediction_data_label"] = measurements.get("prediction_data_label")
    return overlay


def _feature_collection_bbox(collection: Any) -> list[float] | None:
    """Bounding box of a GeoJSON Feature/FeatureCollection as [w, s, e, n]."""
    if not isinstance(collection, dict):
        return None
    coordinates: list[float] = []

    def walk(node: Any) -> None:
        # A GeoJSON position is [x, y] or [x, y, z]; only x and y are wanted.
        if isinstance(node, dict):
            for key in ("geometry", "coordinates", "features"):
                if node.get(key) is not None:
                    walk(node[key])
            return
        if isinstance(node, (list, tuple)):
            if (
                len(node) >= 2
                and all(
                    isinstance(item, (int, float)) and not isinstance(item, bool) for item in node
                )
            ):
                coordinates.extend([float(node[0]), float(node[1])])
                return
            for item in node:
                walk(item)

    walk(collection)
    if not coordinates:
        return None
    xs = coordinates[0::2]
    ys = coordinates[1::2]
    return [round(min(xs), 6), round(min(ys), 6), round(max(xs), 6), round(max(ys), 6)]


__all__ = ["build_facts", "build_map_payload", "results_pending"]