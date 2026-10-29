from __future__ import annotations

import io
import re
from html import escape
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry

from app.models.analysis import Analysis, AnalysisResult

rl_config.invariant = 1
rl_config.useA85 = 1

PAGE_SIZES = {"A4": A4, "LETTER": LETTER}

ACCENT = colors.HexColor("#1d6f8b")
MUTED = colors.HexColor("#5b6b78")
LINE = colors.HexColor("#c9d5dd")


@dataclass(frozen=True)
class ReportSections:
    title: str
    description: str


class ReportService:
    def __init__(self, page_size: str = "A4") -> None:
        self._page_size = PAGE_SIZES.get(page_size.upper(), A4)

    def descriptor(
        self, analysis: Analysis, *, url: str, pdf_url: str, geojson_url: str | None
    ) -> dict[str, Any]:
        return {
            "analysis_id": str(analysis.id),
            "title": f"FloodSight AI situation report - {analysis.name}",
            "description": (
                "Generated from persisted analysis results. All measurements refer to the "
                "area of interest and imagery recorded in this report."
            ),
            "status": analysis.status,
            "format": "pdf",
            "url": url,
            "pdf_url": pdf_url,
            "geojson_url": geojson_url,
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "analysis_completed_at": (
                analysis.completed_at.isoformat() if analysis.completed_at else None
            ),
            "sections": [
                "Analysis metadata",
                "Area of interest",
                "Input imagery",
                "Methodology",
                "Flood extent results",
                "Measurements",
                "Infrastructure and settlement context",
                "Data quality and limitations",
                "Provenance",
            ],
        }

    def render_pdf(self, analysis: Analysis, result: AnalysisResult | None) -> bytes:
        buffer = io.BytesIO()
        document = BaseDocTemplate(
            buffer,
            pagesize=self._page_size,
            title=f"FloodSight AI report {analysis.id}",
            author="FloodSight AI",
            subject="Flood extent and impact situation report",
        )
        frame = Frame(
            18 * mm, 16 * mm, self._page_size[0] - 36 * mm, self._page_size[1] - 34 * mm, id="body"
        )
        document.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=_footer)])
        styles = _styles()
        story = self._build_story(analysis, result, styles)
        document.build(story)
        return buffer.getvalue()

    def _build_story(self, analysis: Analysis, result: AnalysisResult | None, styles):
        story: list[Any] = [
            Paragraph("FLOODSIGHT AI", styles["eyebrow"]),
            Paragraph("Flood extent situation report", styles["title"]),
            Spacer(1, 2 * mm),
            Paragraph(
                f"Analysis {analysis.id}<br/>Generated "
                f"{datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} from persisted "
                "analysis results.",
                styles["muted"],
            ),
            Spacer(1, 6 * mm),
            Paragraph("Analysis metadata", styles["heading"]),
            _kv_table(
                [
                    ("Analysis name", analysis.name),
                    ("Status", analysis.status),
                    ("Flood event date", analysis.flood_date.isoformat()),
                    ("Location label", analysis.location_label or "Not supplied"),
                    ("Requested method", analysis.requested_method),
                    ("Created", _fmt(analysis.created_at)),
                    ("Started", _fmt(analysis.started_at)),
                    ("Completed", _fmt(analysis.completed_at)),
                ],
                styles,
            ),
            Spacer(1, 5 * mm),
            Paragraph("Area of interest", styles["heading"]),
            Paragraph(
                "Bounding box (west, south, east, north): "
                f"{analysis.min_lon:.5f}, {analysis.min_lat:.5f}, "
                f"{analysis.max_lon:.5f}, {analysis.max_lat:.5f}",
                styles["body"],
            ),
            Paragraph(
                f"Area of interest size: {_fmt_number(_km2(analysis.aoi_area_m2))} km2.",
                styles["body"],
            ),
        ]

        if result is None:
            story.append(Spacer(1, 4 * mm))
            story.append(Paragraph(
                "No results are stored for this analysis. Processing did not complete, so no "
                "flood extent, measurement or impact figure is reported.",
                styles["warning"],
            ))
            return story

        story.extend(self._imagery_story(result, styles))
        story.extend(self._synthetic_story(result, styles))
        story.extend(self._method_story(result, styles))
        story.extend(self._measurements_story(result, styles))
        story.extend(self._context_story(result, styles))
        story.extend(self._model_story(result, styles))
        story.extend(self._quality_story(result, styles))
        story.extend(self._map_story(analysis, result, styles))
        return story

    def _synthetic_story(self, result: AnalysisResult, styles) -> list[Any]:
        """State unmistakably that the numbers are synthetic test data."""
        measurements = result.measurements or {}
        if not measurements.get("predictions_are_synthetic"):
            return []
        return [
            Spacer(1, 4 * mm),
            Paragraph("SYNTHETIC TEST DATA - NOT A FLOOD PREDICTION", styles["synthetic_banner"]),
            Paragraph(
                "This analysis was produced by the synthetic test provider because no trained "
                "model checkpoint was available. Every area, count, polygon and map layer below "
                "is a deterministic pipeline test value. Nothing in this report describes real "
                "flooding, and it must not be used for situational awareness, reporting or "
                "decision making.",
                styles["warning"],
            ),
        ]

    def _imagery_story(self, result: AnalysisResult, styles) -> list[Any]:
        imagery = result.imagery or []
        story: list[Any] = [Spacer(1, 4 * mm), Paragraph("Input imagery", styles["heading"])]
        if not imagery:
            story.append(Paragraph("No input imagery was recorded.", styles["body"]))
            return story
        rows = [["Collection", "Platform", "Acquired (UTC)", "Bands", "Resolution"]]
        for record in imagery:
            rows.append(
                [
                    str(record.get("collection") or "-"),
                    str(record.get("platform") or record.get("source") or "-"),
                    str(record.get("acquisition_datetime") or record.get("acquisition_date") or "-"),
                    ", ".join(record.get("bands") or []) or "-",
                    f"{record.get('resolution_m')} m",
                ]
            )
        story.append(_table(rows, styles, colWidths=[38 * mm, 24 * mm, 44 * mm, 26 * mm, 20 * mm]))
        return story

    def _method_story(self, result: AnalysisResult, styles) -> list[Any]:
        story: list[Any] = [Spacer(1, 4 * mm), Paragraph("Methodology", styles["heading"])]
        algorithm = result.algorithm or {}
        steps = algorithm.get("steps") or []
        if steps:
            for index, step in enumerate(steps, start=1):
                story.append(Paragraph(f"{index}. {escape(str(step))}", styles["step"]))
            reference = algorithm.get("reference")
            if reference:
                story.append(Paragraph(f"Reference: {escape(str(reference))}", styles["muted"]))
        else:
            methodology = str(result.methodology or "Not recorded.")
            # Providers often serialize the whole preprocessing pipeline into one
            # semicolon-delimited paragraph. Present each item as a readable step.
            items = [part.strip(" .;\n\t") for part in re.split(r";\s*", methodology) if part.strip(" .;\n\t")]
            for index, item in enumerate(items, start=1):
                story.append(Paragraph(f"{index}. {escape(item)}", styles["step"]))
        return story

    def _measurements_story(self, result: AnalysisResult, styles) -> list[Any]:
        measurements = result.measurements or {}
        synthetic = bool(measurements.get("predictions_are_synthetic"))
        heading = "Synthetic test measurements" if synthetic else "Flood extent results"
        section: list[Any] = [Paragraph(heading, styles["heading"])]
        if result.flood_geometry and not (result.flood_geometry.get("features") or []):
            section.append(Paragraph(
                (
                    "The synthetic test provider produced no mask above the configured minimum "
                    "area within the area of interest. No extent is reported."
                    if synthetic
                    else (
                        "The detection algorithm ran to completion but did not classify any "
                        "pixel as flooded within the area of interest. No flood extent is "
                        "reported."
                    )
                ),
                styles["warning"],
            ))
        rows = [
            ["Measurement", "Value"],
            ["Flooded area", f"{_fmt_number(result.flooded_area_km2)} km2"],
            [
                "Share of area classified as flooded",
                f"{_fmt_number(result.flooded_percentage)} %",
            ],
            ["Flood polygons", _fmt_number(result.flood_polygon_count)],
            ["Detection method", str(result.method or "-")],
            [
                "Prediction provider",
                str(measurements.get("prediction_provider") or "not recorded"),
            ],
            [
                "Raw flood pixels (provider mask)",
                _fmt_number(measurements.get("flood_pixel_count")),
            ],
            [
                "Pixels kept after polygon filtering",
                _fmt_number(measurements.get("retained_flood_pixel_count")),
            ],
            [
                "Minimum polygon area",
                f"{_fmt_number(measurements.get('min_flood_polygon_area_m2'))} m2",
            ],
            ["Valid pixels", _fmt_number(measurements.get("valid_pixel_count"))],
            ["Processing resolution", f"{_fmt_number(measurements.get('resolution_m'))} m"],
            [
                "Detection threshold",
                f"{_fmt_number((result.confidence or {}).get('threshold_db', (result.confidence or {}).get('threshold')))}",
            ],
            [
                "Class separability index",
                f"{_fmt_number((result.confidence or {}).get('index'))}",
            ],
        ]
        section.append(_table(rows, styles, colWidths=[70 * mm, 80 * mm]))
        # Keep the section title and its table together when a page is nearly full.
        story: list[Any] = [Spacer(1, 4 * mm), KeepTogether(section)]
        confidence = result.confidence or {}
        if confidence.get("definition"):
            story.append(Spacer(1, 2 * mm))
            story.append(Paragraph(f"Confidence definition: {confidence['definition']}", styles["muted"]))
        return story

    def _context_story(self, result: AnalysisResult, styles) -> list[Any]:
        story: list[Any] = [
            Spacer(1, 4 * mm),
            Paragraph("Infrastructure and settlement context", styles["heading"]),
        ]
        settlements = result.settlements or {}
        infrastructure_layers = result.infrastructure or {}
        building_count = len(
            [
                feature
                for feature in infrastructure_layers.get("features", [])
                if feature.get("properties", {}).get("kind") == "building"
            ]
        )
        rows = [
            ["Indicator", "Value"],
            ["Buildings intersecting the mapped extent", _fmt_number(building_count)],
            [
                "Road segments intersecting the mapped extent",
                _fmt_number(settlements.get("road_segments_intersecting_flood")),
            ],
            [
                "Settlements inside the mapped extent",
                _fmt_number(settlements.get("settlements_inside_flood_extent")),
            ],
            [
                "Settlements within 2 km of the mapped extent",
                _fmt_number(settlements.get("settlements_within_2km_of_flood_extent")),
            ],
        ]
        if not settlements:
            story.append(Paragraph(
                "OpenStreetMap exposure analysis did not complete for this analysis, so these "
                "indicators are unavailable.",
                styles["warning"],
            ))
        story.append(_table(rows, styles, colWidths=[95 * mm, 55 * mm]))
        return story

    def _model_story(self, result: AnalysisResult, styles) -> list[Any]:
        """Record which weights produced this report and how they were applied."""
        model = (result.provenance or {}).get("model") or {}
        provider = (result.provenance or {}).get("prediction_provider") or {}
        if not model and not provider:
            return []
        if model.get("synthetic") or provider.get("synthetic"):
            return self._synthetic_provenance_story(model, styles)
        story: list[Any] = [
            Spacer(1, 4 * mm),
            Paragraph("Model provenance", styles["heading"]),
        ]
        iou = model.get("val_flood_iou")
        rows: list[list[str]] = [
            ["Architecture", str(model.get("architecture") or "not recorded")],
            ["Configuration", str(model.get("model_config") or "not recorded")],
            ["Checkpoint", str(model.get("checkpoint") or "not recorded")],
            ["Input bands", ", ".join(model.get("input_bands") or []) or "not recorded"],
            ["Input units", str(model.get("input_units") or "not recorded")],
            ["Inference device", str(model.get("device") or "not recorded")],
            [
                "Tiling",
                f"{model.get('patch_size')} px patches, stride {model.get('stride')}, "
                f"{model.get('tiles')} tiles",
            ],
            ["Flood probability threshold", str(model.get("confidence_threshold") or "not recorded")],
            [
                "Temporal consistency filter",
                "applied" if model.get("temporal_filter_applied") else "not applied",
            ],
            [
                "Checkpoint validation flood IoU",
                f"{float(iou):.4f}" if isinstance(iou, (int, float)) else "not recorded",
            ],
        ]
        story.append(_table(rows, styles, colWidths=[62 * mm, 88 * mm]))
        if iou is None:
            story.append(Paragraph(
                "The loaded checkpoint does not record a validation IoU, so the accuracy of "
                "these weights cannot be confirmed from the checkpoint itself.",
                styles["warning"],
            ))
        story.append(Paragraph(
            "Flood extents in this report are model predictions derived from satellite radar "
            "imagery. They are not confirmed ground truth.",
            styles["limitation"],
        ))
        return story

    def _synthetic_provenance_story(self, model: dict, styles) -> list[Any]:
        return [
            Spacer(1, 4 * mm),
            Paragraph("Prediction provenance (synthetic)", styles["heading"]),
            _kv_table(
                [
                    ("Prediction provider", str(model.get("provider") or "synthetic_test")),
                    ("Data label", str(model.get("data_label") or "synthetic_test_data")),
                    ("Pattern", str(model.get("pattern") or "not recorded")),
                    ("Seed", str(model.get("seed", "not recorded"))),
                    ("Target coverage", str(model.get("target_coverage", "not recorded"))),
                    (
                        "Temporal consistency filter",
                        "applied" if model.get("temporal_filter_applied") else "not applied",
                    ),
                    ("Trained checkpoint loaded", "none"),
                ],
                styles,
            ),
            Spacer(1, 2 * mm),
            Paragraph(
                "No trained weights were loaded for this analysis. The values above and in the "
                "measurements table are synthetic test data.",
                styles["warning"],
            ),
        ]

    def _quality_story(self, result: AnalysisResult, styles) -> list[Any]:
        story: list[Any] = [
            Spacer(1, 4 * mm),
            Paragraph("Data quality, warnings and limitations", styles["heading"]),
        ]
        warnings = result.warnings or []
        limitations = result.limitations or []
        if not warnings and not limitations:
            story.append(Paragraph(
                "No warnings or limitations were recorded for this analysis.", styles["body"]
            ))
        for warning in warnings:
            story.append(Paragraph(f"Warning: {escape(str(warning))}", styles["warning"]))
        for limitation in limitations:
            story.append(Paragraph(f"Limitation: {escape(str(limitation))}", styles["limitation"]))
        return story

    def _map_story(self, analysis: Analysis, result: AnalysisResult, styles) -> list[Any]:
        geometry = _geometry_from_feature_collection(result.flood_geometry)
        story: list[Any] = [Spacer(1, 4 * mm)]
        drawing = _map_drawing(analysis, geometry, self._page_size)
        if drawing is None:
            story.append(Paragraph(
                "No flood geometry is available to display.", styles["warning"]
            ))
            return story
        story.append(KeepTogether([
            Paragraph("Flood extent overview", styles["heading"]),
            drawing,
            Paragraph(
                "Flood extent outlines within the area of interest. The figure uses the same "
                "validated geometry returned by the map endpoint.",
                styles["muted"],
            ),
        ]))
        return story


def _footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 10 * mm, "FloodSight AI - machine generated analysis report")
    canvas.drawRightString(
        document.pagesize[0] - 18 * mm, 10 * mm, f"Page {document.page}"
    )
    canvas.restoreState()


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "eyebrow": ParagraphStyle(
            "eyebrow", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=8,
            leading=11, textColor=ACCENT, spaceAfter=2,
        ),
        "title": ParagraphStyle(
            "title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=19,
            leading=23, textColor=colors.HexColor("#10222f"), spaceAfter=3,
        ),
        "heading": ParagraphStyle(
            "heading", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=11,
            leading=14, textColor=colors.HexColor("#10222f"), spaceBefore=4, spaceAfter=3,
        ),
        "body": ParagraphStyle(
            "body", parent=base["BodyText"], fontName="Helvetica", fontSize=8.5, leading=11.5,
        ),
        "step": ParagraphStyle(
            "step", parent=base["BodyText"], fontName="Helvetica", fontSize=8.5, leading=11.5,
            leftIndent=8,
        ),
        "muted": ParagraphStyle(
            "muted", parent=base["BodyText"], fontName="Helvetica", fontSize=7.5, leading=10,
            textColor=MUTED,
        ),
        "warning": ParagraphStyle(
            "warning", parent=base["BodyText"], fontName="Helvetica", fontSize=8.5, leading=11.5,
            textColor=colors.HexColor("#8a4b16"),
        ),
        "limitation": ParagraphStyle(
            "limitation", parent=base["BodyText"], fontName="Helvetica", fontSize=8.5,
            leading=11.5, textColor=colors.HexColor("#5f5a2a"),
        ),
        "synthetic_banner": ParagraphStyle(
            "synthetic_banner", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=12, leading=15, textColor=colors.HexColor("#8c1d18"),
            borderColor=colors.HexColor("#8c1d18"), borderWidth=1,
            borderPadding=5, spaceAfter=3,
        ),
        "cell": ParagraphStyle(
            "cell", parent=base["BodyText"], fontName="Helvetica", fontSize=8, leading=10,
        ),
        "code": ParagraphStyle(
            "code", parent=base["Code"], fontName="Courier", fontSize=6.5, leading=8,
        ),
    }


def _table(rows: list[list[str]], styles, colWidths: list[float]) -> Table:
    data = [[Paragraph(escape(str(cell)), styles["cell"]) for cell in row] for row in rows]
    table = Table(data, colWidths=colWidths, hAlign="LEFT", repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eff3")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#10222f")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.4, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _kv_table(pairs: list[tuple[str, str]], styles) -> Table:
    rows = [
        [Paragraph(escape(str(key)), styles["cell"]), Paragraph(escape(str(value)), styles["cell"])]
        for key, value in pairs
    ]
    table = Table(rows, colWidths=[48 * mm, 102 * mm], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef3f6")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _map_drawing(analysis: Analysis, geometry: BaseGeometry | None, page_size):
    from reportlab.graphics.shapes import Drawing, PolyLine, Rect

    width = 150 * mm
    height = 95 * mm
    drawing = Drawing(width, height)
    drawing.add(Rect(0, 0, width, height, fillColor=colors.white, strokeColor=LINE, strokeWidth=0.7))
    if geometry is None:
        return drawing
    aoi = box(analysis.min_lon, analysis.min_lat, analysis.max_lon, analysis.max_lat)
    extent = aoi
    if geometry is not None and not geometry.is_empty:
        extent = geometry.union(aoi)
    min_lon, min_lat, max_lon, max_lat = extent.bounds
    span_lon = max(max_lon - min_lon, 1e-9)
    span_lat = max(max_lat - min_lat, 1e-9)
    padding = 0.08
    min_lon -= span_lon * padding
    max_lon += span_lon * padding
    min_lat -= span_lat * padding
    max_lat += span_lat * padding
    span_lon = max_lon - min_lon
    span_lat = max_lat - min_lat

    def project_ring(ring) -> list[list[float]]:
        """Project one ring without connecting it to neighboring polygons."""
        return [
            [(x - min_lon) / span_lon * width, (y - min_lat) / span_lat * height]
            for x, y in ring.coords
        ]

    def add_outlines(geom: BaseGeometry, *, stroke, stroke_width: float, dash=None) -> None:
        parts = getattr(geom, "geoms", [geom])
        for part in parts:
            if part.geom_type != "Polygon":
                continue
            for ring in [part.exterior, *part.interiors]:
                drawing.add(
                    PolyLine(
                        project_ring(ring),
                        strokeColor=stroke,
                        strokeWidth=stroke_width,
                        strokeDashArray=dash,
                        fillColor=None,
                    )
                )

    add_outlines(aoi, stroke=ACCENT, stroke_width=0.9, dash=[3, 2])
    if geometry is not None and not geometry.is_empty:
        add_outlines(geometry, stroke=colors.HexColor("#c0504d"), stroke_width=1.1)
    return drawing


def _geometry_from_feature_collection(feature_collection: dict[str, Any] | None):
    if not feature_collection:
        return None
    parts = []
    for feature in feature_collection.get("features") or []:
        geometry = feature.get("geometry")
        if not geometry:
            continue
        if geometry.get("type") == "Polygon":
            parts.append(Polygon(geometry["coordinates"][0]))
    if not parts:
        return None
    from shapely.ops import unary_union

    return unary_union(parts)


def _fmt(value: datetime | None) -> str:
    if value is None:
        return "Not available"
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _fmt_number(value: Any) -> str:
    if value is None:
        return "Not computed"
    if isinstance(value, (int,)):
        return str(value)
    try:
        return f"{float(value):,.4f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(value)


def _km2(area_m2: float | None) -> float | None:
    if area_m2 is None:
        return None
    return area_m2 / 1_000_000.0


__all__ = ["ReportService", "ReportSections"]
