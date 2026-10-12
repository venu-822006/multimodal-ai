"""
generate_report.py  —  FloodSight AI | One-Page Situation Report Generator
===========================================================================
Generates the official one-page executive situation report answering the
three core questions from the problem statement:

  1. WHERE THE FLOOD HIT:
     - Total inundation area (km²), debris candidate area, bounding box coordinates.
  2. WHAT WAS DAMAGED:
     - Affected buildings count, affected road length (km), affected bridges count.
     - Strictly formatted as "affected or potentially affected by mapped inundation".
  3. WHO MAY HAVE LOST ACCESS UNDER THE ROUTE SCENARIO:
     - Settlements by modeled state: CONNECTED, CUT_OFF, and UNKNOWN.
     - A CUT_OFF result is not a confirmed road closure.

EVERY NUMBER COMES DIRECTLY FROM facts.json.
Renders source attributions recorded in facts.json; it does not invent a provider
or licence when input metadata is missing.

Usage:
    python generate_report.py \
        --facts-json outputs/event_001_facts.json \
        --out-html   outputs/situation_report.html \
        [--out-pdf   outputs/situation_report.pdf]
"""

import argparse
from html import escape
import json
import sys
from datetime import datetime

# Conditional attribution references, emitted only if the corresponding input
# provider/product is verified: "Contains modified Copernicus Sentinel data";
# "Copernicus WorldDEM-30 © DLR e.V."; "Airbus Defence and Space GmbH";
# "© OpenStreetMap contributors"; and the "Kuro Siwo" dataset citation.
# Caller-supplied imagery has no inferred provider attribution by default.
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>FloodSight AI — Executive Situation Report</title>
<style>
  @page { size: A4; margin: 15mm; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #1f2937;
    background: #fff;
    margin: 0;
    padding: 20px;
    font-size: 13px;
    line-height: 1.45;
  }
  .header {
    border-bottom: 3px solid #1e40af;
    padding-bottom: 12px;
    margin-bottom: 16px;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
  }
  .title-group h1 {
    font-size: 24px;
    margin: 0;
    color: #1e3a8a;
    font-weight: 800;
    letter-spacing: -0.5px;
  }
  .title-group .subtitle {
    font-size: 12px;
    color: #4b5563;
    font-weight: 500;
  }
  .meta-box {
    text-align: right;
    font-size: 11px;
    color: #6b7280;
  }
  .badge {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
  }
  .badge-danger { background: #fee2e2; color: #991b1b; }
  .badge-warning { background: #fef3c7; color: #92400e; }
  .badge-success { background: #dcfce7; color: #166534; }

  .grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 16px;
  }
  .card {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 12px;
    background: #f9fafb;
  }
  .card-title {
    font-size: 11px;
    text-transform: uppercase;
    color: #6b7280;
    font-weight: 700;
    margin-bottom: 6px;
  }
  .card-stat {
    font-size: 24px;
    font-weight: 800;
    color: #111827;
  }
  .card-subtext {
    font-size: 11px;
    color: #6b7280;
    margin-top: 2px;
  }

  .section {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 14px;
    margin-bottom: 14px;
  }
  .section h2 {
    font-size: 14px;
    margin: 0 0 10px 0;
    color: #1e3a8a;
    font-weight: 700;
    border-bottom: 1px solid #e5e7eb;
    padding-bottom: 4px;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }
  th, td {
    padding: 6px 10px;
    text-align: left;
    border-bottom: 1px solid #e5e7eb;
  }
  th {
    background: #f3f4f6;
    font-weight: 600;
    color: #374151;
  }

  .stat-row {
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    border-bottom: 1px dashed #e5e7eb;
  }
  .stat-row:last-child { border-bottom: none; }

  .footer {
    margin-top: 18px;
    padding-top: 10px;
    border-top: 1px solid #d1d5db;
    font-size: 9.5px;
    color: #6b7280;
    line-height: 1.35;
  }
</style>
</head>
<body>

<div class="header">
  <div class="title-group">
    <h1>FloodSight AI</h1>
    <div class="subtitle">Rapid Spaceborne Flood Damage Assessment | Track B: Multimodal AI Hackathon 2026</div>
  </div>
  <div class="meta-box">
    <div><strong>Report ID:</strong> {{REPORT_ID}}</div>
    <div><strong>Generated:</strong> {{TIMESTAMP}}</div>
    <div><strong>Model Checkpoint:</strong> {{CHECKPOINT}}</div>
  </div>
</div>

<div class="grid">
  <div class="card">
    <div class="card-title">1. Flood Inundation Area</div>
    <div class="card-stat">{{FLOOD_AREA_KM2}} km²</div>
    <div class="card-subtext">{{FLOOD_PIXEL_COUNT}} flooded pixels ({{FLOOD_FRACTION}}% of area)</div>
  </div>
  <div class="card">
    <div class="card-title">2. Buildings Potentially Affected</div>
    <div class="card-stat">{{BUILDINGS_AFFECTED}}</div>
    <div class="card-subtext">{{ROAD_KM_AFFECTED}} km roads | {{BRIDGES_AFFECTED}} bridges</div>
  </div>
  <div class="card">
    <div class="card-title">3. Potential Access Loss (Modelled)</div>
    <div class="card-stat" style="color: #b91c1c;">{{CUT_OFF_COUNT}}</div>
    <div class="card-subtext">{{CONNECTED_COUNT}} connected | {{UNKNOWN_COUNT}} unmapped</div>
  </div>
</div>

<div class="section">
  <h2>Question 1: Where the Flood Hit (Satellite Inundation Extent)</h2>
  <div class="stat-row">
    <span><strong>Bounding Box (WGS-84):</strong></span>
    <span>{{BBOX}}</span>
  </div>
  <div class="stat-row">
    <span><strong>Spatial Resolution:</strong></span>
    <span>{{RESOLUTION}} m / pixel (Sentinel-1 SAR IW)</span>
  </div>
  <div class="stat-row">
    <span><strong>Primary Mapped Inundation (Class A):</strong></span>
    <span>{{FLOOD_AREA_KM2}} km²</span>
  </div>
  <div class="stat-row">
    <span><strong>Model ambiguity (predictive entropy, uncalibrated):</strong></span>
    <span>{{UNCERTAINTY_SUMMARY}}</span>
  </div>
  <div class="stat-row">
    <span><strong>Debris / Surface Change Candidate (Class B - Experimental):</strong></span>
    <span>{{DEBRIS_AREA_KM2}} km² (uncalibrated heuristic score: {{DEBRIS_SCORE}})</span>
  </div>
  <div class="stat-row">
    <span><strong>Terrain-only D8 downslope screen (not a flood forecast):</strong></span>
    <span>{{DEM_FLOW_INFO}}</span>
  </div>
  <div class="stat-row">
    <span><strong>Segmentation Model:</strong></span>
    <span>U-Net (dual-polarization SAR VV/VH segmentation)</span>
  </div>
</div>

<div class="section">
  <h2>Question 2: What Was Damaged (Infrastructure Exposure)</h2>
  <div class="stat-row">
    <span><strong>Building Structures:</strong></span>
    <span><strong>{{BUILDINGS_AFFECTED}}</strong> structures affected or potentially affected</span>
  </div>
  <div class="stat-row">
    <span><strong>Road Network Inundated:</strong></span>
    <span><strong>{{ROAD_KM_AFFECTED}} km</strong> of road segments intersecting flood mask</span>
  </div>
  <div class="stat-row">
    <span><strong>Bridges Impacted:</strong></span>
    <span><strong>{{BRIDGES_AFFECTED}}</strong> bridges within inundation zone (OSM snapshot)</span>
  </div>
  <div style="font-size: 10px; color: #6b7280; margin-top: 6px; font-style: italic;">
    * In accordance with operational standards, exposure numbers represent features affected or potentially affected by mapped inundation.
  </div>
</div>

<div class="section">
  <h2>Question 3: Potential Access Loss (Road Connectivity Scenario)</h2>
  <table>
    <thead>
      <tr>
        <th>Settlement / Community</th>
        <th>Access State</th>
        <th>Target Town / Medical Facility</th>
        <th>Pre-event Modeled Route Distance</th>
      </tr>
    </thead>
    <tbody>
      {{SETTLEMENT_ROWS}}
    </tbody>
  </table>
  <div style="font-size: 10.5px; color: #4b5563; margin-top: 8px;">
    <strong>Modelled scenario:</strong> {{CUT_OFF_COUNT}} communities have no remaining route to a mapped town or medical facility after overlapping road edges are treated as impassable; {{CONNECTED_COUNT}} retain a mapped route; {{UNKNOWN_COUNT}} are unknown. This is potential access loss, not a confirmed road closure. Verify routes in the field.
  </div>
</div>

<div class="section" style="background: #fdfbf7; border-color: #f3e8d2;">
  <h2>कार्यकारी सारांश (Bilingual Rescuer Briefing — Nepali)</h2>
  <div style="font-size: 11px; color: #451a03; line-height: 1.5;">
    <strong>उद्धार तथा राहत टोलीका लागि मुख्य वस्तुस्थिति:</strong><br>
    • कुल बाढी प्रभावित क्षेत्र: <strong>{{FLOOD_AREA_KM2}} वर्ग कि.मि.</strong> (क्षेत्रफलको {{FLOOD_FRACTION}}%)<br>
    • सम्भावित प्रभावित संरचनाहरू: <strong>{{BUILDINGS_AFFECTED}} घर/भवन</strong>, <strong>{{ROAD_KM_AFFECTED}} कि.मि. सडक खण्ड</strong>, र <strong>{{BRIDGES_AFFECTED}} पुल</strong><br>
    • मोडेलअनुसार सम्भावित सडक पहुँच अवरोध: <strong>{{CUT_OFF_COUNT}} बस्तीहरू</strong>मा नक्साङ्कित सडक खण्डहरू बाढी क्षेत्रमा परेका कारण वैकल्पिक मार्ग देखिएन ({{CONNECTED_COUNT}} मा मार्ग देखियो, {{UNKNOWN_COUNT}} अज्ञात)। यो पुष्टि भएको सडक बन्द अवस्था होइन; स्थलगत जाँच आवश्यक छ।
  </div>
</div>

<div class="footer">
  <strong>Data Sources & Attributions:</strong><br>
  {{DATA_ATTRIBUTIONS}}
</div>

</body>
</html>
"""


def render_report(facts_path: Path, out_html_path: Path, out_pdf_path: Path = None):
    with open(facts_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    now = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    report_id = f"FS-{abs(hash(facts_path.stem)) % 100000:05d}"

    # Extract facts with clean fallbacks
    flood_area = data.get("flood_area_km2")
    flood_px = data.get("flood_pixel_count")
    fraction = data.get("flood_fraction")
    flood_frac = round(fraction * 100, 2) if isinstance(fraction, (int, float)) else None
    bbox = str(data.get("bbox", ["N/A"]))
    res = data.get("pixel_resolution_approx_m", data.get("pixel_resolution_m"))
    ckpt = data.get("checkpoint", "best.pt")

    debris_area = data.get("debris_area_km2")
    debris_score = data.get("debris_heuristic_score", "not available")

    bldgs = data.get("buildings_affected")
    roads_km = data.get("road_length_affected_km")
    bridges = data.get("bridges_affected")
    bridges_conf = data.get("bridges_affected_confidence", "not calibrated")
    uncertainty = data.get("uncertainty", {})
    uncertainty_mean = uncertainty.get("mean") if isinstance(uncertainty, dict) else None
    uncertainty_high = uncertainty.get("fraction_at_least_0_5") if isinstance(uncertainty, dict) else None
    uncertainty_summary = (
        f"mean {uncertainty_mean:.3f}; {uncertainty_high * 100:.1f}% of valid pixels ≥ 0.5"
        if isinstance(uncertainty_mean, (int, float)) and isinstance(uncertainty_high, (int, float))
        else "not available"
    )
    attributions = data.get("data_attributions") or [
        "Data provenance and licences were not recorded for this analysis."
    ]
    attribution_html = "<br>".join(escape(str(item)) for item in attributions)

    settlement_counts = data.get("settlement_counts")
    cut_off = settlement_counts.get("CUT_OFF") if isinstance(settlement_counts, dict) else None
    connected = settlement_counts.get("CONNECTED") if isinstance(settlement_counts, dict) else None
    unknown = settlement_counts.get("UNKNOWN") if isinstance(settlement_counts, dict) else None

    settlements = data.get("settlements")
    if isinstance(settlements, list) and settlements:
        rows = []
        for s in settlements[:5]:  # show up to 5 on one-page
            state = s.get("state", "UNKNOWN")
            display_state = "POTENTIALLY CUT OFF (MODEL)" if state == "CUT_OFF" else state
            badge_cls = "badge-danger" if state == "CUT_OFF" else ("badge-success" if state == "CONNECTED" else "badge-warning")
            row = f"""<tr>
              <td><strong>{escape(str(s.get('settlement', 'Village')))}</strong></td>
              <td><span class="badge {badge_cls}">{escape(display_state)}</span></td>
              <td>{escape(str(s.get('target', 'Mapped destination')))}</td>
              <td>{escape(str(s.get('pre_distance_km', 'N/A')))} km</td>
            </tr>"""
            rows.append(row)
        settlement_rows = "\n".join(rows)
    elif isinstance(settlements, list):
        settlement_rows = """<tr>
          <td colspan="4" style="text-align: center; color: #6b7280; font-style: italic;">
            No mapped settlements recorded within this bounding box in pre-event OSM data.
          </td>
        </tr>"""
    else:
        settlement_rows = """<tr>
          <td colspan="4" style="text-align: center; color: #6b7280; font-style: italic;">
            Connectivity analysis was not available for this run.
          </td>
        </tr>"""

    dem_info = data.get("dem_flow_analysis", {})
    dem_len = dem_info.get("downstream_length_km")
    if dem_len is not None and dem_len > 0:
        dem_flow_text = f"{dem_len} km (elevation drop: {dem_info.get('elevation_drop_m', 0)} m, terrain gradient: {dem_info.get('terrain_gradient_m_per_km', dem_info.get('hydraulic_gradient_m_per_km', 0))} m/km)"
    else:
        dem_status = dem_info.get("status")
        if dem_status == "SKIPPED":
            dem_flow_text = "not available (no DEM supplied)"
        elif dem_status == "NO_DEM_COVERAGE":
            dem_flow_text = "not available (DEM does not cover mapped flood pixels)"
        else:
            dem_flow_text = "No terrain-only D8 path could be traced from the available DEM"

    html = HTML_TEMPLATE
    replacements = {
        "{{REPORT_ID}}": report_id,
        "{{TIMESTAMP}}": now,
        "{{CHECKPOINT}}": str(ckpt),
        "{{FLOOD_AREA_KM2}}": f"{flood_area:.2f}" if isinstance(flood_area, (int, float)) else "not available",
        "{{FLOOD_PIXEL_COUNT}}": f"{flood_px:,}" if isinstance(flood_px, int) else "not available",
        "{{FLOOD_FRACTION}}": str(flood_frac) if flood_frac is not None else "not available",
        "{{BBOX}}": bbox,
        "{{RESOLUTION}}": str(res) if res is not None else "not available",
        "{{UNCERTAINTY_SUMMARY}}": uncertainty_summary,
        "{{DEBRIS_AREA_KM2}}": f"{debris_area:.2f}" if isinstance(debris_area, (int, float)) else "not available",
        "{{DEBRIS_SCORE}}": str(debris_score),
        "{{DEM_FLOW_INFO}}": dem_flow_text,
        "{{BUILDINGS_AFFECTED}}": str(bldgs) if bldgs is not None else "not available",
        "{{ROAD_KM_AFFECTED}}": f"{roads_km:.2f}" if isinstance(roads_km, (int, float)) else "not available",
        "{{BRIDGES_AFFECTED}}": str(bridges) if bridges is not None else "not available",
        "{{BRIDGES_CONFIDENCE}}": str(bridges_conf),
        "{{OSM_SNAPSHOT_DATE}}": str(data.get("osm_snapshot_date", "not recorded")),
        "{{DATA_ATTRIBUTIONS}}": attribution_html,
        "{{CUT_OFF_COUNT}}": str(cut_off) if isinstance(cut_off, int) else "not available",
        "{{CONNECTED_COUNT}}": str(connected) if isinstance(connected, int) else "not available",
        "{{UNKNOWN_COUNT}}": str(unknown) if isinstance(unknown, int) else "not available",
        "{{SETTLEMENT_ROWS}}": settlement_rows,
    }

    for k, v in replacements.items():
        html = html.replace(k, v)

    out_html_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  [OK] Saved 1-page Situation Report HTML -> {out_html_path}")

    # Optional PDF rendering via WeasyPrint
    if out_pdf_path:
        try:
            from weasyprint import HTML
            HTML(filename=str(out_html_path)).write_pdf(str(out_pdf_path))
            print(f"  [OK] Saved PDF Report -> {out_pdf_path}")
        except ImportError:
            print("  [!] WeasyPrint not installed. The HTML report is styled for 1-page printing via browser (Ctrl+P -> Save as PDF).")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--facts-json", type=Path, required=True, help="Input facts.json file")
    p.add_argument("--out-html",   type=Path, default=Path("outputs/situation_report.html"), help="Output HTML situation report")
    p.add_argument("--out-pdf",    type=Path, default=None, help="(Optional) Output PDF situation report")
    args = p.parse_args()

    render_report(args.facts_json, args.out_html, args.out_pdf)


if __name__ == "__main__":
    main()
