"""Create a lightweight interactive Leaflet dashboard from a completed job."""

import html
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, mapping
from shapely.ops import unary_union


def flood_geojson(mask_path: Path, simplify_tolerance: float = 0.00003) -> dict:
    """Vectorize positive pixels, preserving georeferencing and nodata holes."""
    with rasterio.open(mask_path) as src:
        if src.count != 1 or src.crs is None:
            raise ValueError("Dashboard flood mask must be a single-band georeferenced raster.")
        band = src.read(1)
        mask = band == 1
        if not mask.any():
            return {"type": "FeatureCollection", "features": []}
        geometries = [shape(geom) for geom, value in shapes(
            band.astype(np.uint8), mask=mask, transform=src.transform, connectivity=8
        ) if value == 1]
        merged = unary_union(geometries).simplify(simplify_tolerance, preserve_topology=True)
        parts = list(merged.geoms) if merged.geom_type == "MultiPolygon" else [merged]
        return {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {"class": "Mapped inundation"}, "geometry": mapping(part)}
                for part in parts if not part.is_empty
            ],
        }


def _safe_json(value) -> str:
    # Prevent OSM tag strings from closing an inline script element.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def render_dashboard(facts_path: Path, flood_mask_path: Path, out_path: Path, osm_data: dict | None = None):
    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    flood = flood_geojson(flood_mask_path)
    roads = (osm_data or {}).get("roads", {"type": "FeatureCollection", "features": []})
    hospitals = (osm_data or {}).get("hospitals", {"type": "FeatureCollection", "features": []})
    settlements = facts.get("settlements", [])
    bbox = facts.get("bbox", [85.1, 27.8, 85.5, 28.1])
    center = [(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2]

    cards = {
        "Flood area": f"{facts.get('flood_area_km2', 'N/A')} km²",
        "Buildings potentially affected": facts.get("buildings_affected", "N/A"),
        "Road length intersecting flood": f"{facts.get('road_length_affected_km', 'N/A')} km",
        "Settlements cut off": facts.get("settlement_counts", {}).get("CUT_OFF", "N/A"),
    }
    card_html = "".join(
        f'<article><span>{html.escape(str(label))}</span><strong>{html.escape(str(value))}</strong></article>'
        for label, value in cards.items()
    )
    attribution = (
        "Contains modified Copernicus Sentinel data 2026. "
        f"© OpenStreetMap contributors (pre-event snapshot: {facts.get('osm_snapshot_date', 'not recorded')}). "
        "Educational prototype; verify every layer before operational use."
    )
    template = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FloodSight | Event map dashboard</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
*{box-sizing:border-box}body{margin:0;background:#f3f6fb;color:#14213d;font:15px system-ui,Segoe UI,sans-serif}
header{padding:18px 24px;background:#102a43;color:white}header h1{margin:0;font-size:22px}header p{margin:4px 0 0;color:#d9e7f2}
.cards{display:grid;grid-template-columns:repeat(4,minmax(140px,1fr));gap:12px;padding:14px 18px}.cards article{background:white;border-radius:10px;padding:12px 14px;box-shadow:0 2px 8px #102a4314}.cards span{display:block;color:#526477;font-size:12px}.cards strong{display:block;margin-top:5px;font-size:19px}
#map{height:calc(100vh - 220px);min-height:420px;margin:0 18px;border-radius:10px;box-shadow:0 3px 14px #102a4320}.note{padding:12px 22px;color:#526477;font-size:12px}
@media(max-width:700px){.cards{grid-template-columns:repeat(2,1fr)}#map{height:65vh;margin:0 10px}header{padding:16px}}
</style></head><body><header><h1>FloodSight · Event map</h1><p>Satellite flood extent, pre-event roads, and settlement access status</p></header>
<section class="cards">__CARDS__</section><div id="map" role="application" aria-label="Interactive flood extent map"></div>
<p class="note">__ATTRIBUTION__ Base-map tiles require an internet connection. Mapped inundation and OSM intersections indicate exposure; they do not confirm physical damage. Sentinel revisit timing, terrain shadow, OSM coverage, and model uncertainty apply.</p>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script><script>
const flood=__FLOOD__, roads=__ROADS__, hospitals=__HOSPITALS__, settlements=__SETTLEMENTS__;
const map=L.map('map').setView(__CENTER__,9);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'© OpenStreetMap contributors'}).addTo(map);
const floodLayer=L.geoJSON(flood,{style:{color:'#c2410c',weight:1,fillColor:'#f97316',fillOpacity:.5},onEachFeature:(f,l)=>l.bindPopup('Mapped inundation')}).addTo(map);
const roadLayer=L.geoJSON(roads,{style:{color:'#64748b',weight:2,opacity:.8},onEachFeature:(f,l)=>{const name=f.properties&&f.properties.name;if(name)l.bindPopup('Road: '+escapeHtml(name))}});
const hospitalLayer=L.geoJSON(hospitals,{pointToLayer:(f,latlng)=>L.circleMarker(latlng,{radius:6,color:'#2563eb',fillOpacity:.9}),onEachFeature:(f,l)=>l.bindPopup('Medical facility: '+escapeHtml((f.properties&&f.properties.name)||'Unnamed'))});
const colors={CUT_OFF:'#dc2626',CONNECTED:'#16a34a',UNKNOWN:'#64748b'};
const settlementLayer=L.layerGroup(settlements.filter(s=>Number.isFinite(s.longitude)&&Number.isFinite(s.latitude)).map(s=>L.circleMarker([s.latitude,s.longitude],{radius:6,color:colors[s.state]||colors.UNKNOWN,fillOpacity:.9}).bindPopup('<b>'+escapeHtml(s.settlement||'Unnamed settlement')+'</b><br>Status: '+escapeHtml(s.state||'UNKNOWN')+(s.target?'<br>Target: '+escapeHtml(s.target):'')+(s.reason?'<br>'+escapeHtml(s.reason):''))));
L.control.layers(null,{'Mapped inundation':floodLayer,'Pre-event roads':roadLayer,'Settlements':settlementLayer,'Medical facilities':hospitalLayer},{collapsed:false}).addTo(map);
map.fitBounds([[__BBOX__[1],__BBOX__[0]],[__BBOX__[3],__BBOX__[2]]]);
function escapeHtml(x){return String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
</script></body></html>'''
    html_doc = (template.replace("__CARDS__", card_html)
        .replace("__ATTRIBUTION__", html.escape(attribution))
        .replace("__FLOOD__", _safe_json(flood))
        .replace("__ROADS__", _safe_json(roads))
        .replace("__HOSPITALS__", _safe_json(hospitals))
        .replace("__SETTLEMENTS__", _safe_json(settlements))
        .replace("__CENTER__", _safe_json(center))
        .replace("__BBOX__", _safe_json(bbox)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc, encoding="utf-8")
    return out_path
