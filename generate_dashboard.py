"""Create a lightweight interactive Leaflet dashboard from a completed job."""

import html
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.warp import transform_bounds
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


def class_geojson(mask_path: Path | None, class_value: int, class_name: str) -> dict:
    """Vectorize one class from the experimental multiclass output."""
    empty = {"type": "FeatureCollection", "features": []}
    if mask_path is None or not Path(mask_path).is_file():
        return empty
    with rasterio.open(mask_path) as src:
        if src.count != 1 or src.crs is None:
            raise ValueError("Multiclass raster must be a single-band georeferenced raster.")
        band = src.read(1)
        class_mask = band == class_value
        if not class_mask.any():
            return empty
        geometries = [shape(geom) for geom, value in shapes(
            band.astype(np.uint8), mask=class_mask, transform=src.transform, connectivity=8
        ) if value == class_value]
        merged = unary_union(geometries).simplify(0.00003, preserve_topology=True)
        parts = list(merged.geoms) if merged.geom_type == "MultiPolygon" else [merged]
        return {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {"class": class_name}, "geometry": mapping(part)}
                for part in parts if not part.is_empty
            ],
        }


def export_uncertainty_overlay(uncertainty_path: Path | None, out_png: Path):
    """Write a transparent red overlay for normalized predictive entropy."""
    if uncertainty_path is None or not Path(uncertainty_path).is_file():
        return None

    with rasterio.open(uncertainty_path) as src:
        if src.count != 1 or src.crs is None:
            raise ValueError("Uncertainty raster must be a single-band georeferenced raster.")
        values = src.read(1).astype(np.float32)
        bounds = src.bounds
        if src.crs.to_epsg() != 4326:
            bounds = transform_bounds(src.crs, "EPSG:4326", *bounds, densify_pts=21)

    valid = np.isfinite(values) & (values >= 0.0)
    rgba = np.zeros((*values.shape, 4), dtype=np.uint8)
    rgba[..., 0] = 220
    rgba[..., 1] = 38
    rgba[..., 2] = 38
    rgba[..., 3] = np.where(valid, np.clip(values, 0.0, 1.0) * 190, 0).astype(np.uint8)
    max_dimension = 2000
    scale = min(1.0, max_dimension / max(values.shape))
    out_png.parent.mkdir(parents=True, exist_ok=True)

    try:
        from PIL import Image
        img = Image.fromarray(rgba, mode="RGBA")
        if scale < 1.0:
            width = max(1, int(values.shape[1] * scale))
            height = max(1, int(values.shape[0] * scale))
            img = img.resize((width, height), resample=Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR)
        img.save(out_png)
    except ImportError:
        import cv2
        if scale < 1.0:
            width = max(1, int(values.shape[1] * scale))
            height = max(1, int(values.shape[0] * scale))
            rgba = cv2.resize(rgba, (width, height), interpolation=cv2.INTER_LINEAR)
        bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
        if not cv2.imwrite(str(out_png), bgra):
            raise RuntimeError(f"Could not write uncertainty overlay: {out_png}")

    west, south, east, north = bounds
    return [[south, west], [north, east]]


def _safe_json(value) -> str:
    # Prevent OSM tag strings from closing an inline script element.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def render_dashboard(
    facts_path: Path,
    flood_mask_path: Path,
    out_path: Path,
    osm_data: dict | None = None,
    multiclass_mask_path: Path | None = None,
):
    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    flood = flood_geojson(flood_mask_path)
    debris = class_geojson(multiclass_mask_path, 2, "Possible debris / surface-change candidate (experimental)")
    roads = (osm_data or {}).get("roads", {"type": "FeatureCollection", "features": []})
    hospitals = (osm_data or {}).get("hospitals", {"type": "FeatureCollection", "features": []})
    towns = (osm_data or {}).get("towns", {"type": "FeatureCollection", "features": []})
    assets = (osm_data or {}).get("potentially_affected_assets", {})
    uncertainty_info = facts.get("uncertainty", {})
    uncertainty_path = uncertainty_info.get("raster") if isinstance(uncertainty_info, dict) else None
    uncertainty_bounds = export_uncertainty_overlay(
        Path(uncertainty_path) if uncertainty_path else None,
        out_path.parent / "uncertainty_entropy_overlay.png",
    )
    uncertainty_layer_js = (
        "L.imageOverlay('uncertainty_entropy_overlay.png', " + _safe_json(uncertainty_bounds) + ", {opacity:0.75})"
        if uncertainty_bounds else "null"
    )
    settlements = facts.get("settlements", [])
    downstream_candidates = facts.get("dem_flow_analysis", {}).get("downstream_communities_near_screening_path", [])
    bbox = facts.get("bbox", [85.1, 27.8, 85.5, 28.1])
    center = [(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2]

    dem_data = facts.get("dem_flow_analysis", {})
    flow_coords = dem_data.get("flow_path_coords", [])
    if len(flow_coords) >= 2:
        flow_path_geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": flow_coords},
                "properties": {
                    "hazard": "Terrain-only D8 downslope screening path (not a forecast)",
                    "length_km": dem_data.get("downstream_length_km", 0.0),
                    "elevation_drop_m": dem_data.get("elevation_drop_m", 0.0),
                    "gradient_m_per_km": dem_data.get("terrain_gradient_m_per_km", dem_data.get("hydraulic_gradient_m_per_km", 0.0)),
                }
            }]
        }
    else:
        flow_path_geojson = {"type": "FeatureCollection", "features": []}

    cards = {
        "Flood area": f"{facts.get('flood_area_km2', 'N/A')} km²",
        "Buildings potentially affected": facts.get("buildings_affected", "N/A"),
        "Road length intersecting flood": f"{facts.get('road_length_affected_km', 'N/A')} km",
        "Potentially cut off (model)": facts.get("settlement_counts", {}).get("CUT_OFF", "N/A"),
    }
    dem_km = dem_data.get("downstream_length_km")
    if dem_km is not None and dem_km > 0:
        cards["Terrain-only D8 screen (not forecast)"] = f"{dem_km} km ({dem_data.get('elevation_drop_m', 0)}m drop)"

    card_html = "".join(
        f'<article><span>{html.escape(str(label))}</span><strong>{html.escape(str(value))}</strong></article>'
        for label, value in cards.items()
    )
    attribution = " ".join(str(item) for item in facts.get("data_attributions", [
        "Data provenance and licences were not recorded for this analysis."
    ])) + " Educational prototype; verify every layer before operational use."
    template = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FloodSight | Event map dashboard</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
*{box-sizing:border-box}body{margin:0;background:#f3f6fb;color:#14213d;font:15px system-ui,Segoe UI,sans-serif}
header{padding:18px 24px;background:#102a43;color:white}header h1{margin:0;font-size:22px}header p{margin:4px 0 0;color:#d9e7f2}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;padding:14px 18px}.cards article{background:white;border-radius:10px;padding:12px 14px;box-shadow:0 2px 8px #102a4314}.cards span{display:block;color:#526477;font-size:12px}.cards strong{display:block;margin-top:5px;font-size:19px}
#map{height:calc(100vh - 220px);min-height:420px;margin:0 18px;border-radius:10px;box-shadow:0 3px 14px #102a4320}.note{padding:12px 22px;color:#526477;font-size:12px}
@media(max-width:700px){.cards{grid-template-columns:repeat(2,1fr)}#map{height:65vh;margin:0 10px}header{padding:16px}}
</style></head><body><header><h1>FloodSight · Event map</h1><p>Satellite flood extent, pre-event roads, and settlement access status</p></header>
<section class="cards">__CARDS__</section><div id="map" role="application" aria-label="Interactive flood extent map"></div>
<p class="note">__ATTRIBUTION__ Red tint shows normalized predictive entropy (higher means more model ambiguity); it is uncalibrated and does not measure geographic-shift uncertainty. Access-loss markers are a model scenario: road edges intersecting the mapped flood candidate plus a 30 m buffer are treated as impassable. This does not confirm road closure; field verification is required. Debris markers are experimental surface-change candidates, not a validated debris classifier. Satellite revisit timing, terrain effects, and OSM coverage apply.</p>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script><script>
const flood=__FLOOD__, debris=__DEBRIS__, roads=__ROADS__, hospitals=__HOSPITALS__, towns=__TOWNS__, settlements=__SETTLEMENTS__, downstreamCandidates=__DOWNSTREAM_CANDIDATES__, flowPath=__FLOW_PATH__, assets=__ASSETS__, uncertaintyLayer=__UNCERTAINTY__;
const map=L.map('map').setView(__CENTER__,9);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'© OpenStreetMap contributors'}).addTo(map);
const floodLayer=L.geoJSON(flood,{style:{color:'#c2410c',weight:1,fillColor:'#f97316',fillOpacity:.5},onEachFeature:(f,l)=>l.bindPopup('Mapped inundation')}).addTo(map);
const debrisLayer=L.geoJSON(debris,{style:{color:'#7c3aed',weight:1,fillColor:'#a78bfa',fillOpacity:.55},onEachFeature:(f,l)=>l.bindPopup('Possible debris / surface-change candidate (experimental)')});
const roadLayer=L.geoJSON(roads,{style:{color:'#64748b',weight:2,opacity:.8},onEachFeature:(f,l)=>{const name=f.properties&&f.properties.name;if(name)l.bindPopup('Road: '+escapeHtml(name))}});
const buildingLayer=L.geoJSON(assets.buildings||{type:'FeatureCollection',features:[]},{style:{color:'#ea580c',weight:1,fillColor:'#fb923c',fillOpacity:.55},onEachFeature:(f,l)=>l.bindPopup('Potentially affected building (mapped inundation overlap)')});
const affectedRoadLayer=L.geoJSON(assets.roads||{type:'FeatureCollection',features:[]},{style:{color:'#dc2626',weight:4},onEachFeature:(f,l)=>l.bindPopup('Road segment intersects mapped inundation; closure unconfirmed')});
const bridgeLayer=L.geoJSON(assets.bridges||{type:'FeatureCollection',features:[]},{style:{color:'#7f1d1d',weight:4},onEachFeature:(f,l)=>l.bindPopup('Bridge within 20 m of mapped inundation; damage unconfirmed')});
const hospitalLayer=L.geoJSON(hospitals,{pointToLayer:(f,latlng)=>L.circleMarker(latlng,{radius:6,color:'#2563eb',fillOpacity:.9}),onEachFeature:(f,l)=>l.bindPopup('Medical facility: '+escapeHtml((f.properties&&f.properties.name)||'Unnamed'))});
const townLayer=L.geoJSON(towns,{pointToLayer:(f,latlng)=>L.circleMarker(latlng,{radius:5,color:'#0f766e',fillOpacity:.9}),onEachFeature:(f,l)=>l.bindPopup('Town: '+escapeHtml((f.properties&&f.properties.name)||'Unnamed'))});
const downstreamLayer=L.layerGroup(downstreamCandidates.filter(s=>Number.isFinite(s.longitude)&&Number.isFinite(s.latitude)).map(s=>L.circleMarker([s.latitude,s.longitude],{radius:7,color:'#7c3aed',fillColor:'#c4b5fd',fillOpacity:.9}).bindPopup('<b>'+escapeHtml(s.name||'Community')+'</b><br>Near terrain-only screening path; not a flood-risk forecast<br>Distance to path: '+escapeHtml(s.distance_to_screening_path_m)+' m')));
const colors={CUT_OFF:'#dc2626',CONNECTED:'#16a34a',UNKNOWN:'#64748b'};
const settlementLayer=L.layerGroup(settlements.filter(s=>Number.isFinite(s.longitude)&&Number.isFinite(s.latitude)).map(s=>{const status=s.state==='CUT_OFF'?'POTENTIALLY CUT OFF (MODEL)':(s.state||'UNKNOWN');return L.circleMarker([s.latitude,s.longitude],{radius:6,color:colors[s.state]||colors.UNKNOWN,fillOpacity:.9}).bindPopup('<b>'+escapeHtml(s.settlement||'Unnamed settlement')+'</b><br>Modelled status: '+escapeHtml(status)+(s.target?'<br>Target: '+escapeHtml(s.target):'')+(s.reason?'<br>'+escapeHtml(s.reason):''))}));
const flowPathLayer=L.geoJSON(flowPath,{style:{color:'#0284c7',weight:4,dashArray:'6, 6'},onEachFeature:(f,l)=>l.bindPopup('<b>Terrain-only D8 downslope screening path (not a flood forecast)</b><br>Length: '+(f.properties?.length_km||0)+' km<br>Elevation drop: '+(f.properties?.elevation_drop_m||0)+' m<br>Terrain gradient: '+(f.properties?.gradient_m_per_km||0)+' m/km')});
if(flowPath.features&&flowPath.features.length>0){flowPathLayer.addTo(map);}
const overlays={'Mapped inundation':floodLayer,'Possible debris / surface change (experimental)':debrisLayer,'Pre-event roads':roadLayer,'Potentially affected buildings':buildingLayer,'Roads intersecting inundation':affectedRoadLayer,'Bridges near inundation':bridgeLayer,'Settlements':settlementLayer,'Towns':townLayer,'Medical facilities':hospitalLayer,'Near terrain screen (review)':downstreamLayer,'Terrain-only path (not forecast)':flowPathLayer};
if(uncertaintyLayer){overlays['Predictive entropy (uncalibrated)']=uncertaintyLayer;}
L.control.layers(null,overlays,{collapsed:false}).addTo(map);
map.fitBounds([[__BBOX__[1],__BBOX__[0]],[__BBOX__[3],__BBOX__[2]]]);
function escapeHtml(x){return String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
</script></body></html>'''
    html_doc = (template.replace("__CARDS__", card_html)
        .replace("__ATTRIBUTION__", html.escape(attribution))
        .replace("__FLOOD__", _safe_json(flood))
        .replace("__DEBRIS__", _safe_json(debris))
        .replace("__ROADS__", _safe_json(roads))
        .replace("__HOSPITALS__", _safe_json(hospitals))
        .replace("__TOWNS__", _safe_json(towns))
        .replace("__DOWNSTREAM_CANDIDATES__", _safe_json(downstream_candidates))
        .replace("__SETTLEMENTS__", _safe_json(settlements))
        .replace("__FLOW_PATH__", _safe_json(flow_path_geojson))
        .replace("__ASSETS__", _safe_json(assets))
        .replace("__UNCERTAINTY__", uncertainty_layer_js)
        .replace("__CENTER__", _safe_json(center))
        .replace("__BBOX__", _safe_json(bbox)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc, encoding="utf-8")
    return out_path
