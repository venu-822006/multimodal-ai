"""
connectivity_analysis.py  -  FloodSight AI | Road Graph & Risk-Stratified Cut-off Analysis
========================================================================================
Implements Step 8 & Gap 3: Road Graph + Risk-Stratified Accessibility Analysis:
  1. Builds pre-event road network graph from OSM road geometries.
  2. For each settlement (village/town), calculates shortest road route to nearest hospital
     or administrative town center.
  3. Applies flood disruption model: removes road segments intersecting the flood candidate
     (buffered by disruption_buffer_m).
  4. Evaluates accessibility and risk stratification:
       - CONNECTED: Direct route remains unimpaired.
       - FUNCTIONAL_DETOUR: Primary route is severed, but alternative detour path exists.
         Calculates additional detour distance (km) and penalty percentage.
       - CRITICAL_ISOLATED (CUT_OFF): Zero alternative routes remain to any mapped facility.
         Flagged as Priority 1 for emergency evacuation / relief airlift.
       - UNKNOWN: Settlement was isolated or unmapped even pre-event.
"""

import argparse
import json
import math
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import networkx as nx
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import LineString, Point, shape
from shapely.ops import transform as transform_geometry, unary_union
from pyproj import CRS, Transformer


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Calculate distance in kilometres between two coordinates."""
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2)**2
    return 2 * r * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def build_road_graph(roads_geojson: dict) -> tuple[nx.Graph, list]:
    """
    Build a NetworkX undirected graph from OSM road line strings.
    Nodes are coordinate tuples (round to 6 decimal places).
    Edges carry distance weight in km.
    """
    g = nx.Graph()
    road_geoms = []

    if not roads_geojson or "features" not in roads_geojson:
        return g, road_geoms

    for feat in roads_geojson["features"]:
        geom = shape(feat["geometry"])
        if geom.geom_type == "LineString":
            lines = [geom]
        elif geom.geom_type == "MultiLineString":
            lines = list(geom.geoms)
        else:
            continue
        for line in lines:
            coords = list(line.coords)
            for i in range(len(coords) - 1):
                p1 = (round(coords[i][0], 6), round(coords[i][1], 6))
                p2 = (round(coords[i + 1][0], 6), round(coords[i + 1][1], 6))
                if p1 == p2:
                    continue
                segment = LineString([p1, p2])
                road_geoms.append(segment)
                dist = haversine_km(p1[0], p1[1], p2[0], p2[1])
                g.add_edge(p1, p2, weight=dist, geom=segment)

    return g, road_geoms


def find_nearest_node(graph: nx.Graph, point: Point) -> tuple[tuple, float]:
    """Find the closest existing graph node to a given settlement or hospital."""
    if len(graph.nodes) == 0:
        return None, float("inf")
    best_node = None
    min_dist = float("inf")
    px, py = point.x, point.y
    for node in graph.nodes:
        d = haversine_km(px, py, node[0], node[1])
        if d < min_dist:
            min_dist = d
            best_node = node
    return best_node, min_dist


def _buffer_flood_in_metres(flood_polygon, buffer_m: float):
    """Buffer an EPSG:4326 flood candidate by a ground distance in metres."""
    if flood_polygon is None or flood_polygon.is_empty or buffer_m <= 0:
        return flood_polygon
    center = flood_polygon.centroid
    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={center.y:.8f} +lon_0={center.x:.8f} "
        "+datum=WGS84 +units=m +no_defs"
    )
    to_local = Transformer.from_crs("EPSG:4326", local_crs, always_xy=True).transform
    to_wgs84 = Transformer.from_crs(local_crs, "EPSG:4326", always_xy=True).transform
    return transform_geometry(to_wgs84, transform_geometry(to_local, flood_polygon).buffer(buffer_m))


def analyze_connectivity(
    flood_polygon,
    road_graph: nx.Graph,
    settlements: list[dict],
    hospitals: list[dict],
    towns: list[dict] = None,
    disruption_buffer_m: float = 30.0,
    max_settlement_snap_km: float = 2.0,
    max_destination_snap_km: float = 2.0,
) -> dict:
    """
    Run risk-stratified connectivity and detour analysis:
      - CONNECTED (direct unimpaired access)
      - FUNCTIONAL_DETOUR (route available via alternative path)
      - CRITICAL_ISOLATED / CUT_OFF (zero remaining routes)
      - UNKNOWN (unmapped / disconnected baseline)
    """
    counts = {
        "CONNECTED": 0,
        "FUNCTIONAL_DETOUR": 0,
        "CRITICAL_ISOLATED": 0,
        "CUT_OFF": 0,
        "UNKNOWN": 0,
    }
    detailed = []
    detour_additions = []

    # Scenario assumption: road edges intersecting the buffered candidate are excluded.
    post_graph = road_graph.copy()
    if flood_polygon is not None and not flood_polygon.is_empty:
        flood_buffered = _buffer_flood_in_metres(flood_polygon, disruption_buffer_m)
        edges_to_remove = []
        for u, v, data in post_graph.edges(data=True):
            geom = data.get("geom")
            if geom and geom.intersects(flood_buffered):
                edges_to_remove.append((u, v))
        post_graph.remove_edges_from(edges_to_remove)

    # Medical facilities and mapped towns as possible destinations.
    target_nodes = []
    for h in hospitals:
        geom = shape(h["geometry"])
        pt = geom.centroid if geom.geom_type != "Point" else geom
        node, dist = find_nearest_node(road_graph, pt)
        if node and dist <= max_destination_snap_km:
            name = h.get("properties", {}).get("name") or "Medical facility"
            target_nodes.append((node, name, "medical_facility", dist))
    for town in towns or []:
        geom = shape(town["geometry"])
        pt = geom.centroid if geom.geom_type != "Point" else geom
        node, dist = find_nearest_node(road_graph, pt)
        if node and dist <= max_destination_snap_km:
            props = town.get("properties", {})
            name = props.get("name") or props.get("place") or "Town"
            target_nodes.append((node, name, "town", dist))

    for s in settlements:
        s_props = s.get("properties", {})
        s_name = s_props.get("name", "Unnamed Village")
        geom = shape(s["geometry"])
        pt = geom.centroid if geom.geom_type != "Point" else geom

        s_node, s_dist = find_nearest_node(road_graph, pt)
        destinations = [
            t for t in target_nodes
            if not (t[2] == "town" and str(t[1]).strip().casefold() == str(s_name).strip().casefold())
        ]

        if s_node is None or s_dist > max_settlement_snap_km or not destinations:
            counts["UNKNOWN"] += 1
            detailed.append({
                "settlement": s_name,
                "state": "UNKNOWN",
                "substate": "UNMAPPED_BASELINE",
                "triage_priority": "UNASSESSED",
                "longitude": float(pt.x),
                "latitude": float(pt.y),
                "reason": "No usable road or mapped town/medical destination in the pre-event snapshot"
            })
            continue

        # Pre-event route
        pre_route_exists = False
        reachable_targets = []

        for t_node, t_name, t_type, t_snap_km in destinations:
            if nx.has_path(road_graph, s_node, t_node):
                pre_route_exists = True
                graph_len = nx.shortest_path_length(road_graph, s_node, t_node, weight="weight")
                path_len = graph_len + s_dist + t_snap_km
                reachable_targets.append((t_node, t_name, t_type, path_len, t_snap_km))

        if not pre_route_exists:
            counts["UNKNOWN"] += 1
            detailed.append({
                "settlement": s_name,
                "state": "UNKNOWN",
                "substate": "UNREACHABLE_PRE_EVENT",
                "triage_priority": "UNASSESSED",
                "longitude": float(pt.x),
                "latitude": float(pt.y),
                "reason": "No route existed even pre-event"
            })
            continue

        best_pre_target = min(reachable_targets, key=lambda t: t[3])
        min_pre_dist = best_pre_target[3]

        # Post-event routing
        remaining_targets = []
        for t_node, t_name, t_type, orig_len, t_snap_km in reachable_targets:
            if nx.has_path(post_graph, s_node, t_node):
                post_graph_len = nx.shortest_path_length(post_graph, s_node, t_node, weight="weight")
                post_path_len = post_graph_len + s_dist + t_snap_km
                remaining_targets.append((t_node, t_name, t_type, post_path_len, t_snap_km, orig_len))

        if remaining_targets:
            chosen = min(remaining_targets, key=lambda t: t[3])
            best_target_name, best_target_type = chosen[1], chosen[2]
            best_target_snap_km = chosen[4]
            post_distance_km = round(chosen[3], 2)
            orig_distance_km = round(chosen[5], 2)
            detour_additional_km = max(0.0, round(post_distance_km - orig_distance_km, 2))
            detour_penalty_pct = round(100.0 * detour_additional_km / max(orig_distance_km, 0.001), 1)

            if detour_additional_km >= 0.5:
                state = "FUNCTIONAL_DETOUR"
                counts["FUNCTIONAL_DETOUR"] += 1
                counts["CONNECTED"] += 1
                triage_priority = "PRIORITY_2_DETOUR_ALERT"
                detour_additions.append(detour_additional_km)
                reason = (
                    f"Primary route disrupted by inundation barrier; detour path available "
                    f"(+{detour_additional_km} km / +{detour_penalty_pct}% distance penalty)."
                )
            else:
                state = "CONNECTED"
                counts["CONNECTED"] += 1
                triage_priority = "PRIORITY_3_NORMAL"
                reason = "Direct mapped route remains unimpaired under the road-disruption scenario."
        else:
            state = "CUT_OFF"
            counts["CUT_OFF"] += 1
            counts["CRITICAL_ISOLATED"] += 1
            best_target_name = best_pre_target[1]
            best_target_type = best_pre_target[2]
            best_target_snap_km = best_pre_target[4]
            post_distance_km = None
            detour_additional_km = None
            detour_penalty_pct = None
            triage_priority = "PRIORITY_1_CRITICAL_ISOLATION"
            reason = (
                "All mapped road routes to emergency medical facilities and town centers are blocked "
                "by modeled flood inundation. Immediate relief/boat/airlift required."
            )

        detailed.append({
            "settlement": s_name,
            "state": state,
            "substate": "CRITICAL_ISOLATED" if state == "CUT_OFF" else state,
            "triage_priority": triage_priority,
            "longitude": float(pt.x),
            "latitude": float(pt.y),
            "target": best_target_name,
            "target_type": best_target_type,
            "pre_distance_km": round(min_pre_dist, 2),
            "post_distance_km": post_distance_km,
            "detour_additional_km": detour_additional_km,
            "detour_penalty_pct": detour_penalty_pct,
            "settlement_to_road_km": round(float(s_dist), 3),
            "target_to_road_km": round(float(best_target_snap_km), 3) if best_target_snap_km is not None else None,
            "reason": reason,
        })

    detour_metrics = {
        "settlements_requiring_detour": len(detour_additions),
        "max_detour_km": round(float(max(detour_additions)), 2) if detour_additions else 0.0,
        "mean_detour_km": round(float(np.mean(detour_additions)), 2) if detour_additions else 0.0,
        "critical_isolation_count": counts["CRITICAL_ISOLATED"],
    }

    return {
        "settlement_counts": counts,
        "detour_metrics": detour_metrics,
        "settlements": detailed,
        "method": "NetworkX road graph with flood-candidate edge exclusion & detour penalty quantification",
        "interpretation": "CUT_OFF / CRITICAL_ISOLATED denotes simulated loss of road route to any mapped destination under assumed impassability.",
        "disclaimer": (
            "Modeled routing accessibility under the assumption that all flooded road edges are impassable. "
            "This constitutes an emergency response planning scenario and does not verify physical structural bridge "
            "destruction without high-resolution field or aerial validation."
        ),
        "disruption_assumption": (
            f"Any mapped road edge intersecting the flood candidate buffered by {disruption_buffer_m:g} m "
            "is treated as impassable. Routes requiring detours >= 0.5 km are classified FUNCTIONAL_DETOUR."
        ),
        "disruption_buffer_m": float(disruption_buffer_m),
        "max_settlement_snap_km": float(max_settlement_snap_km),
        "max_destination_snap_km": float(max_destination_snap_km),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--flood-mask",  type=Path, required=True, help="Flood mask GeoTIFF")
    p.add_argument("--roads",       type=Path, default=None,  help="OSM roads GeoJSON")
    p.add_argument("--settlements", type=Path, default=None,  help="OSM settlements GeoJSON")
    p.add_argument("--hospitals",   type=Path, default=None,  help="OSM hospitals GeoJSON")
    p.add_argument("--towns",       type=Path, default=None,  help="OSM towns/cities GeoJSON")
    p.add_argument("--facts-json",  type=Path, default=None,  help="Existing facts.json to update")
    p.add_argument("--out-json",    type=Path, default=None,  help="Output updated facts.json")
    args = p.parse_args()

    print("=" * 65)
    print(" FloodSight AI - Step 8: Road Graph & Risk-Stratified Cut-off Analysis")
    print("=" * 65)

    with rasterio.open(args.flood_mask) as src:
        mask = src.read(1)
        transform = src.transform

    geoms = [shape(geom) for geom, val in shapes((mask == 1).astype(np.uint8), mask=(mask == 1), transform=transform) if val == 1]
    flood_union = unary_union(geoms) if geoms else None

    r_json = s_json = h_json = t_json = {"features": []}
    if args.roads and args.roads.exists():
        with open(args.roads, "r", encoding="utf-8") as f:
            r_json = json.load(f)
    if args.settlements and args.settlements.exists():
        with open(args.settlements, "r", encoding="utf-8") as f:
            s_json = json.load(f)
    if args.hospitals and args.hospitals.exists():
        with open(args.hospitals, "r", encoding="utf-8") as f:
            h_json = json.load(f)
    if args.towns and args.towns.exists():
        with open(args.towns, "r", encoding="utf-8") as f:
            t_json = json.load(f)

    graph, _ = build_road_graph(r_json)
    print(f"  Road graph constructed: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    results = analyze_connectivity(
        flood_union, graph, s_json.get("features", []), h_json.get("features", []),
        towns=t_json.get("features", []),
    )
    counts = results["settlement_counts"]
    detours = results["detour_metrics"]
    print("  Settlement status:")
    print(f"    - CONNECTED:         {counts['CONNECTED']} (Unimpaired: {counts['CONNECTED'] - counts['FUNCTIONAL_DETOUR']})")
    print(f"    - FUNCTIONAL_DETOUR: {counts['FUNCTIONAL_DETOUR']} (Max detour: {detours['max_detour_km']} km)")
    print(f"    - CRITICAL_ISOLATED: {counts['CRITICAL_ISOLATED']}")
    print(f"    - UNKNOWN:           {counts['UNKNOWN']}")

    if args.facts_json and args.facts_json.exists():
        with open(args.facts_json, "r", encoding="utf-8") as f:
            facts = json.load(f)
        facts["settlement_counts"] = counts
        facts["detour_metrics"] = detours
        facts["settlements"] = results["settlements"]
        facts["connectivity_disclaimer"] = results["disclaimer"]
        out_p = args.out_json if args.out_json else args.facts_json
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(facts, f, indent=2)
        print(f"  [OK] Updated facts.json with risk-stratified connectivity states -> {out_p}")


if __name__ == "__main__":
    main()
