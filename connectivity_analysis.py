"""
connectivity_analysis.py  —  FloodSight AI | Road Graph & Cut-off Analysis
===========================================================================
Implements Step 8 & Gap 3: Road Graph + Cut-off Analysis according to the
FloodSight architecture rules:

  Rule & States:
    1. Build pre-event road graph via NetworkX from OSM road segments.
    2. For each settlement (village/town), find road route to nearest hospital
       (or nearest town if no hospital exists).
    3. Remove roads and bridges that intersect the flood mask (with a buffer).
    4. Re-test the route:
       - CONNECTED: A route still exists after removing flooded roads.
       - CUT_OFF:   No route remains under the explicit road-disruption scenario.
       - UNKNOWN:   No baseline route or usable road/destination was mapped.

  Output:
    Updates facts.json with:
    {
      "settlement_counts": {
        "CONNECTED": int,
        "CUT_OFF": int,
        "UNKNOWN": int
      },
      "settlements": [
        {"name": str, "state": "CONNECTED"|"CUT_OFF"|"UNKNOWN", "target": str, "pre_distance_km": float}
      ]
    }

Usage:
    python connectivity_analysis.py \
        --flood-mask   outputs/event_001_flood_predicted.tif \
        --roads        data/osm/roads.geojson \
        --settlements  data/osm/settlements.geojson \
        --hospitals    data/osm/hospitals.geojson \
        --facts-json   outputs/event_001_facts.json
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
    """Buffer an EPSG:4326 flood candidate by a ground distance, not degrees."""
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
    Run 3-state connectivity classification:
      - CONNECTED
      - CUT_OFF (modelled potential loss, not confirmed closure)
      - UNKNOWN
    """
    counts = {"CONNECTED": 0, "CUT_OFF": 0, "UNKNOWN": 0}
    detailed = []

    # This is a scenario assumption, not evidence that a road is physically
    # impassable: any edge intersecting this buffered candidate is excluded.
    post_graph = road_graph.copy()
    if flood_polygon is not None and not flood_polygon.is_empty:
        flood_buffered = _buffer_flood_in_metres(flood_polygon, disruption_buffer_m)
        edges_to_remove = []
        for u, v, data in post_graph.edges(data=True):
            geom = data.get("geom")
            if geom and geom.intersects(flood_buffered):
                edges_to_remove.append((u, v))
        post_graph.remove_edges_from(edges_to_remove)

    # Medical facilities and mapped towns are possible destinations.
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

        # If village is isolated or road unmapped pre-event -> UNKNOWN
        if s_node is None or s_dist > max_settlement_snap_km or not destinations:
            counts["UNKNOWN"] += 1
            detailed.append({
                "settlement": s_name,
                "state": "UNKNOWN",
                "longitude": float(pt.x),
                "latitude": float(pt.y),
                "reason": "No usable road or mapped town/medical destination in the pre-event snapshot"
            })
            continue

        # Check pre-event route
        pre_route_exists = False
        best_target_name = "None"
        best_target_type = None
        min_pre_dist = float("inf")
        reachable_targets = []
        best_target_snap_km = None

        for t_node, t_name, t_type, t_snap_km in destinations:
            if nx.has_path(road_graph, s_node, t_node):
                pre_route_exists = True
                graph_len = nx.shortest_path_length(road_graph, s_node, t_node, weight="weight")
                path_len = graph_len + s_dist + t_snap_km
                reachable_targets.append((t_node, t_name, t_type, path_len, t_snap_km))
                if path_len < min_pre_dist:
                    min_pre_dist = path_len
                    best_target_name = t_name
                    best_target_type = t_type
                    best_target_snap_km = t_snap_km

        if not pre_route_exists:
            counts["UNKNOWN"] += 1
            detailed.append({
                "settlement": s_name,
                "state": "UNKNOWN",
                "longitude": float(pt.x),
                "latitude": float(pt.y),
                "reason": "No route existed even pre-event"
            })
            continue

        # A settlement remains connected if any previously reachable
        # destination is still reachable after applying the disruption rule.
        remaining_targets = [t for t in reachable_targets if nx.has_path(post_graph, s_node, t[0])]
        if remaining_targets:
            counts["CONNECTED"] += 1
            state = "CONNECTED"
            chosen = min(remaining_targets, key=lambda t: t[3])
            best_target_name, best_target_type, best_target_snap_km = chosen[1], chosen[2], chosen[4]
        else:
            counts["CUT_OFF"] += 1
            state = "CUT_OFF"

        detailed.append({
            "settlement": s_name,
            "state": state,
            "longitude": float(pt.x),
            "latitude": float(pt.y),
            "target": best_target_name,
            "target_type": best_target_type,
            "pre_distance_km": round(min_pre_dist, 2),
            "settlement_to_road_km": round(float(s_dist), 3),
            "target_to_road_km": round(float(best_target_snap_km), 3) if best_target_snap_km is not None else None,
            "reason": (
                "Mapped road edges crossing the flood-candidate buffer were treated as impassable; "
                "potential access loss only, closure not confirmed."
                if state == "CUT_OFF" else
                "At least one mapped route remains under the road-disruption scenario."
            )
        })

    return {
        "settlement_counts": counts,
        "settlements": detailed,
        "method": "NetworkX road graph with flood-candidate edge exclusion scenario",
        "interpretation": "CUT_OFF is a modelled potential access loss, not a confirmed road closure.",
        "disruption_assumption": (
            f"Any mapped road edge intersecting the flood candidate buffered by {disruption_buffer_m:g} m "
            "is treated as impassable; field verification is required. A remaining route to any mapped "
            "pre-event hospital/clinic or town is classified CONNECTED."
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
    print(" FloodSight AI — Step 8: Road Graph & Cut-off Analysis (Gap 3)")
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
    print(f"  Settlement status:")
    print(f"    - CONNECTED: {counts['CONNECTED']}")
    print(f"    - CUT_OFF:   {counts['CUT_OFF']}")
    print(f"    - UNKNOWN:   {counts['UNKNOWN']}")

    if args.facts_json and args.facts_json.exists():
        with open(args.facts_json, "r", encoding="utf-8") as f:
            facts = json.load(f)
        facts["settlement_counts"] = counts
        facts["settlements"] = results["settlements"]
        out_p = args.out_json if args.out_json else args.facts_json
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(facts, f, indent=2)
        print(f"  [OK] Updated facts.json with connectivity states -> {out_p}")


if __name__ == "__main__":
    main()
