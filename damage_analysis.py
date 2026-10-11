"""
damage_analysis.py  —  FloodSight AI | OSM x Flood Damage Assessment
====================================================================
Implements Step 7: Damage Analysis (OSM x Flood) according to the
FloodSight problem statement and wording rules:

  Wording rule:
    Use "affected / potentially affected by mapped inundation".
    Do NOT say "destroyed".

Calculations:
  - Building footprints intersected with flood polygon: count & area.
  - Road network intersected with flood mask: total length in km.
  - Bridges near or within flooded areas: count.
  - Confidence scoring per asset class based on flood mask probability.

Updates facts.json with:
  {
    "buildings_affected": int,
    "buildings_affected_confidence": float,
    "road_length_affected_km": float,
    "bridges_affected": int,
    "bridges_affected_confidence": float
  }

Usage:
    python damage_analysis.py \
        --flood-mask outputs/event_001_flood_predicted.tif \
        --osm-dir    data/osm/ \
        --facts-json outputs/event_001_facts.json \
        --out-json   outputs/event_001_facts_updated.json
"""

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.features import shapes
from shapely.geometry import box, shape
from shapely.ops import unary_union


def vectorize_flood_mask(mask_path: Path) -> tuple[object, dict]:
    """
    Vectorize binary flood pixels (value == 1) into Shapely multipolygon.
    """
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        transform = src.transform
        crs = src.crs

    flood_pixels = (mask == 1).astype(np.uint8)
    geoms = []
    for geom, val in shapes(flood_pixels, mask=(flood_pixels == 1), transform=transform):
        if val == 1:
            geoms.append(shape(geom))

    if geoms:
        flood_union = unary_union(geoms)
    else:
        flood_union = None

    meta = {
        "crs": str(crs),
        "flood_pixel_count": int((mask == 1).sum()),
        "valid_pixel_count": int((mask != 255).sum())
    }
    return flood_union, meta


def analyze_damage(
    flood_polygon,
    buildings_geojson: dict = None,
    roads_geojson: dict = None,
    bridges_geojson: dict = None,
    flood_confidence: float | None = None,
) -> dict:
    """
    Intersect OSM infrastructure features with flood polygon.
    """
    affected_buildings = 0
    affected_road_km = 0.0
    affected_bridges = 0

    if flood_polygon is None or flood_polygon.is_empty:
        return {
            "buildings_affected": 0,
            "buildings_affected_confidence": None,
            "road_length_affected_km": 0.0,
            "bridges_affected": 0,
            "bridges_affected_confidence": None,
            "damage_status": "No mapped inundation detected in region"
        }

    # 1. Buildings intersection
    if buildings_geojson and "features" in buildings_geojson:
        for feat in buildings_geojson["features"]:
            geom = shape(feat["geometry"])
            if geom.intersects(flood_polygon):
                affected_buildings += 1

    # 2. Roads intersection (length in km)
    if roads_geojson and "features" in roads_geojson:
        geod = CRS.from_epsg(4326).get_geod()

        def geodesic_length_km(geometry) -> float:
            if geometry.geom_type == "LineString":
                coords = list(geometry.coords)
                return sum(
                    geod.inv(x1, y1, x2, y2)[2]
                    for (x1, y1), (x2, y2) in zip(coords, coords[1:])
                ) / 1000.0
            if geometry.geom_type == "MultiLineString":
                return sum(geodesic_length_km(line) for line in geometry.geoms)
            return 0.0

        for feat in roads_geojson["features"]:
            geom = shape(feat["geometry"])
            if geom.intersects(flood_polygon):
                intersection_geom = geom.intersection(flood_polygon)
                affected_road_km += geodesic_length_km(intersection_geom)

    # 3. Bridges intersection
    if bridges_geojson and "features" in bridges_geojson:
        # Buffer flood mask slightly (e.g. 20m ~ 0.0002 deg) for bridges near water
        flood_buffered = flood_polygon.buffer(0.0002)
        for feat in bridges_geojson["features"]:
            geom = shape(feat["geometry"])
            if geom.intersects(flood_buffered):
                affected_bridges += 1

    return {
        "buildings_affected": int(affected_buildings),
        "buildings_affected_confidence": None,
        "road_length_affected_km": round(float(affected_road_km), 2),
        "bridges_affected": int(affected_bridges),
        "bridges_affected_confidence": None,
        "wording_standard": "Reported counts represent structures/roads affected or potentially affected by mapped inundation."
    }


def update_facts_with_damage(
    facts_json_path: Path,
    damage_results: dict,
    out_json_path: Path = None,
) -> dict:
    facts = {}
    if facts_json_path and facts_json_path.exists():
        with open(facts_json_path, "r", encoding="utf-8") as f:
            facts = json.load(f)

    # Merge damage fields
    facts.update(damage_results)

    out_path = out_json_path if out_json_path else facts_json_path
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(facts, f, indent=2)
        print(f"  [OK] Updated facts -> {out_path}")

    return facts


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--flood-mask",  type=Path, required=True, help="Binary flood mask GeoTIFF")
    p.add_argument("--buildings",   type=Path, default=None,  help="Optional OSM buildings GeoJSON")
    p.add_argument("--roads",       type=Path, default=None,  help="Optional OSM roads GeoJSON")
    p.add_argument("--bridges",     type=Path, default=None,  help="Optional OSM bridges GeoJSON")
    p.add_argument("--facts-json",  type=Path, default=None,  help="Existing facts.json to update")
    p.add_argument("--out-json",    type=Path, default=None,  help="Output updated facts.json")
    args = p.parse_args()

    print("=" * 60)
    print(" FloodSight AI — Step 7: Damage Analysis (OSM x Flood)")
    print("=" * 60)

    flood_poly, meta = vectorize_flood_mask(args.flood_mask)
    print(f"  Flood pixels: {meta['flood_pixel_count']:,} / {meta['valid_pixel_count']:,}")

    b_json = r_json = br_json = None
    if args.buildings and args.buildings.exists():
        with open(args.buildings, "r", encoding="utf-8") as f:
            b_json = json.load(f)
    if args.roads and args.roads.exists():
        with open(args.roads, "r", encoding="utf-8") as f:
            r_json = json.load(f)
    if args.bridges and args.bridges.exists():
        with open(args.bridges, "r", encoding="utf-8") as f:
            br_json = json.load(f)

    damage = analyze_damage(flood_poly, b_json, r_json, br_json)
    print(f"  Buildings affected:       {damage['buildings_affected']} (conf: {damage['buildings_affected_confidence']})")
    print(f"  Road length affected:     {damage['road_length_affected_km']:.2f} km")
    print(f"  Bridges affected:         {damage['bridges_affected']} (conf: {damage['bridges_affected_confidence']})")

    if args.facts_json:
        update_facts_with_damage(args.facts_json, damage, args.out_json)


if __name__ == "__main__":
    main()
