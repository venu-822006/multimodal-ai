"""
stac_search.py  —  FloodSight AI | Copernicus STAC Scene Search & Pairing
=========================================================================
Implements Step 2: Find Sentinel-1/2 scenes via STAC according to Section 5:

  Core requirements:
    1. Search Sentinel-1 GRD scenes overlapping the bounding box.
    2. Same-orbit pairing: Pair pre-event and post-event scenes with:
       - SAME relative orbit number
       - SAME flight direction (ASCENDING or DESCENDING)
       A different track cannot be compared pixel by pixel.
    3. Score pairs based on:
       - Orbit match: relative orbit exact match (mandatory, score 1.0)
       - Date gap: temporal proximity to flood date (smaller gap = higher score)
    4. Document pairing score and metadata.
    5. Fail gracefully with a clear descriptive message if no valid pair exists.
    6. Sentinel-2 search: query cloud-free post-event optical scenes as supporting evidence.

Usage:
    python stac_search.py \
        --bbox 85.1 27.8 85.5 28.1 \
        --flood-date 2024-09-28 \
        --out-json outputs/scene_pairs.json
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Copernicus Data Space STAC Endpoint
STAC_API_URL = "https://stac.dataspace.copernicus.eu/v1/"


def score_scene_pair(pre_scene: dict, post_scene: dict, flood_date: datetime) -> float:
    """
    Score a pair of Sentinel-1 scenes based on orbital geometry and temporal spacing.

    Score = Orbit_Match_Score (0 or 1) * Date_Proximity_Score (0 to 1)
    """
    # 1. Exact Orbit & Direction match check
    pre_orbit = pre_scene.get("relative_orbit")
    post_orbit = post_scene.get("relative_orbit")
    pre_dir = pre_scene.get("direction")
    post_dir = post_scene.get("direction")

    if not pre_orbit or not post_orbit or not pre_dir or not post_dir:
        return 0.0
    if pre_orbit != post_orbit or str(pre_dir).lower() != str(post_dir).lower():
        return 0.0  # Invalid pair: cannot compare different orbits pixel-by-pixel

    # 2. Date gap penalty
    pre_dt = datetime.fromisoformat(pre_scene["datetime"].replace("Z", "+00:00"))
    post_dt = datetime.fromisoformat(post_scene["datetime"].replace("Z", "+00:00"))

    # Ensure pre is before flood and post is after flood
    target_dt = flood_date.astimezone(pre_dt.tzinfo) if pre_dt.tzinfo else flood_date
    if pre_dt > target_dt or post_dt < target_dt:
        return 0.0

    gap_days = abs((post_dt - pre_dt).total_seconds()) / 86400.0

    # 12-day repeat cycle is ideal (score 1.0); degrade gracefully up to 36 days
    if gap_days <= 12.0:
        temporal_score = 1.0
    else:
        temporal_score = max(0.2, 1.0 - ((gap_days - 12.0) / 48.0))

    return round(float(temporal_score), 3)


def search_sentinel_scenes(
    bbox: list[float],
    flood_date_str: str,
    days_window: int = 21,
    client_mock_offline: bool = False,
) -> dict:
    """
    Query STAC catalog for Sentinel-1 scenes and pair them with exact orbit match.
    """
    flood_date = datetime.strptime(flood_date_str, "%Y-%m-%d")
    start_date = flood_date - timedelta(days=days_window)
    end_date = flood_date + timedelta(days=days_window)

    west, south, east, north = bbox
    print(f"  [STAC] Searching Copernicus catalogue for bbox: [{west}, {south}, {east}, {north}]")
    print(f"         Time window: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    scenes_found = []
    try:
        from pystac_client import Client
        client = Client.open(STAC_API_URL)
        search = client.search(
            collections=["sentinel-1-grd"],
            bbox=bbox,
            datetime=f"{start_date.isoformat()}Z/{end_date.isoformat()}Z",
            query={"sar:instrument_mode": {"eq": "IW"}},
            max_items=30
        )
        for item in search.items():
            props = item.properties
            dt_str = props.get("datetime")
            rel_orbit = props.get("sat:relative_orbit") or props.get("s1:relative_orbit_number")
            direction = props.get("sat:orbit_state") or props.get("s1:orbit_direction")

            scenes_found.append({
                "id": item.id,
                "datetime": dt_str,
                "relative_orbit": rel_orbit,
                "direction": direction,
                "bbox": item.bbox,
                "assets": {key: asset.href for key, asset in item.assets.items()}
            })
    except Exception as e:
        print(f"  [STAC] Note: Live Copernicus catalogue search returned: {e}")
        print("         STAC returned no usable metadata; no scene download will be attempted.")

    # Pair scenes
    pairs = []
    pre_scenes = [s for s in scenes_found if datetime.fromisoformat(s["datetime"].replace("Z", "")) < flood_date]
    post_scenes = [s for s in scenes_found if datetime.fromisoformat(s["datetime"].replace("Z", "")) >= flood_date]

    for pre in pre_scenes:
        for post in post_scenes:
            score = score_scene_pair(pre, post, flood_date)
            if score > 0.0:
                pairs.append({
                    "pre_scene": pre,
                    "post_scene": post,
                    "pairing_score": score,
                    "relative_orbit": pre["relative_orbit"],
                    "direction": pre["direction"],
                })

    # Sort by highest score
    pairs.sort(key=lambda p: p["pairing_score"], reverse=True)

    result = {
        "bbox": bbox,
        "flood_date": flood_date_str,
        "stac_endpoint": STAC_API_URL,
        "total_scenes_found": len(scenes_found),
        "valid_same_orbit_pairs": len(pairs),
        "best_pair": pairs[0] if pairs else None,
        "all_pairs": pairs[:5]
    }

    if not pairs:
        result["status"] = "NO_VALID_PAIR_FOUND"
        result["message"] = (
            f"No same-orbit pre- and post-flood Sentinel-1 scene pair found for "
            f"flood date {flood_date_str} within ±{days_window} days. Different orbital tracks "
            f"cannot be compared pixel-by-pixel."
        )
    else:
        result["status"] = "PAIR_FOUND"
        best = pairs[0]
        result["message"] = (
            f"Found optimal same-orbit pair (Orbit {best['relative_orbit']}, {best['direction']}) "
            f"with pairing score {best['pairing_score']:.3f}."
        )

    # Discover cloud-light Sentinel-2 scenes as optional supporting evidence.
    result["sentinel2_scenes"] = []
    try:
        s2_start = flood_date
        s2_end = flood_date + timedelta(days=14)
        s2_search = client.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{s2_start.isoformat()}Z/{s2_end.isoformat()}Z",
            query={"eo:cloud_cover": {"lt": 20}},
            max_items=20,
        )
        for item in s2_search.items():
            result["sentinel2_scenes"].append({
                "id": item.id,
                "datetime": item.properties.get("datetime"),
                "cloud_cover": item.properties.get("eo:cloud_cover"),
                "assets": {key: asset.href for key, asset in item.assets.items()},
            })
    except Exception as e:
        result["sentinel2_search_error"] = str(e)

    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    p.add_argument("--flood-date", type=str, required=True, help="Flood date YYYY-MM-DD")
    p.add_argument("--window", type=int, default=21, help="Search window in days (default: 21)")
    p.add_argument("--out-json", type=Path, default=Path("outputs/scene_pairs.json"))
    args = p.parse_args()

    results = search_sentinel_scenes(args.bbox, args.flood_date, days_window=args.window)

    print(f"\n  Status: {results['status']}")
    print(f"  {results['message']}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"  [OK] Saved results -> {args.out_json}")


if __name__ == "__main__":
    main()
