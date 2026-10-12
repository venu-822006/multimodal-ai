"""
dem_flow_analysis.py  —  FloodSight AI | DEM Downstream Flood-Path Analysis
=============================================================================
Implements the Track B Bonus Requirement:
  "Downstream flood-path analysis using DEM:
   Compute elevation gradients and steepest descent flow paths from mapped
   flood inundation downriver to identify downstream communities and infrastructure
   at immediate risk."

Core Capabilities:
  1. Computes D8 steepest-descent flow direction from Copernicus WorldDEM-30.
  2. Traces a terrain-only D8 downslope screening path from the mapped inundation.
  3. Reports path length (km), elevation drop (m), and terrain gradient (m/km).
  4. Flags mapped settlements near the screening path for human review.
  5. Exports GeoJSON downstream vector path and structured risk metrics for facts.json.
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

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_bounds
from rasterio.warp import reproject, transform as transform_coords
from pyproj import Geod, Transformer
from shapely.geometry import LineString, Point, mapping, shape
from shapely.ops import transform as transform_geometry


D8_OFFSETS = {
    1:   (0, 1),    # East
    2:   (1, 1),    # South-East
    4:   (1, 0),    # South
    8:   (1, -1),   # South-West
    16:  (0, -1),   # West
    32:  (-1, -1),  # North-West
    64:  (-1, 0),   # North
    128: (-1, 1),   # North-East
}


def generate_synthetic_copernicus_dem(
    bbox: list[float], shape: tuple[int, int] = (256, 256)
) -> tuple[np.ndarray, Affine, CRS]:
    """
    Generate representative Copernicus WorldDEM-30 topography for the Trishuli /
    Bhote Koshi canyon corridor (Nepal Himalayas).

    Elevations descend from ~3,800m at high northern ridges down to ~650m in the
    southern Trishuli valley floor.
    """
    h, w = shape
    west, south, east, north = bbox
    transform = from_bounds(west, south, east, north, w, h)
    crs = CRS.from_epsg(4326)

    y_grid, x_grid = np.mgrid[0:h, 0:w]

    # Regional north-to-south mountain descent
    regional_slope = 3500.0 - (y_grid / float(h)) * 2600.0

    # V-shaped gorge canyon profile along the Trishuli River
    river_axis = (y_grid - 0.7 * x_grid - 30).astype(np.float32)
    valley_depth = np.abs(river_axis) * 45.0

    # Ridge roughness
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 15.0, (h, w)).astype(np.float32)

    dem = (regional_slope + valley_depth + noise).clip(550.0, 4800.0).astype(np.float32)
    return dem, transform, crs


def compute_d8_flow_directions(
    dem: np.ndarray,
    transform: Affine = None,
    crs: CRS = None,
    cell_size_m: float = 30.0,
) -> np.ndarray:
    """
    Compute D8 steepest-descent flow direction for each grid cell.
    Returns array of D8 direction codes (1, 2, 4, 8, 16, 32, 64, 128).
    """
    h, w = dem.shape
    flow_dir = np.zeros((h, w), dtype=np.uint8)

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            center = dem[r, c]
            if not np.isfinite(center):
                continue
            if transform is None or crs is None:
                dx_m = dy_m = cell_size_m
            elif crs.is_geographic:
                latitude = transform.f + (r + 0.5) * transform.e
                dx_m = abs(transform.a) * 111_320.0 * max(0.01, math.cos(math.radians(latitude)))
                dy_m = abs(transform.e) * 111_139.0
            else:
                dx_m = math.hypot(transform.a, transform.d)
                dy_m = math.hypot(transform.b, transform.e)
            max_drop = 0.0
            best_code = 0

            neighbors = [
                (r, c + 1, dx_m, 1),
                (r + 1, c + 1, math.hypot(dx_m, dy_m), 2),
                (r + 1, c, dy_m, 4),
                (r + 1, c - 1, math.hypot(dx_m, dy_m), 8),
                (r, c - 1, dx_m, 16),
                (r - 1, c - 1, math.hypot(dx_m, dy_m), 32),
                (r - 1, c, dy_m, 64),
                (r - 1, c + 1, math.hypot(dx_m, dy_m), 128),
            ]

            for nr, nc, dist, code in neighbors:
                if not np.isfinite(dem[nr, nc]) or dist <= 0:
                    continue
                diff = center - dem[nr, nc]
                gradient = diff / dist
                if gradient > max_drop:
                    max_drop = gradient
                    best_code = code

            flow_dir[r, c] = best_code

    return flow_dir


def trace_downstream_flood_path(
    flood_mask: np.ndarray,
    dem: np.ndarray,
    transform: Affine,
    max_steps: int = 400,
    crs: CRS = None,
) -> dict:
    """
    Trace downstream flow path starting from the lowest inundation boundary
    cells and routing downhill following the D8 steepest-descent path.
    """
    h, w = flood_mask.shape
    flood_coords = np.argwhere((flood_mask == 1) & np.isfinite(dem))

    if len(flood_coords) == 0:
        return {
            "status": "NO_FLOOD_PIXELS",
            "downstream_length_km": 0.0,
            "elevation_drop_m": 0.0,
            "flow_path_coords": [],
        }

    # Prefer the lowest valid cell on the flood boundary. This is a terrain
    # screening seed only; it does not estimate flood volume or hydraulics.
    flood_set = {tuple(rc) for rc in flood_coords.tolist()}
    boundary_coords = []
    h, w = flood_mask.shape
    for r, c in flood_coords:
        if any(
            (r + dr, c + dc) not in flood_set
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if (dr or dc) and 0 <= r + dr < h and 0 <= c + dc < w
        ):
            boundary_coords.append((r, c))
    seed_coords = np.asarray(boundary_coords or flood_coords.tolist(), dtype=np.int64)
    flood_elevations = [dem[r, c] for r, c in seed_coords]
    min_idx = int(np.argmin(flood_elevations))
    start_r, start_c = seed_coords[min_idx]

    flow_dirs = compute_d8_flow_directions(dem, transform=transform, crs=crs)

    path_cells = [(start_r, start_c)]
    visited = {(start_r, start_c)}
    curr_r, curr_c = start_r, start_c

    for _ in range(max_steps):
        code = flow_dirs[curr_r, curr_c]
        if code not in D8_OFFSETS:
            # Fallback: search lowest immediate neighbor
            best_drop = 0.0
            next_step = None
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = curr_r + dr, curr_c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited and np.isfinite(dem[nr, nc]):
                        drop = dem[curr_r, curr_c] - dem[nr, nc]
                        if drop > best_drop:
                            best_drop = drop
                            next_step = (nr, nc)
            if next_step is None:
                break
            curr_r, curr_c = next_step
        else:
            dr, dc = D8_OFFSETS[code]
            next_r, next_c = curr_r + dr, curr_c + dc
            if not (0 <= next_r < h and 0 <= next_c < w) or (next_r, next_c) in visited:
                break
            curr_r, curr_c = next_r, next_c

        visited.add((curr_r, curr_c))
        path_cells.append((curr_r, curr_c))

    # Convert raster coordinates to geographic (Lon, Lat) coordinates
    geo_coords = []
    for r, c in path_cells:
        x, y = rasterio.transform.xy(transform, r, c)
        if crs is not None and crs.to_epsg() != 4326:
            lon, lat = transform_coords(crs, "EPSG:4326", [x], [y])
            x, y = lon[0], lat[0]
        geo_coords.append([round(x, 6), round(y, 6)])

    # Compute length and elevation drop
    start_elev = float(dem[path_cells[0][0], path_cells[0][1]])
    end_elev = float(dem[path_cells[-1][0], path_cells[-1][1]])
    elev_drop = max(0.0, start_elev - end_elev)

    # Approximate length in km using geodesic line
    if len(geo_coords) >= 2:
        geod = Geod(ellps="WGS84")
        length_m = sum(
            geod.inv(x1, y1, x2, y2)[2]
            for (x1, y1), (x2, y2) in zip(geo_coords, geo_coords[1:])
        )
        length_km = round(length_m / 1000.0, 2)
    else:
        length_km = 0.0

    gradient_m_per_km = round(elev_drop / max(length_km, 0.1), 1)

    return {
        "status": "SUCCESS",
        "downstream_length_km": length_km,
        "elevation_start_m": round(start_elev, 1),
        "elevation_end_m": round(end_elev, 1),
        "elevation_drop_m": round(elev_drop, 1),
        "hydraulic_gradient_m_per_km": gradient_m_per_km,
        "terrain_gradient_m_per_km": gradient_m_per_km,
        "flow_path_steps": len(geo_coords),
        "flow_path_coords": geo_coords,
    }


def analyze_downstream_exposure(
    flow_coords: list[list[float]],
    settlements: list[dict],
    buffer_m: float = 300.0,
) -> list[dict]:
    """Flag mapped settlements near a terrain screening line for human review."""
    if len(flow_coords) < 2:
        return []

    path_line = LineString(flow_coords)
    center = path_line.centroid
    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={center.y:.8f} +lon_0={center.x:.8f} "
        "+datum=WGS84 +units=m +no_defs"
    )
    to_local = Transformer.from_crs("EPSG:4326", local_crs, always_xy=True).transform
    local_line = transform_geometry(to_local, path_line)
    screening_corridor = local_line.buffer(buffer_m)

    exposed = []
    for s in settlements:
        geom = shape(s.get("geometry", {}))
        pt = geom.centroid if geom.geom_type != "Point" else geom
        if transform_geometry(to_local, pt).intersects(screening_corridor):
            exposed.append({
                "name": s.get("properties", {}).get("name", "Downstream Community"),
                "status": "NEAR_TERRAIN_SCREENING_PATH",
                "warning": "Terrain-only screening proximity; not a validated flood-risk forecast",
            })

    return exposed


def run_dem_flow_analysis(
    flood_mask_path: Path,
    dem_path: Path = None,
    out_dir: Path = Path("outputs"),
) -> dict:
    """Execute complete DEM downstream flow-path analysis from flood mask."""
    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(flood_mask_path) as src:
        flood_mask = src.read(1)
        transform = src.transform
        crs = src.crs
        if src.crs is None:
            raise ValueError("Flood mask must have a CRS for terrain alignment.")
        dem_transform = src.transform
        dem_crs = src.crs
        dem_shape = (src.height, src.width)

    if dem_path is not None:
        if not dem_path.is_file():
            raise FileNotFoundError(f"Requested DEM file not found: {dem_path}")
        with rasterio.open(dem_path) as dem_src:
            if dem_src.crs is None:
                raise ValueError("DEM must have a CRS for terrain alignment.")
            source_dem = dem_src.read(1, masked=True).astype(np.float32).filled(np.nan)
            dem = np.full(dem_shape, np.nan, dtype=np.float32)
            reproject(
                source=source_dem,
                destination=dem,
                src_transform=dem_src.transform,
                src_crs=dem_src.crs,
                src_nodata=np.nan if dem_src.nodata is None else dem_src.nodata,
                dst_transform=dem_transform,
                dst_crs=dem_crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
    else:
        # Check standard data/terrain directory
        default_dem = Path("data/terrain/copernicus_worlddem30.tif")
        if default_dem.exists():
            with rasterio.open(default_dem) as dem_src:
                if dem_src.crs is None:
                    raise ValueError("DEM must have a CRS for terrain alignment.")
                source_dem = dem_src.read(1, masked=True).astype(np.float32).filled(np.nan)
                dem = np.full(dem_shape, np.nan, dtype=np.float32)
                reproject(
                    source=source_dem,
                    destination=dem,
                    src_transform=dem_src.transform,
                    src_crs=dem_src.crs,
                    src_nodata=np.nan if dem_src.nodata is None else dem_src.nodata,
                    dst_transform=dem_transform,
                    dst_crs=dem_crs,
                    dst_nodata=np.nan,
                    resampling=Resampling.bilinear,
                )
        else:
            print("  [DEM Note] No DEM raster provided. Skipping the optional terrain screen.")
            return {
                "status": "SKIPPED",
                "reason": "No DEM elevation raster was provided or found in the configured local data path.",
                "downstream_length_km": 0.0,
                "elevation_start_m": None,
                "elevation_end_m": None,
                "elevation_drop_m": 0.0,
            "hydraulic_gradient_m_per_km": 0.0,
            "terrain_gradient_m_per_km": 0.0,
                "flow_path_steps": 0,
                "flow_path_coords": [],
                "geojson_path": None,
                "downstream_communities_near_screening_path": [],
            }

    if not np.any((flood_mask == 1) & np.isfinite(dem)):
        return {
            "status": "NO_DEM_COVERAGE",
            "reason": "The supplied DEM does not overlap any mapped flood pixels after reprojection.",
            "downstream_length_km": 0.0,
            "elevation_start_m": None,
            "elevation_end_m": None,
            "elevation_drop_m": 0.0,
            "terrain_gradient_m_per_km": 0.0,
            "hydraulic_gradient_m_per_km": 0.0,
            "flow_path_steps": 0,
            "flow_path_coords": [],
            "geojson_path": None,
            "downstream_communities_near_screening_path": [],
        }

    flow_results = trace_downstream_flood_path(flood_mask, dem, transform, crs=crs)
    flow_results["interpretation"] = (
        "Terrain-only D8 downslope screening path from the mapped flood boundary; "
        "not a hydrodynamic forecast, travel-time estimate, or validated hazard prediction."
    )

    # Export GeoJSON flow path
    coords = flow_results.get("flow_path_coords", [])
    if len(coords) >= 2:
        geojson_feature = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coords,
                    },
                    "properties": {
                    "hazard": "Terrain-only D8 downslope screening path (not a forecast)",
                        "length_km": flow_results["downstream_length_km"],
                        "elevation_drop_m": flow_results["elevation_drop_m"],
                        "gradient_m_per_km": flow_results.get("terrain_gradient_m_per_km", flow_results.get("hydraulic_gradient_m_per_km")),
                    },
                }
            ],
        }
        geojson_path = out_dir / "downstream_flow_path.geojson"
        with open(geojson_path, "w", encoding="utf-8") as f:
            json.dump(geojson_feature, f, indent=2)
        flow_results["geojson_path"] = str(geojson_path)

    return flow_results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--flood-mask", type=Path, required=True, help="Predicted flood mask GeoTIFF")
    p.add_argument("--dem", type=Path, default=None, help="Optional georeferenced DEM GeoTIFF")
    p.add_argument("--out-dir", type=Path, default=Path("outputs"))
    p.add_argument("--out-json", type=Path, default=None)
    args = p.parse_args()

    results = run_dem_flow_analysis(args.flood_mask, args.dem, args.out_dir)

    print("\n" + "=" * 65)
    print(" D8 DEM TERRAIN-ONLY SCREENING (BONUS)")
    print("=" * 65)
    print(f"  Downstream Propagation Path: {results['downstream_length_km']} km")
    print(f"  Total Elevation Drop:        {results['elevation_drop_m']} m")
    print(f"  Terrain Gradient:            {results.get('terrain_gradient_m_per_km', results['hydraulic_gradient_m_per_km'])} m/km")
    if "geojson_path" in results:
        print(f"  GeoJSON Path:                {results['geojson_path']}")
    print("=" * 65)

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"  [OK] Saved results -> {args.out_json}\n")


if __name__ == "__main__":
    main()
