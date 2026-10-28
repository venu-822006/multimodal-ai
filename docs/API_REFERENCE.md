# FloodSight AI — AI Engine API & Integration Reference

This document provides the programmatic specification for backend and frontend engineers consuming the FloodSight AI module.

---

## 1. CLI Execution Reference

### Job Worker (prepared-scene workflow)
Runs flood inference, historical OSM exposure/connectivity analysis, and report
generation when real analysis-ready Sentinel-1 scenes are provided:
```bash
python worker.py \
    --bbox 85.1 27.8 85.5 28.1 \
    --flood-date 2024-09-28 \
    --pre-image data/scenes/pre_event_vv_vh_db.tif \
    --post-image data/scenes/post_event_vv_vh_db.tif \
    --job-id job_101 \
    --out-dir outputs/job_101
```

Both rasters must be real, preprocessed two-band VV/VH dB GeoTIFFs in EPSG:4326,
with matching grids and full AOI coverage. The operator must verify their event
dates and same-orbit relationship. `stac_search.py` discovers candidate metadata;
it does not download products or verify that the supplied files match its result.
Raw GRD preprocessing and automatic Sentinel-1/Sentinel-2 acquisition are not
implemented. Missing Sentinel files fail the job instead of substituting
simulated observations. Sentinel-2 is optional and requires real `--s2-green` and
`--s2-nir` rasters; its cloud screen is only a reflectance proxy.

The connectivity result is a scenario: any mapped road edge intersecting the
flood candidate plus a 30 m buffer is treated as impassable. `CUT_OFF` means no
mapped route remains to any mapped town or medical facility under that assumption;
it is not a confirmed road closure. The optional DEM path is terrain-only, not a
flood forecast. See [the full Track B checklist](HACKATHON_REQUIREMENTS.md).

### Standalone Inference
```bash
python predict.py \
    --scene data/processed/images/event_post.tif \
    --checkpoint checkpoints/best.pt \
    --out-dir outputs/event_run
```

---

## 2. Job State Machine & Status Codes

Jobs transition through the following states, logged in `status.json` and `job.log`:

| State | Description |
|---|---|
| `QUEUED` | Job created and registered. |
| `VALIDATING` | Coordinate validity checked (`-180 <= lon <= 180`, `-90 <= lat <= 90`, `west < east`, `south < north`). |
| `SEARCHING_DATA` | STAC Copernicus query for matching Sentinel-1 same-orbit pairs. |
| `DOWNLOADING` | Validating caller-supplied Sentinel-1 rasters (automatic download is not implemented). |
| `PREPROCESSING` | Checking that rasters are analysis-ready, aligned VV/VH dB inputs (raw preprocessing is not implemented). |
| `FLOOD_MAPPING` | Dual-pol U-Net inference + anti-hallucination spatial filtering. |
| `DAMAGE_ANALYSIS` | OSM building, road, and bridge exposure overlay. |
| `CONNECTIVITY_ANALYSIS` | NetworkX road graph 3-state cut-off routing. |
| `GENERATING_REPORT` | One-page situation report and interactive map dashboard generation. |
| `COMPLETED` | Output assets ready (`situation_report.html`, `dashboard.html`, GeoTIFFs, and `facts.json`). |
| `FAILED` | An unrecoverable exception occurred (diagnostic message in `error_message`). |

---

## 3. `facts.json` Schema Specification
All frontend components, situation reports, and summary statistics read strictly from `facts.json`. The values below are illustrative schema examples, not measured results. The worker currently supplies no DEM slope mask, so its slope-filter count should be zero unless this capability is wired in for that run:

```json
{
  "flood_area_km2": 45.67,
  "flood_pixel_count": 456700,
  "total_valid_pixels": 1200000,
  "flood_fraction": 0.3805,
  "bbox": [85.1, 27.8, 85.5, 28.1],
  "crs": "EPSG:4326",
  "pixel_resolution_approx_m": 10.0,
  "checkpoint": "best.pt",
  "val_flood_iou": 0.3894,
  "anti_hallucination_metrics": {
    "raw_candidates": 996,
    "slope_hallucinations_filtered": 210,
    "temporal_false_positives_filtered": 180,
    "speckle_noise_pixels_filtered": 154,
    "retained_flood_candidate_pixels": 452
  },
  "buildings_affected": 142,
  "road_length_affected_km": 18.4,
  "bridges_affected": 3,
  "settlement_counts": {
    "CONNECTED": 12,
    "CUT_OFF": 4,
    "UNKNOWN": 1
  },
  "settlements": [
    {
      "settlement": "Khopasi Village",
      "state": "CUT_OFF",
      "target": "Dhulikhel Hospital",
      "pre_distance_km": 14.2
    }
  ],
  "wording_standard": "Reported counts represent structures/roads affected or potentially affected by mapped inundation."
}
```

Building and bridge counts are exposure counts based on the selected OSM
snapshot; the worker does not calculate calibrated confidence scores for them.

---

## 4. Output Artifacts

* `*_flood_predicted.tif`: Georeferenced uint8 raster (`0` = background, `1` = flood, `255` = nodata).
* `*_multiclass_mask.tif`: Georeferenced uint8 raster (`0` = bg, `1` = core flood, `2` = possible debris).
* `facts.json`: Complete computed disaster metrics.
* `situation_report.html`: Printable 1-page executive situation report.
* `dashboard.html`: Interactive Leaflet map with flood extent, pre-event roads, medical facilities, and settlement access markers (requires network for the basemap).
