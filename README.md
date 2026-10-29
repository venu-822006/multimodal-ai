# FloodSight AI

FloodSight AI is a research prototype for mapping flood candidates from Sentinel-1 imagery and summarizing possible impacts to buildings, roads, and access routes. It combines a compact U-Net segmentation model with geospatial analysis and produces maps and a human-review situation report.

> **Research use only.** The held-out flood results are weak. Outputs are not suitable on their own for evacuation, dispatch, road-closure, or other operational decisions.

## What it does

- Predicts a binary flood/background mask from Sentinel-1 VV/VH backscatter in decibels.
- Writes a normalized predictive-entropy raster as an uncertainty proxy. It is not calibrated confidence.
- Optionally screens supplied pre/post SAR imagery for surface change, compares supplied Sentinel-2 Green/NIR bands, and traces a terrain-only path from a supplied DEM.
- Intersects candidate flood pixels with a dated OpenStreetMap snapshot to estimate potential asset exposure and model route scenarios.
- Produces JSON facts, an HTML situation report, and an interactive map dashboard.

These are candidate detections and modeled scenarios. They do not establish physical damage, verified road closures, or a flood forecast.

## Model benchmark

The current clean event-held-out run trained on Events 1111002, 1111003, and 1111004; selected its checkpoint and threshold on validation Event 1111006; and evaluated once on unseen Event 1111005 (1,369 scenes). The validation-selected threshold was 0.49.

| Measure | Event 1111005 result |
|---|---:|
| Flood IoU, all-scene macro | 13.95% |
| F1, all-scene macro | 15.42% |
| Precision, all-scene macro | 16.90% |
| Recall, all-scene macro | 15.80% |
| Flood IoU, pixel pooled | 28.58% |
| F1, pixel pooled | 44.45% |
| Pixel accuracy | 94.85% |

There were 179 flood-positive and 1,190 zero-flood scenes; only 123 zero-flood scenes had no false-positive flood pixels. Pixel accuracy is dominated by background and must not be presented as flood-detection accuracy or 90% reliability. See [AI readiness](docs/HACKATHON_AI_READINESS.md) and the full [Himalayan evaluation](docs/HIMALAYAN_EVALUATION.md).

## Important limitations

- The worker does **not** download Sentinel imagery or preprocess raw GRD products. You must provide analysis-ready pre- and post-event Sentinel-1 GeoTIFFs.
- STAC search discovers candidate scene metadata; it does not establish that your supplied raster files are those exact products.
- The current benchmark model uses post-event VV/VH only. Its training data does not contain paired pre/post SAR for the model input.
- Debris detection is a surface-change heuristic, not a trained or independently validated debris classifier.
- Sentinel-2 cloud screening is a simple proxy. Optical imagery is not downloaded automatically.
- DEM analysis is optional, terrain-only screening, not hydraulic modeling.
- A road intersecting a flood candidate plus a 30 m buffer is assumed impassable for the route scenario. This is not a confirmed closure.
- The benchmark tiles have placeholder georeferencing; area and bounding-box estimates may be withheld when pixel scale is implausible.
- Upstream training-label provenance has not been fully audited, and independent regional validation is limited.

The full requirement-by-requirement status and remaining work are in [HACKATHON_REQUIREMENTS.md](docs/HACKATHON_REQUIREMENTS.md).


## Full-Stack Automated Architecture

In addition to the standalone research CLI tools, FloodSight AI includes a complete full-stack automated platform that eliminates manual raster downloading:

1. **Automated Planetary Computer STAC Ingestion**:
   - Queries Microsoft Planetary Computer STAC API for **Sentinel-1 RTC** (Radiometrically Terrain Corrected SAR at 10m) and **Sentinel-2 L2A** (surface reflectance).
   - Automatically signs access tokens, discovers same-orbit pre/post pairs, and reprojects assets onto a unified 20m UTM analysis grid.
2. **Asynchronous Execution & Database Queue**:
   - Powered by FastAPI and PostgreSQL 16, an autonomous daemon worker claims jobs through a 9-stage lifecycle (`validating` → `searching_data` → `downloading` → `preprocessing` → `flood_mapping` → `damage_analysis` → `connectivity_analysis` → `generating_report` → `completed`).
3. **Interactive Web Workspace & Map Dashboard**:
   - Built with React 19 and MapLibre GL, featuring 1-click flood disaster benchmarks (Assam, Sylhet, Emilia-Romagna, Kerala), real-time stage progress tracking, and layer visibility toggles (Flood, Infrastructure, Settlements, AOI).
4. **Automated PDF Situation Reports**:
   - Generates FEMA-standard PDF situation reports and vector GeoJSON exports directly from persisted facts.

## Setup

Use Python 3.10 or later. From the repository directory, create and activate an environment, then install the dependencies:

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt

PyTorch is listed in requirements. For a specific CUDA setup, install the PyTorch build that matches your system using the official PyTorch installation guidance.

This repository does not include the satellite rasters, labeled training data, or model weight files. Prepare or obtain those separately and follow their data-use terms. The model training and input conventions are described in [MODEL_TRAINING.md](MODEL_TRAINING.md).

## Run the analysis worker

The worker needs a compatible checkpoint, network access for scene discovery and the dated OSM query, and a prepared SAR pair. Each SAR GeoTIFF must contain VV and VH bands in that order, in dB, use EPSG:4326, share the same dimensions and pixel grid, and cover the requested bounding box. Verify that the pre/post scenes are an appropriate pair; metadata tags are checked when present but do not prove product identity.

Example (replace the example coordinates, date, and file paths with your event data):

    python worker.py --bbox 85 27 86 28 --flood-date 2024-07-01 --pre-image data/pre_event.tif --post-image data/post_event.tif --out-dir outputs/event_001

Optional arguments include --s2-green and --s2-nir together for supplied Sentinel-2 bands, and --dem for a supplied DEM. See the CLI options in worker.py. The job writes status.json, job.log, scene_search.json, facts.json, situation_report.html, dashboard.html, and generated analysis rasters under the output directory. A failed infrastructure-data request fails the job rather than reporting missing exposure data as zero.

Set a checkpoint explicitly when needed:

    $env:FLOODSIGHT_CHECKPOINT = "checkpoints/path/to/best.pt"

The default checkpoint lookup prefers checkpoints/hackathon_holdout_1111005/best.pt, but weight files are ignored by Git and are not included in this repository. Train or provide a compatible checkpoint before running inference. FLOODSIGHT_THRESHOLD can override the prediction threshold; use a threshold selected on validation data, never the final test event.

## Train and evaluate

Training expects paired GeoTIFFs under data/processed/images and data/processed/masks. Images contain two VV/VH bands; masks use 0 for background, 1 for flood, and 255 for nodata. Keep complete events separate across training, validation, and test splits.

For the documented clean event split, with the corresponding labeled data prepared:

    python train.py --data-dir data/processed --output-dir checkpoints/hackathon_holdout_1111005 --val-events 1111006 --test-events 1111005 --epochs 30 --patience 8 --batch-size 4 --patch-size 224 --flood-weight 8.0 --event-balanced-sampling

The validation event is used for checkpoint selection and threshold calibration. Keep the test event untouched until the model and settings are fixed. See [MODEL_TRAINING.md](MODEL_TRAINING.md) for data preparation, and [HIMALAYAN_EVALUATION.md](docs/HIMALAYAN_EVALUATION.md) for the clean evaluation protocol and results.

## Repository guide

- model.py, dataset.py, train.py — model, data loading, and training
- predict.py, evaluate.py, calibrate_threshold.py — inference, metrics, and threshold calibration
- worker.py, stac_search.py — job orchestration and scene metadata discovery
- debris.py, optical_analysis.py, dem_flow_analysis.py — optional candidate screens
- damage_analysis.py, connectivity_analysis.py — exposure and route scenarios
- generate_report.py, generate_dashboard.py — result presentation
- docs/ — requirements, readiness, evaluation, API, and testing notes
- tests/ — automated test suite

## Track B requirements and references

- [Track B requirement status](docs/HACKATHON_REQUIREMENTS.md)
- [AI model readiness and benchmark](docs/HACKATHON_AI_READINESS.md)
- [Himalayan event-held-out evaluation](docs/HIMALAYAN_EVALUATION.md)
- [API reference](docs/API_REFERENCE.md)
- [Testing strategy](docs/TESTING_STRATEGY.md)
- [Training guide](MODEL_TRAINING.md)

KuroSiwo dataset information and attribution: [official dataset repository](https://github.com/Orion-AI-Lab/KuroSiwo). Dataset terms apply to any data you use. This repository currently contains no LICENSE file, so no software license is declared here.
