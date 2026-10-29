# FloodSight AI

FloodSight AI is an educational and operational-readiness decision support system for mapping flood inundation from Sentinel-1 SAR imagery and quantifying critical impacts on infrastructure, road networks, and isolated settlements.

> **Research & Rapid-Response Decision Support Prototype.** Evaluated on event-held-out benchmarks; all outputs include uncertainty metadata and require human-in-the-loop validation before guiding field deployments.

---

## What It Does

1. **AI Inundation Segmentation**: Predicts high-resolution flood masks from Sentinel-1 dual-polarization (VV/VH dB) radar, cutting through heavy monsoon cloud cover.
2. **Physics-Grounded Anti-Hallucination**: Eliminates mountain radar shadow false alarms using Copernicus DEM topographic slope gating ($\le 8^\circ$) and temporal backscatter drop consistency ($\Delta\text{VV} \ge 1.5\text{ dB}$).
3. **Uncertainty Mapping**: Emits normalized Shannon entropy rasters alongside binary masks to pinpoint boundary ambiguity.
4. **Multi-Modal Debris & Surface Change Classifier**: Fuses SAR roughness surges ($\Delta\sigma^0_{\text{VH}} \ge 1.5\text{ dB}$), Sentinel-2 optical scoured vegetation signatures ($\Delta\text{NDVI} < 0.25$), and geomorphometric valley runout corridors ($1^\circ \le \text{slope} \le 25^\circ$) into a calibrated confidence score.
5. **Cross-Sensor Optical Agreement**: Automatically compares SAR flood extent with clear-sky Sentinel-2 McFeeters NDWI to verify surface water.
6. **Copernicus DEM Downstream Flow Analysis**: Traces D8 steepest-descent downhill runout corridors to identify communities and infrastructure at immediate downstream risk.
7. **Risk-Stratified Road Graph & Detour Quantification**: Constructs pre-event OpenStreetMap road networks and models flood closures, classifying settlements into `CRITICAL_ISOLATED` (Priority 1: zero routes remaining), `FUNCTIONAL_DETOUR` (Priority 2: alternative detour available with quantified $+X\text{ km}$ distance penalty), and `CONNECTED` (unimpaired).
8. **Automated Situation Reports & Interactive Dashboard**: Produces FEMA-standard PDF Situation Reports, interactive Leaflet web dashboards, and vector GeoJSONs.

---

## Full-Stack & Standalone Architecture

FloodSight AI provides two execution modalities:

### 1. Standalone End-to-End CLI (`worker.py` & `download_scenes.py`)
Enter a bounding box and event date; the system automatically queries Microsoft Planetary Computer STAC, downloads and speckle-filters same-orbit Sentinel-1 RTC GeoTIFFs, retrieves clear-sky Sentinel-2 L2A bands, and pulls Copernicus 30m DEM tiles:

```bash
# Automated cloud-native ingestion & analysis:
python worker.py --bbox 91.5 26.0 92.0 26.5 --flood-date 2024-07-02 --out-dir outputs/assam_flood

# Or run with local preprocessed GeoTIFFs:
python worker.py --bbox 85.2 27.6 85.4 27.8 --flood-date 2024-07-01 \
    --pre-image data/pre.tif --post-image data/post.tif \
    --s2-green data/s2_b03.tif --s2-nir data/s2_b08.tif --dem data/dem_30m.tif \
    --out-dir outputs/event_001
```

### 2. Full-Stack Asynchronous Platform (Web UI + FastAPI + PostgreSQL)
- **Web UI (React 19 + MapLibre GL)**: Interactive map workspace featuring 1-click global flood benchmarks (Assam, Sylhet, Emilia-Romagna, Kerala), real-time pipeline stage progress bars, and map layer toggles (Flood Extent, Infrastructure Exposure, Isolated Settlements, Bounding Box).
- **FastAPI Daemon Worker**: PostgreSQL 16 `FOR UPDATE SKIP LOCKED` distributed queue executing a 10-state lifecycle (`VALIDATING` → `SEARCHING_DATA` → `DOWNLOADING` → `PREPROCESSING` → `FLOOD_MAPPING` → `DAMAGE_ANALYSIS` → `CONNECTIVITY_ANALYSIS` → `GENERATING_REPORT` → `COMPLETED`).
- **Instant Reporting**: Direct one-click PDF situation report downloads and GeoJSON exports.

---

## Model Benchmark & Held-Out Evaluation

The model was evaluated on unseen Himalayan Event 1111005 (1,369 scenes) across raw and operationally hardened configurations:

| Metric | Raw Baseline (Unfiltered) | Balanced Response Mode | High-Precision Mode | Delta (Balanced) |
|---|---:|---:|---:|---:|
| **All-Scene Macro IoU** | 13.95% | **56.49%** | **61.20%** | **+42.54%** |
| **All-Scene Macro F1** | 15.42% | **57.52%** | **62.85%** | **+42.10%** |
| **All-Scene Macro Precision** | 16.90% | **57.80%** | **71.45%** | **+40.90%** |
| **All-Scene Macro Recall** | 15.80% | **59.42%** | **57.10%** | **+43.62%** |
| **Positive-Scene Macro Precision** | 60.52% | 52.30% | **72.80%** | +12.28% (HP) |
| **Positive-Scene Macro Recall** | 52.14% | **64.35%** | 58.40% | **+12.21%** |
| **Clean Zero-Flood Scenes** | 10.34% (123) | **58.65% (705)** | **82.35% (980)** | **+48.31% pts** |
| **Pixel False Alarm Rate (FPR)** | 9.45% | **2.12%** | **0.84%** | **-77.6%** |

See [Himalayan Evaluation Report](docs/HIMALAYAN_EVALUATION.md) and [Evaluation Comparison JSON](outputs/held_out_eval/before_after_filter_comparison.json) for reproducible benchmark runs.

---

## Official Hackathon AI Transparency & Disclosure Statement

In accordance with official hackathon submission rules regarding AI assistance and transparency:
- **Core AI & Computer Vision Models**:
  - The flood segmentation engine employs **FloodUNet** (PyTorch, 2-channel SAR input VV/VH in decibels), trained on public Copernicus Sentinel-1 datasets.
  - Spatial prediction uncertainty is modeled via **Monte Carlo Dropout predictive entropy**, computing calibrated pixel-level ambiguity without ad-hoc heuristics.
  - Anti-hallucination screening incorporates deterministic physical constraints: Copernicus 30m DEM slope gradients ($\le 8^\circ$) and temporal backscatter drop thresholds ($\Delta\text{VV} \ge 1.5\text{ dB}$).
- **AI Coding Assistance**:
  - **Google Antigravity / Gemini** was utilized as an interactive AI coding assistant for code scaffolding, Planetary Computer STAC client integration, and unit test suite implementation.
- **Human Review & Verification**:
  - All satellite geometry validation logic (same-orbit, same flight direction), geospatial reprojections, graph connectivity algorithms, and benchmark scripts were designed, reviewed, and independently tested by the human engineering team.

---

See [Track B requirement status](docs/HACKATHON_REQUIREMENTS.md) and [Himalayan evaluation](docs/HIMALAYAN_EVALUATION.md) for full compliance documentation.

---

## Setup & Verification

```bash
# 1. Setup Python Environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Run Multi-Modal AI Tests
pytest multimodal-ai/tests

# 3. Run Backend API Tests
pytest backend/tests

# 4. Run Frontend Tests
npm test
```

---

## Repository Guide

- `worker.py` — 10-state asynchronous pipeline orchestrator with auto-STAC ingestion
- `download_scenes.py` — Cloud-native Planetary Computer STAC download & speckle filtering
- `debris.py` — Multi-modal Class B debris classifier (SAR roughness + optical NDVI + slope corridor)
- `predict.py` — U-Net model inference with DEM slope & temporal anti-hallucination filters
- `connectivity_analysis.py` — NetworkX road network graph with risk-stratified detour quantification
- `optical_analysis.py` — McFeeters NDWI calculation and SAR-optical consensus matrix
- `dem_flow_analysis.py` — Copernicus WorldDEM-30 D8 flow routing and downstream community risk
- `damage_analysis.py` — OpenStreetMap historical infrastructure exposure analysis
- `generate_report.py`, `generate_dashboard.py` — Situation report and Leaflet dashboard generation
- `docs/HACKATHON_REQUIREMENTS.md` — Detailed Track B compliance and remediation checklist

---

## Attributions & Licences
- **Satellite Data**: Copernicus Sentinel-1 and Sentinel-2 data processed via Microsoft Planetary Computer STAC.
- **Topography**: Copernicus WorldDEM-30 © European Space Agency.
- **Infrastructure**: OpenStreetMap data retrieved via HeiGIT ohsome API © OpenStreetMap contributors (ODbL).
- **Benchmark Dataset**: KuroSiwo (Bountos et al., 2024).
