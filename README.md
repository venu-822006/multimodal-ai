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

The model was trained on KuroSiwo Events 1111002, 1111003, and 1111004; calibrated on Event 1111006; and evaluated on unseen Himalayan Event 1111005 (1,369 scenes):

| Metric | Event 1111005 Result | Operational Hardening |
|---|---:|---|
| **Flood IoU (all-scene macro)** | 13.95% | Enhanced by slope ($\le 8^\circ$) & temporal change filter |
| **F1 (all-scene macro)** | 15.42% | Suppresses false alarms on dry mountain scenes |
| **Precision (all-scene macro)** | 16.90% | Eliminates steep mountain radar shadows |
| **Recall (all-scene macro)** | 15.80% | Retains coherent flood expansion clusters |
| **Pixel-Pooled F1** | 44.45% | Robust on large contiguous inundations |
| **Pixel Accuracy** | 94.85% | Background-dominated; not presented as reliability |

See [Track B requirement status](docs/HACKATHON_REQUIREMENTS.md) and [Himalayan evaluation](docs/HIMALAYAN_EVALUATION.md) for full audit reports.

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
