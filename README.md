# FloodSight AI

FloodSight is an educational prototype for flood segmentation from dual-band
Sentinel-1 data, with optional Sentinel-2 support, pre-event OpenStreetMap
exposure/connectivity analysis, and a situation report. The challenge's
requirements are in `Space Track.pdf` (keep that file beside this README if you
want the checklist available in the repository).

## Current scope

- Trains a U-Net flood segmentation model on prepared KuroSiwo data, with an
  event-held-out validation split, early stopping, and IoU/F1/precision/recall
  metrics.
- Runs inference on real, prepared Sentinel-1 VV/VH dB rasters and compares
  pre-event and post-event data when both are supplied.
- Can query a historical OSM snapshot through ohsome and produce flood
  exposure, road connectivity, and an HTML situation report.
- Can compare a prediction with a supplied EMSR927 reference mask in a
  separate evaluation-only step.

It is **not yet end-to-end from raw satellite data**: STAC search discovers
metadata but does not download or calibrate/process raw GRD products. The worker
requires prepared, aligned two-band VV/VH dB GeoTIFFs in EPSG:4326. Sentinel-2
bands must also be supplied separately. It does not currently ingest Copernicus
DEM or calculate an upstream flood path. Do not describe the project as
operational or as fully meeting the challenge until these gaps are closed.

No genuine EMSR927 benchmark result is bundled. Previous draft comparison
polygons and reported scores were fabricated and have been removed. See
`docs/CASE_STUDY_EMSR927.md` for the real-reference workflow and
`docs/HIMALAYAN_EVALUATION.md` for the strict holdout evaluation procedure.

## Setup

Use Python 3.10 or later, then install the project dependencies:

```powershell
python -m pip install -r requirements.txt
```

Training data setup and conversion instructions are in `setup_data.py` and
`MODEL_TRAINING.md`. The resulting processed folder is expected to contain
`data/processed/images/` and `data/processed/masks/`.

## Train and evaluate

```powershell
python run_pipeline.py --epochs 100 --patience 15 --batch-size 4
```

The pipeline trains (unless `--skip-train` is set), infers on the saved
event-held-out manifest, and writes evaluation outputs under `outputs/`.
Inspect the training history and per-scene evaluation report; a higher epoch
count does not guarantee better generalization.

For the dedicated Himalayan holdout report:

```powershell
python evaluate_himalayan.py --manifest checkpoints/val_manifest.json `
  --checkpoint checkpoints/best.pt --out-dir outputs/himalayan_eval
```

This command refuses to substitute unrelated events if the manifest lacks the
specified Himalayan event IDs.

## Analyze a new area/date with prepared Sentinel-1 scenes

The input rasters must be real, analysis-ready two-band GeoTIFFs with VV and VH
in dB, the same dimensions and transform, EPSG:4326, and full coverage of the
requested bounding box. Supply a distinct pre-event and post-event scene from
the same relative orbit/direction. The worker also needs a checkpoint in
`checkpoints/best.pt` or `checkpoints/latest.pt` and internet access to the
Copernicus STAC and ohsome services.

```powershell
python worker.py --bbox 85.1 27.8 85.5 28.1 --flood-date 2026-08-26 `
  --pre-image data/scenes/prepared_pre.tif `
  --post-image data/scenes/prepared_post.tif `
  --job-id trishuli --out-dir outputs/trishuli
```

Optional Sentinel-2 green and NIR rasters must be provided together with
`--s2-green` and `--s2-nir`. The worker uses the OSM snapshot dated one day
before the event. If the requested snapshot is unavailable from ohsome, the job
will fail rather than substituting current OSM data.

## EMSR927 evaluation only

Prepare a verified official-reference mask separately from the prediction:
`1` flood, `0` valid mapped non-flood, `255` nodata/outside the reference
product's coverage. Then run:

```powershell
python case_study_emsr927.py --pred outputs/trishuli/post_flood_predicted.tif `
  --reference data/evaluation_only/emsr927_reference.tif `
  --out-json outputs/trishuli_emsr927_benchmark.json
```

Never use EMSR927, UNOSAT, or other published damage maps as model inputs,
training labels, preprocessing data, or tuning targets. The reference is only
for independent final checking.

## Outputs and limitations

Each worker job writes `status.json`, `job.log`, `facts.json`, STAC metadata,
cached inputs, a predicted flood GeoTIFF, a one-page HTML situation report, and
an interactive Leaflet map dashboard (`dashboard.html`). The dashboard needs an
internet connection for its basemap and map library.
Review `facts.json` and `status.json` before communicating results. Exposure
means features intersecting the mapped flood footprint; it does not prove
physical damage. OSM completeness, cloud cover, radar shadow/layover, model
generalization, revisit latency, and label quality all affect confidence.
This prototype must not be used as the sole basis for emergency decisions.

Every submission must include the challenge's required source and training
data attributions. Include a data-source attribution only when that source was
actually used, and never claim WorldDEM was used unless the project really used
it. KuroSiwo citation/license details should be checked against the exact
download and its accompanying terms.

# multimodal-ai
