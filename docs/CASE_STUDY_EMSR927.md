# Trishuli / EMSR927 evaluation status

The challenge asks for an August 2026 Trishuli flood map and an independent
comparison with the official Copernicus EMS activation EMSR927. This repository
does **not** currently contain verified Trishuli Sentinel-1 input scenes or an
official EMSR927 reference mask, so it has no defensible case-study metrics.

Earlier drafts of this document and `case_study_emsr927.py` contained invented
reference polygons, acquisition metadata, impact figures, and comparison scores.
Those values were not derived from EMSR927 and must not be used or presented as
results. The comparison code now requires an actual reference-mask GeoTIFF from
the official EMSR927 product. EMSR927 remains evaluation-only: do not use it to
train, tune thresholds, choose model parameters, preprocess input imagery, or
generate predictions.

## Run a valid comparison

1. Obtain the official EMSR927 product and document its provenance and mapped
   coverage. Convert the relevant flood delineation to a single-band mask on a
   georeferenced grid: `1` for flood, `0` for valid mapped non-flood, and `255`
   for nodata or locations outside the product's coverage.
2. Produce the FloodSight prediction using only permitted pre-event and event
   inputs.
3. Run the evaluation-only comparator:

   ```powershell
   python case_study_emsr927.py --pred outputs\event_flood_predicted.tif `
       --reference data\evaluation_only\emsr927_reference.tif `
       --out-json outputs\trishuli_emsr927_benchmark.json
   ```

The comparator aligns the reference mask to the prediction grid using nearest
neighbour resampling and scores only pixels valid in both. Confirm that the
reference mask represents the official product's actual coverage; a partial
mapping must not be filled with non-flood labels outside its coverage.

## Current limitations

- No official EMSR927-derived mask or verified August 2026 Trishuli prediction
  is included in this repository; benchmark results remain pending.
- The operational worker still requires caller-prepared Sentinel-1 VV/VH dB
  GeoTIFFs. It does not download or calibrate raw Copernicus products.
- No event-specific Himalayan validation score is claimed here. Generate one
  from a manifest that holds out the named Himalayan event(s); the benchmark
  now fails if those events are absent.
- Sentinel revisit time, terrain shadow/layover, cloud cover, OSM completeness,
  and label uncertainty limit interpretation. This is an educational prototype,
  not an operational rescue system.

## Attribution

For an actual comparison, include: “Contains modified Copernicus Sentinel data
2026.” and “Evaluated against European Union, Copernicus Emergency Management
Service data (EMSR927).”
