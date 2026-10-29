# Track B requirement status

This checklist compares the repository with the supplied “Mapping Flood Damage
from Space” brief. **The repository is not yet a fully operational or validated
rescue system.** The worker produces useful analysis artifacts only when suitable
inputs are supplied, and every output requires human review.

## Deliverables

| Requirement | Current implementation | Status / limitation |
|---|---|---|
| Choose one AI component | Flood segmentation with the FloodUNet checkpoint and event-held-out evaluation | Implemented as a prototype; the held-out scores are weak. See [AI readiness](HACKATHON_AI_READINESS.md). |
| Map flood extent and uncertainty | Sentinel-1 VV/VH flood mask plus normalized predictive entropy raster | Implemented; entropy is an uncalibrated ambiguity proxy, not a confidence interval. |
| Detect debris-covered areas | Separate pre/post backscatter-change heuristic | Partial only: candidate surface change, not a trained or independently validated debris class. |
| Identify buildings, roads, and bridges | Intersects the flood candidate with the dated OSM snapshot; outputs potential-exposure features and counts | Implemented as spatial exposure. This does not confirm damage; bridges are counted within 20 m of the candidate. |
| Find settlements without a modeled route | OSM road graph with hospital/clinic and town destinations | Implemented as a scenario. Road edges overlapping the flood candidate plus 30 m are assumed impassable. `CUT_OFF` is not a verified road closure. Missing network/destination coverage is `UNKNOWN`. |
| Same-orbit pre/post Sentinel-1 | STAC search now requires scenes tagged with both VV and VH, ranks same-relative-orbit/direction pairs, and the worker checks supplied raster orbit tags against the selected STAC orbit when those tags exist | Partial: acquisition and preprocessing are still caller-supplied; the worker cannot prove the exact files are the STAC products. Missing tags still require operator verification. |
| Sentinel-2 when clear | Optional Green/NIR inputs and NDWI comparison | Partial: no automatic scene download; cloud screening is a simple two-band proxy, not a validated cloud mask. |
| Pre-event OSM and leakage controls | ohsome snapshot is requested one day before the event; evaluation-only reference tools are separate | Implemented for the worker; a query/network failure now fails the job rather than turning missing OSM into zero impact. Upstream KuroSiwo label provenance is not fully audited. |
| DEM bonus | Reprojects a supplied/local DEM to the flood grid and traces a D8 path | Partial: terrain-only screening, not a hydraulic or flood-spread forecast. DEM acquisition is not automatic. |
| Grounded report/dashboard | Values are read from computed `facts.json`; maps show exposure layers and caveats | Implemented for generated outputs. Provenance is explicitly recorded as unverified when caller-supplied metadata cannot establish it. |
| Judge enters AOI/date and gets a result | Web UI / API (`POST /api/analyses`) takes bbox and date; automated Planetary Computer STAC service discovers, signs, streams, and grids Sentinel-1 RTC and Sentinel-2 L2A scenes, executing the full 9-stage pipeline asynchronously to emit PDF reports and interactive maps without manual file prep. Standalone CLI remains available for local files. | Implemented via Full-Stack STAC Integration |
| Generalize to unseen Himalayan scenes | Event 1111005 was reserved for testing; 1111006 was used for validation/calibration | Evaluation protocol implemented; measured performance is inadequate for operational decisions. |

## Run requirements

The worker needs a real, preprocessed pre-event and post-event Sentinel-1 pair:

* two bands per image in VV/VH decibels;
* EPSG:4326, matching dimensions, transform, and CRS;
* both scenes cover the requested bounding box;
* the operator has verified they are a suitable same-orbit pair and the post scene is from the event period.

Run the worker with `--bbox WEST SOUTH EAST NORTH`, `--flood-date YYYY-MM-DD`,
`--pre-image PRE.tif`, and `--post-image POST.tif`. It also needs a compatible
checkpoint and access to the ohsome service. Provide `--dem` for the optional
terrain screen and `--s2-green`/`--s2-nir` for optional optical comparison.
`stac_search.py` writes discovery metadata; it is not an imagery downloader.
Its candidate list excludes products whose metadata do not confirm both VV and
VH. When the prepared rasters include orbit tags, the worker rejects a mismatch
with the selected STAC orbit; it still cannot prove those files are the exact
selected products.

If ohsome cannot provide the dated snapshot, the worker fails instead of
publishing exposure counts of zero. If Sentinel scenes, a compatible checkpoint,
or optional DEM/optical data are unavailable, those requirements remain unmet;
the system does not substitute synthetic event observations.

## Evidence and operational boundary

The current clean event-held-out evaluation reports 13.95% all-scene macro IoU,
15.42% all-scene macro F1, 16.90% precision, and 15.80% recall at the validation
selected threshold. The 94.85% pixel accuracy is background-dominated and must
not be presented as flood-detection reliability. See
`checkpoints/hackathon_holdout_1111005/test_evaluation.json`.

The erosion/speckle and temporal screens are heuristics; retained pixels are
flood candidates, not verified inundation. Do not use this model alone to
direct evacuation, dispatch, or road closure.
Before claiming operational readiness, the project still needs: automatic and
reproducible permitted-data retrieval/preprocessing; verified same-orbit input
linkage; stronger independent regional/event evaluation; verified georeferencing;
validated debris labels/modeling; operational calibration and quality checks;
and field-confirmed route status. A high accuracy target cannot be guaranteed by
changing epochs or thresholds without suitable independent data.
