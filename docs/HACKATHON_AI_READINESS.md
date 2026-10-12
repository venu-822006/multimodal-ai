# Hackathon AI model readiness

Scope: the flood-segmentation AI component in Track B. This is not a claim
that the full satellite-to-rescue-intelligence system is ready.

## Requirement check

| Track B AI requirement | Current evidence | Status |
|---|---|---|
| Choose one AI component | Flood segmentation with a compact U-Net | Implemented |
| Map flood extent from satellite imagery | Binary background/flood mask from Sentinel-1 VV/VH | Implemented as a prototype |
| Generate an uncertainty layer | Inference writes normalized binary predictive entropy to `*_uncertainty.tif` | Implemented; uncalibrated ambiguity proxy |
| Report IoU, precision, recall, and F1 | Strict event-held-out evaluation reports these measures | Implemented |
| Test Himalayan scenes excluded from training and model selection | New split trains on events 1111002/1111003/1111004, validates on 1111006, and reserves all of 1111005 for testing | Split implemented and evaluated |
| Avoid reference-map leakage | This run's manifest points to the KuroSiwo `data/processed` scenes and masks; no EMSR927, UNOSAT, or OSM references are listed as model inputs. Upstream label provenance cannot be verified from the workspace alone. | No leakage found in the recorded run; upstream provenance audit remains |
| Distinguish debris from water | The neural model predicts flood/background only. The separate debris module is heuristic surface-change detection, not a learned or validated debris class. | Not met as a learned class |
| Use paired pre/post imagery | Current clean benchmark candidate was trained on two-band post-event scenes because paired pre-event imagery is not present in `data/processed`. Inference supports 4-band checkpoint contracts, but this candidate does not use that input. | Not met for this candidate |

## Clean event-held-out result

The checkpoint in `checkpoints/hackathon_holdout_1111005/best.pt` was trained
without Event 1111005. Its probability threshold (`0.49`) was selected using
validation Event 1111006 only, then applied once to all 1,369 test scenes in
Event 1111005.

| Metric | All-scene macro | Flood-positive-scene macro | Pixel pooled |
|---|---:|---:|---:|
| IoU | 13.95% | 37.97% | 28.58% |
| F1 | 15.42% | 49.22% | 44.45% |
| Precision | 16.90% | 60.52% | 33.28% |
| Recall | 15.80% | 52.14% | 66.90% |
| Pixel accuracy | 94.85% | — | 94.85% |

There were 179 flood-positive scenes and 1,190 zero-flood scenes. Only 123
zero-flood scenes (10.34%) had no false-positive flood pixels. The high pixel
accuracy is dominated by background and must not be presented as 90% flood
segmentation accuracy. Full results are in
`checkpoints/hackathon_holdout_1111005/test_evaluation.json`.

## Readiness decision

The model meets the **prototype and evaluation-method** portions of the AI
track, but it does not meet a reasonable reliability bar for operational use.
The clean held-out event has low scene-macro scores and substantial false
alarms. Do not claim 90% reliability or use these outputs alone for flood
decisions. Require human review.

The remaining model-data blockers are: labeled paired pre/post SAR for the
training split; independently verified debris labels if debris is to become a
learned class; more independent flood events and regions; and verified
georeferencing. The current benchmark tiles have placeholder pixel scales, so
inference suppresses area and bounding-box reporting when scale is implausible.
