# FloodSight flood model

The segmentation model is a compact U-Net built in `model.py`. It is initialized
with random weights and does not download or use a pretrained encoder. It maps
either post-event VV/VH or paired pre/post-event VV/VH channels (in dB) to
background and flood logits. The checkpoint records the channel contract and
normalization values; inference rejects incompatible inputs.

## Data layout

Prepare labeled, analysis-ready GeoTIFFs with the same radiometric processing
used for inference. The image must have two bands in VV, VH order. The mask must
have one band with `0` for background, `1` for flood, and `255` for nodata.

```text
data/processed/
  images/
    event_001_post.tif
    event_002_post.tif
  masks/
    event_001.tif
    event_002.tif
```

An image is paired to a mask by event name; `_pre`, `_post`, `_before`, and
`_after` suffixes are removed when matching. If both before and after images
exist for an event, training uses the post-event image. Image and mask dimensions
must match. The training script holds out whole events for validation to avoid
leaking neighboring patches from one event across the split.

VV/VH means and standard deviations are estimated from valid pixels in the
training split only. Those same values normalize validation inputs and are
stored in each checkpoint for inference.

## Train

Install `requirements.txt`, then run:

```bash
python train.py --data-dir data/processed --output-dir checkpoints
```

For a final benchmark, reserve a complete event for testing and use a different
event for validation/model selection. For example:

```bash
python train.py --data-dir data/processed --output-dir checkpoints/ready_run \
  --val-events 1111005 --test-events 1111003 --epochs 100 --patience 15 \
  --flood-weight 8
```

The training command writes separate validation and test manifests. The test
event is excluded from training and checkpoint selection; use it only after the
model and prediction settings are fixed.

The flood-class weight defaults to 8.0 to reduce the severe overprediction seen
with the earlier 49.5 setting. Threshold calibration must use the validation
event only. For example, predict validation probabilities with
`predict.py --manifest checkpoints/ready_run/val_manifest.json --checkpoint
checkpoints/ready_run/best.pt --out-dir outputs/validation --save-probability`,
then run `calibrate_threshold.py --manifest checkpoints/ready_run/val_manifest.json
--prob-dir outputs/validation --out-json outputs/validation/threshold.json`.
Apply that selected threshold once to the separate test manifest for the final
report; do not retune it using test results.

To prepare the official KuroSiwo GRD WebDataset shard, convert its paired
pre/post SAR and flood labels into the project's four-band training format:

```bash
python import_kuro_webdataset.py data/official_kuro/train_GRD/shard-00000.tar \
  --out-dir data/official_kuro/processed
python train.py --data-dir data/official_kuro/processed \
  --output-dir checkpoints/official_run --val-events 502 --test-events 555 \
  --epochs 100 --patience 15 --flood-weight 8
```

These event IDs refer to the imported shard. The importer converts source
linear backscatter to dB, orders bands as pre VV/VH then post VV/VH, maps
source flood class 2 to class 1, and preserves invalid pixels as 255.
KuroSiwo is distributed under CC BY; retain attribution and cite the
[official dataset repository](https://github.com/Orion-AI-Lab/KuroSiwo) when
redistributing this dataset or reporting results.

The defaults use native 224-pixel tiles, batch size 4, one sampled patch per
training tile per epoch, up to 100 epochs, and mixed precision when CUDA is
available. Training stops after 15 epochs without an improvement in validation
flood IoU; adjust this with `--patience`. Validation evaluates a deterministic
grid over complete held-out tiles. The history records flood IoU, precision,
recall, specificity, balanced accuracy, F1, pixel accuracy, and pixel error
rate.
Flood IoU and F1 are the more useful measures for imbalanced flood masks; pixel
accuracy alone can look high when the model misses flood pixels. If memory is
limited, reduce `--batch-size` or `--patch-size`. Training stops with a clear
error if it cannot find image/mask pairs or at least two distinct events for an
event-held-out validation split.

Inference reads the input channel count and normalization values from the
checkpoint. A 2-channel checkpoint takes one post-event VV/VH raster. A
4-channel checkpoint takes pre-event VV/VH followed by post-event VV/VH; pass
the two aligned 2-band rasters separately with:

```bash
python predict.py --scene path/to/post_event.tif --pre-scene path/to/pre_event.tif \
  --checkpoint checkpoints/best.pt --out-dir outputs/predictions
```

The pre- and post-event rasters must have matching dimensions, CRS, and pixel
alignment, and each must contain exactly two VV/VH bands. A single 4-band raster is also supported when its bands are ordered
pre-VV, pre-VH, post-VV, post-VH.

For an application worker, set `FLOODSIGHT_CHECKPOINT` to the validated model
artifact path. When present, the default is
`checkpoints/official_run/best.pt`; otherwise it falls back to
`checkpoints/best.pt`. `FLOODSIGHT_THRESHOLD` controls the probability cutoff
and defaults to `0.92`, the pooled-F1 optimum selected using validation event
502. A lower `0.55` cutoff is a recall-oriented alternative, with more false
alarms. On a partial import of the official test shard (4,029 complete scenes,
8 events; source download was truncated), neither setting was reliable across
events. At `0.92`, pixel-weighted IoU/precision/recall were 0.594/0.695/0.805,
but only 42% of zero-flood scenes were completely clean; at `0.55`, only 11%
were clean. Event performance varied substantially. These scores are
diagnostic and must not be used to claim operational readiness. Keep checkpoint
and threshold settings identical between batch inference and the application.

Every inference facts record labels the output as research-only and requires
human review. Do not treat a flood mask or area estimate as an operational
warning without independent geospatial verification and domain review.

Each inference also writes `*_uncertainty.tif`, a normalized binary predictive
entropy map (0 = low model ambiguity, 1 = high model ambiguity; `-1` = nodata).
This is an uncertainty proxy, not calibrated confidence: it does not capture
geographic or sensor distribution shift, and a low value is not proof that a
prediction is correct. The facts JSON records the raster path, mean entropy,
and the share of valid pixels with entropy at least 0.5.

## Outputs and interpretation

- `checkpoints/best.pt`: model and optimizer state for the epoch with the best
  validation flood IoU, plus model and normalization configuration.
- `checkpoints/latest.pt`: most recent epoch checkpoint.
- `checkpoints/training_history.csv`: training and validation loss and
  segmentation metrics for each epoch.

The current four-channel run is summarized in
`checkpoints/official_run/readiness_report.json`, with per-scene results in
`outputs/official_run_test/evaluation_report.json`. Its held-out event was
reserved from the downloaded training shard, not from KuroSiwo's official test
partition. The score is a research result: held-out flood recall remains low,
and the WebDataset metadata lacks a CRS, so imported predictions are not
geographically located until georeferenced source rasters are supplied.

Flood IoU is computed only for class `1`; nodata pixels (`255`) are excluded.
The checkpoint is a trained artifact only after this command has run over real,
labeled data. No weights are included in this repository, and random/dummy
samples are never used by the training command.

Checkpoints created before explicit per-dataset SAR statistics were passed to
data-loader workers cannot be resumed with the corrected pipeline. The
`official_run` checkpoint uses the corrected pipeline. Its held-out report is
still a limited research benchmark; a broader evaluation should also use the
dataset's official test partition and additional geographic regions.

The current artifact is not cleared for operational flood decisions. The
partial official-test evaluation shows substantial event-to-event error, and
source tiles lack a CRS.
When CRS is absent, inference now reports pixel counts/fractions but leaves
square-kilometer area and metric resolution unknown. Before operational use,
evaluate on the official test split and independent regions, georeference the
inputs, set and document the operating threshold on validation data, and have
domain reviewers assess missed-flood and false-alarm costs.

An event-balanced sampling experiment was run separately in
`checkpoints/official_event_balanced`. It stopped after six epochs; its best
validation flood IoU was `0.3584`, below the selected baseline's `0.3762`, so
its checkpoint was not promoted. This experiment does not close the external
validation gap. No evidence currently supports a 90% operational reliability
claim. Pixel accuracy alone is unsuitable for that claim because non-flood
background dominates the image area. Keep operational decisions disabled and
require analyst review while the documented deployment blockers remain.
