# FloodSight flood model

The segmentation model is a compact U-Net built in `model.py`. It is initialized
with random weights and does not download or use a pretrained encoder. It maps
two Sentinel-1 channels (VV and VH, in dB) to background and flood logits.

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

The defaults use 256-pixel patches, batch size 4, up to 100 epochs, and mixed
precision when CUDA is available. Training stops after 15 epochs without an
improvement in held-out flood IoU; adjust this with `--patience`. Each scene
contributes 16 randomly cropped training patches per epoch; validation evaluates
a deterministic grid of patches across each held-out event. The history file
records flood IoU, precision, recall, F1, pixel accuracy, and pixel error rate.
Flood IoU and F1 are the more useful measures for imbalanced flood masks; pixel
accuracy alone can look high when the model misses flood pixels. If memory is
limited, reduce `--batch-size` or `--patch-size`. Training stops with a clear
error if it cannot find image/mask pairs or at least two distinct events for an
event-held-out validation split.

## Outputs and interpretation

- `checkpoints/best.pt`: model and optimizer state for the epoch with the best
  validation flood IoU, plus model and normalization configuration.
- `checkpoints/latest.pt`: most recent epoch checkpoint.
- `checkpoints/training_history.csv`: training and validation loss, flood IoU,
  and pixel accuracy for each epoch.

Flood IoU is computed only for class `1`; nodata pixels (`255`) are excluded.
The checkpoint is a trained artifact only after this command has run over real,
labeled data. No weights are included in this repository, and random/dummy
samples are never used by the training command.
