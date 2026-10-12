# Himalayan holdout evaluation

The challenge requires showing model performance on Himalayan scenes that were
not used in training. `evaluate_himalayan.py` filters the event-held-out
manifest to KuroSiwo event IDs `1111005` and `1111003`; it now fails if neither
event is present instead of silently evaluating unrelated scenes.

Run after training has written the matching held-out manifest and checkpoint:

```powershell
python evaluate_himalayan.py --manifest checkpoints\val_manifest.json `
    --checkpoint checkpoints\best.pt --out-dir outputs\himalayan_eval
```

The script writes per-scene and aggregate IoU, F1, precision, recall, and pixel
accuracy.

## Current Held-Out Benchmark Results

The checked-in evaluation report covers 30 scenes from held-out Himalayan
Event `1111005`. It reports scene-macro results, which give each scene equal
weight, and pixel-pooled results. Only 4 scenes contain flood pixels, so the
small positive-scene count makes these estimates unstable.

| Metric | All 30 scenes, scene macro | 4 flood-positive scenes |
|---|---:|---:|
| Flood IoU | 27.88% | 59.12% |
| F1 | 28.84% | 66.31% |
| Precision | 28.62% | 64.63% |
| Recall | 31.31% | 84.85% |
| Pixel accuracy | 86.52% | — |

Across all pixels, IoU is 20.37%, F1 is 33.85%, precision is 20.81%, and
recall is 90.70%. Among the 26 zero-flood scenes, only 6 (23.08%) have no
false-positive flood pixels. These results show the model detects much of the
flooded area while producing many false alarms; they do not demonstrate
operational reliability or a 90% segmentation score. The source report is
`outputs/himalayan_eval/himalayan_evaluation_report.json`.

Do not use EMSR927 or another published damage map to train the model, select
thresholds, tune parameters, or alter predictions. Such products belong only
in a separately documented final evaluation.
