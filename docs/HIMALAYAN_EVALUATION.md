# Himalayan holdout evaluation

The challenge requires testing on Himalayan scenes excluded from training.
For a final, leakage-resistant result, use the dedicated split below: Event
`1111005` is excluded from training and validation, and its threshold is fixed
using Event `1111006`. `evaluate_himalayan.py` now requires a dedicated test
manifest and checks that the checkpoint records the same test events, disjoint
from its training and validation events. The older 30-scene report below was
created before that guard and is retained only as a validation-overlap example.

The clean split was trained and evaluated with:

```powershell
python train.py --data-dir data\processed `
    --output-dir checkpoints\hackathon_holdout_1111005 `
    --val-events 1111006 --test-events 1111005 `
    --epochs 30 --patience 8 --event-balanced-sampling
python predict.py --manifest checkpoints\hackathon_holdout_1111005\val_manifest.json `
    --checkpoint checkpoints\hackathon_holdout_1111005\best.pt `
    --out-dir outputs\hackathon_holdout_1111005_val --save-probability
python calibrate_threshold.py `
    --manifest checkpoints\hackathon_holdout_1111005\val_manifest.json `
    --prob-dir outputs\hackathon_holdout_1111005_val `
    --out-json checkpoints\hackathon_holdout_1111005\threshold_calibration.json
python predict.py --manifest checkpoints\hackathon_holdout_1111005\test_manifest.json `
    --checkpoint checkpoints\hackathon_holdout_1111005\best.pt `
    --out-dir outputs\hackathon_holdout_1111005_test_rerun `
    --confidence-threshold 0.49
python evaluate.py `
    --manifest checkpoints\hackathon_holdout_1111005\test_manifest.json `
    --pred-dir outputs\hackathon_holdout_1111005_test_rerun `
    --out-json checkpoints\hackathon_holdout_1111005\test_evaluation.json
```

The generated test report contains per-scene and aggregate IoU, F1, precision,
recall, specificity, and pixel accuracy. Do not use this final test event to
change the model or threshold.

## Clean Final Event-Held-Out Results

The final benchmark covers all 1,369 scenes in Event 1111005, which was
excluded from both training and model selection. The model checkpoint was
selected on Event 1111006, and threshold 0.49 was calibrated there. Results:

| Metric | All-scene macro | Flood-positive-scene macro | Pixel pooled |
|---|---:|---:|---:|
| Flood IoU | 13.95% | 37.97% | 28.58% |
| F1 | 15.42% | 49.22% | 44.45% |
| Precision | 16.90% | 60.52% | 33.28% |
| Recall | 15.80% | 52.14% | 66.90% |
| Pixel accuracy | 94.85% | — | 94.85% |

There were 179 flood-positive scenes and 1,190 zero-flood scenes. Only 123
zero-flood scenes (10.34%) had no false-positive flood pixels. The high pixel
accuracy is caused by background dominance; this result does not demonstrate
90% flood reliability. Full results are in
`checkpoints/hackathon_holdout_1111005/test_evaluation.json`.

## Prior Validation-Overlap Report (Not an Independent Final Test)

The older report covers 30 scenes from Himalayan
Event `1111005`. It reports scene-macro results, which give each scene equal
weight, and pixel-pooled results. Only 4 scenes contain flood pixels, so the
small positive-scene count makes these estimates unstable. Event 1111005 was
also present in the prior validation manifest, so do not present this report as
an independent final test.

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
