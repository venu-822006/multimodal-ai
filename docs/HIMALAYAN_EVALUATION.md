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
accuracy. No score is asserted in this document: publish the values generated
from the checkpoint and manifest you actually used, alongside the event IDs,
scene count, and limitations. The current dataset split and model require
independent review before describing the results as geographic generalization.

Do not use EMSR927 or another published damage map to train the model, select
thresholds, tune parameters, or alter predictions. Such products belong only
in a separately documented final evaluation.
