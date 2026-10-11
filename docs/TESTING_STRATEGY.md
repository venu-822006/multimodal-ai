# FloodSight AI — Testing & Verification Strategy

## 1. Overview
The testing architecture ensures safety, air-gapped compliance, numerical robustness, and zero data leakage.

```bash
# Run the complete test suite:
python -m pytest tests/
```

---

## 2. Test Suites

### Suite 1: AI Model & Algorithmic Regression (`tests/test_ai_model.py`)
* `test_model_forward_pass_and_shapes`: Validates that `FloodUNet` produces identical spatial dimensions and proper logits.
* `test_model_odd_dimensions_handling`: Verifies that bilinear interpolation handles odd dimensions (e.g. $225 \times 225$) without tensor shape mismatches.
* `test_anti_hallucination_slope_rejection`: Verifies that steep slopes ($> 8^{\circ}$) are zeroed out by the anti-hallucination filter.
* `test_anti_hallucination_speckle_removal`: Verifies that isolated single-pixel radar noise is removed.
* `test_debris_candidate_detection`: Validates that surface change with backscatter increase is labeled as Class B (Debris).
* `test_baseline_log_ratio_drop`: Validates that the classical change baseline identifies significant backscatter drops as inundation.

### Suite 2: Data Rules & Air-Gapped Isolation (`tests/test_data_rules.py`)
* `test_no_emsr927_leakage_in_worker_and_training`: Scans all source files to verify that no model or worker component references the EMSR927 validation directory.
* `test_damage_wording_compliance`: Enforces that the prohibited term "destroyed" is absent, and the operational phrase "affected / potentially affected" is present.
* `test_mandatory_attributions_present`: Verifies that Copernicus Sentinel 2026, WorldDEM-30, OpenStreetMap, and Kuro Siwo citations are included in output reports.

---

## 3. Evaluation on Held-Out Events (Gap 2)
To prevent spatial data leakage, evaluation is conducted strictly across **held-out events** rather than random spatial patches:
```bash
python evaluate.py \
    --pred-dir outputs/predictions \
    --truth-dir data/processed/masks \
    --out-json outputs/eval_metrics.json
```
Metrics reported:
* **Flood IoU (Jaccard Index)**
* **Precision & Recall**
* **F1-Score**
* **Pixel Accuracy**
* **Confusion Matrix** (True Positives, False Positives, True Negatives, False Negatives)
