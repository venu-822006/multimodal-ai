# FloodSight AI — AI Model Architecture Specification

## 1. Executive Summary
The current clean event-held-out FloodSight checkpoint is a prototype
segmentation network for Sentinel-1 flood extent. It consumes post-event VV/VH
and predicts background versus flood. The architecture and inference code
support four-band paired pre/post VV/VH checkpoints, but no paired pre-event
training data is available for this candidate. The model does not learn a
debris class; the separate debris candidate module is heuristic and is not
verified as debris by this model.

---

## 2. Model Architecture: `FloodUNet`
The network is implemented in PyTorch (`model.py`) and initialized completely from scratch using Kaiming normal initialization. It requires no external pre-trained weights, conforming strictly to hackathon air-gapped guidelines.

### Layer Blueprint
* **Input Layer:** the current clean checkpoint uses 2 post-event channels
  ($C=2$), Sentinel-1 VV/VH in decibels. Four-band checkpoints use pre-event
  VV/VH followed by post-event VV/VH. Training records the selected input
  contract in the checkpoint.
* **Encoder:** 4 downsampling stages with double $3\times3$ convolutions, Batch Normalization, and ReLU activations:
  - Stage 1: $C \rightarrow 24$ channels (spatial resolution: $H \times W$)
  - Stage 2: $24 \rightarrow 48$ channels (Spatial resolution: $H/2 \times W/2$)
  - Stage 3: $48 \rightarrow 96$ channels (Spatial resolution: $H/4 \times W/4$)
  - Bottleneck: $96 \rightarrow 192$ channels (Spatial resolution: $H/8 \times W/8$)
* **Decoder:** 3 upsampling stages via transposed convolutions ($2\times2$, stride 2) with skip-connections concatenated along the channel dimension.
  - Bilinear interpolation guard on skip connections guarantees odd-dimension input safety (e.g. $225 \times 225$).
* **Segmentation Head:** $1\times1$ convolution mapping 24 channels to 2 logits:
  - Class 0: Background / Non-flooded land
  - Class 1: Flood Inundation (Core water extent)
* **Parameter Footprint:** 1,084,490 parameters in the current `model.py` configuration. Training checkpoints also store optimizer and resume state, so their file size is larger than the model weights alone.

At inference, the system writes a normalized binary predictive-entropy raster
(`*_uncertainty.tif`) alongside the flood mask. This is an ambiguity proxy,
not calibrated confidence and not a guarantee under geographic or sensor
distribution shift. The raster and facts record are marked research-only and
require human review.

---

## 3. Loss Function & Optimization (Addressing Class Imbalance)

In SAR flood mapping datasets (such as KuroSiwo), flood pixels account for only ~2.02% of the scene, creating extreme class imbalance. A naive Cross-Entropy loss causes the model to collapse toward predicting background.

FloodSight employs a **Tri-Partite Composite Loss (`ComboLoss`)**:
$$\mathcal{L}_{\text{total}} = 0.35 \cdot \mathcal{L}_{\text{wCE}} + 0.35 \cdot \mathcal{L}_{\text{Focal}} + 0.30 \cdot \mathcal{L}_{\text{Dice}}$$

1. **Class-Weighted Cross Entropy ($\mathcal{L}_{\text{wCE}}$):**
   - Applies a balanced class weight ($w_{\text{bg}} = 1.0, w_{\text{flood}} = 6.0\text{--}8.0$) to counter minority-class sparsity while preventing false-alarm saturation on dry terrain. Combining moderate CE/Focal weights with Soft Dice Loss ensures high spatial overlap without over-predicting water across dry mountain slopes.
2. **Focal Loss ($\mathcal{L}_{\text{Focal}}$):**
   - Introduces focusing parameter $\gamma = 2.0$:
     $$\mathcal{L}_{\text{Focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
   - Suppresses gradients from easy-negative background pixels and concentrates learning on hard boundary pixels.
3. **Soft Dice Loss ($\mathcal{L}_{\text{Dice}}$):**
   - Directly optimizes the spatial overlap score (IoU surrogate):
     $$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum (p_{\text{flood}} \cdot y_{\text{flood}}) + \epsilon}{\sum p_{\text{flood}} + \sum y_{\text{flood}} + \epsilon}$$

### Optimization & Stability
* **Optimizer:** AdamW ($\text{lr} = 3 \times 10^{-4}, \text{weight decay} = 10^{-4}$)
* **Gradient Clipping:** Max norm $\|\mathbf{g}\| \le 1.0$ to prevent gradient spikes induced by focal loss.
* **Learning Rate Scheduler:** `CosineAnnealingWarmRestarts` with $T_0 = \max(5, \lfloor\text{epochs}/4\rfloor)$, $T_{\text{mult}} = 2$, and $\eta_{\min} = 10^{-6}$.

---

## 4. Radiometric Calibration & Normalization
SAR backscatter values vary over orders of magnitude in linear power. Preprocessing converts linear values to decibels (dB):
$$\sigma^0_{\text{dB}} = 10 \cdot \log_{10}(\sigma^0_{\text{linear}})$$

`setup_data.py` assumes calibrated linear Sigma0 power and converts it using
$10\log_{10}(x)$, then clips to $[-40, 0]$ dB. The official KuroSiwo GRD SNAP
configuration specifies Sigma output with dB scaling disabled. The raw source
TIFFs are not present in this workspace, so the exact stored-value encoding of
the downloaded files has not been inspected here.

### Dynamic Training Normalization
Normalization constants are estimated strictly from valid pixels in the training split before training begins:
$$x_{\text{norm}} = \frac{x_{\text{dB}} - \mu_{\text{train}}}{\sigma_{\text{train}} + \epsilon}$$
These parameters are serialized in each checkpoint. The dataset now receives
them explicitly so Windows data-loader workers use the same values as the main
process and inference.

---

## 5. Multi-Stage Anti-Hallucination Pipeline
SAR segmentation neural networks commonly suffer from false-positive hallucinations in mountainous or urban terrain. FloodSight AI implements a multi-stage physics-grounded anti-hallucination suppression filter:

1. **Temperature Scaling (optional, not fitted calibration):**
   - The default is $\tau = 1.0$, which leaves logits unchanged. Other values are
     user-configurable but have not been calibrated on a separate calibration set:
     $$p_i = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}$$
   - Prevents artificial overconfidence spikes before thresholding.
2. **Probability Thresholding:**
   - The current clean-event checkpoint uses $0.49$, selected to maximize
     pooled F1 on validation Event 1111006. The untouched Event 1111005 test
     still shows substantial errors, so this remains a research operating
     point, not an operationally validated threshold. Inference reads the
     matching checkpoint calibration automatically when available.
3. **Terrain filtering:**
   - A slope mask can be passed to the prediction filter, but the current worker does not acquire or supply a DEM. Do not claim DEM-based filtering for worker outputs unless a real, aligned DEM slope raster is integrated and recorded in the run provenance. A fixed slope cutoff also cannot reliably distinguish radar layover from real water in every terrain/aspect/orbit configuration.
4. **Temporal Change Consistency Check:**
   - When pre-event SAR is available, pixels must exhibit a backscatter drop:
     $$\Delta\text{dB} = \text{dB}_{\text{pre}} - \text{dB}_{\text{post}} \ge 1.5\text{ dB}$$
     Permanent water and pre-existing dark terrain ($\Delta\text{dB} \approx 0$) are rejected.
5. **Morphological Filtering:**
   - Small-component removal is disabled by default (`--min-cluster-px 0`); it can
     be enabled as a user-selected heuristic. It is not validated as a general
     false-positive filter.

---

## 6. Debris Candidate Detection (Class B: Experimental)
The separate debris module uses heuristic thresholds: surface changes with
increased backscatter ($\Delta\text{dB} \ge +2.0\text{ dB}$) may be flagged as
**Possible Debris / Surface Change**. This is not a learned class and has not
been independently validated.
* Labeled with an explicit lower confidence score ($0.40 - 0.75$).
* Combined with Class A to produce a multi-class GeoTIFF raster:
  - `0` = Background
  - `1` = Class A: Inundation (Core)
  - `2` = Class B: Possible Debris (Experimental)
  - `255` = Nodata

## 7. Current evaluation status

The model was retrained from scratch with explicit per-worker SAR normalization
and balanced composite loss (`flood_weight = 8.0`).

### Training & Validation Progression (checkpoints/training_history.csv):
* **Best Validation Flood IoU:** **22.91%** (Epoch 6)
* **Validation Pixel Accuracy:** **90.24%**
* **Validation Specificity:** **90.15%** (false positives reduced by >90% compared to earlier unweighted runs)
* **Validation F1-Score:** **0.3728**

### Clean Event-Held-Out Benchmark (Event 1111005):
* Trained on Events 1111002/1111003/1111004, selected checkpoint on 1111006,
  and evaluated once on 1,369 scenes from 1111005. The threshold was selected
  using validation data only.
* All-scene macro IoU/F1/precision/recall: **13.95% / 15.42% / 16.90% / 15.80%**.
* Flood-positive-scene macro IoU/F1/precision/recall: **37.97% / 49.22% / 60.52% / 52.14%**.
* Pixel-pooled IoU/F1/precision/recall: **28.58% / 44.45% / 33.28% / 66.90%**.
* Pixel accuracy is 94.85%, but only 10.34% of zero-flood scenes are completely
  clean. Background-heavy accuracy is not a flood-reliability score. Full
  results: `checkpoints/hackathon_holdout_1111005/test_evaluation.json`.
