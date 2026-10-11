# FloodSight AI — AI Model Architecture Specification

## 1. Executive Summary
The FloodSight AI model is a prototype semantic segmentation network for
post-event flood inundation using two Sentinel-1 SAR bands (VV and VH). The
learned network predicts background versus flood; it does not learn a debris
class or compare pre- and post-event imagery inside the network.

---

## 2. Model Architecture: `FloodUNet`
The network is implemented in PyTorch (`model.py`) and initialized completely from scratch using Kaiming normal initialization. It requires no external pre-trained weights, conforming strictly to hackathon air-gapped guidelines.

### Layer Blueprint
* **Input Layer:** 2 channels ($C=2$):
  - Band 1: Sentinel-1 VV polarization (in decibels, dB)
  - Band 2: Sentinel-1 VH polarization (in decibels, dB)
* **Encoder:** 4 downsampling stages with double $3\times3$ convolutions, Batch Normalization, and ReLU activations:
  - Stage 1: $2 \rightarrow 24$ channels (Spatial resolution: $H \times W$)
  - Stage 2: $24 \rightarrow 48$ channels (Spatial resolution: $H/2 \times W/2$)
  - Stage 3: $48 \rightarrow 96$ channels (Spatial resolution: $H/4 \times W/4$)
  - Bottleneck: $96 \rightarrow 192$ channels (Spatial resolution: $H/8 \times W/8$)
* **Decoder:** 3 upsampling stages via transposed convolutions ($2\times2$, stride 2) with skip-connections concatenated along the channel dimension.
  - Bilinear interpolation guard on skip connections guarantees odd-dimension input safety (e.g. $225 \times 225$).
* **Segmentation Head:** $1\times1$ convolution mapping 24 channels to 2 logits:
  - Class 0: Background / Non-flooded land
  - Class 1: Flood Inundation (Core water extent)
* **Parameter Footprint:** 1,084,490 parameters in the current `model.py` configuration. Training checkpoints also store optimizer and resume state, so their file size is larger than the model weights alone.

---

## 3. Loss Function & Optimization (Addressing Class Imbalance)

In SAR flood mapping datasets (such as KuroSiwo), flood pixels account for only ~2.02% of the scene, creating extreme class imbalance. A naive Cross-Entropy loss causes the model to collapse toward predicting background.

FloodSight employs a **Tri-Partite Composite Loss (`ComboLoss`)**:
$$\mathcal{L}_{\text{total}} = 0.35 \cdot \mathcal{L}_{\text{wCE}} + 0.35 \cdot \mathcal{L}_{\text{Focal}} + 0.30 \cdot \mathcal{L}_{\text{Dice}}$$

1. **Class-Weighted Cross Entropy ($\mathcal{L}_{\text{wCE}}$):**
   - Applies an inverse class frequency weight ($w_{\text{bg}} = 1.0, w_{\text{flood}} = 49.5$) to counter the ~50:1 background-to-flood pixel ratio.
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
   - The default flood cutoff is $0.5$, equivalent to binary argmax. The cutoff is
     configurable; it is not calibrated by the training pipeline.
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

The local epoch-6 checkpoint was evaluated on 1,388 event-held-out scenes from
events `1111005` and `1111003`. The report records mean scene IoU 0.0461 and F1
0.0559; pixel-pooled precision 0.1115 and recall 0.9967. This indicates severe
overprediction. More importantly, this checkpoint predates the explicit
normalization handoff to Windows data-loader workers. Its training IoU (0.1945)
does not agree with the full held-out pixel-pooled IoU (0.1115), so neither
score should be presented as the final corrected-model result. Retrain with the
current pipeline and reserve an event-level test set for the final evaluation.
