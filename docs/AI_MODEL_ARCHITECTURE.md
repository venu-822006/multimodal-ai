# FloodSight AI — AI Model Architecture Specification

## 1. Executive Summary
The FloodSight AI model is an earth-observation deep learning segmentation network designed for rapid spaceborne flood inundation and surface change detection using dual-polarization Sentinel-1 Synthetic Aperture Radar (SAR) Ground Range Detected (GRD) imagery.

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
* **Parameter Footprint:** ~1.63M parameters (~13 MB checkpoint), optimized for low-latency execution and low VRAM footprint (< 4 GB) on entry-level GPUs (RTX 3050).

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
* **Learning Rate Scheduler:** `CosineAnnealingWarmRestarts` ($T_0 = 5, T_{\text{mult}} = 2, \eta_{\min} = 10^{-6}$), preventing premature plateaus and ensuring continued plasticity across extended training.

---

## 4. Radiometric Calibration & Normalization
SAR backscatter values vary over orders of magnitude in linear power. Preprocessing converts linear values to decibels (dB):
$$\sigma^0_{\text{dB}} = 10 \cdot \log_{10}(\sigma^0_{\text{linear}})$$

Values are clipped to the physically realistic SAR terrestrial dynamic range $[-40.0, 0.0] \text{ dB}$.

### Dynamic Training Normalization
Normalization constants are estimated strictly from valid pixels in the training split before training begins:
$$x_{\text{norm}} = \frac{x_{\text{dB}} - \mu_{\text{train}}}{\sigma_{\text{train}} + \epsilon}$$
These parameters are permanently serialized into `checkpoints/best.pt` under `normalization`, ensuring training-inference distribution alignment.

---

## 5. Multi-Stage Anti-Hallucination Pipeline
SAR segmentation neural networks commonly suffer from false-positive hallucinations in mountainous or urban terrain. FloodSight AI implements a multi-stage physics-grounded anti-hallucination suppression filter:

1. **Temperature Scaling (fixed heuristic, not fitted calibration):**
   - Raw logits are scaled by temperature $\tau = 1.5$:
     $$p_i = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}$$
   - Prevents artificial overconfidence spikes before thresholding.
2. **Calibrated Softmax Thresholding:**
   - Marginal predictions ($p < 0.55$) are rejected.
3. **Terrain filtering:**
   - A slope mask can be passed to the prediction filter, but the current worker does not acquire or supply a DEM. Do not claim DEM-based filtering for worker outputs unless a real, aligned DEM slope raster is integrated and recorded in the run provenance. A fixed slope cutoff also cannot reliably distinguish radar layover from real water in every terrain/aspect/orbit configuration.
4. **Temporal Change Consistency Check:**
   - When pre-event SAR is available, pixels must exhibit a backscatter drop:
     $$\Delta\text{dB} = \text{dB}_{\text{pre}} - \text{dB}_{\text{post}} \ge 1.5\text{ dB}$$
     Permanent water and pre-existing dark terrain ($\Delta\text{dB} \approx 0$) are rejected.
5. **Morphological Speckle Cleaning:**
   - Connected component analysis removes isolated speckle clusters $< 20\text{ pixels}$ and fills enclosed interior voids.
   - Per-scene metrics (`noise_reduction_pct`, `speckle_noise_pixels_filtered`) are recorded in `facts.json`.

---

## 6. Debris Candidate Detection (Class B: Experimental)
To close **Gap 1**, surface changes in valley corridors where radar backscatter increases ($\Delta\text{dB} \ge +2.0\text{ dB}$) rather than decreases are flagged as **Class B: Possible Debris / Surface Change**.
* Labeled with an explicit lower confidence score ($0.40 - 0.75$).
* Combined with Class A to produce a multi-class GeoTIFF raster:
  - `0` = Background
  - `1` = Class A: Inundation (Core)
  - `2` = Class B: Possible Debris (Experimental)
  - `255` = Nodata
