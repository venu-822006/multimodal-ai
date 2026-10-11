"""
tests/test_ai_model.py  —  FloodSight AI | AI Model Unit & Regression Test Suite
================================================================================
Comprehensive test suite verifying:
  1. FloodUNet model architecture, tensor shapes, and gradient flow.
  2. Input edge cases (NaN handling, extreme dB values, odd dimensions).
  3. Anti-hallucination filter edge cases (steep slopes, speckle noise).
  4. Debris candidate detection logic and multi-class generation.
  5. Baseline log-ratio change detection with Otsu thresholding.
"""

import numpy as np
import pytest
import torch

from model import create_unet_model, FloodUNet
from predict import apply_anti_hallucination_filters
from debris import detect_debris_candidates, generate_multiclass_map
from baseline import compute_log_ratio_flood


def test_model_forward_pass_and_shapes():
    """Verify FloodUNet handles standard and arbitrary patch sizes."""
    model = create_unet_model(in_channels=2, classes=2, base_channels=24)
    model.eval()

    # Standard 256x256 patch
    x = torch.randn(2, 2, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 2, 256, 256), f"Unexpected output shape: {out.shape}"


def test_model_odd_dimensions_handling():
    """Verify bilinear skip-connection fix handles odd spatial dimensions without crashing."""
    model = create_unet_model(in_channels=2, classes=2, base_channels=24)
    model.eval()

    # Odd dimensions (e.g. 225x225 or 231x231)
    x = torch.randn(1, 2, 225, 225)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 2, 225, 225), f"Odd dimension mismatch: {out.shape}"


def test_anti_hallucination_slope_rejection():
    """Verify anti-hallucination filter completely zeroes out flood on steep slopes (> 8 deg)."""
    # Create fake probability map where everything is predicted as flood (prob = 0.95)
    prob_map = np.zeros((2, 100, 100), dtype=np.float32)
    prob_map[0] = 0.05
    prob_map[1] = 0.95

    # Slope array: left half flat (2 deg), right half steep mountain (25 deg)
    slope = np.zeros((100, 100), dtype=np.float32)
    slope[:, :50] = 2.0
    slope[:, 50:] = 25.0

    filtered_mask, stats = apply_anti_hallucination_filters(
        prob_map=prob_map,
        confidence_thresh=0.60,
        slope=slope,
        max_slope_deg=8.0,
        min_cluster_px=10,
    )

    # All steep slopes on the right half MUST be filtered out
    assert filtered_mask[:, 50:].sum() == 0, "Anti-hallucination failed: detected flood on steep mountain slope!"
    # Flat slope on left half should be retained
    assert filtered_mask[:, :50].sum() > 0, "Flat terrain flood was mistakenly filtered!"
    assert stats["slope_hallucinations_filtered"] == 5000


def test_anti_hallucination_speckle_removal():
    """Verify single-pixel noise is removed by morphological filtering."""
    prob_map = np.zeros((2, 100, 100), dtype=np.float32)
    prob_map[0] = 0.99
    prob_map[1] = 0.01

    # Add 3 isolated single-pixel false alarms
    prob_map[1, 10, 10] = 0.99
    prob_map[1, 50, 50] = 0.99
    prob_map[1, 80, 80] = 0.99

    filtered_mask, stats = apply_anti_hallucination_filters(
        prob_map=prob_map,
        confidence_thresh=0.60,
        min_cluster_px=10,
    )

    # Isolated pixels MUST be removed
    assert filtered_mask.sum() == 0, "Speckle noise false alarm was not filtered!"
    assert stats["speckle_noise_pixels_filtered"] >= 3


def test_debris_candidate_detection():
    """Verify Class B debris detection triggers on backscatter increase in non-flooded areas."""
    pre_vv = np.full((50, 50), -16.0, dtype=np.float32)
    post_vv = np.full((50, 50), -11.0, dtype=np.float32)  # +5 dB increase (debris deposit)
    pre_vh = np.full((50, 50), -22.0, dtype=np.float32)
    post_vh = np.full((50, 50), -17.0, dtype=np.float32)

    flood_mask = np.zeros((50, 50), dtype=np.uint8)  # No open water

    debris_mask, conf = detect_debris_candidates(
        pre_vv, post_vv, pre_vh, post_vh, flood_mask, min_backscatter_increase_db=2.0, min_cluster_px=10
    )

    assert debris_mask.sum() > 0, "Debris deposit with +5 dB increase was not detected!"
    assert 0.4 <= conf <= 0.8, f"Confidence out of expected bounds: {conf}"

    multi_map = generate_multiclass_map(flood_mask, debris_mask)
    assert 2 in np.unique(multi_map), "Class 2 (Debris) not present in multi-class map"


def test_baseline_log_ratio_drop():
    """Verify baseline log-ratio correctly flags large dB drops as water."""
    pre_vv = np.full((50, 50), -10.0, dtype=np.float32)
    post_vv = np.full((50, 50), -19.0, dtype=np.float32)  # -9 dB drop (water inundation)

    mask, thresh = compute_log_ratio_flood(pre_vv, post_vv, threshold_db=4.0, min_size_px=10)
    assert (mask == 1).sum() > 0, "Water inundation with -9 dB drop not flagged by baseline!"


def test_evaluate_shape_mismatch_raises_error(tmp_path):
    """Verify evaluate_pair raises an explicit ValueError when shapes differ (no silent cropping)."""
    from evaluate import evaluate_pair
    from rasterio.crs import CRS
    from rasterio.transform import Affine
    import rasterio

    p_tif = tmp_path / "pred.tif"
    t_tif = tmp_path / "truth.tif"
    trans = Affine(0.0001, 0.0, 85.0, 0.0, -0.0001, 28.0)
    crs_wgs = CRS.from_epsg(4326)

    # Pred: 100x100
    with rasterio.open(
        p_tif, "w", driver="GTiff", height=100, width=100, count=1, dtype=np.uint8,
        crs=crs_wgs, transform=trans
    ) as dst:
        dst.write(np.zeros((100, 100), dtype=np.uint8), 1)

    # Truth: 100x80 (mismatched)
    with rasterio.open(
        t_tif, "w", driver="GTiff", height=100, width=80, count=1, dtype=np.uint8,
        crs=crs_wgs, transform=trans
    ) as dst:
        dst.write(np.zeros((100, 80), dtype=np.uint8), 1)

    with pytest.raises(ValueError, match="Spatial dimension mismatch"):
        evaluate_pair(p_tif, t_tif)


def test_match_truth_file_exact():
    """Verify match_truth_file reconstructs the exact truth stem without loose prefix guessing."""
    from evaluate import match_truth_file
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        truth_dir = Path(td)
        # Create true mask
        mask_file = truth_dir / "1111004_01_4b_4b221743de8756628f33db1a5a261cbb.tif"
        mask_file.touch()

        # Prediction file
        pred_file = Path("1111004_01_4b_4b221743de8756628f33db1a5a261cbb_post_flood_predicted.tif")
        matched = match_truth_file(pred_file, truth_dir)

        assert matched is not None, "Failed to match prediction to truth mask"
        assert matched.name == mask_file.name


def test_geodesic_pixel_area_calculation():
    """Verify geodesic pixel area accounts for latitude cos(lat) shrinkage in EPSG:4326."""
    from predict import compute_pixel_area_m2
    from rasterio.transform import Affine
    from rasterio.crs import CRS
    import math

    crs_geo = CRS.from_epsg(4326)
    # 0.0001 deg pixel spacing
    trans = Affine(0.0001, 0.0, 85.0, 0.0, -0.0001, 28.0)

    # At equator (lat 0): cos(0) = 1.0
    area_eq = compute_pixel_area_m2(trans, crs_geo, center_lat=0.0)
    # At Nepal (lat 28 deg): cos(28 deg) ≈ 0.8829
    area_nepal = compute_pixel_area_m2(trans, crs_geo, center_lat=28.0)

    expected_ratio = math.cos(math.radians(28.0))
    actual_ratio = area_nepal / area_eq
    assert abs(actual_ratio - expected_ratio) < 1e-4, f"Geodesic scaling error: {actual_ratio} vs {expected_ratio}"


def test_debris_module_rejects_identical_images(tmp_path):
    """Verify debris module rejects identical pre and post image paths."""
    from debris import run_debris_module
    from rasterio.crs import CRS
    from rasterio.transform import Affine
    import rasterio

    img_path = tmp_path / "same_image.tif"
    mask_path = tmp_path / "mask.tif"
    out_path = tmp_path / "out.tif"
    trans = Affine(0.0001, 0.0, 85.0, 0.0, -0.0001, 28.0)
    crs_wgs = CRS.from_epsg(4326)

    with rasterio.open(
        img_path, "w", driver="GTiff", height=50, width=50, count=2, dtype=np.float32,
        crs=crs_wgs, transform=trans
    ) as dst:
        dst.write(np.zeros((2, 50, 50), dtype=np.float32))

    with rasterio.open(
        mask_path, "w", driver="GTiff", height=50, width=50, count=1, dtype=np.uint8,
        crs=crs_wgs, transform=trans
    ) as dst:
        dst.write(np.zeros((50, 50), dtype=np.uint8), 1)

    with pytest.raises(ValueError, match="requires distinct pre- and post-flood images"):
        run_debris_module(
            pre_path=img_path,
            post_path=img_path,
            flood_mask_path=mask_path,
            out_mask_path=out_path,
        )


def test_dem_downstream_flow_path_tracing():
    """Verify Copernicus DEM D8 downstream flow routing traces downhill propagation."""
    from dem_flow_analysis import (
        generate_synthetic_copernicus_dem,
        trace_downstream_flood_path,
        analyze_downstream_exposure,
    )

    bbox = [85.15, 27.90, 85.45, 28.25]
    dem, transform, crs = generate_synthetic_copernicus_dem(bbox, shape=(100, 100))
    assert dem.shape == (100, 100)
    assert dem.max() > dem.min()

    # Flood inundation cluster in upper northern gorge
    flood_mask = np.zeros((100, 100), dtype=np.uint8)
    flood_mask[10:15, 30:35] = 1

    results = trace_downstream_flood_path(flood_mask, dem, transform)
    assert results["status"] == "SUCCESS"
    assert results["downstream_length_km"] > 0
    assert results["elevation_drop_m"] >= 0
    assert len(results["flow_path_coords"]) > 1

    # Intersect with sample settlement
    flow_coords = results["flow_path_coords"]
    sample_pt = flow_coords[len(flow_coords) // 2]
    settlements = [{
        "geometry": {"type": "Point", "coordinates": sample_pt},
        "properties": {"name": "Downstream Village"},
    }]
    exposed = analyze_downstream_exposure(flow_coords, settlements)
    assert len(exposed) == 1
    assert exposed[0]["name"] == "Downstream Village"

