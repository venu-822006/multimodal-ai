"""
test_weakness_fixes.py - Verification suite for Track B audit remediations
===========================================================================
Tests:
  1. Multi-modal debris detection scoring and bug-free execution
  2. Risk-stratified road connectivity and detour penalty quantification
  3. Anti-hallucination topographic slope and temporal consistency filtering
  4. Topographic slope gradient calculation from DEM
"""

import sys
from pathlib import Path

# Add multimodal-ai root to path
WORKSPACE_ROOT = Path(__file__).parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon

from debris import detect_debris_candidates, generate_multiclass_map
from connectivity_analysis import analyze_connectivity, build_road_graph
from predict import apply_anti_hallucination_filters, compute_slope_from_dem


def test_multimodal_debris_detection_and_scoring():
    """Verify multi-modal debris detection with SAR roughness, optical vegetation drop, and slope."""
    H, W = 64, 64
    pre_vv = np.full((H, W), -15.0, dtype=np.float32)
    post_vv = np.full((H, W), -15.0, dtype=np.float32)
    pre_vh = np.full((H, W), -22.0, dtype=np.float32)
    post_vh = np.full((H, W), -22.0, dtype=np.float32)
    flood_mask = np.zeros((H, W), dtype=np.uint8)

    # Core flood in top-left
    flood_mask[:20, :20] = 1

    # Simulate debris deposit in center:
    # 1. SAR backscatter surge (roughness increase +3 dB)
    post_vv[25:55, 25:55] = -12.0
    post_vh[25:55, 25:55] = -18.0

    # 2. Valley slope: 10 degrees (within 1 to 25 deg runout corridor)
    slope = np.full((H, W), 10.0, dtype=np.float32)

    # 3. Sentinel-2 Optical: low NDVI (scoured vegetation / bare gravel)
    s2_green = np.full((H, W), 0.15, dtype=np.float32)
    s2_nir = np.full((H, W), 0.18, dtype=np.float32)  # NDVI = (0.18-0.15)/(0.18+0.15) = ~0.09

    debris_mask, conf_score, metadata = detect_debris_candidates(
        pre_vv=pre_vv,
        post_vv=post_vv,
        pre_vh=pre_vh,
        post_vh=post_vh,
        flood_mask=flood_mask,
        slope=slope,
        s2_green=s2_green,
        s2_nir=s2_nir,
        min_cluster_px=10,
    )

    # Must detect debris in the simulated deposit area
    assert debris_mask[30, 30] == 1, "Expected debris candidate in runout zone"
    # Must NOT flag core flood as debris
    assert debris_mask[10, 10] == 0, "Core flood must not be marked as debris"
    # Confidence score must be calculated and non-zero
    assert 0.40 <= conf_score <= 1.0, f"Confidence score out of bounds: {conf_score}"
    assert metadata["scoring_mode"] == "multimodal_sar_optical_terrain"
    assert metadata["optical_used"] is True
    assert metadata["slope_used"] is True

    # Test multi-class map labeling
    multi_map = generate_multiclass_map(flood_mask, debris_mask)
    assert multi_map[10, 10] == 1  # Flood (Class A)
    assert multi_map[30, 30] == 2  # Debris (Class B)
    assert multi_map[0, 50] == 0   # Background


def test_risk_stratified_connectivity_and_detour():
    """Verify route scenarios: CRITICAL_ISOLATED, FUNCTIONAL_DETOUR, CONNECTED."""
    # Build a simple triangular road network with alternative route:
    # Node S (Settlement): (91.0, 26.0)
    # Node J (Junction):   (91.0, 26.1)
    # Node H (Hospital):   (91.1, 26.1)
    # Node D (Detour via): (90.9, 26.05)
    g = nx.Graph()
    p_s = (91.0, 26.0)
    p_j = (91.0, 26.1)
    p_h = (91.1, 26.1)
    p_d = (90.9, 26.05)

    # Primary route: S -> J -> H (distance ~ 22 km)
    g.add_edge(p_s, p_j, weight=11.1, geom=LineString([p_s, p_j]))
    g.add_edge(p_j, p_h, weight=10.0, geom=LineString([p_j, p_h]))

    # Alternative detour route: S -> D -> J -> H (longer detour)
    g.add_edge(p_s, p_d, weight=12.0, geom=LineString([p_s, p_d]))
    g.add_edge(p_d, p_j, weight=12.0, geom=LineString([p_d, p_j]))

    settlement = {
        "properties": {"name": "Test Village"},
        "geometry": {"type": "Point", "coordinates": [91.0, 26.0]},
    }
    hospital = {
        "properties": {"name": "Regional Hospital"},
        "geometry": {"type": "Point", "coordinates": [91.1, 26.1]},
    }

    # Case 1: Flood covers ONLY primary segment S-J, forcing detour S -> D -> J -> H
    flood_on_primary = Polygon([
        (90.99, 26.03), (91.01, 26.03), (91.01, 26.07), (90.99, 26.07), (90.99, 26.03)
    ])

    results_detour = analyze_connectivity(
        flood_polygon=flood_on_primary,
        road_graph=g,
        settlements=[settlement],
        hospitals=[hospital],
        disruption_buffer_m=10.0,
    )

    counts = results_detour["settlement_counts"]
    assert counts["FUNCTIONAL_DETOUR"] == 1, "Should classify as FUNCTIONAL_DETOUR"
    assert counts["CONNECTED"] == 1, "Should be counted as connected via detour"
    assert counts["CRITICAL_ISOLATED"] == 0

    s_res = results_detour["settlements"][0]
    assert s_res["state"] == "FUNCTIONAL_DETOUR"
    assert s_res["detour_additional_km"] > 0, "Detour additional km must be positive"
    assert s_res["triage_priority"] == "PRIORITY_2_DETOUR_ALERT"

    # Case 2: Complete isolation: flood covers both S-J and S-D
    flood_full = Polygon([
        (90.85, 25.95), (91.15, 25.95), (91.15, 26.15), (90.85, 26.15), (90.85, 25.95)
    ])
    results_isolated = analyze_connectivity(
        flood_polygon=flood_full,
        road_graph=g,
        settlements=[settlement],
        hospitals=[hospital],
        disruption_buffer_m=10.0,
    )
    assert results_isolated["settlement_counts"]["CRITICAL_ISOLATED"] == 1
    assert results_isolated["settlement_counts"]["CUT_OFF"] == 1
    assert results_isolated["settlements"][0]["triage_priority"] == "PRIORITY_1_CRITICAL_ISOLATION"
    assert "disclaimer" in results_isolated


def test_anti_hallucination_slope_and_temporal_filtering():
    """Verify topographic slope and temporal consistency filters eliminate radar shadow false alarms."""
    H, W = 100, 100
    prob_map = np.zeros((2, H, W), dtype=np.float32)
    prob_map[0] = 0.1
    prob_map[1] = 0.9  # Raw model is confident it's water (0.90)

    # 1. Slope test: steep mountain slope (18 degrees > 8 degree threshold)
    slope = np.full((H, W), 18.0, dtype=np.float32)
    cleaned_mask, stats = apply_anti_hallucination_filters(
        prob_map=prob_map,
        confidence_thresh=0.50,
        slope=slope,
        max_slope_deg=8.0,
    )
    assert cleaned_mask.sum() == 0, "Steep slopes must have all flood candidates filtered out"
    assert stats["slope_hallucinations_filtered"] == H * W

    # 2. Temporal change test: Flat terrain (slope 2 deg), but pre and post backscatter show NO drop
    flat_slope = np.full((H, W), 2.0, dtype=np.float32)
    pre_img = np.full((H, W, 2), -18.0, dtype=np.float32)
    post_img = np.full((H, W, 2), -18.0, dtype=np.float32)

    cleaned_mask_temp, stats_temp = apply_anti_hallucination_filters(
        prob_map=prob_map,
        confidence_thresh=0.50,
        slope=flat_slope,
        max_slope_deg=8.0,
        pre_image=pre_img,
        post_image=post_img,
        min_drop_db=1.5,
    )
    assert cleaned_mask_temp.sum() == 0, "Pixels with no temporal backscatter drop must be filtered out"
    assert stats_temp["temporal_false_positives_filtered"] == H * W

    # 3. Valid flood test: flat terrain + significant backscatter drop (pre: -10 dB, post: -20 dB -> drop: 10 dB)
    post_flooded = np.full((H, W, 2), -20.0, dtype=np.float32)
    pre_unflooded = np.full((H, W, 2), -10.0, dtype=np.float32)
    cleaned_valid, stats_valid = apply_anti_hallucination_filters(
        prob_map=prob_map,
        confidence_thresh=0.50,
        slope=flat_slope,
        max_slope_deg=8.0,
        pre_image=pre_unflooded,
        post_image=post_flooded,
        min_drop_db=1.5,
    )
    assert cleaned_valid.sum() == H * W, "Legitimate flood pixels must be retained"
    assert stats_valid["retained_flood_candidate_pixels"] == H * W


def test_dem_slope_gradient():
    """Verify DEM gradient computation produces expected slope degrees."""
    dem = np.zeros((50, 50), dtype=np.float32)
    for r in range(50):
        dem[r, :] = r * 3.0  # 3m rise per 30m cell = 10% slope
    slope = compute_slope_from_dem(dem, cell_size_m=30.0)
    interior_slope = slope[1:-1, 1:-1]
    assert np.all(interior_slope >= 5.0) and np.all(interior_slope <= 6.5)


def test_evaluate_filters_comparison():
    """Verify before/after anti-hallucination filter evaluation calculations."""
    from evaluate_filters import generate_filter_comparison
    res = generate_filter_comparison()
    assert "metrics_comparison" in res
    asm = res["metrics_comparison"]["all_scene_macro"]
    assert asm["iou"]["filtered"] > asm["iou"]["baseline"]
    assert asm["f1"]["filtered"] > asm["f1"]["baseline"]
    zfa = res["metrics_comparison"]["zero_flood_false_alarms"]
    assert zfa["clean_scenes_pct"]["filtered"] > zfa["clean_scenes_pct"]["baseline"]
    assert zfa["scenes_with_false_positives"]["eliminated_fp_scenes"] > 0


def test_same_orbit_enforcement_raises():
    """Verify cross-orbit pairs are strictly rejected without silent fallback."""
    import pytest
    from unittest.mock import MagicMock, patch
    from datetime import date

    mock_pre = MagicMock()
    mock_pre.datetime.date.return_value = date(2024, 6, 25)
    mock_pre.properties = {"sat:relative_orbit": 41, "sat:orbit_state": "ascending"}

    mock_post = MagicMock()
    mock_post.datetime.date.return_value = date(2024, 7, 5)
    # Different orbit!
    mock_post.properties = {"sat:relative_orbit": 99, "sat:orbit_state": "ascending"}

    with patch("pystac_client.Client.open") as mock_client_open:
        mock_client = MagicMock()
        mock_client_open.return_value = mock_client
        mock_search = MagicMock()
        mock_search.items.return_value = [mock_pre, mock_post]
        mock_client.search.return_value = mock_search

        from download_scenes import discover_and_download
        with pytest.raises(RuntimeError, match="none share both a matching relative orbit and matching flight direction"):
            discover_and_download([91.5, 26.0, 91.6, 26.1], "2024-07-01", "outputs/test_orbit_fail")


def test_same_orbit_different_flight_direction_raises():
    """Verify matching relative orbit with opposing flight directions is rejected."""
    import pytest
    from unittest.mock import MagicMock, patch
    from datetime import date

    mock_pre = MagicMock()
    mock_pre.datetime.date.return_value = date(2024, 6, 25)
    # Same orbit 41, but Ascending pass
    mock_pre.properties = {"sat:relative_orbit": 41, "sat:orbit_state": "ascending"}

    mock_post = MagicMock()
    mock_post.datetime.date.return_value = date(2024, 7, 5)
    # Same orbit 41, but Descending pass!
    mock_post.properties = {"sat:relative_orbit": 41, "sat:orbit_state": "descending"}

    with patch("pystac_client.Client.open") as mock_client_open:
        mock_client = MagicMock()
        mock_client_open.return_value = mock_client
        mock_search = MagicMock()
        mock_search.items.return_value = [mock_pre, mock_post]
        mock_client.search.return_value = mock_search

        from download_scenes import discover_and_download
        with pytest.raises(RuntimeError, match="none share both a matching relative orbit and matching flight direction"):
            discover_and_download([91.5, 26.0, 91.6, 26.1], "2024-07-01", "outputs/test_dir_fail")


def test_optical_and_sar_same_orbit_direction_manifest_structure(tmp_path):
    """Verify pairing succeeds and writes direction and optical fields when matching."""
    from unittest.mock import MagicMock, patch
    from datetime import date

    mock_pre = MagicMock()
    mock_pre.id = "S1A_IW_GRDH_PRE"
    mock_pre.datetime.date.return_value = date(2024, 6, 25)
    mock_pre.properties = {"sat:relative_orbit": 41, "sat:orbit_state": "descending", "platform": "sentinel-1a"}

    mock_post = MagicMock()
    mock_post.id = "S1A_IW_GRDH_POST"
    mock_post.datetime.date.return_value = date(2024, 7, 5)
    mock_post.properties = {"sat:relative_orbit": 41, "sat:orbit_state": "descending", "platform": "sentinel-1a"}

    with patch("pystac_client.Client.open") as mock_client_open, \
         patch("planetary_computer.sign") as mock_pc_sign, \
         patch("rasterio.open") as mock_rio_open, \
         patch("download_scenes.reproject") as mock_reproject:

        mock_client = MagicMock()
        mock_client_open.return_value = mock_client
        
        mock_s1_search = MagicMock()
        mock_s1_search.items.return_value = [mock_pre, mock_post]

        mock_s2 = MagicMock()
        mock_s2.id = "S2A_MSIL2A_TEST"
        mock_s2.datetime.date.return_value = date(2024, 6, 28)
        mock_s2.properties = {"eo:cloud_cover": 12.5}

        mock_s2_search = MagicMock()
        mock_s2_search.items.return_value = [mock_s2]

        mock_dem = MagicMock()
        mock_dem.id = "COP30_DEM_TEST"
        mock_dem_search = MagicMock()
        mock_dem_search.items.return_value = [mock_dem]

        mock_client.search.side_effect = [mock_s1_search, mock_s2_search, mock_dem_search]

        signed_item = MagicMock()
        signed_item.assets = {
            "vv": MagicMock(href="http://example.com/vv.tif"),
            "vh": MagicMock(href="http://example.com/vh.tif"),
            "B03": MagicMock(href="http://example.com/b03.tif"),
            "B08": MagicMock(href="http://example.com/b08.tif"),
            "data": MagicMock(href="http://example.com/dem.tif"),
        }
        mock_pc_sign.return_value = signed_item

        mock_src = MagicMock()
        mock_src.read.return_value = np.zeros((64, 64), dtype=np.float32)
        mock_src.transform = MagicMock()
        mock_src.crs = "EPSG:4326"
        mock_src.dtypes = ("float32", "float32")
        mock_src.shape = (64, 64)
        mock_rio_open.return_value.__enter__.return_value = mock_src

        from download_scenes import discover_and_download
        out_dir = tmp_path / "scenes"
        manifest = discover_and_download([91.5, 26.0, 91.55, 26.05], "2024-07-01", out_dir)

        assert manifest["status"] == "success"
        assert manifest["pre_event"]["orbit"] == 41
        assert manifest["pre_event"]["direction"] == "descending"
        assert manifest["post_event"]["orbit"] == 41
        assert manifest["post_event"]["direction"] == "descending"
        assert manifest["optical"] is not None
        assert manifest["optical"]["item_id"] == "S2A_MSIL2A_TEST"
        assert manifest["optical"]["cloud_cover"] == 12.5
        assert manifest["dem"] is not None
        assert manifest["dem"]["item_id"] == "COP30_DEM_TEST"
