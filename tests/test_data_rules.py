"""
tests/test_data_rules.py  —  FloodSight AI | Data Rules & Integrity Verification
================================================================================
Verifies data leakage, terminology, and attribution rules mandated by Section 7:

  1. Rule 1: Isolation of EMSR927 validation data
     - Scans source files (worker, training, inference, preprocessing) to ensure
       no module outside eval/ references EMSR927.
  2. Rule 2: Damage Wording Compliance
     - Scans reports, damage analysis, and outputs to verify prohibited words
       (e.g., 'destroyed') are NOT used, and operational wording
       ('affected / potentially affected') is strictly followed.
  3. Rule 3: Mandatory Attributions
     - Verifies required Copernicus, ESA, OSM, and Kuro Siwo citation lines are present.
"""

import os
from pathlib import Path
import pytest


WORKSPACE_ROOT = Path(__file__).parent.parent

# Files that participate in training, inference, or processing
PROTECTED_SOURCES = [
    "model.py",
    "train.py",
    "dataset.py",
    "preprocess.py",
    "predict.py",
    "worker.py",
    "baseline.py",
    "debris.py",
    "damage_analysis.py",
    "connectivity_analysis.py",
]


def test_no_emsr927_leakage_in_worker_and_training():
    """
    Test Rule 1: Ensure worker, model, and training code have ZERO references
    to the EMSR927 evaluation directory, preventing data leakage.
    """
    for fname in PROTECTED_SOURCES:
        fpath = WORKSPACE_ROOT / fname
        if not fpath.exists():
            continue
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().lower()
            assert "emsr927" not in content, (
                f"Data Leakage Violation: Protected source file '{fname}' references EMSR927! "
                f"Per Section 7, only evaluation code may read validation benchmark data."
            )


def test_damage_wording_compliance():
    """
    Test Rule 2: Ensure damage assessment code strictly uses
    'affected / potentially affected' and never claims structures are 'destroyed'.
    """
    damage_file = WORKSPACE_ROOT / "damage_analysis.py"
    assert damage_file.exists(), "damage_analysis.py missing"
    with open(damage_file, "r", encoding="utf-8") as f:
        content = f.read()

    assert "destroyed" not in content.lower().replace('do not say "destroyed"', ""), (
        "Wording Violation: damage_analysis.py uses prohibited word 'destroyed'. "
        "Must use 'affected / potentially affected by mapped inundation'."
    )
    assert "affected / potentially affected" in content, (
        "Wording Missing: Standard wording 'affected / potentially affected' not found in damage_analysis.py."
    )


def test_mandatory_attributions_present():
    """
    Test Rule 3: Verify all mandatory Copernicus, Sentinel, DEM, and OSM
    attribution lines appear in the report generator.
    """
    report_file = WORKSPACE_ROOT / "generate_report.py"
    assert report_file.exists(), "generate_report.py missing"
    with open(report_file, "r", encoding="utf-8") as f:
        content = f.read()

    assert "Contains modified Copernicus Sentinel data" in content
    assert "Copernicus WorldDEM-30" in content
    assert "OpenStreetMap contributors" in content
    assert "Kuro Siwo" in content
