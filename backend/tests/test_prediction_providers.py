"""Tests for the prediction provider contract.

Covers both providers, the explicit selection rules, and the guarantee that the
synthetic provider is never silently substituted for the trained model.
"""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine
from pyproj import CRS

from app.core.config import Settings
from app.services.prediction_providers import (
    PROVIDER_SYNTHETIC,
    PROVIDER_TRAINED_MODEL,
    SYNTHETIC_DATA_LABEL,
    TRAINED_DATA_LABEL,
    PredictionProviderError,
    SyntheticTestProvider,
    TrainedModelProvider,
    get_prediction_provider,
    reset_prediction_provider_registry,
    resolve_provider_id,
    validate_prediction,
)

UTM = CRS.from_epsg(32645)
TRANSFORM = Affine(20.0, 0.0, 500000.0, 0.0, -20.0, 4000000.0)


@pytest.fixture(autouse=True)
def _clear_registry():
    reset_prediction_provider_registry()
    yield
    reset_prediction_provider_registry()


def _raster(height: int = 120, width: int = 160, *, valid: np.ndarray | None = None):
    rng = np.random.default_rng(3)
    vv = rng.normal(-13.0, 3.0, (height, width)).astype("float32")
    vh = rng.normal(-19.0, 2.5, (height, width)).astype("float32")
    stack = np.stack([vv, vh], axis=-1)
    mask = np.ones((height, width), dtype=bool) if valid is None else valid
    return stack, mask


def _settings(tmp_path, **overrides) -> Settings:
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2] / "multimodal-ai"
    base = {
        "environment": "test",
        "database_url": "postgresql+psycopg://unused:unused@localhost/unused",
        "artifact_root": tmp_path / "artifacts",
        "model_repo_path": repo,
        "model_path": None,
        "model_device": "cpu",
        "osm_enabled": False,
        "log_format": "text",
        "log_level": "ERROR",
        "prediction_provider": overrides.get("prediction_provider", "trained_model"),
    }
    base.update(overrides)
    return Settings(**base)


class TestSyntheticProviderRunsWithoutWeights:
    def test_produces_a_mask_without_any_checkpoint(self, tmp_path):
        provider = SyntheticTestProvider(_settings(tmp_path, model_path=None))
        image, valid = _raster()
        prediction = provider.predict(image, TRANSFORM, UTM, valid)

        assert prediction.synthetic is True
        assert prediction.data_label == SYNTHETIC_DATA_LABEL
        assert prediction.mask.shape == image.shape[:2]
        assert set(np.unique(prediction.mask)).issubset({0, 1, 255})
        assert prediction.mask[valid].max() == 1, "expected some flood pixels in the test pattern"

    def test_declares_itself_synthetic_in_provenance(self, tmp_path):
        provider = SyntheticTestProvider(_settings(tmp_path))
        image, valid = _raster()
        prediction = provider.predict(image, TRANSFORM, UTM, valid)
        provenance = prediction.provenance
        assert provenance["synthetic"] is True
        assert provenance["data_label"] == SYNTHETIC_DATA_LABEL
        assert provenance["provider"] == PROVIDER_SYNTHETIC
        assert "not trained" in provenance["note"] or "No trained" in provenance["note"]
        # No checkpoint identity may be implied.
        assert provenance["checkpoint"] is None

    def test_descriptor_reports_synthetic_and_available(self, tmp_path):
        descriptor = SyntheticTestProvider(_settings(tmp_path)).describe()
        assert descriptor.available is True
        assert descriptor.synthetic is True
        assert descriptor.data_label == SYNTHETIC_DATA_LABEL
        payload = descriptor.as_dict()
        assert payload["synthetic"] is True
        assert payload["status"] == "ok"
        assert payload["warning"]


class TestSyntheticDeterminism:
    def test_same_seed_gives_identical_mask(self, tmp_path):
        image, valid = _raster()
        first = SyntheticTestProvider(_settings(tmp_path, synthetic_seed=7)).predict(
            image, TRANSFORM, UTM, valid
        )
        second = SyntheticTestProvider(_settings(tmp_path, synthetic_seed=7)).predict(
            image, TRANSFORM, UTM, valid
        )
        np.testing.assert_array_equal(first.mask, second.mask)

    def test_different_seed_gives_a_different_mask(self, tmp_path):
        image, valid = _raster()
        first = SyntheticTestProvider(_settings(tmp_path, synthetic_seed=7)).predict(
            image, TRANSFORM, UTM, valid
        )
        second = SyntheticTestProvider(_settings(tmp_path, synthetic_seed=8)).predict(
            image, TRANSFORM, UTM, valid
        )
        assert not np.array_equal(first.mask, second.mask)

    @pytest.mark.parametrize("pattern", ["blocky", "blobs", "gradient", "checkerboard"])
    def test_every_pattern_preserves_georeferencing(self, tmp_path, pattern):
        image, valid = _raster()
        provider = SyntheticTestProvider(_settings(tmp_path, synthetic_pattern=pattern))
        prediction = provider.predict(image, TRANSFORM, UTM, valid)
        assert prediction.transform == TRANSFORM
        assert prediction.crs == UTM
        assert prediction.mask.shape == image.shape[:2]

    @pytest.mark.parametrize("pattern", ["blocky", "blobs"])
    def test_blocky_patterns_produce_connected_regions(self, tmp_path, pattern):
        """Filtering small patches must be exercised, not bypassed by pure speckle."""
        image, valid = _raster()
        prediction = SyntheticTestProvider(
            _settings(tmp_path, synthetic_pattern=pattern)
        ).predict(image, TRANSFORM, UTM, valid)
        from skimage.measure import label as label_regions

        assert label_regions(prediction.mask == 1, connectivity=2).max() < 200


class TestNodataHandling:
    def test_invalid_pixels_are_marked_nodata(self, tmp_path):
        image, _ = _raster()
        valid = np.ones(image.shape[:2], dtype=bool)
        valid[0:40, 0:40] = False
        prediction = SyntheticTestProvider(_settings(tmp_path)).predict(
            image, TRANSFORM, UTM, valid
        )
        assert (prediction.mask[~valid] == 255).all()
        assert prediction.mask[valid].max() == 1

    def test_fully_invalid_grid_produces_no_flood(self, tmp_path):
        image, _ = _raster()
        valid = np.zeros(image.shape[:2], dtype=bool)
        prediction = SyntheticTestProvider(_settings(tmp_path)).predict(
            image, TRANSFORM, UTM, valid
        )
        assert (prediction.mask == 255).all()


class TestTrainedProviderReportsMissingWeights:
    def test_unavailable_without_checkpoint(self, tmp_path):
        descriptor = TrainedModelProvider(_settings(tmp_path, model_path=None)).describe()
        assert descriptor.available is False
        assert descriptor.synthetic is False
        assert descriptor.data_label == TRAINED_DATA_LABEL
        assert "MODEL_PATH" in (descriptor.reason or "")

    def test_unavailable_when_checkpoint_path_does_not_exist(self, tmp_path):
        settings = _settings(tmp_path, model_path=tmp_path / "missing.pt")
        descriptor = TrainedModelProvider(settings).describe()
        assert descriptor.available is False
        assert "does not exist" in (descriptor.reason or "")

    def test_ensure_available_raises_a_typed_error(self, tmp_path):
        with pytest.raises(PredictionProviderError) as excinfo:
            TrainedModelProvider(_settings(tmp_path, model_path=None)).ensure_available()
        assert "MODEL_PATH" in str(excinfo.value)

    def test_never_falls_back_to_synthetic(self, tmp_path):
        """A missing checkpoint must not silently produce synthetic output."""
        provider = TrainedModelProvider(_settings(tmp_path, model_path=None))
        with pytest.raises(PredictionProviderError):
            provider.ensure_available()
        assert provider.synthetic is False


class TestProviderSelectionIsExplicit:
    def test_default_is_the_trained_model(self, tmp_path):
        assert resolve_provider_id(_settings(tmp_path)) == PROVIDER_TRAINED_MODEL

    def test_unknown_provider_is_rejected(self, tmp_path):
        with pytest.raises(PredictionProviderError):
            resolve_provider_id(_settings(tmp_path), "not_a_provider")

    def test_synthetic_requires_development_or_test(self, tmp_path):
        settings = _settings(tmp_path, environment="staging")
        with pytest.raises(PredictionProviderError) as excinfo:
            resolve_provider_id(settings, PROVIDER_SYNTHETIC)
        assert "local configuration" in str(excinfo.value)

    def test_synthetic_allowed_in_development(self, tmp_path):
        settings = _settings(tmp_path, environment="development")
        assert resolve_provider_id(settings, PROVIDER_SYNTHETIC) == PROVIDER_SYNTHETIC

    def test_production_refuses_to_start_with_synthetic(self, tmp_path):
        with pytest.raises(ValueError) as excinfo:
            _settings(tmp_path, environment="production", prediction_provider=PROVIDER_SYNTHETIC)
        assert "synthetic_test_data" in str(excinfo.value)

    def test_production_can_opt_in_explicitly(self, tmp_path):
        settings = _settings(
            tmp_path,
            environment="production",
            prediction_provider=PROVIDER_SYNTHETIC,
            allow_synthetic_provider_in_production=True,
        )
        assert settings.synthetic_provider_selected is True

    def test_invalid_provider_setting_is_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            _settings(tmp_path, prediction_provider="guess")

    def test_invalid_pattern_setting_is_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            _settings(tmp_path, synthetic_pattern="swirl")

    def test_factory_returns_matching_provider(self, tmp_path):
        settings = _settings(tmp_path, prediction_provider=PROVIDER_SYNTHETIC)
        provider = get_prediction_provider(settings)
        assert isinstance(provider, SyntheticTestProvider)
        assert provider.synthetic is True

    def test_factory_reuses_one_instance(self, tmp_path):
        settings = _settings(tmp_path, prediction_provider=PROVIDER_SYNTHETIC)
        assert get_prediction_provider(settings) is get_prediction_provider(settings)


class TestPredictionValidation:
    def _prediction(self, tmp_path, **overrides):
        image, valid = _raster()
        provider = SyntheticTestProvider(_settings(tmp_path))
        prediction = provider.predict(image, TRANSFORM, UTM, valid)
        for key, value in overrides.items():
            setattr(prediction, key, value)
        return prediction

    def test_accepts_a_well_formed_prediction(self, tmp_path):
        prediction = self._prediction(tmp_path)
        validate_prediction(
            prediction,
            expected_shape=prediction.valid_mask.shape,
            expected_transform=TRANSFORM,
            expected_crs=UTM,
        )

    def test_rejects_shape_mismatch(self, tmp_path):
        prediction = self._prediction(tmp_path)
        with pytest.raises(PredictionProviderError):
            validate_prediction(
                prediction,
                expected_shape=(prediction.mask.shape[0] + 1, prediction.mask.shape[1]),
                expected_transform=TRANSFORM,
                expected_crs=UTM,
            )

    def test_rejects_transform_mismatch(self, tmp_path):
        prediction = self._prediction(tmp_path)
        with pytest.raises(PredictionProviderError):
            validate_prediction(
                prediction,
                expected_shape=prediction.valid_mask.shape,
                expected_transform=Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0),
                expected_crs=UTM,
            )

    def test_rejects_crs_mismatch(self, tmp_path):
        prediction = self._prediction(tmp_path)
        with pytest.raises(PredictionProviderError):
            validate_prediction(
                prediction,
                expected_shape=prediction.valid_mask.shape,
                expected_transform=TRANSFORM,
                expected_crs=CRS.from_epsg(4326),
            )

    def test_rejects_unexpected_class_values(self, tmp_path):
        prediction = self._prediction(tmp_path)
        prediction.mask = prediction.mask.copy()
        prediction.mask[0, 0] = 7
        with pytest.raises(PredictionProviderError):
            validate_prediction(
                prediction,
                expected_shape=prediction.valid_mask.shape,
                expected_transform=TRANSFORM,
                expected_crs=UTM,
            )

    def test_rejects_missing_georeferencing(self, tmp_path):
        prediction = self._prediction(tmp_path)
        prediction.transform = None
        with pytest.raises(PredictionProviderError):
            validate_prediction(
                prediction,
                expected_shape=prediction.valid_mask.shape,
                expected_transform=TRANSFORM,
                expected_crs=UTM,
            )