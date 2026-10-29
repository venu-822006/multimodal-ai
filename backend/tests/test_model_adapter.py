from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from affine import Affine
from pyproj import CRS

from app.services.model_service import (
    FloodSegmentationModel,
    ModelUnavailableError,
    get_segmentation_model,
)

UTM = CRS.from_epsg(32645)
REPO = None


@pytest.fixture()
def repo_path():
    from pathlib import Path

    return Path(__file__).resolve().parents[2] / "multimodal-ai"


@pytest.fixture()
def model_repo(repo_path):
    if not (repo_path / "model.py").is_file():
        pytest.skip("The model repository is not available in this checkout.")
    return repo_path


@pytest.fixture()
def checkpoint(model_repo, tmp_path):
    """A checkpoint written in the exact format train.py produces."""
    import sys

    sys.path.insert(0, str(model_repo))
    from model import create_unet_model

    model = create_unet_model(in_channels=2, classes=2, base_channels=24)
    path = tmp_path / "best.pt"
    torch.save(
        {
            "epoch": 7,
            "model_config": {"in_channels": 2, "classes": 2, "base_channels": 24},
            "model_state_dict": model.state_dict(),
            "normalization": {"mean_db": [-12.5, -18.0], "std_db": [4.5, 4.0]},
            "patch_size": 256,
            "val_flood_iou": 0.8123,
        },
        path,
    )
    return path


@pytest.fixture()
def settings_factory(model_repo, tmp_path):
    from app.core.config import Settings

    def build(**overrides):
        base = {
            "environment": "test",
            "database_url": "postgresql+psycopg://unused:unused@localhost/unused",
            "artifact_root": tmp_path / "artifacts",
            "model_repo_path": model_repo,
            "model_path": None,
            "model_device": "cpu",
            "log_format": "text",
            "log_level": "ERROR",
        }
        base.update(overrides)
        return Settings(**base)

    return build


def _grid(width=300, height=300, pixel=10.0):
    from app.services.geospatial_service import TargetGrid

    transform = Affine(pixel, 0.0, 500000.0, 0.0, -pixel, 4000000.0)
    return TargetGrid(crs=UTM, transform=transform, width=width, height=height, pixel_size=pixel)


class TestCheckpointLoading:
    def test_loads_architecture_from_repository(self, checkpoint, settings_factory):
        model = FloodSegmentationModel(settings_factory(model_path=checkpoint))
        model.load()
        assert model._model.__class__.__name__ == "FloodUNet"
        params = sum(p.numel() for p in model._model.parameters())
        assert params == 1_084_490

    def test_loads_once_per_process(self, checkpoint, settings_factory):
        model = FloodSegmentationModel(settings_factory(model_path=checkpoint))
        model.load()
        first = model._model
        model.load()
        assert model._model is first

    def test_registry_returns_same_instance(self, checkpoint, settings_factory):
        from app.services.model_service import reset_model_registry

        settings = settings_factory(model_path=checkpoint)
        reset_model_registry()
        assert get_segmentation_model(settings) is get_segmentation_model(settings)
        reset_model_registry()

    def test_reads_normalisation_from_checkpoint(self, checkpoint, settings_factory):
        model = FloodSegmentationModel(settings_factory(model_path=checkpoint))
        model.load()
        assert np.allclose(model._mean, [-12.5, -18.0])
        assert np.allclose(model._std, [4.5, 4.0])

    def test_reads_patch_size_from_checkpoint(self, checkpoint, settings_factory):
        model = FloodSegmentationModel(settings_factory(model_path=checkpoint))
        model.load()
        assert model._patch_size == 256
        assert model._stride == 128

    def test_eval_mode_and_no_grad(self, checkpoint, settings_factory):
        model = FloodSegmentationModel(settings_factory(model_path=checkpoint))
        model.load()
        assert model._model.training is False
        assert all(not p.requires_grad for p in model._model.parameters())

    def test_device_is_cpu_when_forced(self, checkpoint, settings_factory):
        descriptor = FloodSegmentationModel(
            settings_factory(model_path=checkpoint, model_device="cpu")
        ).describe()
        assert descriptor.available is True
        assert descriptor.device == "cpu"


class TestFailurePaths:
    def test_missing_checkpoint(self, settings_factory, tmp_path):
        model = FloodSegmentationModel(settings_factory(model_path=tmp_path / "absent.pt"))
        assert model.is_available() is False
        assert "does not exist" in (model.describe().reason or "")

    def test_unconfigured_checkpoint(self, settings_factory):
        model = FloodSegmentationModel(settings_factory(model_path=None))
        assert model.is_available() is False
        assert "No trained checkpoint is configured" in (model.describe().reason or "")

    def test_missing_model_repository(self, checkpoint, settings_factory, tmp_path):
        model = FloodSegmentationModel(
            settings_factory(model_path=checkpoint, model_repo_path=tmp_path / "nowhere")
        )
        assert model.is_available() is False
        assert "model.py" in (model.describe().reason or "")

    def test_incompatible_weights(self, model_repo, settings_factory, tmp_path):
        path = tmp_path / "bad.pt"
        torch.save(
            {
                "model_state_dict": {"enc1.layers.0.weight": torch.zeros(3, 3, 3, 3)},
                "model_config": {"in_channels": 2, "classes": 2, "base_channels": 24},
            },
            path,
        )
        model = FloodSegmentationModel(settings_factory(model_path=path))
        assert model.is_available() is False
        assert "incompatible" in (model.describe().reason or "").lower()

    def test_wrong_input_channel_count(self, model_repo, settings_factory, tmp_path):
        import sys

        sys.path.insert(0, str(model_repo))
        from model import create_unet_model

        path = tmp_path / "c3.pt"
        torch.save(
            {
                "model_state_dict": create_unet_model(in_channels=3).state_dict(),
                "model_config": {"in_channels": 3, "classes": 2, "base_channels": 24},
                "normalization": {"mean_db": [-12.5, -18.0], "std_db": [4.5, 4.0]},
                "patch_size": 256,
            },
            path,
        )
        model = FloodSegmentationModel(settings_factory(model_path=path))
        assert model.is_available() is False
        assert "input channels" in (model.describe().reason or "")

    def test_not_a_floodsight_checkpoint(self, settings_factory, tmp_path):
        path = tmp_path / "random.pt"
        torch.save({"state_dict": {}}, path)
        model = FloodSegmentationModel(settings_factory(model_path=path))
        assert model.is_available() is False
        assert "model_state_dict" in (model.describe().reason or "")

    def test_malformed_normalisation(self, model_repo, settings_factory, tmp_path):
        import sys

        sys.path.insert(0, str(model_repo))
        from model import create_unet_model

        path = tmp_path / "badnorm.pt"
        torch.save(
            {
                "model_state_dict": create_unet_model().state_dict(),
                "model_config": {"in_channels": 2, "classes": 2, "base_channels": 24},
                "normalization": {"mean_db": [1.0], "std_db": [1.0]},
                "patch_size": 256,
            },
            path,
        )
        assert FloodSegmentationModel(settings_factory(model_path=path)).is_available() is False

    def test_cuda_request_without_cuda(self, checkpoint, settings_factory):
        if torch.cuda.is_available():
            pytest.skip("CUDA is present, so the unavailable branch cannot be exercised.")
        model = FloodSegmentationModel(
            settings_factory(model_path=checkpoint, model_device="cuda")
        )
        assert model.is_available() is False
        assert "cuda" in (model.describe().reason or "").lower()

    def test_errors_are_sanitised(self, settings_factory, tmp_path):
        model = FloodSegmentationModel(settings_factory(model_path=tmp_path / "x.pt"))
        with pytest.raises(ModelUnavailableError):
            model.load()


class TestPreprocessing:
    def test_clip_then_zscore(self, checkpoint, settings_factory):
        """Values beyond mean +/- 3 sigma are clipped before normalisation."""
        model = FloodSegmentationModel(settings_factory(model_path=checkpoint))
        model.load()
        mean, std = model._mean, model._std
        image = np.full((4, 4, 2), 60.0, dtype=np.float32)
        normalised = model.normalise(image)
        # Far above the upper bound, so both channels clip to exactly +3 sigma.
        assert np.allclose(normalised[:, :, 0], 3.0, atol=1e-3)
        assert np.allclose(normalised[:, :, 1], 3.0, atol=1e-3)
        image = np.full((4, 4, 2), -90.0, dtype=np.float32)
        normalised = model.normalise(image)
        assert np.allclose(normalised[:, :, 0], -3.0, atol=1e-3)
        assert np.allclose(normalised[:, :, 1], -3.0, atol=1e-3)

    def test_values_inside_bounds_are_not_clipped(self, checkpoint, settings_factory):
        model = FloodSegmentationModel(settings_factory(model_path=checkpoint))
        model.load()
        # 0 dB sits inside the VV range of -26 dB to +1 dB, so it is preserved.
        image = np.zeros((4, 4, 2), dtype=np.float32)
        normalised = model.normalise(image)
        assert np.allclose(normalised[:, :, 0], 12.5 / 4.5, atol=1e-3)

    def test_within_range_values_are_centred(self, checkpoint, settings_factory):
        model = FloodSegmentationModel(settings_factory(model_path=checkpoint))
        model.load()
        image = np.tile(np.array([-12.5, -18.0], dtype=np.float32), (4, 4, 1))
        assert np.allclose(model.normalise(image), 0.0, atol=1e-4)


class TestInference:
    def test_output_matches_input_shape(self, checkpoint, settings_factory, synthetic_db_raster):
        from app.services.preprocessing_service import read_prepared_scene

        path, grid, _, _ = synthetic_db_raster
        image, valid, transform, crs = read_prepared_scene(path)
        model = FloodSegmentationModel(settings_factory(model_path=checkpoint))
        prediction = model.predict(image, transform, crs, valid)
        assert prediction.mask.shape == image.shape[:2]
        assert prediction.probabilities.shape == (2,) + image.shape[:2]

    def test_georeferencing_preserved(self, checkpoint, settings_factory, synthetic_db_raster):
        from app.services.preprocessing_service import read_prepared_scene

        path, _, _, _ = synthetic_db_raster
        image, valid, transform, crs = read_prepared_scene(path)
        model = FloodSegmentationModel(settings_factory(model_path=checkpoint))
        prediction = model.predict(image, transform, crs, valid)
        assert prediction.transform == transform
        assert prediction.crs == crs

    def test_probabilities_are_normalised(self, checkpoint, settings_factory, synthetic_db_raster):
        from app.services.preprocessing_service import read_prepared_scene

        path, _, _, _ = synthetic_db_raster
        image, valid, transform, crs = read_prepared_scene(path)
        model = FloodSegmentationModel(settings_factory(model_path=checkpoint))
        prediction = model.predict(image, transform, crs, valid)
        total = prediction.probabilities[0] + prediction.probabilities[1]
        assert np.allclose(total, 1.0, atol=1e-4)

    def test_nodata_preserved(self, checkpoint, settings_factory, synthetic_db_raster):
        from app.services.preprocessing_service import read_prepared_scene

        path, _, _, _ = synthetic_db_raster
        image, valid, transform, crs = read_prepared_scene(path)
        valid = valid.copy()
        valid[:50, :50] = False
        model = FloodSegmentationModel(settings_factory(model_path=checkpoint))
        prediction = model.predict(image, transform, crs, valid)
        assert np.all(prediction.mask[:50, :50] == 255)

    def test_small_raster_is_padded_not_dropped(self, checkpoint, settings_factory):
        rng = np.random.default_rng(2)
        image = rng.normal(-13.0, 2.0, (40, 60, 2)).astype(np.float32)
        valid = np.ones((40, 60), dtype=bool)
        grid = _grid(width=60, height=40)
        model = FloodSegmentationModel(settings_factory(model_path=checkpoint))
        prediction = model.predict(image, grid.transform, grid.crs, valid)
        assert prediction.mask.shape == (40, 60)

    def test_edge_tiles_cover_full_extent(self, checkpoint, settings_factory):
        rng = np.random.default_rng(4)
        image = rng.normal(-13.0, 2.0, (300, 300, 2)).astype(np.float32)
        valid = np.ones((300, 300), dtype=bool)
        grid = _grid()
        model = FloodSegmentationModel(settings_factory(model_path=checkpoint))
        prediction = model.predict(image, grid.transform, grid.crs, valid)
        # A 300 px scene at 256 patch / 128 stride needs 2x2 tiles to cover the edge.
        assert prediction.inference["tiles"] == 4

    def test_temporal_filter_applied_when_pre_image_given(
        self, checkpoint, settings_factory, synthetic_db_raster
    ):
        from app.services.preprocessing_service import read_prepared_scene

        path, _, _, _ = synthetic_db_raster
        image, valid, transform, crs = read_prepared_scene(path)
        model = FloodSegmentationModel(settings_factory(model_path=checkpoint))
        prediction = model.predict(
            image, transform, crs, valid, pre_image_db=image.copy()
        )
        assert prediction.inference["temporal_filter_applied"] is True
        # Nothing changed, so the temporal filter removes every candidate.
        assert int((prediction.mask == 1).sum()) == 0

    def test_mismatched_pre_image_rejected(self, checkpoint, settings_factory, synthetic_db_raster):
        from app.services.preprocessing_service import read_prepared_scene

        from app.core.exceptions import PermanentProcessingError

        path, _, _, _ = synthetic_db_raster
        image, valid, transform, crs = read_prepared_scene(path)
        model = FloodSegmentationModel(settings_factory(model_path=checkpoint))
        with pytest.raises(PermanentProcessingError):
            model.predict(image, transform, crs, valid, pre_image_db=image[:, :10, :])

    def test_wrong_band_count_rejected(self, checkpoint, settings_factory):
        from app.core.exceptions import PermanentProcessingError

        image = np.zeros((64, 64, 3), dtype=np.float32)
        model = FloodSegmentationModel(settings_factory(model_path=checkpoint))
        with pytest.raises(PermanentProcessingError):
            model.predict(image, None, UTM, np.ones((64, 64), dtype=bool))

    def test_single_band_pre_image_accepted(self, checkpoint, settings_factory, synthetic_db_raster):
        """The pipeline passes a single VV band, so a (H, W, 1) array must be accepted."""
        from app.services.preprocessing_service import read_prepared_scene

        path, _, _, _ = synthetic_db_raster
        image, valid, transform, crs = read_prepared_scene(path)
        model = FloodSegmentationModel(settings_factory(model_path=checkpoint))
        prediction = model.predict(
            image, transform, crs, valid, pre_image_db=image[:, :, :1].copy()
        )
        assert prediction.mask.shape == image.shape[:2]
        assert prediction.inference["temporal_filter_applied"] is True

    def test_two_band_pre_image_accepted(self, checkpoint, settings_factory, synthetic_db_raster):
        from app.services.preprocessing_service import read_prepared_scene

        path, _, _, _ = synthetic_db_raster
        image, valid, transform, crs = read_prepared_scene(path)
        model = FloodSegmentationModel(settings_factory(model_path=checkpoint))
        prediction = model.predict(image, transform, crs, valid, pre_image_db=image.copy())
        assert prediction.inference["temporal_filter_applied"] is True

    def test_confidence_threshold_respected(self, checkpoint, settings_factory, synthetic_db_raster):
        from app.services.preprocessing_service import read_prepared_scene

        path, _, _, _ = synthetic_db_raster
        image, valid, transform, crs = read_prepared_scene(path)
        strict = FloodSegmentationModel(
            settings_factory(model_path=checkpoint, model_confidence_threshold=0.999)
        )
        prediction = strict.predict(image, transform, crs, valid)
        assert int((prediction.mask == 1).sum()) <= int(
            (
                FloodSegmentationModel(
                    settings_factory(model_path=checkpoint, model_confidence_threshold=0.0)
                )
                .predict(image, transform, crs, valid)
                .mask
                == 1
            ).sum()
        )

    def test_inference_is_deterministic(self, checkpoint, settings_factory, synthetic_db_raster):
        from app.services.preprocessing_service import read_prepared_scene

        path, _, _, _ = synthetic_db_raster
        image, valid, transform, crs = read_prepared_scene(path)
        model = FloodSegmentationModel(settings_factory(model_path=checkpoint))
        first = model.predict(image, transform, crs, valid).mask
        second = model.predict(image, transform, crs, valid).mask
        assert np.array_equal(first, second)

    def test_cpu_inference_runs(self, checkpoint, settings_factory, synthetic_db_raster):
        from app.services.preprocessing_service import read_prepared_scene

        path, _, _, _ = synthetic_db_raster
        image, valid, transform, crs = read_prepared_scene(path)
        model = FloodSegmentationModel(settings_factory(model_path=checkpoint, model_device="cpu"))
        prediction = model.predict(image, transform, crs, valid)
        assert prediction.inference["device"] == "cpu"
        assert set(np.unique(prediction.mask)) <= {0, 1, 255}


class TestProvenance:
    def test_descriptor_reports_checkpoint_without_path(
        self, checkpoint, settings_factory, settings
    ):
        descriptor = FloodSegmentationModel(settings_factory(model_path=checkpoint)).describe()
        assert descriptor.available is True
        assert descriptor.checkpoint_name == "best.pt"
        assert "/" not in descriptor.checkpoint_identifier
        assert str(settings_factory(model_path=checkpoint).model_path) not in str(
            descriptor.checkpoint_identifier
        )

    def test_descriptor_reports_validation_iou(self, checkpoint, settings_factory):
        descriptor = FloodSegmentationModel(settings_factory(model_path=checkpoint)).describe()
        assert descriptor.val_flood_iou == pytest.approx(0.8123)
        assert descriptor.epoch == 7

    def test_descriptor_states_input_contract(self, checkpoint, settings_factory):
        descriptor = FloodSegmentationModel(settings_factory(model_path=checkpoint)).describe()
        assert descriptor.input_bands == ("VV", "VH")
        assert "decibel" in descriptor.input_units