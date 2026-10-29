from __future__ import annotations

import numpy as np
import pytest
import rasterio
from affine import Affine
from pyproj import CRS

from app.core.exceptions import PermanentProcessingError, UpstreamError
from app.processing.validators import (
    AreaOfInterest,
    geodesic_area_m2,
    validate_bbox,
)
from app.services.preprocessing_service import (
    clip_db,
    lee_sigma_filter,
    linear_to_db,
    prepare_model_input,
    read_prepared_scene,
)


class TestValidators:
    def test_accepts_ordered_bbox(self, settings):
        aoi = validate_bbox([[85.1, 27.8], [85.3, 28.0]], settings)
        assert aoi.min_lon == 85.1 and aoi.max_lat == 28.0
        assert aoi.area_m2 > 0

    @pytest.mark.parametrize(
        "bbox",
        [
            [[85.3, 27.8], [85.1, 28.0]],
            [[85.1, 28.0], [85.3, 27.8]],
            [[85.1, 27.8]],
            [[200.0, 27.8], [200.2, 28.0]],
            [[85.1, -95.0], [85.3, -94.0]],
        ],
    )
    def test_rejects_invalid(self, settings, bbox):
        from app.core.exceptions import DomainValidationError

        with pytest.raises(DomainValidationError):
            validate_bbox(bbox, settings)

    def test_zero_area_rejected(self, settings):
        from app.core.exceptions import DomainValidationError

        with pytest.raises(DomainValidationError):
            validate_bbox([[85.1, 27.8], [85.1, 27.9]], settings)

    def test_oversized_rejected(self, settings):
        from app.core.exceptions import DomainValidationError

        strict = settings.model_copy(update={"max_bbox_dimension_deg": 0.01})
        with pytest.raises(DomainValidationError):
            validate_bbox([[85.1, 27.8], [85.3, 28.0]], strict)


class TestGeodesicArea:
    def test_one_degree_square_at_equator(self):
        from shapely.geometry import box

        # A 1 degree cell at the equator is close to 111 km on a side.
        area = geodesic_area_m2(box(0.0, 0.0, 1.0, 1.0))
        assert 1.22e10 < area < 1.24e10

    def test_degrees_are_not_treated_as_metres(self):
        from shapely.geometry import box

        area = geodesic_area_m2(box(85.0, 27.0, 86.0, 28.0))
        # A naive planar reading of degrees would give ~1 m^2; the real area is ~1e10.
        assert area > 1e10

    def test_high_latitude_cell_is_smaller_than_equatorial(self):
        from shapely.geometry import box

        assert geodesic_area_m2(box(0, 0, 1, 1)) > geodesic_area_m2(box(60, 60, 61, 61))

    def test_multipolygon_sums_parts(self):
        from shapely.geometry import MultiPolygon, box

        combined = MultiPolygon([box(0, 0, 0.5, 0.5), box(1, 1, 1.5, 1.5)])
        single = box(0, 0, 0.5, 0.5)
        # Geodesic area is not exactly additive across widely separated parts on an
        # ellipsoid, so compare with a tolerance rather than exact equality.
        assert geodesic_area_m2(combined) == pytest.approx(
            geodesic_area_m2(single) * 2, rel=1e-3
        )


class TestPolygonize:
    def test_small_polygon_dropped_by_min_area(self, grid_10m):
        from app.services.geospatial_service import polygonize_mask

        mask = np.zeros((100, 100), dtype=bool)
        mask[10:14, 10:14] = True  # 16 px = 1600 m^2, below a 5000 m^2 minimum
        geometry, stats = polygonize_mask(mask, grid_10m, min_area_m2=5000.0)
        assert geometry is None
        assert stats["dropped_polygons"] == 1

    def test_large_polygon_retained(self, grid_10m):
        from app.services.geospatial_service import polygonize_mask

        mask = np.zeros((100, 100), dtype=bool)
        mask[10:70, 10:70] = True  # 3600 px = 360000 m^2
        geometry, stats = polygonize_mask(mask, grid_10m, min_area_m2=5000.0)
        assert geometry is not None
        assert stats["retained_polygons"] == 1

    def test_min_area_boundary_is_inclusive_at_scale(self, grid_10m):
        from app.services.geospatial_service import polygonize_mask

        mask = np.zeros((100, 100), dtype=bool)
        mask[10:17, 10:17] = True  # 49 px = 4900 m^2
        assert polygonize_mask(mask, grid_10m, min_area_m2=4900.0)[0] is not None
        assert polygonize_mask(mask, grid_10m, min_area_m2=5000.0)[0] is None

    def test_empty_mask_returns_none(self, grid_10m):
        from app.services.geospatial_service import polygonize_mask

        geometry, stats = polygonize_mask(
            np.zeros((100, 100), dtype=bool), grid_10m, min_area_m2=5000.0
        )
        assert geometry is None
        assert stats["retained_polygons"] == 0

    def test_geographic_grid_uses_geodesic_area(self):
        from app.services.geospatial_service import TargetGrid, polygonize_mask

        transform = Affine(0.001, 0.0, 85.0, 0.0, -0.001, 28.0)
        grid = TargetGrid(
            crs=CRS.from_epsg(4326), transform=transform, width=100, height=100, pixel_size=111.0
        )
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:70, 10:70] = True
        geometry, _ = polygonize_mask(mask, grid, min_area_m2=1.0)
        assert geometry is not None
        assert geometry.geom_type in {"Polygon", "MultiPolygon"}


class TestMeasurement:
    def test_percentage_uses_valid_area_when_supplied(self, grid_10m):
        from app.services.geospatial_service import measure_geometry, polygonize_mask

        mask = np.zeros((100, 100), dtype=bool)
        mask[10:70, 10:70] = True  # 360000 m^2
        geometry, _ = polygonize_mask(mask, grid_10m, min_area_m2=5000.0)
        result = measure_geometry(geometry, aoi_area_m2=1e9, valid_area_m2=1e6)
        assert result["flooded_area_m2"] == pytest.approx(360000, rel=0.01)
        assert result["flooded_percentage"] == pytest.approx(36.0, rel=0.01)
        assert result["percentage_denominator"] == "valid_analysis_area"

    def test_nodata_excluded_from_denominator(self, grid_10m):
        """Half the grid is nodata, so the percentage doubles against valid pixels."""
        from app.services.geospatial_service import measure_geometry, polygonize_mask

        mask = np.zeros((100, 100), dtype=bool)
        mask[10:70, 10:70] = True
        geometry, _ = polygonize_mask(mask, grid_10m, min_area_m2=5000.0)
        full = measure_geometry(geometry, aoi_area_m2=1e6)
        partial = measure_geometry(geometry, aoi_area_m2=1e6, valid_area_m2=5e5)
        assert partial["flooded_percentage"] == pytest.approx(
            full["flooded_percentage"] * 2, rel=0.02
        )

    def test_none_geometry_reports_null_not_zero(self):
        from app.services.geospatial_service import measure_geometry

        result = measure_geometry(None, aoi_area_m2=1e6)
        assert result["flooded_area_km2"] is None
        assert result["flooded_percentage"] is None


class TestPreprocessingStages:
    def test_linear_to_db(self):
        result = linear_to_db(np.array([[1.0, 10.0, 100.0]], dtype=np.float32))
        assert result[0, 0] == pytest.approx(0.0, abs=1e-4)
        assert result[0, 1] == pytest.approx(10.0, abs=1e-4)
        assert result[0, 2] == pytest.approx(20.0, abs=1e-4)

    def test_linear_to_db_handles_zero_without_inf(self):
        result = linear_to_db(np.array([[0.0]], dtype=np.float32))
        assert np.isfinite(result).all()

    def test_clip_db_enforces_training_range(self):
        result = clip_db(np.array([[-80.0, -10.0, 5.0]], dtype=np.float32))
        assert result[0, 0] == pytest.approx(-40.0)
        assert result[0, 1] == pytest.approx(-10.0)
        assert result[0, 2] == pytest.approx(0.0)

    def test_lee_sigma_reduces_variance(self):
        rng = np.random.default_rng(1)
        noisy = rng.normal(-13.0, 4.0, (64, 64)).astype(np.float32)
        filtered = lee_sigma_filter(noisy, size=3)
        assert filtered.std() < noisy.std()

    def test_lee_sigma_preserves_mean(self):
        rng = np.random.default_rng(1)
        noisy = rng.normal(-13.0, 4.0, (64, 64)).astype(np.float32)
        assert lee_sigma_filter(noisy).mean() == pytest.approx(noisy.mean(), rel=0.05)


class TestPrepareModelInput:
    def test_produces_two_band_db_geotiff(self, settings, local_scene, aoi, tmp_path):
        prepared = prepare_model_input(
            local_scene, aoi, settings, output_path=tmp_path / "prepared.tif"
        )
        with rasterio.open(prepared.path) as dataset:
            assert dataset.count == 2
            assert dataset.dtypes[0] == "float32"
            assert dataset.crs is not None
            assert dataset.nodata is not None
            band = dataset.read(1)
            finite = band[np.isfinite(band)]
            assert finite.size > 0
            # Values land in the dB range the model was trained on.
            assert finite.max() <= 0.5
            assert finite.min() >= -60.0

    def test_bands_are_vv_then_vh(self, settings, local_scene, aoi, tmp_path):
        prepared = prepare_model_input(
            local_scene, aoi, settings, output_path=tmp_path / "prepared.tif"
        )
        with rasterio.open(prepared.path) as dataset:
            assert dataset.tags(1)["band"] == "VV_dB"
            assert dataset.tags(2)["band"] == "VH_dB"

    def test_georeferencing_preserved(self, settings, local_scene, aoi, tmp_path, grid_10m):
        prepared = prepare_model_input(
            local_scene, aoi, settings, grid=grid_10m, output_path=tmp_path / "p.tif"
        )
        with rasterio.open(prepared.path) as dataset:
            assert dataset.crs == grid_10m.crs
            assert dataset.transform == grid_10m.transform
            assert (dataset.width, dataset.height) == (grid_10m.width, grid_10m.height)

    def test_alignment_to_target_grid(self, settings, local_scene, aoi, tmp_path, grid_10m):
        prepared = prepare_model_input(
            local_scene, aoi, settings, grid=grid_10m, output_path=tmp_path / "p.tif"
        )
        image, valid, transform, crs = read_prepared_scene(prepared.path)
        assert image.shape == (100, 100, 2)
        assert transform == grid_10m.transform
        assert crs == grid_10m.crs
        assert valid.shape == (100, 100)

    def test_reports_low_coverage_warning(self, settings, aoi, tmp_path):
        import rasterio

        from app.services.imagery_service import SceneAsset
        from datetime import datetime, timezone

        # A tiny raster far from the AOI, so almost everything is nodata.
        path = tmp_path / "far.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=20,
            width=20,
            count=1,
            dtype="float32",
            crs=CRS.from_epsg(32645),
            transform=Affine(10.0, 0, 800000.0, 0, -10.0, 3900000.0),
        ) as dataset:
            dataset.write(np.full((20, 20), 0.05, dtype="float32"), 1)
        scene = SceneAsset(
            source="sentinel-1-rtc",
            collection="sentinel-1-rtc",
            item_id="FAR",
            platform="S1A",
            acquisition=datetime(2024, 9, 20, tzinfo=timezone.utc),
            bbox=[0, 0, 1, 1],
            hrefs={"vv": str(path), "vh": str(path)},
            resolution_m=10.0,
            nodata_values=(),
        )
        from app.services.geospatial_service import TargetGrid

        grid = TargetGrid(
            crs=CRS.from_epsg(32645),
            transform=Affine(10.0, 0, 500000.0, 0, -10.0, 4000000.0),
            width=100,
            height=100,
            pixel_size=10.0,
        )
        with pytest.raises(PermanentProcessingError):
            prepare_model_input(scene, aoi, settings, grid=grid, output_path=tmp_path / "p.tif")

    def test_missing_band_rejected(self, settings, local_scene, aoi, tmp_path):
        from dataclasses import replace

        broken = replace(local_scene, hrefs={"vv": local_scene.hrefs["vv"]})
        with pytest.raises(PermanentProcessingError):
            prepare_model_input(broken, aoi, settings, output_path=tmp_path / "p.tif")

    def test_unreadable_source_raises_upstream(self, settings, local_scene, aoi, tmp_path):
        from dataclasses import replace

        broken = replace(local_scene, hrefs={"vv": str(tmp_path / "nope.tif"),
                                             "vh": str(tmp_path / "nope.tif")})
        with pytest.raises(UpstreamError):
            prepare_model_input(broken, aoi, settings, output_path=tmp_path / "p.tif")


class TestPreparedRead:
    def test_valid_mask_excludes_nodata(self, synthetic_db_raster):
        path, _, vv, vh = synthetic_db_raster
        image, valid, _, _ = read_prepared_scene(path)
        assert valid.all()
        assert image.shape[2] == 2
        assert np.isfinite(image).all()

    def test_nan_pixels_are_invalid(self, tmp_path, grid_10m):
        from app.services.preprocessing_service import write_model_input

        vv = np.full((100, 100), -13.0, dtype=np.float32)
        vh = np.full((100, 100), -19.0, dtype=np.float32)
        vv[0:10, 0:10] = np.nan
        path = tmp_path / "nan.tif"
        write_model_input(path, vv, vh, grid_10m)
        image, valid, _, _ = read_prepared_scene(path)
        assert not valid[:10, :10].any()
        assert valid[10:, 10:].all()