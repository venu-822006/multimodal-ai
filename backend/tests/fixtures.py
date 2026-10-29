from __future__ import annotations

import numpy as np
import pytest
import rasterio
from affine import Affine
from pyproj import CRS

from app.processing.validators import AreaOfInterest
from app.services.geospatial_service import TargetGrid

UTM_CRS = CRS.from_epsg(32645)
# UTM 45N origin that covers the AOI used by these fixtures (lon 85.05-85.45, lat 27.75-28.05).
ORIGIN = (308339.0, 3104275.0)
AOI_ORIGIN = (312000.0, 3099000.0)


@pytest.fixture()
def grid_10m() -> TargetGrid:
    """A 100x100 grid at 10 m in UTM 45N inside the AOI; each pixel is 100 m^2."""
    transform = Affine(10.0, 0.0, AOI_ORIGIN[0], 0.0, -10.0, AOI_ORIGIN[1])
    return TargetGrid(crs=UTM_CRS, transform=transform, width=100, height=100, pixel_size=10.0)


@pytest.fixture()
def aoi() -> AreaOfInterest:
    return AreaOfInterest(
        min_lon=85.10, min_lat=27.80, max_lon=85.40, max_lat=28.00, area_m2=1_000_000_000.0
    )


def write_geotiff(path, array, *, crs=UTM_CRS, transform=None, nodata=None, count=1):
    transform = transform or Affine(10.0, 0.0, ORIGIN[0], 0.0, -10.0, ORIGIN[1])
    if array.ndim == 2:
        array = array[None, ...]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[1],
        width=array.shape[2],
        count=count,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dataset:
        for index in range(array.shape[0]):
            dataset.write(array[index].astype("float32"), index + 1)
    return path


@pytest.fixture()
def synthetic_db_raster(tmp_path):
    """A deterministic two band dB raster with a dark (water-like) region."""
    rng = np.random.default_rng(11)
    height, width = 300, 300
    vv = rng.normal(-13.0, 3.0, (height, width)).astype("float32")
    vh = rng.normal(-19.0, 2.5, (height, width)).astype("float32")
    vv[80:220, 100:300] -= 9.0
    vh[80:220, 100:300] -= 11.0
    path = tmp_path / "prepared.tif"
    from app.services.preprocessing_service import write_model_input

    transform = Affine(10.0, 0.0, ORIGIN[0], 0.0, -10.0, ORIGIN[1])
    grid = TargetGrid(
        crs=UTM_CRS, transform=transform, width=width, height=height, pixel_size=10.0
    )
    write_model_input(path, vv, vh, grid)
    return path, grid, vv, vh


@pytest.fixture()
def local_scene(tmp_path):
    """A SceneAsset backed by a local GeoTIFF so tests never hit the network."""
    from datetime import datetime, timezone

    from app.services.imagery_service import SceneAsset

    rng = np.random.default_rng(5)
    # Covers the whole AOI at 20 m so an AOI-derived target grid always overlaps.
    height, width = 1200, 1600
    linear = rng.gamma(2.0, 0.03, (height, width)).astype("float32")
    linear[300:800, 400:1200] *= 0.02
    transform = Affine(20.0, 0.0, AOI_ORIGIN[0], 0.0, -20.0, AOI_ORIGIN[1])
    path = write_geotiff(
        tmp_path / "s1_vv.tif", linear, transform=transform, nodata=-32768.0
    )
    return SceneAsset(
        source="sentinel-1-rtc",
        collection="sentinel-1-rtc",
        item_id="TEST_S1_ITEM",
        platform="S1A",
        acquisition=datetime(2024, 9, 20, tzinfo=timezone.utc),
        bbox=[85.10, 27.80, 85.40, 28.00],
        hrefs={"vv": str(path), "vh": str(path)},
        resolution_m=10.0,
        nodata_values=(-32768.0,),
        properties={},
    )