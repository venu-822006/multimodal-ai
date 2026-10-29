from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from tests.fixtures import (  # noqa: F401  (re-exported fixtures)
    aoi,
    grid_10m,
    local_scene,
    synthetic_db_raster,
    write_geotiff,
)

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))


def _database_url() -> str:
    return os.getenv("TEST_DATABASE_URL") or os.getenv(
        "DATABASE_URL", "postgresql+psycopg://floodsight:floodsight@127.0.0.1:5432/floodsight"
    )


@pytest.fixture(scope="session")
def database_url() -> str:
    url = _database_url()
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 2})
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        engine.dispose()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"PostgreSQL is required for database tests: {type(exc).__name__}")
    return url


@pytest.fixture(scope="session")
def settings(tmp_path_factory):
    from app.core.config import Settings, reset_settings_cache

    artifacts = tmp_path_factory.mktemp("artifacts")
    reset_settings_cache()
    return Settings(
        environment="test",
        database_url=_database_url(),
        artifact_root=artifacts,
        model_repo_path=BACKEND_ROOT.parent / "multimodal-ai",
        model_path=None,
        osm_enabled=False,
        max_bbox_area_km2=100_000.0,
        max_bbox_dimension_deg=5.0,
        log_format="text",
        log_level="WARNING",
    )


@pytest.fixture()
def db_schema(database_url, settings):
    """Create the schema once per session and truncate between tests."""
    from sqlalchemy import create_engine, text

    from app.core.enums import TERMINAL_STATUSES  # noqa: F401
    from app.db.base import Base
    from app import models  # noqa: F401

    engine = create_engine(settings.database_url, pool_pre_ping=True)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield engine
    with engine.begin() as connection:
        for table in reversed(Base.metadata.sorted_tables):
            connection.execute(text(f"TRUNCATE {table.name} CASCADE"))
    engine.dispose()


@pytest.fixture()
def session(db_schema, settings):
    from sqlalchemy.orm import sessionmaker

    factory = sessionmaker(bind=db_schema, autoflush=False, expire_on_commit=False)
    with factory() as db_session:
        yield db_session


@pytest.fixture()
def client(settings, db_schema):
    from fastapi.testclient import TestClient

    from app.core.config import get_settings
    from app.db.session import dispose_engine
    from app.main import create_app

    get_settings.cache_clear()
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client
    dispose_engine()