from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(BACKEND_ROOT / ".env", BACKEND_ROOT.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "FloodSight AI API"
    app_version: str = "1.0.0"
    environment: str = "development"
    debug: bool = False
    api_prefix: str = "/api"

    database_url: str = "postgresql+psycopg://floodsight:floodsight@127.0.0.1:5432/floodsight"
    database_pool_size: int = 5
    database_max_overflow: int = 10
    database_pool_timeout_seconds: int = 30
    database_pool_recycle_seconds: int = 1800

    cors_allow_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:4173",
            "http://127.0.0.1:4173",
        ]
    )
    cors_allow_credentials: bool = False

    # ------------------------------------------------------------ authentication
    # Server-side sessions: the browser holds an opaque, signed-into-database token
    # in an HttpOnly cookie. Only a SHA-256 hash of the token is persisted.
    auth_session_cookie_name: str = "floodsight_session"
    auth_session_ttl_seconds: int = 60 * 60 * 24 * 7
    auth_session_cookie_secure: bool = False
    auth_session_cookie_samesite: str = "lax"
    auth_session_cookie_domain: str | None = None
    auth_password_min_length: int = 8
    auth_password_max_length: int = 128
    auth_login_max_attempts: int = 5
    auth_login_lockout_seconds: int = 900
    auth_login_touch_interval_seconds: int = 60

    log_level: str = "INFO"
    log_format: str = "json"
    log_request_body: bool = False

    artifact_root: Path = BACKEND_ROOT / "var" / "artifacts"
    artifact_base_url_prefix: str = "/api/artifacts"
    max_artifact_bytes: int = 64 * 1024 * 1024

    max_bbox_area_km2: float = 5000.0
    min_bbox_area_km2: float = 0.05
    max_bbox_dimension_deg: float = 5.0
    max_flood_lookback_days: int = 120
    max_flood_lookahead_days: int = 120
    default_lookback_days: int = 45
    default_lookahead_days: int = 30
    max_name_length: int = 200
    max_description_length: int = 2000
    max_queued_analyses: int = 25
    max_active_analyses: int = 10
    default_page_size: int = 20
    max_page_size: int = 100

    stac_endpoint: str = "https://planetarycomputer.microsoft.com/api/stac/v1"
    stac_timeout_seconds: float = 30.0
    imagery_max_candidates: int = 40
    imagery_connect_timeout_seconds: float = 20.0
    imagery_read_timeout_seconds: float = 60.0
    imagery_user_agent: str = "FloodSight-AI/1.0 (+https://github.com/kushan2210-tech/FloodSight-AI)"
    raster_target_resolution_m: float = 20.0
    raster_max_pixels: int = 25_000_000
    min_flood_polygon_area_m2: float = 5000.0
    flood_min_component_pixels: int = 12

    osm_enabled: bool = True
    # overpass-api.de is the primary because it is the most reliable public mirror.
    # Several mirrors answer HTTP 406 to browser-like User-Agents, so the plain
    # application identifier below is deliberate and must not be replaced with one.
    osm_endpoint: str = "https://overpass-api.de/api/interpreter"
    osm_fallback_endpoints: list[str] = Field(
        default_factory=lambda: [
            "https://overpass.kumi.systems/api/interpreter",
            "https://overpass.private.coffee/api/interpreter",
        ]
    )
    osm_timeout_seconds: float = 45.0
    osm_max_elements: int = 6000

    model_repo_path: Path = BACKEND_ROOT.parent / "multimodal-ai"
    model_path: Path | None = None
    model_device: str = "auto"
    model_batch_size: int = 1
    model_patch_size: int | None = None
    model_stride: int | None = None
    model_temperature: float = 1.0
    model_confidence_threshold: float = 0.5
    model_min_cluster_px: int = 0
    model_require_checkpoint: bool = True
    model_temporal_filter: bool = True

    # ------------------------------------------------------------ prediction provider
    # Selects what produces the flood prediction mask. The provider is never chosen
    # implicitly: 'synthetic_test' must be set here, or per analysis request in a
    # non-production environment, and it is never used as a fallback.
    prediction_provider: str = "trained_model"
    # Synthetic provider knobs. They only affect the synthetic test provider.
    synthetic_pattern: str = "blocky"
    synthetic_seed: int = 20260921
    synthetic_coverage: float = 0.06
    synthetic_min_blob_px: int = 24
    synthetic_apply_temporal_filter: bool = False
    # Escape hatch for staging environments that must never serve synthetic output.
    allow_synthetic_provider_in_production: bool = False

    worker_poll_interval_seconds: float = 2.0
    worker_idle_sleep_seconds: float = 1.0
    worker_max_attempts: int = 3
    worker_retry_base_seconds: float = 10.0
    worker_retry_max_seconds: float = 300.0
    worker_stale_after_seconds: int = 900
    worker_batch_size: int = 1
    worker_shutdown_grace_seconds: float = 20.0

    report_page_size: str = "A4"
    report_public_base_url: str | None = None

    @field_validator("environment")
    @classmethod
    def _normalise_environment(cls, value: str) -> str:
        allowed = {"development", "test", "staging", "production"}
        normalised = value.strip().lower()
        if normalised not in allowed:
            raise ValueError(f"environment must be one of {sorted(allowed)}")
        return normalised

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("osm_fallback_endpoints", mode="before")
    @classmethod
    def _split_endpoints(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("artifact_root", "model_repo_path")
    @classmethod
    def _resolve_artifact_root(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    @field_validator("auth_session_cookie_samesite")
    @classmethod
    def _validate_samesite(cls, value: str) -> str:
        normalised = value.strip().lower()
        if normalised not in {"lax", "strict", "none"}:
            raise ValueError("auth_session_cookie_samesite must be one of lax, strict, none")
        return normalised

    @field_validator("auth_session_cookie_domain", mode="before")
    @classmethod
    def _blank_domain_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("model_path", mode="before")
    @classmethod
    def _blank_model_path_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        if isinstance(value, Path):
            return value.expanduser().resolve()
        return value

    @field_validator("model_device")
    @classmethod
    def _validate_device(cls, value: str) -> str:
        normalised = value.strip().lower()
        if normalised not in {"auto", "cpu", "cuda"}:
            raise ValueError("model_device must be one of auto, cpu, cuda")
        return normalised

    @field_validator("model_temperature")
    @classmethod
    def _validate_temperature(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("model_temperature must be positive")
        return value

    @field_validator("model_confidence_threshold")
    @classmethod
    def _validate_threshold(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("model_confidence_threshold must be between 0 and 1")
        return value

    @field_validator("model_min_cluster_px", "model_batch_size")
    @classmethod
    def _validate_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("this setting cannot be negative")
        return value

    @field_validator("prediction_provider")
    @classmethod
    def _validate_prediction_provider(cls, value: str) -> str:
        from app.services.prediction_providers import PROVIDER_CHOICES

        normalised = value.strip().lower()
        if normalised not in PROVIDER_CHOICES:
            raise ValueError(f"prediction_provider must be one of {sorted(PROVIDER_CHOICES)}")
        return normalised

    @field_validator("synthetic_pattern")
    @classmethod
    def _validate_synthetic_pattern(cls, value: str) -> str:
        from app.services.prediction_providers import SYNTHETIC_PATTERN_CHOICES

        normalised = value.strip().lower()
        if normalised not in SYNTHETIC_PATTERN_CHOICES:
            raise ValueError(
                f"synthetic_pattern must be one of {sorted(SYNTHETIC_PATTERN_CHOICES)}"
            )
        return normalised

    @field_validator("synthetic_coverage")
    @classmethod
    def _validate_synthetic_coverage(cls, value: float) -> float:
        if not 0.0 < value <= 1.0:
            raise ValueError("synthetic_coverage must be in (0, 1]")
        return value

    @field_validator("synthetic_seed", "synthetic_min_blob_px")
    @classmethod
    def _validate_synthetic_positive(cls, value: int) -> int:
        if value < 0:
            raise ValueError("this setting cannot be negative")
        return value

    @model_validator(mode="after")
    def _check_synthetic_provider_guard(self) -> "Settings":
        """Refuse to start a production server that would emit synthetic predictions."""
        from app.services.prediction_providers import PROVIDER_SYNTHETIC

        if (
            self.environment == "production"
            and self.prediction_provider == PROVIDER_SYNTHETIC
            and not self.allow_synthetic_provider_in_production
        ):
            raise ValueError(
                "prediction_provider 'synthetic_test' emits synthetic_test_data, not flood "
                "predictions. It is not permitted in production unless "
                "allow_synthetic_provider_in_production is explicitly enabled."
            )
        return self

    @model_validator(mode="after")
    def _check_security_invariants(self) -> "Settings":
        if self.environment == "production":
            if "*" in self.cors_allow_origins:
                raise ValueError("cors_allow_origins must not contain '*' in production")
            if self.debug:
                raise ValueError("debug must be disabled in production")
            # Session cookies must be HTTPS-only in production. Force rather than
            # fail so a misconfigured deployment cannot silently ship an insecure cookie.
            self.auth_session_cookie_secure = True
        if self.auth_session_cookie_samesite == "none" and not self.auth_session_cookie_secure:
            raise ValueError(
                "auth_session_cookie_samesite 'none' requires auth_session_cookie_secure"
            )
        if self.auth_password_min_length < 8:
            raise ValueError("auth_password_min_length must be at least 8")
        if self.auth_password_max_length < self.auth_password_min_length:
            raise ValueError("auth_password_max_length must be >= auth_password_min_length")
        if self.auth_login_max_attempts < 1:
            raise ValueError("auth_login_max_attempts must be at least 1")
        if self.database_pool_size < 1:
            raise ValueError("database_pool_size must be at least 1")
        if not (0 < self.default_page_size <= self.max_page_size):
            raise ValueError("default_page_size must be within (0, max_page_size]")
        return self

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def allows_per_request_prediction_provider(self) -> bool:
        """Whether a single request may override the configured provider.

        Synthetic output is a local development affordance, so the override only
        exists outside production.
        """
        return self.environment in ("development", "test")

    @property
    def synthetic_provider_selected(self) -> bool:
        from app.services.prediction_providers import PROVIDER_SYNTHETIC

        return self.prediction_provider == PROVIDER_SYNTHETIC

    @property
    def database_url_for_alembic(self) -> str:
        return self.database_url.replace("postgresql+psycopg://", "postgresql+psycopg://", 1)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()