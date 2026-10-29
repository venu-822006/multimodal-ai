from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

Longitude = Annotated[float, Field(ge=-180.0, le=180.0, allow_inf_nan=False)]
Latitude = Annotated[float, Field(ge=-90.0, le=90.0, allow_inf_nan=False)]


class ApiModel(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        str_strip_whitespace=True,
        extra="forbid",
        ser_json_timedelta="iso8601",
    )


class CoordinatePair(ApiModel):
    model_config = ConfigDict(
        from_attributes=True, populate_by_name=True, extra="forbid", ser_json_inf_nan="null"
    )

    longitude: Longitude = Field(alias="lon", description="WGS84 longitude in decimal degrees")
    latitude: Latitude = Field(alias="lat", description="WGS84 latitude in decimal degrees")


class BoundingBox(ApiModel):
    model_config = ConfigDict(
        from_attributes=True, populate_by_name=True, extra="forbid", ser_json_inf_nan="null"
    )

    south_west: CoordinatePair = Field(
        alias="southWest", description="South-west corner of the area of interest"
    )
    north_east: CoordinatePair = Field(
        alias="northEast", description="North-east corner of the area of interest"
    )


class PaginationMeta(ApiModel):
    page: int = Field(ge=1)
    page_size: int = Field(ge=1)
    total_items: int = Field(ge=0)
    total_pages: int = Field(ge=0)
    has_next: bool
    has_previous: bool


class HealthResponse(ApiModel):
    status: str
    service: str
    version: str
    environment: str
    checks: dict[str, Any] | None = None