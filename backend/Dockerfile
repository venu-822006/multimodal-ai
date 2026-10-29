# syntax=docker/dockerfile:1

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# GDAL and geospatial shared libraries required by rasterio, fiona and shapely.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gdal-bin \
        libgdal-dev \
        libgeos-dev \
        libproj-dev \
        proj-data \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY alembic.ini ./
COPY migrations ./migrations
COPY app ./app

# Artifacts live outside the source tree so nothing generated is ever committed.
ENV ARTIFACT_ROOT=/var/lib/floodsight/artifacts \
    PYTHONPATH=/app
RUN mkdir -p /var/lib/floodsight/artifacts /var/lib/floodsight/staging \
    && useradd --create-home --uid 10001 floodsight \
    && chown -R floodsight:floodsight /var/lib/floodsight /app
USER floodsight

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health/live || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]