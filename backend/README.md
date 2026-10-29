# FloodSight AI — Backend

FastAPI service that acquires Sentinel-1 radar imagery, prepares it for the
custom-trained flood segmentation model, runs tiled inference and serves the
geospatial results the React frontend renders.

## Architecture

A modular monolith: one API process, one worker process, one PostgreSQL
database. There is no message broker and no cache; jobs are queued in
PostgreSQL and claimed with `FOR UPDATE SKIP LOCKED`.

```
app/
  main.py                     FastAPI app factory, middleware, lifespan
  core/
    config.py                 pydantic-settings configuration
    enums.py                  lifecycle, job and failure codes with a transition table
    exceptions.py             typed errors and sanitised API responses
    logging.py                structured JSON / text logging with request ids
    security.py               Argon2id hashing and opaque session tokens
  api/
    router.py                 route registration
    routes/health.py          liveness, readiness, prediction provider readiness
    routes/analyses.py        the analysis endpoints
    routes/auth.py            register, login, logout and current-user endpoints
  schemas/                    Pydantic request and response contracts
  models/analysis.py          SQLAlchemy 2.x analysis tables
  models/user.py              SQLAlchemy 2.x user, session and login-attempt tables
  repositories/               database access
  services/
    prediction_providers.py   PredictionProvider contract, trained and synthetic providers
    model_service.py          adapter around the trained U-Net
    preprocessing_service.py  Sentinel-1 to model-ready VV/VH dB rasters
    flood_detection_service.py provider output to measurements and geometry
    imagery_service.py        STAC search and scene pairing
    geospatial_service.py     grids, reprojection, polygonisation, measurement
    result_service.py         facts and map payloads
    report_service.py         PDF situation report
    analysis_service.py       create and lifecycle operations
    auth_service.py           registration, sign-in, sessions and lockout
    damage_assessment_service.py  OpenStreetMap exposure context
  processing/
    pipeline.py               ordered processing stages
    validators.py             AOI, date and GeoJSON validation
  db/                         engine and session lifecycle
  workers/analysis_worker.py  PostgreSQL-backed job worker
```

## Requirements

- Python 3.12
- PostgreSQL 14+ (16 recommended)
- Trained model weights (see below)

## Model setup

The model architecture lives in the `aimodel` branch of
<https://github.com/venu-822006/multimodal-ai>. Check it out next to this
repository:

```bash
git clone https://github.com/venu-822006/multimodal-ai.git
cd multimodal-ai && git checkout aimodel
```

`model.py` in that repository is the single source of truth for the
architecture. The backend imports it by path; it never redefines it.

### Trained weights

**The repository does not ship trained weights.** `checkpoints/*.pt` is
git-ignored upstream, and no checkpoint is present in any branch or in git
history. Obtain `best.pt` from whoever trained the model and point `MODEL_PATH`
at it:

```bash
cp /path/to/best.pt ./models/best.pt
```

The checkpoint is loaded with the keys written by `train.py`:

| Key | Purpose |
| --- | --- |
| `model_state_dict` | U-Net weights (required) |
| `model_config` | `in_channels`, `classes`, `base_channels` |
| `normalization` | `mean_db`, `std_db` used for z-score normalisation |
| `patch_size` | Inference patch size, default 256 |
| `val_flood_iou` | Recorded validation score, reported as provenance |
| `epoch` | Recorded epoch |

Without a checkpoint the API still starts and every endpoint works, but
`/health/model` reports `model_unavailable` and analyses fail with
`dependency_unavailable` rather than producing fabricated results. To develop
locally without weights, select the synthetic provider explicitly; see
[Prediction providers](#prediction-providers).

### Model contract

The adapter preserves the trained model's contract exactly. It does not
substitute a different detector, threshold or normalisation.

- **Input**: two-band Sentinel-1 VV/VH float32 GeoTIFF, decibels.
- **Normalisation**: clip at ±3σ, then `(x - mean) / (std + 1e-6)` using the
  checkpoint's `normalization` block.
- **Tiling**: `patch_size` windows at stride `patch_size // 2`, with the final
  row and column snapped to the raster edge so no valid pixel is dropped.
- **Blending**: overlapping tiles are averaged in probability space, not logit
  space.
- **Output**: argmax over the two classes; class 1 is flood, 255 is nodata.
- **Device**: `auto`, `cpu` or `cuda`. CPU is fully supported and CUDA is never
  required.

## Prediction providers

The flood-mapping stage never talks to a checkpoint directly. It depends on the
`PredictionProvider` contract in `app/services/prediction_providers.py`:

| Method | Purpose |
| --- | --- |
| `describe()` | Readiness and provenance without raising, for health reporting |
| `ensure_available()` | Load or validate, raising if predictions are impossible |
| `predict(...)` | Return a binary mask aligned to the prepared raster |

Every prediction carries `provenance` (provider id, data label, checkpoint or
pattern/seed) and a `synthetic` flag. `validate_prediction()` then enforces the
georeferencing contract: mask shape, valid-mask shape, affine transform, CRS and
class values must all match the analysis grid.

### `trained_model`

The real FloodSight U-Net, loaded from a checkpoint through
`app/services/model_service.py`. It validates the checkpoint's `model_config`,
normalisation statistics and input channel count, and fails loudly when the
checkpoint is missing or incompatible.

### `synthetic_test`

A deterministic stand-in for local development without trained weights. It
produces a reproducible mask from the configured pattern and seed, preserves the
raster dimensions, CRS and geotransform, and is labelled `synthetic_test_data` in
the measurements, the GeoJSON properties, the GeoTIFF tags, the facts payload,
the map legend and the PDF report.

### Selection rules

Selection is always explicit. **The synthetic provider is never used as a
fallback when the trained model is unavailable.**

- `PREDICTION_PROVIDER` in `backend/.env` sets the default.
- A single analysis may override it with `"prediction_provider"` in the request
  body, but only in `development` or `test`.
- `environment=production` with `prediction_provider=synthetic_test` refuses to
  start unless `ALLOW_SYNTHETIC_PROVIDER_IN_PRODUCTION=true` is set explicitly.
- An unknown provider name is rejected, never coerced to a default.

Synthetic output is a pipeline test result. It is not a flood prediction and must
not be used for situational awareness or reporting.

## Local development

### 1. Python environment

```bash
cd /path/to/FloodSight-AI
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
pip install -r backend/requirements-dev.txt   # optional, for the test suite
```

### 2. PostgreSQL

```bash
docker compose up -d postgres
docker compose ps                   # wait until the healthcheck reports healthy
```

### 3. Configuration

```bash
cp backend/.env.example backend/.env
```

### 4. Migrations

```bash
cd backend
alembic upgrade head
cd ..
```

### 5. API

```bash
source .venv/bin/activate
uvicorn app.main:app --app-dir backend --reload     # http://127.0.0.1:8000
```

### 6. Worker (second terminal)

```bash
cd backend
source ../.venv/bin/activate
python -m app.workers.analysis_worker
```

### 7. Frontend (third terminal)

```bash
npm install
npm run dev                         # http://localhost:3000
```

The dev server proxies `/api` and `/health` to `http://127.0.0.1:8000`.

### Running without trained weights

Set the provider explicitly in `backend/.env`:

```bash
PREDICTION_PROVIDER=synthetic_test
SYNTHETIC_PATTERN=blocky
SYNTHETIC_SEED=20260921
SYNTHETIC_COVERAGE=0.06
```

Restart the API and the worker, then submit an analysis:

```bash
curl -X POST http://127.0.0.1:8000/api/analyses \
  -H 'Content-Type: application/json' \
  -d '{
    "location": "Koshi basin, Nepal",
    "flood_date": "2026-09-21",
    "bbox": [[85.35, 26.95], [85.65, 27.25]],
    "min_flood_area_m2": 5000
  }'
```

Alternatively select it for a single request without changing configuration
(development and test only):

```bash
curl -X POST http://127.0.0.1:8000/api/analyses \
  -H 'Content-Type: application/json' \
  -d '{
    "location": "Koshi basin, Nepal",
    "flood_date": "2026-09-21",
    "bbox": [[85.35, 26.95], [85.65, 27.25]],
    "prediction_provider": "synthetic_test"
  }'
```

Confirm what is active:

```bash
curl -s http://127.0.0.1:8000/health/model | python -m json.tool
curl -s http://127.0.0.1:8000/health/ready/capabilities | python -m json.tool
```

`/health/model` reports `"synthetic": true` and
`/health/ready/capabilities` reports `"produces_real_predictions": false`.

### Tests

Tests drop and recreate the schema; use a dedicated test database, not the
development one:

```bash
cd backend
source ../.venv/bin/activate
export DATABASE_URL=postgresql+psycopg://floodsight:floodsight@127.0.0.1:5433/floodsight_test
export TEST_DATABASE_URL="$DATABASE_URL"
python -m pytest tests/
```

Database tests skip automatically when PostgreSQL is unreachable. No test
requires trained weights or network access.

## Configuration

Every setting is an environment variable; see `.env.example` for the full list.

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | local postgres | SQLAlchemy URL |
| `MODEL_REPO_PATH` | `../multimodal-ai` | Directory containing `model.py` |
| `MODEL_PATH` | empty | Trained checkpoint; empty means unavailable |
| `MODEL_DEVICE` | `auto` | `auto`, `cpu` or `cuda` |
| `MODEL_BATCH_SIZE` | `1` | Tiles per forward pass |
| `MODEL_PATCH_SIZE` | from checkpoint | Override the checkpoint patch size |
| `MODEL_TEMPERATURE` | `1.0` | Logit temperature before softmax |
| `MODEL_CONFIDENCE_THRESHOLD` | `0.5` | Flood probability cutoff |
| `MODEL_TEMPORAL_FILTER` | `true` | Require a pre-event scene for change consistency |
| `PREDICTION_PROVIDER` | `trained_model` | `trained_model` or `synthetic_test` |
| `SYNTHETIC_PATTERN` | `blocky` | Synthetic mask pattern |
| `SYNTHETIC_SEED` | `20260921` | Synthetic mask seed |
| `SYNTHETIC_COVERAGE` | `0.06` | Target share of valid pixels marked flood |
| `STAC_ENDPOINT` | Microsoft Planetary Computer | Open STAC catalogue |
| `RASTER_TARGET_RESOLUTION_M` | `20.0` | Analysis grid resolution |
| `MIN_FLOOD_POLYGON_AREA_M2` | `5000.0` | Discard smaller predicted patches |
| `OSM_ENABLED` | `true` | Enable OpenStreetMap exposure context |
| `OSM_ENDPOINT` | `overpass-api.de` | Primary Overpass mirror |
| `ARTIFACT_ROOT` | `var/artifacts` | Generated files, kept out of git |
| `CORS_ALLOW_ORIGINS` | localhost dev origins | Comma separated origins |
| `AUTH_SESSION_COOKIE_NAME` | `floodsight_session` | Session cookie name |
| `AUTH_SESSION_TTL_SECONDS` | `604800` (7 days) | Server-side session lifetime |
| `AUTH_SESSION_COOKIE_SECURE` | `false` | HTTPS-only cookie; forced `true` in production |
| `AUTH_SESSION_COOKIE_SAMESITE` | `lax` | `lax`, `strict` or `none`; `none` needs a secure cookie |
| `AUTH_SESSION_COOKIE_DOMAIN` | empty | Optional cookie domain |
| `AUTH_PASSWORD_MIN_LENGTH` | `8` | Minimum accepted password length |
| `AUTH_PASSWORD_MAX_LENGTH` | `128` | Maximum accepted password length |
| `AUTH_LOGIN_MAX_ATTEMPTS` | `5` | Failed sign-ins before a temporary lockout |
| `AUTH_LOGIN_LOCKOUT_SECONDS` | `900` | Lockout duration |

Never commit `.env`, checkpoints or generated artifacts.

## API

Interactive documentation is served at `/docs`; the OpenAPI schema is at
`/openapi.json`.

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/auth/register` | Create an account. Returns `201` and sets the session cookie. |
| `POST` | `/api/auth/login` | Sign in. Returns `200`; `remember: false` omits `Max-Age`. |
| `POST` | `/api/auth/logout` | Revoke the current session. Returns `204`. |
| `GET` | `/api/auth/me` | The authenticated user. Returns `401` when unauthenticated. |
| `POST` | `/api/analyses` | Create an analysis and schedule a job. Returns `202`. |
| `GET` | `/api/analyses` | Paginated list with status filters. |
| `GET` | `/api/analyses/{id}` | Metadata and result references. |
| `GET` | `/api/analyses/{id}/status` | Persisted lifecycle and stage history. |
| `GET` | `/api/analyses/{id}/facts` | Measurements, provenance, warnings. |
| `GET` | `/api/analyses/{id}/map` | GeoJSON layers for MapLibre. |
| `GET` | `/api/analyses/{id}/report` | Report descriptor. |
| `GET` | `/api/analyses/{id}/report/download` | PDF situation report. |
| `GET` | `/api/analyses/{id}/artifacts/{key}` | Generated GeoTIFF or GeoJSON. |
| `POST` | `/api/analyses/{id}/cancel` | Request cancellation. |
| `GET` | `/health/live` | Process liveness. |
| `GET` | `/health/ready` | Database readiness. |
| `GET` | `/health/model` | Prediction provider readiness and provenance. |
| `GET` | `/health/ready/capabilities` | Whether new analyses are accepted. |

`POST /api/analyses` accepts:

```json
{
  "location": "Koshi basin, Nepal",
  "flood_date": "2026-09-21",
  "bbox": [[85.35, 26.95], [85.65, 27.25]]
}
```

`bbox` is `[[west, south], [east, north]]` in WGS84 decimal degrees and must be
ordered south-west first. A `202` confirms the job was recorded, not that flood
detection succeeded.

### Status lifecycle

```
queued -> validating -> searching_data -> downloading -> preprocessing
       -> flood_mapping -> damage_analysis -> connectivity_analysis
       -> generating_report -> completed
```

Any stage may transition to `failed`. `cancelled` is terminal. Transitions
outside this table are rejected, so a job can never appear to skip work.

## Authentication

Authentication is server-side sessions, not bearer tokens. The browser holds an
opaque random token in an `HttpOnly` cookie and nothing else; only the SHA-256
hash of that token is stored, so a database read cannot reconstruct a session.
Passwords are stored as Argon2id hashes.

### Design

- **Hashing** — `argon2-cffi` with a random per-password salt. Verification uses
  the library's constant-time comparison and transparently re-hashes when the
  parameters change.
- **Sessions** — `auth_sessions` stores `token_hash`, the owning `user_id`,
  `created_at`, `last_used_at`, `expires_at` and an optional `revoked_at`. A
  request is authenticated only while `revoked_at IS NULL AND expires_at > now`.
  Session rows cascade-delete with their user.
- **Cookie** — `HttpOnly`, `SameSite=Lax`, `Path=/`, `Secure` in production,
  `Max-Age` equal to `AUTH_SESSION_TTL_SECONDS` unless `remember` is `false`.
  The default lifetime is 7 days.
- **CSRF** — `SameSite=Lax` blocks cross-site cookie transmission; in addition,
  state-changing auth requests reject a declared `Origin` that is not in
  `CORS_ALLOW_ORIGINS`. Non-browser clients that send no `Origin` are unaffected.
- **Login throttling** — failures are counted per `sha256(normalised email |
  client IP)` in `auth_login_attempts`. After `AUTH_LOGIN_MAX_ATTEMPTS` failures
  the key is locked for `AUTH_LOGIN_LOCKOUT_SECONDS` and further attempts receive
  `429` with a `retry_after_seconds` hint. A successful sign-in clears the count.
- **Logout** — revokes the current session row and clears the cookie. An expired,
  revoked, unknown or malformed token is treated identically: `401`.

### Errors

Unknown email and wrong password both return the same generic
`401 invalid_credentials` body, and an unknown email still performs a dummy hash
so the timing does not reveal whether an account exists. Request bodies and
responses never include the password hash, and passwords/tokens are never logged.

### Scoping and remaining work

Only `/api/auth/*` (notably `/api/auth/me`) is protected at the API layer. The
analysis endpoints remain unauthenticated and are not yet owned by a user, so the
frontend gates workspace routes on `/api/auth/me` while the API stays open to the
existing analysis contract. Email verification, password change/reset, account
deletion and third-party (Google) sign-in are not implemented; the frontend shows
these as unavailable rather than simulating them.

## Processing pipeline


1. **validating** — re-validate the stored area of interest and date window.
2. **searching_data** — query the STAC catalogue for Sentinel-1 GRD RTC scenes
   overlapping the area; pair the latest pre-event scene with the earliest
   post-event scene.
3. **downloading** — resolve signed asset URLs.
4. **preprocessing** — read VV and VH, convert linear power to decibels, apply a
   3×3 Lee-sigma speckle filter, clip to the training range of −40 dB to 0 dB,
   and reproject both polarisations onto one aligned analysis grid.
5. **flood_mapping** — run the selected prediction provider over the prepared
   raster; the trained model uses tiled inference with the temporal consistency
   filter when a pre-event scene is available.
6. **damage_analysis** / **connectivity_analysis** — OpenStreetMap features
   intersecting the predicted extent. These describe exposure only; an
   intersection does not establish damage.
7. **generating_report** — persist measurements, write artifacts, render the PDF.

Flooded percentage is measured against the **valid** analysis area, so pixels
without radar coverage never inflate or deflate the result. Areas are computed
geodesically on the WGS84 ellipsoid, never by treating degrees as metres.

### Raw mask versus filtered extent

Two rasters are written, because minimum-area filtering is a vector concern and
must not rewrite the prediction:

| Artifact | Contents |
| --- | --- |
| `flood_mask.tif` | Raw provider output: 0 background, 1 flood, 255 nodata. Never filtered. |
| `flood_extent_mask.tif` | Rasterised `flood_extent.geojson`, i.e. after minimum-area filtering. |
| `flood_extent.geojson` | Filtered polygon geometry in WGS84. |
| `analysis.json` | Measurements, provenance, warnings and limitations. |

Both rasters carry GeoTIFF tags recording the provider, the data label and
whether the prediction is synthetic. When filtering discards every polygon,
`flooded_area_km2` is `null` but `flood_mask.tif` still contains the full
prediction, and `flood_pixel_count` still reports it.

### OpenStreetMap exposure

Building, road and settlement lookups are queried **separately** so that a dense
category cannot exhaust the element limit and starve the others. A category that
failed or was truncated reports `null` rather than `0`; a category that genuinely
returned no features reports `0`. Exposure never fails the analysis.

## Tests

Tests drop and recreate the schema, so they must run against a dedicated,
throwaway database — never the development database. Point both
`DATABASE_URL` and `TEST_DATABASE_URL` at the same test database:

```bash
cd backend
export DATABASE_URL=postgresql+psycopg://floodsight:floodsight@127.0.0.1:5433/floodsight_test
export TEST_DATABASE_URL="$DATABASE_URL"
python -m pytest tests/
```

The suite covers health endpoints, the full analysis contract, validation,
pagination and filtering, unfinished and failed analyses, artifact traversal
protection, the model adapter (loading, failure modes, normalisation, tiling,
georeferencing, nodata), preprocessing, geodesic measurement, and the worker
(atomic claiming, retries, recovery, idempotency, duplicate prevention).

It also covers the prediction providers: the synthetic provider running without
a checkpoint, determinism, georeferencing preservation, nodata handling, the
explicit selection rules including the production guard, prediction validation,
raw mask versus filtered extent, measurement consistency across facts, map,
summary and report, and OpenStreetMap handling for successful, empty, truncated
and failed responses.

Authentication coverage includes registration and normalised-email uniqueness,
password hashing (hashes are never echoed back), invalid input, generic
invalid-credential responses for unknown email and wrong password, sessions
accepted/expired/revoked, logout and stale-cookie reuse, login lockout and reset,
origin checks, and `remember` cookie persistence.

No test requires trained weights or network access.

Database tests skip automatically when PostgreSQL is unreachable.

## Docker

```bash
docker compose up -d --build
docker compose exec api alembic upgrade head
```

The API is served on `http://localhost:8000` and the worker runs as a separate
service against the same queue.

## Troubleshooting

| Symptom | Cause | Action |
| --- | --- | --- |
| `/health/model` reports unavailable | `PREDICTION_PROVIDER=trained_model` but `MODEL_PATH` is unset or the file is absent | Copy `best.pt` and set `MODEL_PATH`, or select `synthetic_test` for local development |
| `dependency_unavailable` on analysis | Provider failed to load | Check `/health/model` for the reason |
| Overpass returns HTTP 406 | A browser-like `User-Agent` was configured | Use a plain application identifier; several mirrors reject browser agents |
| Exposure statistics are `null` | The category lookup failed or hit the element limit | Check the warnings; a `null` means unavailable, `0` means genuinely empty |
| `no_imagery` | No Sentinel-1 scene overlaps the area and window | Narrow the area, move the flood date, or widen `lookback_days` / `lookahead_days` |
| `imagery_unavailable` | Catalogue or asset host unreachable | Check network egress; the job retries automatically |
| Analysis stuck in `queued` | No worker is running | Start `python -m app.workers.analysis_worker` |
| `flooded_percentage` is null | No polygon survived minimum-area filtering | Lower `MIN_FLOOD_POLYGON_AREA_M2` or inspect `flood_mask.tif`, which still holds the raw prediction |
| Frontend shows "API NOT CONNECTED" | Backend unreachable | Start the API; `npm run dev` proxies `/api` to `127.0.0.1:8000` |
| `flood_date is outside the supported archive window` | Date beyond `MAX_FLOOD_LOOKBACK_DAYS` | Use a more recent date or raise the limit |

## Known limitations

- **Trained weights are not in the repository.** Without `best.pt` the system
  cannot produce flood extents. The synthetic provider exists so the rest of the
  pipeline can be exercised and tested; it produces `synthetic_test_data`, not
  flood predictions.
- **Genuine flood detection is unverified** until a trained checkpoint is
  available *and* has been evaluated on a known scene against reference data.
- Predictions are model output, not confirmed ground truth.
- A model trained on a 12-day revisit interval degrades when pre- and post-event
  scenes are far apart in time.
- OpenStreetMap exposure is supporting context only; no damage is inferred.
- OpenStreetMap figures can be incomplete where the element limit is reached;
  those categories are reported as unavailable rather than as zero.
- **Analysis endpoints are not yet user-scoped.** The backend protects
  `/api/auth/*` only; `POST/GET /api/analyses` remain unauthenticated and results
  are not associated with a user. The frontend gates workspace routes on
  `/api/auth/me`, but a direct API caller is not blocked. Adding ownership would
  change the analysis contract and its tests.
- **No email verification, password change/reset, account deletion or third-party
  sign-in.** The frontend presents these actions as unavailable instead of
  simulating success.