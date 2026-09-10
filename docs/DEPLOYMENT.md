# Deployment

A reproducible container setup for running the app beyond `python3 app.py`. The
target is a single node (see [DECISIONS.md](DECISIONS.md#8-in-process-jobs-instead-of-celeryredis));
this is not a horizontally-scaled service.

## What is and isn't in the image

- **In:** the app code and its Python dependencies + gunicorn.
- **Not in (mount at run time):** the trained YOLO weights (large, gitignored),
  the SQLite database, and the evidence directory. These are provided via volumes
  so the image is generic and data persists across restarts.

## Build

```bash
docker build -t tws .
```

## Run

```bash
docker run --rm -p 5000:5000 \
    -v "$PWD/weights:/weights:ro" \
    -v tws-data:/data \
    -e MODEL_PATH=/weights/best.pt \
    -e TRAFFIC_DB_PATH=/data/traffic.db \
    -e EVIDENCE_DIR=/data/evidence \
    -e DETECT_API_KEY="$(openssl rand -hex 24)" \
    tws
```

- `weights/best.pt` on the host is mounted read-only at `/weights`.
- `tws-data` is a named volume holding the DB + evidence, so violations and
  evidence survive container restarts.
- `DETECT_API_KEY` protects the write API. Generate it, don't hard-code it.

Health: `GET /health` (the image also declares a Docker `HEALTHCHECK`).

## Production server

The image serves with **gunicorn, one worker + threads**:

```
gunicorn --workers 1 --threads 8 --timeout 600 --bind 0.0.0.0:5000 app:app
```

A single worker is deliberate: cached models, the in-process video `JobManager`,
and the rate limiter are per-process state, so multiple workers would fragment
them (a status poll could hit a worker that never ran the job). Threads provide
request concurrency without splitting that state. The long timeout accommodates
model loading on the first upload; per-video processing is separately bounded by
`MAX_VIDEO_SECONDS`.

The Flask dev server (`python3 app.py`) is for local use only — its debugger can
execute arbitrary code and it is single-threaded by default.

## Configuration

All knobs are environment variables — see the table in the
[README](../README.md#configuration). For production, at minimum:

| Variable | Recommended |
|---|---|
| `FLASK_DEBUG` | `0` (default) — never `1` in production |
| `DETECT_API_KEY` | a strong random secret |
| `MODEL_PATH` | the mounted weights path |
| `TRAFFIC_DB_PATH`, `EVIDENCE_DIR` | paths on a persistent volume |
| `MAX_VIDEO_MB`, `MAX_VIDEO_SECONDS`, `MAX_CONCURRENT_VIDEO_JOBS` | size to the host |

## Persistence & retention

The DB and evidence grow over time. Back up the volume, and apply a retention
policy (see [PRIVACY.md](PRIVACY.md)) — evidence images are personal data.

## Reverse proxy (optional)

Front gunicorn with nginx/Caddy for TLS, real client IPs (so the rate limiter
sees the right address — configure `X-Forwarded-For` handling if you rely on it),
and static caching of `/evidence`. Keep the app bound to localhost or an internal
interface behind the proxy.

## Secrets

Never commit secrets. `DETECT_API_KEY` and any future credentials come from the
environment / your orchestrator's secret store. `.env` and `*.log` are gitignored.
