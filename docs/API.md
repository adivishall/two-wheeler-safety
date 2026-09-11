# API reference

Base URL in local dev: `http://127.0.0.1:5000`. All responses are JSON unless
noted. Fields shown are representative; see `app.py` for the source of truth.

**Auth.** Two independent mechanisms, both **off by default** for local use:
- `/detect` can require a matching `X-API-Key` header if `DETECT_API_KEY` is set
  (401 otherwise).
- Role-based access (viewer < reviewer < admin) via role keys: once configured,
  the mutating review/payment endpoints require **reviewer** and `/api/audit`
  requires **admin** (403 otherwise). With no role keys set, every request is
  treated as an anonymous admin (unchanged local/demo behaviour). See
  [SECURITY.md](SECURITY.md).

**Rate limiting.** The write/upload routes (`/detect`, `/analyze`,
`/analyze_video`, `/api/violations/<id>/review`, `/api/violations/<id>/payment`)
are limited per IP to `RATE_LIMIT_PER_MIN` requests/minute (429 when exceeded).

**Errors.** 404 / 405 / 413 / 500 return generic JSON (`{"error": …}`), never an
HTML page or a traceback.

---

## GET /health

Liveness/readiness. Never exposes secrets.

```json
{ "status": "ok", "models_loaded": false, "debug": false }
```

## POST /analyze

Analyze an uploaded image (multipart form field `image`), record any violations,
return what was found. Rejects non-images by extension **and** magic bytes.

```bash
curl -F "image=@photo.jpg" http://127.0.0.1:5000/analyze
```

```json
{
  "plate": "MH12AB1234",
  "evidence": "/evidence/MH12AB1234_1699999999.jpg",
  "violations": [{ "type": "no_helmet", "amount": 500 }]
}
```

No plate read → `{ "plate": null, "message": "..." }`. Errors → `400` with
`{ "error": "..." }`.

## POST /analyze_video

Start a background video job (multipart field `video`). Returns immediately.

```bash
curl -F "video=@clip.mp4" http://127.0.0.1:5000/analyze_video
```

```json
{ "job_id": "9f2c…" }
```

`503` if the concurrency cap is reached; `400` for an unsupported/oversized file.

## GET /video_status/&lt;job_id&gt;

Poll a job. While processing:

```json
{ "status": "processing", "done": 120, "total": 400, "result": null, "error": null }
```

When finished (`status` ∈ `done` / `error` / `cancelled`); a `done` result:

```json
{
  "status": "done",
  "result": {
    "frames": 400, "plates_tracked": 2, "fps": 25.0,
    "output": "/evidence/annotated_ab12.mp4",
    "violations": [
      { "plate": "MH12AB1234", "violation": "no_helmet", "amount": 500,
        "confidence": 0.71, "evidence": "/evidence/…_annotated.jpg",
        "metadata": "/evidence/….json", "confidence_breakdown": { "...": 0.x } }
    ]
  }
}
```

Unknown id → `404`.

## POST /video_cancel/&lt;job_id&gt;

Cooperatively cancel a running job. `404` if not found or not cancellable.

```json
{ "message": "cancellation requested" }
```

## POST /detect

Record one violation. **API-key gated** if `DETECT_API_KEY` is set.

```bash
curl -X POST http://127.0.0.1:5000/detect \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $DETECT_API_KEY" \
  -d '{"plate":"MH12AB1234","violation":"no_helmet","image_path":"evidence/x.jpg"}'
```

`{ "message": "Violation Recorded" }`. `400` for missing fields / invalid plate /
invalid violation / unsafe `image_path`; `401` for a bad key.

## GET /get_fines/&lt;plate&gt;

Fines for a plate (case/whitespace-insensitive) + unpaid total + decoded region.

```json
{
  "fines": [
    { "violation": "no_helmet", "amount": 500, "status": "unpaid",
      "time": "2026-09-01 10:00:00", "image": "/evidence/x.jpg" }
  ],
  "total": 500,
  "vehicle": { "recognized": true, "state": "Maharashtra",
               "rto_code": "12", "rto": "Pune", "plate": "MH12AB1234" }
}
```

## GET /fines

Every fine in the legacy flat shape (`id, plate, violation, amount, image_path,
timestamp, status`).

## GET /api/stats

Dashboard overview (all real aggregates).

```json
{
  "total_violations": 5, "total_fines": 3200,
  "unpaid_fines": 2500, "paid_fines": 700, "vehicles": 4,
  "by_type": [{ "type": "no_helmet", "n": 3, "amount": 1500 }],
  "by_review_status": { "pending": 4, "confirmed": 1 },
  "recent": [{ "id": 5, "plate": "…", "type": "no_helmet", "amount": 500,
               "status": "unpaid", "confidence": 0.66, "timestamp": "…",
               "review_status": "pending" }]
}
```

Optional query: `recent` (count of recent rows, default 5).

## GET /api/analytics

Deeper dashboard analytics, all computed from stored rows (nothing fabricated).
Optional query: `days` (over-time window, default 30), `conf_buckets`
(histogram bins, default 10).

```json
{
  "over_time": [{ "day": "2026-09-10", "n": 3, "amount": 1700 }],
  "by_type": [{ "type": "no_helmet", "n": 3, "amount": 1500 }],
  "by_payment_status": { "unpaid": 4, "paid": 1 },
  "by_review_status": { "pending": 4, "confirmed": 1 },
  "review_outcomes": { "pending": 4, "confirmed": 1, "dismissed": 0,
                       "decided": 1, "confirmation_rate": 1.0 },
  "confidence": { "buckets": 10, "histogram": [0,0,1,2,…], "edges": [0.0,0.1,…],
                  "count": 5, "missing": 0, "mean": 0.68 },
  "plate_recognition": { "decoded": 4, "total": 5, "rate": 0.8 },
  "sessions": { "by_status": { "completed": 2 },
                "throughput_fps": { "runs": 2, "avg": 11.9, "min": 9.2, "max": 14.6 },
                "totals": { "frames": 2760, "vehicles_tracked": 55, "violations_detected": 14 },
                "by_model": [{ "model": "traffic-4class@1.0.0", "runs": 2 }] }
}
```

`confidence.histogram` is over **raw scores**, not calibrated probabilities
(see [EVALUATION.md](EVALUATION.md)).

## GET /api/violations

Filtered, paginated list. All query params optional:

| Param | Meaning |
|---|---|
| `plate` | normalized-plate match |
| `type` | `no_helmet` / `triple_riding` / `overspeed` / … |
| `status` | `unpaid` / `paid` |
| `review_status` | `pending` / `confirmed` / `dismissed` |
| `session_id` | only violations produced by that processing run |
| `min_confidence`, `max_confidence` | 0–1 (rows with NULL confidence excluded) |
| `date_from`, `date_to` | timestamp bounds (`YYYY-MM-DD[ HH:MM:SS]`) |
| `sort` | `id` / `timestamp` / `amount` / `confidence` / `type` / `status` (whitelisted) |
| `order` | `asc` / `desc` (default `desc`) |
| `limit` | page size (clamped to 500) |
| `offset` | page offset |

```bash
curl "http://127.0.0.1:5000/api/violations?type=no_helmet&min_confidence=0.5&limit=10"
```

```json
{ "items": [ { "id": 5, "plate": "…", "registration_state": "Maharashtra",
               "rto_name": "Pune", "type": "no_helmet", "amount": 500,
               "status": "unpaid", "confidence": 0.66,
               "review_status": "pending", "evidence": "/evidence/…" } ],
  "total": 3, "limit": 10, "offset": 0 }
```

## GET /api/violations/&lt;id&gt;

Full detail incl. evidence package URLs. `404` if not found.

```json
{ "id": 5, "plate": "MH12AB1234", "type": "no_helmet", "amount": 500,
  "confidence": 0.66, "status": "unpaid", "review_status": "pending",
  "registration_state": "Maharashtra", "rto_name": "Pune",
  "evidence": { "original": "/evidence/…_original.jpg",
                "annotated": "/evidence/…_annotated.jpg",
                "plate_crop": "/evidence/…_plate.jpg",
                "metadata": "/evidence/….json" } }
```

## POST /api/violations/&lt;id&gt;/review

Record a human decision. Rate-limited.

```bash
curl -X POST http://127.0.0.1:5000/api/violations/5/review \
  -H "Content-Type: application/json" \
  -d '{"review_status":"confirmed","decision":"valid","notes":"clear plate"}'
```

Body: `review_status` (required, one of `pending`/`confirmed`/`dismissed`),
`decision` (optional), `notes` (optional). Returns
`{ "message": "review recorded", "review_status": "confirmed" }`. `400` for a
missing/invalid status; `404` for an unknown id.

## POST /api/violations/&lt;id&gt;/payment

Move a violation along its **payment** lifecycle (independent of review):
`unpaid → paid | cancelled`; `paid`/`cancelled` are terminal. Reviewer role
required when auth is on; rate-limited.

```bash
curl -X POST http://127.0.0.1:5000/api/violations/5/payment \
  -H "Content-Type: application/json" -d '{"status":"paid"}'
```

Body: `status` (required, one of `unpaid`/`paid`/`cancelled`). `400` for an
unknown status or an illegal transition; `404` for an unknown id.

## GET /api/violations/&lt;id&gt;/trace

The bounded per-frame detection trace supporting a confirmed violation
(reconstructs *why* it was confirmed). Returns
`{ "trace": [{ "track_id", "label", "confidence", "box", "frame_index", "timestamp" }] }`.

## GET /api/sessions

Processing sessions, most recent first. Optional `limit` (default 50).
Returns `{ "sessions": [ … ] }`.

## GET /api/sessions/&lt;id&gt;

One session plus the violations it produced (per-run drill-down). Optional
`limit` (default 100) on the embedded violations. `404` if not found.

```json
{ "id": "…", "source": "clip.mp4", "status": "completed",
  "model_version": "traffic-4class@1.0.0", "processing_fps": 14.6,
  "frames_processed": 1820, "vehicles_tracked": 37,
  "violations": { "items": [ … ], "total": 9, "limit": 100, "offset": 0 } }
```

## GET /api/audit

Recent audit-log events (append-only). **Admin role required when auth is on.**
Optional query: `limit` (default 100), `record_id` (filter to one record).
Returns `{ "events": [{ "event", "record_type", "record_id", "actor",
"timestamp", "metadata" }] }`.

## GET /evidence/&lt;file&gt;

Serve an evidence image/video by **basename only** (traversal-proof, media-type
allowlist). `404` for subpaths or disallowed types.
