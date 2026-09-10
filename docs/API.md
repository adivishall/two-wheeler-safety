# API reference

Base URL in local dev: `http://127.0.0.1:5000`. All responses are JSON unless
noted. Fields shown are representative; see `app.py` for the source of truth.

**Auth.** Only `/detect` can be protected: if `DETECT_API_KEY` is set, requests
must send a matching `X-API-Key` header (401 otherwise). All other routes are
unauthenticated (local prototype).

**Rate limiting.** The write/upload routes (`/detect`, `/analyze`,
`/analyze_video`, `/api/violations/<id>/review`) are limited per IP to
`RATE_LIMIT_PER_MIN` requests/minute (429 when exceeded).

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

## GET /api/violations

Filtered, paginated list. All query params optional:

| Param | Meaning |
|---|---|
| `plate` | normalized-plate match |
| `type` | `no_helmet` / `triple_riding` / `overspeed` / … |
| `status` | `unpaid` / `paid` |
| `review_status` | `pending` / `confirmed` / `dismissed` |
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

## GET /evidence/&lt;file&gt;

Serve an evidence image/video by **basename only** (traversal-proof, media-type
allowlist). `404` for subpaths or disallowed types.
