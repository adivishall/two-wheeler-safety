# Security review

Scope: the Flask web app (`app.py`) and its request-handling helpers
(`modules/validation.py`, `modules/jobs.py`, `modules/db.py`). This is a
prototype detection/review tool, not a public multi-tenant service; the controls
below match that threat model and are each backed by a test in
`tests/test_security.py` (plus `tests/test_app_auth.py`, `tests/test_validation.py`).

## Threat model

Assumed deployment: a single organization running the app behind its own network
boundary (`127.0.0.1` bind by default; gunicorn + a reverse proxy in production,
see [DEPLOYMENT.md](DEPLOYMENT.md)). The assets worth protecting are the fine
records and the evidence files. The realistic adversaries are: a malicious
uploader (crafted file/plate), a cross-site page trying to drive the API with a
victim's browser, and anyone who can reach the port trying to read or write
fines they shouldn't.

## Controls, by area

### Authentication & authorization
- `/detect` can require an `X-API-Key` (`DETECT_API_KEY`); unset only for local
  demo use.
- Role-based access (viewer < reviewer < admin) gates **every endpoint that
  creates or changes records** — `/analyze` and `/analyze_video` (they write
  fines), `/video_cancel`, review, payment — plus the audit log, **once role
  keys are configured**; with no keys set the app is open for local/demo use
  (documented, opt-in hardening). The upload routes used to be left open even
  with auth on, so anyone who could reach the port could still create fines;
  fixed and pinned in `tests/test_hardening.py`. Also `tests/test_app_auth.py`.
- **No cookies or server-side sessions are used for auth.** Credentials are a
  header API key, carried explicitly by the caller.

### CSRF
Structurally low-risk, and by design: CSRF needs an *ambient* credential (a
cookie) that the browser attaches automatically. This app has none — auth is an
explicit `X-API-Key` header a cross-site page cannot set (it would require a CORS
preflight the server never approves). The sensitive mutations (`/api/.../review`,
`/api/.../payment`) also consume a JSON body; a cross-site HTML form can only
send `application/x-www-form-urlencoded`/`multipart`, which `request.get_json`
rejects. So no CSRF token scheme is needed for this design; if cookie-based
sessions were ever added, one would be required.

### CORS
No `Access-Control-Allow-Origin` header is emitted, so browsers apply the
same-origin default and no other site can read the API with a user's context. A
test asserts the app never returns a wildcard ACAO.

### Rate limiting
An in-process per-IP sliding-window limiter (`RateLimiter`) guards the write /
upload routes (`/detect`, `/analyze`, `/analyze_video`, review, payment);
over-limit callers get `429`. Configurable via `RATE_LIMIT_PER_MIN`. Idle keys
are swept once a minute — it used to keep a deque for every IP that ever called,
an unbounded-memory path under a scan. Tested.
(Per-process only — a multi-worker deployment should add a shared limiter at the
proxy; noted in DEPLOYMENT.md.)

### Input validation & path traversal
- Uploads: extension allowlist **and** content sniffing — image magic bytes
  (a RIFF container counts as an image only if it says `WEBP`; it used to accept
  any RIFF, i.e. AVI/WAV) and, newly, video container signatures (ISO-BMFF
  `ftyp`, AVI, EBML) so a renamed non-video is rejected before a worker thread
  hands it to the decoder. Server-controlled temp filenames (never the client's),
  a global request-size cap (`MAX_CONTENT_LENGTH` → `413`), per-video size +
  wall-clock ceilings, and temp files removed on every exit path.
- Evidence is served only as a bare basename with a media-type allowlist, so
  `/evidence/../app.py` and non-media names are `404` — traversal-proof
  (`send_from_directory` is itself safe; this is defense in depth). A
  caller-supplied `image_path` is reduced to a safe basename before storage, and
  names reducing to `.`/`..` or containing a NUL are rejected.
- Plate/violation inputs are normalized/validated before they reach SQL.

### Data integrity under concurrency
Payment status moves along a lifecycle with terminal states. The update used to
validate the current status and then write unconditionally, so two concurrent
requests (paid vs cancelled) could both pass validation and the second silently
overwrote the first's terminal state. It is now a compare-and-set
(`UPDATE … WHERE id = ? AND status = ?`), and a lost race is reported rather than
applied. Pinned by a deterministic race test and a threaded test in
`tests/test_hardening.py`. `/api/stats?recent=` is clamped: SQLite treats a
negative `LIMIT` as unlimited.

### SQL injection
All queries are parameterized. The one place a column name is interpolated
(`list_violations` sort) is whitelisted against a fixed set (`_SORTABLE`); the
sort/filter values themselves are always bound parameters. Covered by the DB
filter/sort tests.

### Secrets handling
API-key/role keys come from the environment and are never logged or returned.
`/health` reports only `status` / `models_loaded` / `debug` — no secrets, no
paths. The debugger is **off by default** (the Werkzeug debugger is an RCE
surface if exposed); enabling it is an explicit `FLASK_DEBUG=1` opt-in.

### Error handling / information leakage
- 404 / 405 / 413 / 500 return generic JSON, not HTML error pages or tracebacks.
- A failed video job surfaces a generic "video processing failed" to the
  browser; the real exception is logged and stored server-side only, never sent
  to the client. Tested.

## Known limitations (honest)
- The rate limiter and job registry are per-process (fine for one worker; a
  scaled-out deployment needs shared state).
- No CSP beyond the header defaults — the dashboard intentionally uses inline
  styles/scripts, so a strict CSP would need a refactor; acceptable for a
  same-origin internal tool, called out rather than hidden.
- Auth is off until keys are configured. That is the correct default for local
  demo use but **must** be enabled for any shared deployment (DEPLOYMENT.md).
