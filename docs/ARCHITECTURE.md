# Architecture

This document describes how the system is put together. For *why* the important
choices were made, see [DECISIONS.md](DECISIONS.md).

## Overview

Three loosely-coupled layers share one detection core:

1. **Detection & reasoning** (`modules/`) — turns pixels into confirmed,
   confidence-scored violations with evidence. Pure Python + OpenCV, with the
   heavy model libraries (torch / ultralytics / easyocr) isolated behind lazy
   imports.
2. **Persistence** (`modules/db.py`) — a normalized SQLite store.
3. **Delivery** (`app.py` + `templates/frontend.html`, and the CLIs) — a Flask
   API + dashboard, plus `main.py` / `main_ocr.py` for headless use.

```mermaid
flowchart LR
    subgraph Entrypoints
        A1[app.py /analyze]
        A2[app.py /analyze_video]
        C1[main_ocr.py]
        C2[main.py]
    end
    A1 --> D[detector.analyze_image]
    C1 --> D
    A2 --> V[video_detector.process_video]
    C2 --> V
    D --> DB[(db.Database)]
    V --> DB
    DB --> UI[Dashboard + JSON API]
```

## Image pipeline

`modules/detector.py :: analyze_image` — single frame, no temporal state:

```mermaid
flowchart TD
    IMG[image] --> M[YOLOv8 predict]
    M --> P{Plate box?}
    P -->|yes| OCR[EasyOCR on plate crop] --> N[clean + look-alike correction]
    M --> H[collect helmet/no-helmet boxes]
    H --> CN{WithoutHelmet overlaps<br/>a WithHelmet box?}
    CN -->|yes: contradiction| SKIP[report neither]
    CN -->|no| NH[no_helmet]
    M --> TR[TripleRiding present? → triple_riding]
    N --> ANN[annotate + save evidence]
    NH --> ANN
    TR --> ANN
    ANN --> OUT[plate, violations, evidence_file]
```

This same function backs both the `/analyze` route and `main_ocr.py`, so the
web and CLI paths can't diverge.

## Video pipeline

`modules/video_detector.py :: process_video` — per-call state (its own tracker,
stabilizers, state machines, reported-set), safe to run from a web request:

```mermaid
flowchart TD
    F[frame] --> DETS[YOLO boxes → DetBox list]
    DETS --> TRK[VehicleTracker.update]
    TRK --> INST[per-vehicle instances<br/>body + plate, stable IDs]
    INST --> OCR[PlateStabilizer.add → stable plate]
    INST --> HSM[HelmetStateMachine.update]
    INST --> TSM[TripleRidingStateMachine.update]
    INST --> SPD[SpeedEstimator.estimate<br/>video time + calibration]
    HSM --> REC{confirmed and<br/>plate readable?}
    TSM --> REC
    SPD --> REC
    REC -->|once per track,violation| CONF[compute_confidence]
    CONF --> EV[build_evidence package]
    EV --> RF[record_fn → DB or API]
    F --> WR[write annotated frame]
```

### Tracking (`modules/vehicle_tracker.py`, `modules/vehicle.py`)

`VehicleTracker` maintains a set of `VehicleTrack`s with a lifecycle
(`tentative → confirmed → lost`) and stable integer IDs. Each frame it builds
candidate per-vehicle instances from the raw boxes, then matches them to existing
tracks by IoU/centroid distance. A track survives a few missed frames before it
is dropped, so a brief occlusion doesn't spawn a new ID.

### Association (`modules/association.py`, `modules/geometry.py`)

Within a frame, plate boxes and rider/violation boxes are matched **one-to-one**
with the Hungarian algorithm over a geometric cost (overlap / containment /
distance). This is the core "which plate belongs to which rider" step and is what
makes multi-bike frames tractable, versus the old nearest-plate heuristic.

## OCR pipeline (`modules/plate_recognizer.py`, `modules/plate_info.py`)

```mermaid
flowchart LR
    C[plate crop] --> R[EasyOCR detail=1]
    R --> A[PlateStabilizer.add<br/>text + min box conf]
    A --> VOTE[vote across frames]
    VOTE --> VAL[normalize + structural validation]
    VAL --> STABLE[stable_plate + confidence]
    STABLE --> DEC[plate_info.decode_plate<br/>state + RTO district]
```

The pipeline fines only against `stable_plate`. `plate_info` decodes the
registration *region* from the public plate-code scheme; it never resolves owner
identity.

## Violation decision engine

- `modules/violation_state.py` — `HelmetStateMachine` and
  `TripleRidingStateMachine`. States confirm only after a persistence window and
  a confidence floor, decay gracefully, and never un-confirm (so a fine is issued
  at most once). The helmet machine reports `ambiguous` on a contradiction.
- `modules/confidence.py` — `compute_confidence` merges detection / temporal /
  association / OCR into a transparent weighted mean. **Not a probability.**

## Evidence (`modules/evidence.py`)

`build_evidence` writes the original + annotated frame, plate + violation crops,
and a JSON sidecar (timestamp, frame index, track id, plate, violation, full
confidence breakdown, supporting frames, speed/calibration). Uniquely tokenized
names; basenames stored for portability.

## Database (`modules/db.py`)

```mermaid
erDiagram
    vehicles ||--o{ violations : has
    violations ||--o{ evidence : has
    vehicles {
        int id PK
        text normalized_plate UK
        text registration_state
        text rto_code
        text rto_name
    }
    violations {
        int id PK
        int vehicle_id FK
        text type
        int amount
        real confidence
        text status
        text review_status
        text reviewer_decision
        datetime reviewed_at
        int track_id
    }
    evidence {
        int id PK
        int violation_id FK
        text original_path
        text annotated_path
        text plate_crop_path
        text metadata_path
    }
    processing_jobs {
        text id PK
        text status
        real progress
    }
```

`initialize()` is idempotent: creates tables/indexes `IF NOT EXISTS`, migrates a
legacy flat `fines` table once, and `ALTER`s in the review columns for older DBs.
A fresh connection per operation, foreign keys on, related inserts in one
transaction.

## Web application (`app.py`, `templates/frontend.html`)

Flask routes (see [API.md](API.md)) for detection, lookup, analytics/filtering,
review, and evidence serving. Models load lazily and are cached behind a lock.
The single-page dashboard (vanilla JS, Apple-style CSS) drives the JSON API.

## Asynchronous video jobs (`modules/jobs.py`)

`JobManager` runs each video in a background thread with a typed lifecycle
(`processing / done / error / cancelled`), a concurrency cap, cooperative
cancellation via an event the worker polls, progress reporting, and lazy
expiry of finished jobs. In-process and dependency-free — see
[DECISIONS.md](DECISIONS.md) for why not Celery/Redis.

## Configuration & observability

- `modules/config.py` — one typed tree (`AppConfig`/`ServerConfig`/
  `DetectionConfig`) built from the environment via `from_env`, read at import so
  test reloads pick up patched env.
- `modules/logging_setup.py` — an idempotent project logger (`tws.*`) for
  application events; the CLIs keep friendly `print()` output.
