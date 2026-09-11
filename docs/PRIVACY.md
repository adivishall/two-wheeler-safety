# Privacy & data retention

The system processes number plates and evidence images of people and vehicles on
public roads. This note states plainly what it stores, what it deliberately does
*not* know, and how to handle the data responsibly. It is a prototype detection &
review assistant, not an enforcement system.

## What is stored — exact field inventory

Every field the system persists, by table (`modules/db.py` schema). "Personal"
marks data that relates to an identifiable vehicle/person; note that even these
resolve only to a *plate* and its *region*, never an owner (see below).

| Table | Fields | Personal? |
|-------|--------|:---------:|
| `vehicles` | `plate`, `normalized_plate`, `registration_state`, `rto_code`, `rto_name`, `created_at`, `updated_at` | **yes** (plate + region only) |
| `violations` | `type`, `amount`, `confidence`, `timestamp`, `status` (payment), `track_id`, `detection_status`, `review_status`, `reviewer_decision`, `reviewed_at`, `review_notes`, `session_id` | indirect (links to a vehicle) |
| `evidence` | `original_path`, `annotated_path`, `plate_crop_path`, `metadata_path` | **yes** (images of people/vehicles) |
| `detections` (trace) | `track_id`, `label`, `confidence`, `box`, `frame_index`, `timestamp` — **bounded** to the last N supporting frames per violation (`DETECTION_TRACE_MAX_FRAMES`, default 20) | no (coordinates only) |
| `sessions` | `source` (uploaded filename), `model_version`, `pipeline_version`, frame/vehicle/violation counts, `processing_fps`, `output_path`, `status`, `error` | no |
| `audit_log` | `event`, `record_type`, `record_id`, `actor`, `timestamp`, `metadata` | no (see actor note) |
| `processing_jobs` | `type`, `status`, `progress`, timestamps, `error`, `output`, `source` | no (transient) |

Notes that matter for privacy:

- **`actor` is a role label, not a person.** It is one of `viewer` / `reviewer`
  / `admin` / `system` / `anonymous` (or `unknown`) — derived from the API key's
  role, never a name, email, or user id. The audit log records *what role* did
  *what*, not *who*.
- **`metadata` / evidence sidecars never contain image bytes** — only detection
  coordinates, confidences, the model/pipeline version, and a config snapshot.
- **No free-text about people** except `review_notes`, which a reviewer types;
  keep those factual (about the evidence), not about the individual.

## Why it is stored

To detect candidate violations, let a human review them against the evidence, and
keep an auditable record of what was flagged and why. The evidence and confidence
breakdown exist precisely so a person can verify a flag rather than trust the
model.

## What the prototype does NOT know

- **Owner identity.** Plate → owner requires the access-restricted VAHAN
  database. This project **never** queries it and has no owner name, address,
  phone, vehicle make/model, or any personal detail beyond the plate and the
  region its code encodes. `modules/plate_info.py` decodes only the public,
  static state/RTO scheme.
- **Ground truth.** A recorded violation is a model prediction with a confidence
  score, not a verified fact — hence mandatory human review.

## Why human review is appropriate

Computer-vision predictions have false positives and a documented
"confidently-wrong" failure mode on unfamiliar footage. Treating an automated
flag as proof would be both technically unjustified and unfair to the person
depicted. The `pending → confirmed | dismissed` workflow keeps a human
accountable for any decision, and the UI never presents a flag as legal certainty.

## Recommended retention policy

This repo does not auto-delete data (a prototype), but a real deployment should:

- **Set a retention window** for evidence images and dismissed violations (e.g.
  delete evidence for `dismissed` records promptly, and purge all evidence older
  than a fixed period). Evidence images are the most sensitive artifact.
- **Minimize.** Keep only the crops/frames needed for review; drop full frames
  once a record is confirmed or dismissed if your process allows.
- **Restrict access.** Serve `/evidence` only to authorized reviewers (behind
  auth / an internal network), not the public internet.
- **Secure the store.** Encrypt the volume at rest; back it up with the same
  access controls.
- **Log carefully.** The application logger avoids dumping image bytes; keep it
  that way and don't log plate strings at large scale without a reason.

## Erasure — how to actually delete a record

Because storage is normalized, deleting one vehicle's data means removing its
rows across tables (child rows first, to respect foreign keys):

1. Delete the on-disk **evidence files** referenced by that vehicle's
   `evidence` rows (`original_path`, `annotated_path`, `plate_crop_path`,
   `metadata_path`) — the DB stores paths, not the image bytes, so a row delete
   alone leaves files behind.
2. Delete `detections` → `evidence` → `violations` → the `vehicles` row.
   `Database.reset()` does this wipe for *all* records (used by the demo
   seeder); a per-vehicle erasure follows the same order filtered by
   `vehicle_id`.
3. **The `audit_log` is append-only by design** and keeps the *event* history
   (e.g. "violation_review_dismissed"), which references records by id but holds
   no plate or image. Decide per policy whether an erasure also tombstones the
   audit entries; keeping them (id-only) is usually the accountable choice.

**Evidence integrity interaction:** each evidence package carries a SHA-256 over
its files (tamper-evidence, `modules/evidence.py`). Deleting evidence is a
legitimate retention action, but it necessarily invalidates that package's
verification — that is expected, not tampering. Record erasures in your own
retention log so a later "hash missing" is explained.

## Explicitly out of scope

- No integration with, or documentation of how to access, any private
  vehicle-owner database.
- No identity resolution, cross-source profiling, or tracking of individuals
  beyond a single clip's per-vehicle track IDs (which are ephemeral).
