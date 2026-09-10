# Privacy & data retention

The system processes number plates and evidence images of people and vehicles on
public roads. This note states plainly what it stores, what it deliberately does
*not* know, and how to handle the data responsibly. It is a prototype detection &
review assistant, not an enforcement system.

## What is stored

- **Plate strings** (normalized) and the **registration region** decoded from
  them (state + RTO district).
- **Violation records**: type, fine amount, timestamp, confidence score, payment
  status, and human-review state.
- **Evidence images**: the original frame, an annotated frame, and plate/violation
  crops, plus a JSON sidecar with detection metadata.
- **Processing-job** bookkeeping (transient).

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

## Explicitly out of scope

- No integration with, or documentation of how to access, any private
  vehicle-owner database.
- No identity resolution, cross-source profiling, or tracking of individuals
  beyond a single clip's per-vehicle track IDs (which are ephemeral).
