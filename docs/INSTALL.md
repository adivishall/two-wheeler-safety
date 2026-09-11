# Installation & reproducible environments

This project has two very different dependency profiles, and they are kept
separate on purpose:

| Profile | What it's for | Install | Pins |
|---------|---------------|---------|------|
| **Contributor / CI** (model-free) | Run the test suite + linter. No torch, no weights. | `requirements-ci.txt` | `constraints-ci.txt` |
| **Full runtime** | Actually run detection on images/video. Pulls torch + ultralytics + easyocr (~2 GB). | `requirements.txt` | `constraints-runtime.txt` |
| **Production** | Full runtime + gunicorn WSGI server. | `requirements-prod.txt` | `constraints-runtime.txt` |

The whole test suite is **model-free** — heavy inference is isolated behind lazy
imports and never exercised in tests (`tests/test_pipeline_integration.py` uses
fakes) — so contributors and CI never need the multi-gigabyte ML stack.

## Supported Python

**3.11 – 3.13.** CI runs 3.11 and 3.12; development is on 3.13. The pinned test
stack (`numpy 2.4`) requires **Python ≥ 3.11**, so 3.10 is not supported even
though the application code itself targets 3.10+ syntax.

## Contributor setup (test + lint, ~1 minute)

```bash
python3 -m venv .venv && source .venv/bin/activate   # Python 3.11–3.13
python -m pip install --upgrade pip
pip install -r requirements-ci.txt -c constraints-ci.txt
pip check          # sanity: no conflicting versions
pytest -q          # expect: all tests pass
ruff check .       # expect: All checks passed!
```

This is byte-for-byte what CI does (`.github/workflows/ci.yml`), so a green
local run means a green CI run. A fresh install from these two files was
verified to install cleanly and pass the full suite.

## Full runtime setup (run the model)

The heavy stack is platform-specific because of **torch**:

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip

# 1. Install torch/torchvision for YOUR platform FIRST:
#    - CPU / Apple Silicon (MPS): the default wheel is fine:
pip install "torch~=2.12" "torchvision~=0.27"
#    - NVIDIA CUDA: install the matching CUDA build from https://pytorch.org
#      (that command decides your torch version) — do this instead of the line
#      above, then continue.

# 2. Install the rest, pinned to the known-good closure:
pip install -r requirements.txt -c constraints-runtime.txt
```

`constraints-runtime.txt` records the exact versions the measured results in
[EVALUATION.md](EVALUATION.md) were produced on. It intentionally does **not**
force a CPU torch pin onto a CUDA host — see the caveat at the top of that file.

You also need the trained weights locally (not shipped in the repo):
`runs/detect/traffic_model-2/weights/best.pt`. See
[MODEL_VERSIONING.md](MODEL_VERSIONING.md) and [DATASET.md](DATASET.md).

## Regenerating the pins after an intentional upgrade

From a clean venv on Python 3.12:

```bash
pip install -r requirements-ci.txt          # unpinned, picks latest compatible
pip freeze | sort                           # copy the closure into constraints-ci.txt
pytest -q                                    # confirm still green BEFORE committing the new pins
```

Never edit a pin by hand to a version you haven't installed and tested — the
point of the file is that every version in it is one the suite actually passed
on.
