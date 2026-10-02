# Contributing

Thanks for looking. This is a prototype for research and review, not an
enforcement system; contributions that make its claims easier to check are the
most useful kind.

Everything below works from a clean clone with **no model weights and no
dataset**. Measured on a laptop (macOS, Apple Silicon): clone to green tests in
under a minute.

## 1. Set up

Python 3.11-3.13 (CI runs 3.11 and 3.12).

```bash
git clone https://github.com/adivishall/two-wheeler-safety.git
cd two-wheeler-safety
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-ci.txt -c constraints-ci.txt   # pinned, model-free
pip check
```

The full stack (PyTorch, Ultralytics, EasyOCR, about 2 GB) is only needed to run
the real detector; see [docs/INSTALL.md](docs/INSTALL.md). No test needs it.

## 2. Run the checks

```bash
make check        # = ruff check . && mypy && pytest --cov  (the CI gates)
pytest tests/test_association.py -q                      # one file
pytest tests/test_association.py::test_hungarian_beats_greedy_nearest   # one test
```

- `ruff` (lint, line length 100) and `mypy` (scoped to the core modules listed
  in `pyproject.toml`) must be clean.
- Branch coverage of `modules/` must stay at or above 90% (currently ~94%).
- CI also runs the model-free evaluation CLIs; run them yourself if you touch
  `modules/*_eval.py` or an `evaluate_*.py` script, and write to a scratch
  directory, because the defaults overwrite the committed files in
  `eval/results/`:

  ```bash
  python3 evaluate_pipeline.py --out /tmp/eval
  python3 evaluate_system.py --json /tmp/eval/system.json
  python3 evaluate_ocr.py --simulate --out /tmp/eval --name ocr_sim
  ```

## 3. Run the dashboard locally

```bash
make demo            # seeds labelled synthetic records, serves http://127.0.0.1:5000
python3 demo.py --port 5001 --no-serve   # seed only
make serve           # serve whatever traffic.db holds
```

The demo needs no model. Photo and video uploads need the full stack and
weights (`MODEL_PATH`); without them those routes fail when they try to load
the model. Configuration is environment variables only, listed in the README
(`TRAFFIC_DB_PATH`, `EVIDENCE_DIR`, `DETECT_API_KEY`, ...). Never set
`FLASK_DEBUG=1` on anything reachable from another machine.

## 4. Where things are

| Path | What |
|---|---|
| `modules/` | importable logic: association, tracking, OCR voting, state machines, confidence, evidence, DB, validation |
| `app.py`, `templates/frontend.html` | Flask API and the single-page dashboard |
| `main.py`, `main_ocr.py` | video / photo CLIs |
| `evaluate_*.py`, `audit_dataset.py`, `compare_models.py`, `benchmark.py` | evaluation CLIs; results go to `eval/results/` |
| `tests/` | model-free tests (fakes for the model and OCR reader) |
| `docs/` | start with [VERIFICATION.md](docs/VERIFICATION.md) and [DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md) |

Not in git, and never to be committed: datasets, weights (`*.pt`), `runs/`,
`traffic.db`, `evidence/`, `reports/`, videos. `eval/results/*.{json,md,csv}`
**is** committed: it is the source the docs cite.

## 5. Making a change

1. Pick or open an issue. Issues labelled `good first issue` are model-free
   and say how to test them.
2. Branch from `main` (`git checkout -b fix/short-name`). Note that
   `feature/flagship-hardening` and PR #18 rewrite large parts of `app.py`,
   `modules/video_detector.py` and the docs; if your change touches those,
   say so in the PR.
3. For a bug: write the test first and show it fails, then fix. Name tests
   after the behaviour (`test_detect_rejects_...`), not the function.
4. Keep commits small and their messages factual: what was wrong, how it was
   found, what changed, which test pins it.
5. Run `make check`, push, open a PR against `main`, and fill in the template.
   CI must be green.

Rules for numbers: never quote a metric without its source file in
`eval/results/` (or say it has none); state dataset, split, metric and sample
size; keep synthetic and real-data results apart. If a change alters a
committed result, regenerate the file in the same PR and say why it changed.

## 6. Reporting a bug

Use the bug template. The most useful report has:

- the commit (`git rev-parse --short HEAD`) and how you ran it (CLI, web route,
  test);
- a minimal input: for detection or association problems, the frame (or a
  synthetic one) **and the boxes** (class, xyxy, confidence), because most
  pipeline bugs reproduce from boxes alone without the model, like the tests
  in `tests/test_association.py`;
- expected vs actual output (plate, violation, confidence breakdown, HTTP status
  and body);
- logs, with paths and keys removed.

Do not attach images of real people or real number plates you do not have the
right to share; the photos in `samples/` or synthetic boxes are enough.
Security problems go through private reporting, not issues: see
[SECURITY.md](SECURITY.md).
