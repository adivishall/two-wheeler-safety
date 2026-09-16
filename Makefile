# Convenience targets. `make demo` is the one-command demo; the rest mirror CI
# so a local `make check` matches the pipeline exactly.
.PHONY: help demo demo-real serve test cov lint type check install-ci \
        eval eval-model eval-audit eval-compare eval-ocr

help:
	@echo "Targets:"
	@echo "  make demo        Seed a self-contained demo dataset and serve it"
	@echo "  make demo-real   Seed with the real detector (needs weights + torch)"
	@echo "  make serve       Run the app against the current traffic.db"
	@echo "  make test        Run the test suite"
	@echo "  make cov         Run tests with branch coverage"
	@echo "  make lint        ruff check"
	@echo "  make type        mypy (core modules)"
	@echo "  make check       lint + type + cov (what CI runs)"
	@echo ""
	@echo "Evaluation (see docs/MODEL_EVALUATION.md, docs/END_TO_END_EVALUATION.md):"
	@echo "  make eval         Pipeline evaluation — no weights or dataset needed"
	@echo "  make eval-audit   Audit split hygiene + build de-leaked splits"
	@echo "  make eval-model   Detector metrics + failure artifacts (needs weights)"
	@echo "  make eval-compare A/B every checkpoint on the de-leaked test split"
	@echo "  make eval-ocr     Single-frame vs temporal OCR (simulated noise)"
	@echo "  make install-ci  Install the pinned model-free deps"

demo:
	python3 demo.py

demo-real:
	python3 demo.py --real

serve:
	python3 app.py

test:
	pytest -q

cov:
	pytest -q --cov --cov-report=term-missing

lint:
	ruff check .

type:
	mypy

check: lint type cov

install-ci:
	pip install -r requirements-ci.txt -c constraints-ci.txt

# ---- Evaluation ------------------------------------------------------------
# `make eval` and `make eval-ocr` need no weights, no dataset and no network,
# so they run anywhere. The rest need the local weights + dataset, which are
# gitignored. MODEL and DATA can be overridden:
#   make eval-model MODEL=runs/detect/other/weights/best.pt
MODEL ?= runs/detect/traffic_model-2/weights/best.pt
DATA  ?= master_traffic_violation_dataset/data.yaml
CLEAN_DATA ?= eval/clean_splits/data.yaml

eval:
	python3 evaluate_pipeline.py

eval-ocr:
	python3 evaluate_ocr.py --simulate --sweep --out eval/results \
	    --name ocr_policy_simulation

eval-audit:
	python3 audit_dataset.py --data $(DATA) --write-clean-split eval/clean_splits
	python3 dataset_manifest.py --data $(DATA)

# Evaluates the held-out TEST split, de-leaked — run `make eval-audit` first.
eval-model:
	python3 evaluate_model.py --model $(MODEL) --data $(CLEAN_DATA) \
	    --split test --save-artifacts eval --benchmark \
	    --name eval_$(notdir $(patsubst %/weights/best.pt,%,$(MODEL)))_test_clean

eval-compare:
	python3 compare_models.py --data $(CLEAN_DATA) --split test
