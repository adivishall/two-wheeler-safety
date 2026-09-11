# Convenience targets. `make demo` is the one-command demo; the rest mirror CI
# so a local `make check` matches the pipeline exactly.
.PHONY: help demo demo-real serve test cov lint type check install-ci

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
