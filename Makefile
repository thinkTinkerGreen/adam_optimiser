.PHONY: setup lint type test smoke p1 p2 p3 p4 p5 figures readme check all clean

VENV := .venv
PYTHON := $(VENV)/bin/python
RUFF := $(VENV)/bin/ruff
MYPY := $(VENV)/bin/mypy
PYTEST := $(VENV)/bin/pytest

setup:
	uv venv --python 3.12 $(VENV)
	uv pip install --python $(PYTHON) -e ".[dev]"

lint:
	$(RUFF) check .
	$(RUFF) format --check .

lint-fix:
	$(RUFF) check --fix .
	$(RUFF) format .

type:
	$(MYPY) src/

test:
	$(PYTEST) tests/ --cov=src/adamlab --cov-report=term-missing --cov-report=html

smoke:
	$(PYTHON) -m scripts.smoke_test

p1:
	$(PYTHON) -m experiments.p1_adam_by_hand

p2:
	$(PYTHON) -m experiments.p2_bias_correction

p3:
	$(PYTHON) -m experiments.p3_update_ratio

p4:
	$(PYTHON) -m experiments.p4_cosine_vs_wsd

p5:
	$(PYTHON) -m experiments.p5_lr_sweep

figures:
	$(PYTHON) -m scripts.make_figures

readme:
	$(PYTHON) -m scripts.render_readme

check: lint type test
	$(PYTHON) -m scripts.check_readme

all: p1 p2 p3 p4 p5 figures readme check

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage dist build *.egg-info
