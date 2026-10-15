# Uses Python 3.12 (eval-protocol's hydra dependency failed to install on 3.13 when this was written).
PY ?= .venv/bin/python

setup:            ## create .venv and install everything
	uv venv -p 3.12 .venv || python3.12 -m venv .venv
	uv pip install -p .venv -r requirements-local.txt || .venv/bin/pip install -r requirements-local.txt

demo:             ## whole local pipeline on a synthetic market (no downloads, no API key)
	$(PY) scripts/02_build_panel.py --synthetic
	$(PY) scripts/03_baselines.py
	$(PY) scripts/04_make_rft_data.py --no-think
	$(PY) scripts/07_report.py

check:            ## offline checks of the scoring, reward and leak logic
	$(PY) -m pytest checks -q

data:             ## step 1: SEC + prices (needs SEC_USER_AGENT)
	$(PY) scripts/01_fetch_data.py
panel:            ## step 2
	$(PY) scripts/02_build_panel.py
baselines:        ## step 3: v0, v1, optimizer, leak ablation
	$(PY) scripts/03_baselines.py
rft-data:         ## step 4
	$(PY) scripts/04_make_rft_data.py --no-think
upload:           ## step 5 (needs FIREWORKS_API_KEY)
	$(PY) scripts/05_upload_datasets.py
base-model:       ## step 6: zero-shot and best-of-16 with FINAGIC_EVAL_MODEL
	$(PY) scripts/06_sample_model.py --label base --n 16 --no-think
report:           ## step 7: tables, charts, GUIDE.md
	$(PY) scripts/07_report.py

.PHONY: setup demo check data panel baselines rft-data upload base-model report
