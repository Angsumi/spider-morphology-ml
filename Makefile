# Reproduce everything: make all   (inside the spider-morphology-ml environment)
PY ?= python

.PHONY: all data evaluate notebook shap diagnostics key test
all: data evaluate notebook shap diagnostics key

data:
	$(PY) src/prepare_data.py

evaluate:
	$(PY) src/train_evaluate.py

notebook:
	$(PY) src/reproduce_notebook.py

shap:
	$(PY) src/explain_shap.py

diagnostics:
	$(PY) src/data_diagnostics.py

key:
	$(PY) src/build_key.py

test:
	$(PY) -m pytest -q
