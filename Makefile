.PHONY: install data train tune evaluate app test lint clean

PY ?= python

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -c "from src.data.loader import download_ames; download_ames()"

train:
	$(PY) -m src.pipeline --model all

tune:
	$(PY) -m src.pipeline --model all --tune

evaluate:
	$(PY) -c "from src.pipeline import run; from src.config import CONFIG; run(CONFIG, tune=False)"

app:
	$(PY) -m streamlit run app.py

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -f models/*.joblib reports/figures/*.png reports/*.csv reports/experiment_log.json
