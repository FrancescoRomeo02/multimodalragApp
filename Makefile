# Makefile per automatizzare task comuni
.PHONY: help install test lint format clean setup-dev run docker-build docker-run

# Variabili
PYTHON := python3
PIP := pip
PYTEST := pytest
BLACK := black
FLAKE8 := flake8
ISORT := isort
MYPY := mypy

help: ## Mostra questo help
	@echo "Comandi disponibili:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Installa le dipendenze
	$(PIP) install -r requirements.txt

install-dev: ## Installa dipendenze di sviluppo
	$(PIP) install -r requirements.txt
	pre-commit install

test: ## Esegue tutti i test
	$(PYTEST) tests/ -v

test-unit: ## Esegue solo i test unitari
	$(PYTEST) tests/unit/ -v

test-integration: ## Esegue solo i test di integrazione
	$(PYTEST) tests/integration/ -v

test-cov: ## Esegue test con coverage
	$(PYTEST) tests/ --cov=src --cov-report=html --cov-report=term

lint: ## Controlla stile del codice
	$(FLAKE8) src/ tests/
	$(MYPY) src/

format: ## Formatta il codice
	$(BLACK) src/ tests/
	$(ISORT) src/ tests/

format-check: ## Controlla se il codice è formattato correttamente
	$(BLACK) --check src/ tests/
	$(ISORT) --check src/ tests/

clean: ## Pulisce file temporanei
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/

setup-dev: install-dev ## Setup completo ambiente sviluppo
	@echo "Ambiente di sviluppo configurato!"

run: ## Avvia l'applicazione Streamlit
	streamlit run streamlit_app/Home.py

run-indexer: ## Esegue l'indicizzazione dei documenti
	$(PYTHON) scripts/reindex_with_context.py

docker-build: ## Build dell'immagine Docker
	docker build -t multimodal-rag .

docker-run: ## Esegue l'applicazione in Docker
	docker run -p 8501:8501 multimodal-rag

qdrant-start: ## Avvia Qdrant in Docker
	docker run -d -p 6333:6333 qdrant/qdrant

check-all: format-check lint test ## Esegue tutti i controlli di qualità

ci: install-dev check-all ## Pipeline CI completa
