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


# Makefile per automatizzare task comuni
.PHONY: help install test lint format clean setup-dev run docker-build docker-run test-structure

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

setup-dev: install-dev ## Setup completo ambiente di sviluppo
	@echo "Setup ambiente di sviluppo completato"
	@echo "Ricorda di copiare .env.example in .env e configurare le API keys"


lint: ## Esegue linting del codice
	$(FLAKE8) src/ streamlit_app/ tests/ --max-line-length=88 --extend-ignore=E203,W503
	$(MYPY) src/ --ignore-missing-imports

format: ## Formatta il codice
	$(BLACK) src/ streamlit_app/ tests/ scripts/ --line-length=88
	$(ISORT) src/ streamlit_app/ tests/ scripts/ --profile black

format-check: ## Controlla la formattazione senza modificare
	$(BLACK) --check src/ streamlit_app/ tests/ scripts/ --line-length=88
	$(ISORT) --check-only src/ streamlit_app/ tests/ scripts/ --profile black

quality-check: ## Esegue controlli di qualità completi
	$(PYTHON) scripts/quality_check.py

clean: ## Pulisce file temporanei e cache
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -name ".DS_Store" -delete
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf temp/

run: ## Avvia l'applicazione in modalità locale
	$(PYTHON) scripts/run.py local

run-dev: ## Avvia l'applicazione in modalità sviluppo
	$(PYTHON) scripts/run.py dev

docker-build: ## Costruisce l'immagine Docker
	docker build -t multimodalrag .

docker-run: docker-build ## Avvia l'applicazione con Docker
	$(PYTHON) scripts/run.py docker

docker-compose-up: ## Avvia con Docker Compose
	$(PYTHON) scripts/run.py docker-compose

docker-compose-down: ## Ferma Docker Compose
	docker-compose down

logs: ## Mostra i log dell'applicazione
	@if [ -d "logs" ]; then tail -f logs/*.log; else echo "📝 Nessun file di log trovato"; fi

check-deps: ## Controlla le dipendenze
	$(PIP) check

update-deps: ## Aggiorna requirements.txt con le dipendenze attuali  
	$(PIP) freeze > requirements.txt

security-check: ## Controlla vulnerabilità di sicurezza
	$(PIP) install safety
	safety check

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
