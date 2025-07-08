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

# === INSTALLATION ===
install: ## Installa le dipendenze
	$(PIP) install -r requirements.txt

install-dev: ## Installa dipendenze di sviluppo
	$(PIP) install -r requirements.txt
	$(PIP) install pre-commit black flake8 isort bandit safety pytest pytest-cov mypy
	pre-commit install

setup-dev: install-dev ## Setup completo ambiente di sviluppo
	@echo "Setup ambiente di sviluppo completato"
	@echo "Ricorda di copiare .env.example in .env e configurare le API keys"

# === CODE QUALITY ===
lint: ## Esegue linting del codice
	$(FLAKE8) src/ streamlit_app/ tests/ scripts/ --max-line-length=88 --extend-ignore=E203,W503
	$(MYPY) src/ --ignore-missing-imports

format: ## Formatta il codice
	$(BLACK) src/ streamlit_app/ tests/ scripts/ --line-length=88
	$(ISORT) src/ streamlit_app/ tests/ scripts/ --profile black

format-check: ## Controlla la formattazione senza modificare
	$(BLACK) --check src/ streamlit_app/ tests/ scripts/ --line-length=88
	$(ISORT) --check-only src/ streamlit_app/ tests/ scripts/ --profile black

quality-check: ## Esegue controlli di qualità completi
	$(PYTHON) scripts/quality_check.py

# === RUN COMMANDS (integrati con scripts/run.py) ===
run: ## Avvia l'applicazione in modalità locale
	$(PYTHON) scripts/run.py local

run-dev: ## Avvia l'applicazione in modalità sviluppo
	$(PYTHON) scripts/run.py dev

# === DOCKER COMMANDS (integrati con scripts/run.py) ===
docker-build: ## Costruisce l'immagine Docker
	docker build -t multimodalrag .

docker-run: docker-build ## Avvia l'applicazione con Docker
	$(PYTHON) scripts/run.py docker

docker-compose-up: ## Avvia con Docker Compose
	$(PYTHON) scripts/run.py docker-compose

docker-compose-down: ## Ferma Docker Compose
	docker-compose down

# === QDRANT COMMANDS ===
qdrant-start: ## Avvia Qdrant in Docker
	docker run -d -p 6333:6333 --name qdrant qdrant/qdrant

qdrant-stop: ## Ferma Qdrant
	docker stop qdrant && docker rm qdrant

# === UTILITY COMMANDS ===
clean: ## Pulisce file temporanei e cache
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -name ".DS_Store" -delete
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf temp/
	rm -rf data/temp/

logs: ## Mostra i log dell'applicazione
	@if [ -d "logs" ]; then tail -f logs/*.log; else echo "📝 Nessun file di log trovato"; fi

check-deps: ## Controlla le dipendenze
	$(PIP) check

update-deps: ## Aggiorna requirements.txt con le dipendenze attuali  
	$(PIP) freeze > requirements.txt

security-check: ## Controlla vulnerabilità di sicurezza
	safety check
	bandit -r src/ streamlit_app/ scripts/ -f json

# === COMPOUND COMMANDS ===
check-all: format-check lint test ## Esegue tutti i controlli di qualità

ci: install-dev check-all ## Pipeline CI completa

dev-setup: setup-dev qdrant-start ## Setup completo per sviluppo (include Qdrant)
	@echo "Ambiente di sviluppo pronto!"
	@echo "Prossimi passi:"
	@echo "   1. Copia .env.example in .env"
	@echo "   2. Configura GROQ_API_KEY nel file .env"
	@echo "   3. Esegui 'make run' per avviare l'applicazione"
