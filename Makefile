# ===========================
# MULTIMODAL RAG - MAKEFILE
# ===========================

.PHONY: help install setup clean run reindex test docker-up docker-down

# Mostra questo messaggio di aiuto
help:
	@echo "MultimodalRAG - Makefile Commands"
	@echo "=================================="
	@echo ""
	@echo "Setup e Installazione:"
	@echo "  make install     - Installa tutte le dipendenze Python"
	@echo "  make setup       - Setup completo del progetto"
	@echo ""
	@echo "Esecuzione:"
	@echo "  make run         - Avvia l'applicazione Streamlit"
	@echo "  make reindex     - Re-indicizza tutti i documenti"
	@echo ""
	@echo "Database:"
	@echo "  make docker-up   - Avvia Qdrant con Docker"
	@echo "  make docker-down - Ferma Qdrant"
	@echo ""
	@echo "Utilità:"
	@echo "  make clean       - Pulisce file temporanei e cache"
	@echo "  make test        - Esegue i test (quando disponibili)"

# Installa le dipendenze
install:
	pip install -r requirements.txt

# Setup completo del progetto
setup: install
	@echo "Verificando configurazione..."
	@if [ ! -f .env ]; then \
		echo "ATTENZIONE: File .env non trovato!"; \
		echo "Copia .env.example in .env e configura le tue API keys"; \
		cp .env.example .env; \
	fi
	@echo "Setup completato!"

# Pulisce file temporanei
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name ".DS_Store" -delete 2>/dev/null || true
	@echo "Pulizia completata!"

# Avvia l'applicazione
run:
	streamlit run streamlit_app/Home.py

# Re-indicizza i documenti
reindex:
	python scripts/reindex_with_context.py

# Avvia Qdrant con Docker
docker-up:
	docker run -d -p 6333:6333 --name qdrant-multimodal qdrant/qdrant

# Ferma Qdrant
docker-down:
	docker stop qdrant-multimodal || true
	docker rm qdrant-multimodal || true

# Test (placeholder per futuri test)
test:
	@echo "Test non ancora implementati"
