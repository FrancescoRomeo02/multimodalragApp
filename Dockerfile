# Multi-stage build per ridurre dimensioni immagine finale
FROM python:3.11-slim as builder

# Installa dipendenze di sistema necessarie
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-ita \
    tesseract-ocr-eng \
    poppler-utils \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Crea directory di lavoro
WORKDIR /app

# Copia requirements e installa dipendenze Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage finale
FROM python:3.11-slim

# Installa solo le dipendenze runtime necessarie
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-ita \
    tesseract-ocr-eng \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Crea utente non-root per sicurezza
RUN useradd --create-home --shell /bin/bash app

# Copia dipendenze Python dal builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Crea directory app e imposta permessi
WORKDIR /app
RUN chown -R app:app /app

# Switcha all'utente non-root
USER app

# Copia codice applicazione
COPY --chown=app:app . .

# Crea directory per log e dati
RUN mkdir -p logs data/raw

# Esponi porta Streamlit
EXPOSE 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Comando di avvio
CMD ["streamlit", "run", "streamlit_app/Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
