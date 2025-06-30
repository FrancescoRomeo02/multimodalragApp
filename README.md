# 🧠 MultimodalRAG

Un sistema RAG (Retrieval-Augmented Generation) multimodale avanzato per interrogare documenti PDF contenenti testo, immagini e tabelle. Utilizza embedding vettoriali, retrieval semantico e Large Language Models via Groq per fornire risposte accurate e contestualizzate.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Caratteristiche principali

- 📄 **Upload multipli di PDF** con processing automatico
- 🔍 **Estrazione intelligente** di testo, immagini e tabelle
- 🧠 **Embedding multimodali** (testo e immagini) tramite modelli CLIP/BGE
- 🗂️ **Indicizzazione vettoriale** su database Qdrant
- 🔎 **Ricerca semantica** avanzata con similarità vettoriale
- 🤖 **Generazione risposte** tramite LLM (GPT-4o, Claude, Llama)
- 🖥️ **Interfaccia web** moderna e user-friendly con Streamlit
- 📊 **Supporto tabelle** con estrazione contesto e caption
- 🖼️ **Analisi immagini** con OCR e descrizioni automatiche
- 📈 **Monitoring e metriche** delle performance integrate
- 🐳 **Deploy containerizzato** con Docker/Docker Compose
- 🧪 **Suite di test** completa con coverage

---

## 🛠️ Stack Tecnologico

| Componente | Tecnologia | Descrizione |
|------------|------------|-------------|
| **Frontend** | Streamlit | Interface web interattiva |
| **Database** | Qdrant | Database vettoriale per embedding |
| **Embedding** | OpenAI/BGE/CLIP | Modelli per encoding multimodale |
| **LLM** | Groq API | Claude, GPT-4o, Llama via API |
| **Processing** | PyMuPDF, Tesseract | Estrazione testo, tabelle e OCR |
| **Orchestrazione** | LangChain | Framework per pipeline LLM |
| **Testing** | Pytest | Test unitari e di integrazione |
| **CI/CD** | Pre-commit, GitHub Actions | Quality assurance automatizzata |

---

## 🚀 Quick Start

### ⚡ Avvio Rapido

```bash
# 1. Clona il repository
git clone <repository-url>
cd multimodalrag

# 2. Configura la chiave API (OBBLIGATORIO)
cp .env.example .env
# Modifica il file .env e aggiungi la tua GROQ_API_KEY

# 3. Avvia tutto con Docker (include Qdrant + App)
docker-compose up -d

# 4. Accedi all'applicazione
open http://localhost:8501
```

### 🔧 Setup Locale (Alternativo)

```bash
# 1. Installa dipendenze Python
pip install -r requirements.txt

# 2. Avvia Qdrant in Docker
docker run -d -p 6333:6333 qdrant/qdrant

# 3. Avvia l'applicazione
streamlit run streamlit_app/Home.py
```

### 📋 Checklist

- [ ] **Docker installato** (per Qdrant e deployment)
- [ ] **Python 3.8+** installato
- [ ] **GROQ_API_KEY** configurata nel file `.env`
- [ ] **Porte libere**: 8501 (Streamlit), 6333 (Qdrant)

### 🎯 Test Funzionalità Principali

1. **Upload PDF**: Carica un documento di esempio
2. **Query Testuale**: "Di cosa parla questo documento?"
3. **Query Multimodale**: "Mostrami le immagini relative a..."
4. **Query Tabelle**: "Quali dati ci sono nelle tabelle?"

### 🐛 Risoluzione Problemi Comuni

**Errore Qdrant non raggiungibile:**
```bash
docker ps  # Verifica che Qdrant sia running
curl http://localhost:6333/health  # Test connessione
```

**Errore GROQ API:**
```bash
# Verifica che GROQ_API_KEY sia nel file .env
cat .env | grep GROQ_API_KEY
```

**Porta già in uso:**
```bash
# Cambia porta Streamlit
streamlit run streamlit_app/Home.py --server.port=8502
```

---

## 🔧 Setup Avanzato per Sviluppatori

### Con Docker Compose

```bash
# 1. Clona il repository
git clone <repository-url>
cd multimodalrag

# 2. Configura ambiente
cp .env.example .env
# Modifica .env con la tua GROQ_API_KEY

# 3. Avvia tutti i servizi
docker-compose up -d

# 4. Accedi all'applicazione
open http://localhost:8501
```

### Setup Locale con Makefile

```bash
# 1. Installa dipendenze
make setup-dev

# 2. Avvia Qdrant
make qdrant-start

# 3. Avvia applicazione
make run
```

---

## 🎓 Guida per Comandi Makefile

Il progetto include un **Makefile** per automatizzare le operazioni comuni:

### 📋 Comandi Principali

```bash
# Vedi tutti i comandi disponibili
make help

# Setup ambiente di sviluppo completo (include pre-commit hooks)
make setup-dev

# Avvia l'applicazione Streamlit
make run

# Avvia Qdrant in Docker
make qdrant-start

# Esegui re-indicizzazione documenti
make run-indexer
```

### 🧪 Testing e Quality Assurance

```bash
# Esegui tutti i test
make test

# Test con coverage HTML (apri htmlcov/index.html dopo)
make test-cov

# Solo test unitari
make test-unit

# Solo test di integrazione  
make test-integration

# Controlli qualità del codice (lint + type checking)
make lint

# Formattazione automatica del codice
make format

# Pipeline CI completa (test + lint + format)
make check-all
```

### 🐳 Docker Commands

```bash
# Build immagine Docker dell'app
make docker-build

# Esegui app in Docker
make docker-run

# Avvia Qdrant
make qdrant-start
```

### 🧹 Manutenzione

```bash
# Pulisci file temporanei e cache
make clean

# Pipeline CI completa per integrazione continua
make ci
```

---

## 📊 Monitoring e Performance

### Attivazione Monitoring

Per abilitare il tracking delle performance, nel file `.env`:
```bash
ENABLE_PERFORMANCE_MONITORING=true
```

### Monitoring Completo con Grafana

```bash
# Avvia stack completo con Grafana + Prometheus
docker-compose --profile monitoring up -d

# Accedi ai servizi:
# App: http://localhost:8501
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
```

### Metriche Disponibili

- **Tempo di risposta** per ogni query
- **Success rate** delle operazioni
- **Uso memoria e CPU** in tempo reale
- **Tipi di query** più frequenti
- **Documenti recuperati** per query

---

## 🚀 Deployment

Vedi [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) per istruzioni dettagliate su:
- Deployment locale e cloud
- Configurazioni di produzione
- Monitoring e backup
- Troubleshooting

---

## 🛠️ Sviluppo

### Setup Ambiente

```bash
# Setup completo per sviluppo
make setup-dev

# Formattazione codice
make format

# Controlli qualità
make lint

# Clean del progetto
make clean
```

### Contribuire

1. Fork del progetto
2. Crea feature branch (`git checkout -b feature/amazing-feature`)
3. Commit delle modifiche (`git commit -m 'Add amazing feature'`)
4. Push al branch (`git push origin feature/amazing-feature`)
5. Apri una Pull Request

---

## 📄 Licenza

Questo progetto è rilasciato sotto licenza MIT. Vedi il file `LICENSE` per dettagli.

### 3. Configura le variabili d'ambiente
Crea un file `.env` nella root del progetto:
```bash
# API Keys
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_key_here  # Optional, per embedding OpenAI

# Database
QDRANT_URL=http://localhost:6333

# Logging
LOG_LEVEL=INFO
```

### 4. Avvia Qdrant
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 5. Indicizza i documenti
```bash
python scripts/reindex_with_context.py
```

### 6. Avvia l'applicazione
```bash
streamlit run streamlit_app/Home.py
```

---

## 📁 Struttura del Progetto

```
multimodalrag/
├── src/                    # Core dell'applicazione
│   ├── config.py          # Configurazioni centrali
│   ├── core/              # Logica di business
│   ├── llm/               # Client per LLM
│   ├── pipeline/          # Pipeline indicizzazione/retrieval
│   └── utils/             # Utilità condivise
├── data/                  # Dati del progetto
│   ├── models/           # Modelli ML pre-addestrati
│   └── raw/              # PDF da processare
├── scripts/              # Script di utilità
├── streamlit_app/        # Interface web Streamlit
└── logs/                 # File di log
```

---

## 🔧 Utilizzo

1. **Upload PDF**: Carica i tuoi documenti tramite l'interfaccia web
2. **Indicizzazione**: I documenti vengono processati automaticamente
3. **Query**: Poni domande sui contenuti dei documenti
4. **Risultati**: Ricevi risposte contestualizzate con riferimenti alle fonti

---

## 🤝 Contribuire

Le contribuzioni sono benvenute! Per favore:
1. Fai un fork del progetto
2. Crea un branch per la tua feature (`git checkout -b feature/AmazingFeature`)
3. Committa le modifiche (`git commit -m 'Add some AmazingFeature'`)
4. Pusha al branch (`git push origin feature/AmazingFeature`)
5. Apri una Pull Request

---

## 📄 Licenza

Questo progetto è rilasciato sotto licenza MIT. Vedi il file `LICENSE` per dettagli.

---

## 📚 Documentazione Completa

- 📋 **[Guida Rapida Avvio](docs/GUIDA_AVVIO.md)** - Setup e avvio
- 🧪 **[Testing Guide](tests/README.md)** - Documentazione suite di test
- 🛠️ **[Makefile Commands](#-guida-per-comandi-makefile)** - Automazione sviluppo

---