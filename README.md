# 🧠 MultimodalRAG

Un sistema RAG (Retrieval-Augmented Generation) multimodale avanzato per interrogare documenti PDF contenenti testo, immagini e tabelle. Utilizza embedding vettoriali, retrieval semantico e Large Language Models via Groq per fornire risposte accurate e contestualizzate.

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

---

## 🚀 Installazione e Setup

### Prerequisiti
- **Python 3.8+**
- **Docker** (per Qdrant)
- **Tesseract OCR** installato nel sistema
- **Poppler** per processing PDF

### 1. Clona il repository
```bash
git clone <repository-url>
cd multimodalrag
```

### 2. Installa le dipendenze
```bash
pip install -r requirements.txt
```

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
├── app/                    # Core dell'applicazione
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