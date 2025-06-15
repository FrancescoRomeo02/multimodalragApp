# 🧠 MultimodalRAG

Un sistema RAG multimodale per interrogare PDF contenenti testo, immagini e tabelle, sfruttando embedding, retrieval semantico e LLM via gorq. Interfaccia web tramite Streamlit.

---

## 🚀 Funzionalità

- 📄 Upload multipli di PDF
- ✂️ Estrazione automatica di testo, immagini e tabelle
- 🧠 Generazione di embedding (testo e immagini)
- 🗂️ Indicizzazione su Qdrant
- 🔍 Recupero dei chunk rilevanti tramite similarità vettoriale
- 🤖 Risposte generate da LLM con contesto documentale
- 🖥️ Interfaccia web user-friendly con Streamlit

---

## 🛠️ Stack Tecnologico

| Componente    | Tecnologia                     |
|---------------|--------------------------------|
| Frontend      | Streamlit                      |
| Retrieval     | Qdrant                         |
| Embedding     | OpenAI / BGE / CLIP            |
| OCR & Tabelle | PyMuPDF, pdfplumber, Tesseract |
| LLM API       | gorq (Claude, GPT-4o, ecc.)    |
| Orchestrazione| Python + moduli custom         |

---

## 📦 App Necessarie
- Python 3.8+
- pip install -r requirements.txt
- Qdrant server in esecuzione
- Chiavi API per OpenAI e gorq
- Tesseract OCR installato
- Applicazione 'poppler' per l'estrazione di tabelle