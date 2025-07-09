# MultimodalRAG

**MultimodalRAG** is an advanced Retrieval-Augmented Generation (RAG) system designed to process and query PDF documents containing **text, images, and tables**. It leverages **multimodal embeddings**, **semantic retrieval**, and **Large Language Models (LLMs)** via Groq to deliver accurate, source-grounded answers.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Key Features

* **Multi-PDF Upload** with automatic preprocessing
* **Smart Extraction** of text, images, and tabular data
* **Advanced Semantic Chunking** using hybrid strategies
* **Multimodal Embeddings** (text and image) via CLIP/BGE
* **Vector Indexing** with Qdrant
* **Context-Aware Retrieval** using vector similarity
* **LLM-Driven Response Generation** (GPT-4o, Claude, LLaMA via Groq API)
* **Modern Web Interface** powered by Streamlit
* **Table Interpretation** with caption/context extraction
* **OCR and Image Analysis** with automatic captioning
* **Integrated Monitoring & Performance Metrics**
* **Containerized Deployment** with Docker & Docker Compose
* **Comprehensive Test Suite** with coverage support

---

## Technology Stack

| Component  | Technology                 | Description                     |
| ---------- | -------------------------- | ------------------------------- |
| Frontend   | Streamlit                  | Interactive web UI              |
| Vector DB  | Qdrant                     | High-performance vector search  |
| Embeddings | CLIP / BGE / OpenAI        | Multimodal encoding models      |
| LLMs       | Groq API                   | Access to GPT, Claude, LLaMA    |
| Parsing    | PyMuPDF / Tesseract        | Text/table/OCR extraction       |
| Pipeline   | LangChain                  | RAG orchestration framework     |
| CI/CD      | Pre-commit, GitHub Actions | Dev workflow automation         |

---

## Data Flow

<img width="932" height="584" alt="Image" src="https://github.com/user-attachments/assets/7c0a80fd-6252-4012-96f9-4383873c0418" />

---


## Quick Start

### 1. Clone the repository

```bash
git clone <repository-url>
cd multimodalrag
```

### 2. Set up API keys

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 3. Run the full stack (App + Qdrant)

```bash
docker-compose up -d
```

### 4. Open the app

```bash
open http://localhost:8501
```

---

## Local Setup (Alternative)

```bash
pip install -r requirements.txt
docker run -d -p 6333:6333 qdrant/qdrant
streamlit run streamlit_app/Home.py
```

---

## Pre-Launch Checklist

* [ ] Python 3.8+ installed
* [ ] Docker & Docker Compose available
* [ ] `.env` configured with valid GROQ\_API\_KEY
* [ ] Ports 8501 (Streamlit) and 6333 (Qdrant) available

---

## Functional Test Scenarios

1. **Upload** a sample PDF file
2. **Textual query**: "What is this document about?"
3. **Image query**: "Show me related diagrams or illustrations"
4. **Table query**: "What data is shown in the tables?"

---

## Troubleshooting

**Qdrant not reachable**

```bash
docker ps
curl http://localhost:6333/health
```

**GROQ API not responding**

```bash
cat .env | grep GROQ_API_KEY
```

**Port conflict**

```bash
streamlit run streamlit_app/Home.py --server.port=8502
```

---

## Developer Setup

### Docker-based Setup

```bash
cp .env.example .env
# Add GROQ_API_KEY
docker-compose up -d
```

### Makefile-based Setup

```bash
make setup-dev       # Install dependencies & pre-commit hooks
make qdrant-start    # Launch Qdrant
make run             # Launch Streamlit App
```

---

## Makefile Commands Reference

```bash
make help                  # List all commands
make setup-dev             # Full local dev setup
make run                   # Launch the main app
make run-test-interface    # Launch test & evaluation interface
make run-indexer           # Re-index all documents
make run-metrics-demo      # Run metrics demonstration
make clean                 # Clean temporary files
make docker-build          # Build the Docker image
make ci                    # Run CI pipeline
```

---

## Metrics & Evaluation

The system includes a comprehensive metrics framework to evaluate RAG performance:

### 🧪 Test Interface

Launch the dedicated test and evaluation interface:

```bash
# Start test interface (port 8503)
make run-test-interface

# Or directly:
python scripts/run_test_interface.py
```

The test interface allows you to:
- Input queries with expected answers
- Specify relevant pages/documents for ground truth
- Automatically calculate all RAG metrics
- View detailed performance analysis
- Export results for further analysis

### Metrics Categories

* **Retrieval Metrics**: Recall@k, Precision@k, Mean Reciprocal Rank (MRR)
* **Generation Metrics**: BERTScore, Exact Match (EM)
* **Multimodal Metrics**: Image/Text accuracy
* **Cost & Efficiency Metrics**: Token count, response time, embedding efficiency

### Usage Example

```python
from metrics.main_metrics import track_retrieval, track_generation

# Track retrieval performance
track_retrieval(
    query_id="q1",
    relevant_docs=["doc1", "doc2"],
    retrieved_docs=["doc1", "doc3", "doc2"],
    retrieval_time=0.15
)

# Track generation quality
track_generation(
    query_id="q1", 
    generated_text="Roma è la capitale d'Italia",
    reference_text="Roma è la capitale italiana",
    generation_time=1.2,
    token_count=15
)
```

### Demo & Export

```bash
# Run metrics demo
python metrics/demo_metrics.py

# Export metrics to JSON
from metrics.main_metrics import global_metrics_collector
global_metrics_collector.export_metrics("my_metrics.json")
```

See [metrics/README.md](metrics/README.md) for detailed documentation.

---

## Project Structure

```
multimodalrag/
├── src/                  
│   ├── config.py         
│   ├── core/             
│   ├── llm/              
│   ├── pipeline/         
│   └── utils/            
├── data/                 
│   ├── models/           
│   └── raw/              
├── scripts/              
├── streamlit_app/        
├── logs/                 
```

---

## Workflow

1. **Upload PDF** via UI
2. **Automatic Processing** and semantic chunking
3. **Multimodal Indexing** (text + image)
4. **Query** using text, image or hybrid inputs
5. **Response Generation** using LLMs

---

## Running Indexer Manually

```bash
python scripts/reindex_with_context.py
```

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for more details.

---

## Documentation

* [Quick Start Guide](docs/GUIDA_AVVIO.md)
* [Semantic Chunking](docs/CHUNKING_SEMANTICO.md)
* [Makefile Command Reference](#makefile-commands-reference)

---
