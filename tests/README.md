# Tests per MultimodalRAG

Questa directory contiene i test per il progetto MultimodalRAG.

## Struttura Test

```
tests/
├── __init__.py
├── unit/                   # Test unitari
│   ├── test_embedder.py
│   ├── test_pdf_parser.py
│   ├── test_qdrant_utils.py
│   └── test_models.py
├── integration/            # Test di integrazione
│   ├── test_indexing_pipeline.py
│   └── test_retrieval_pipeline.py
├── fixtures/               # Dati di test
│   ├── sample.pdf
│   └── test_data.json
└── conftest.py            # Configurazione pytest
```

## Esecuzione Test

```bash
# Installa pytest
pip install pytest pytest-cov pytest-mock

# Esegui tutti i test
pytest

# Esegui test con coverage
pytest --cov=src --cov-report=html

# Esegui solo test unitari
pytest tests/unit/

# Esegui test specifico
pytest tests/unit/test_embedder.py -v
```

## Convenzioni

- Test unitari: testano singole funzioni/classi in isolamento
- Test integrazione: testano interazioni tra componenti
- Mock per dipendenze esterne (Qdrant, Groq API)
- Fixture per dati di test riusabili
