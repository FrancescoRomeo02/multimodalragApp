import pytest
import sys
import os
from unittest.mock import Mock

# Aggiungi src al path per import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

@pytest.fixture
def sample_embedder():
    """Mock embedder per test"""
    mock = Mock()
    mock.embedding_dim = 384
    mock.embed_query.return_value = [0.1] * 384
    mock.embed_documents.return_value = [[0.1] * 384] * 3
    return mock

@pytest.fixture
def sample_text_element():
    """Elemento testo di esempio"""
    from src.core.models import TextElement, TextMetadata
    return TextElement(
        text="Questo è un testo di esempio per i test.",
        metadata=TextMetadata(
            source="test.pdf",
            page=1,
            type="text"
        )
    )

@pytest.fixture
def sample_pdf_content():
    """Contenuto PDF di esempio per test"""
    return {
        "texts": [
            {
                "text": "Introduzione al machine learning",
                "metadata": {"source": "test.pdf", "page": 1, "type": "text", "content_type": "text"}
            }
        ],
        "images": [
            {
                "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
                "metadata": {"source": "test.pdf", "page": 1, "type": "image", "content_type": "image"},
                "page_content": "Descrizione immagine test"
            }
        ],
        "tables": [
            {
                "table_data": {"cells": [["A", "B"], ["1", "2"]], "headers": ["Col1", "Col2"], "shape": [2, 2]},
                "table_markdown": "| Col1 | Col2 |\n|------|------|\n| A    | B    |\n| 1    | 2    |",
                "metadata": {"source": "test.pdf", "page": 1, "type": "table", "content_type": "table"}
            }
        ]
    }

@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Mock delle variabili d'ambiente per i test"""
    monkeypatch.setenv("GROQ_API_KEY", "test_key")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
