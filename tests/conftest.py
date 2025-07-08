"""
Configurazione pytest per MultimodalRAG.
"""

import os
import sys
from pathlib import Path
import pytest

# Aggiungi il root del progetto al Python path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

# Set environment variables for testing
os.environ["TESTING"] = "true"
os.environ["LOG_LEVEL"] = "ERROR"


@pytest.fixture(scope="session")
def project_root():
    """Fixture che restituisce il path del root del progetto."""
    return ROOT_DIR


@pytest.fixture(scope="session")
def test_data_dir():
    """Fixture che restituisce il path della directory test data."""
    test_data = ROOT_DIR / "tests" / "data"
    test_data.mkdir(exist_ok=True)
    return test_data


@pytest.fixture
def sample_config():
    """Fixture che restituisce una configurazione di test."""
    return {
        "groq_api_key": "test_key",
        "qdrant_url": "http://localhost:6333",
        "collection_name": "test_collection",
        "llm_model_name": "llama3-8b-8192",
        "log_level": "ERROR"
    }


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Fixture che imposta variabili d'ambiente mock per i test."""
    env_vars = {
        "GROQ_API_KEY": "test_groq_key",
        "QDRANT_URL": "http://localhost:6333",
        "COLLECTION_NAME": "test_collection",
        "LLM_MODEL_NAME": "llama3-8b-8192",
        "LOG_LEVEL": "ERROR"
    }
    
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    
    return env_vars
