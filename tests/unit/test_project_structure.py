"""
Test di esempio per verificare la configurazione di base.
"""

import pytest
from pathlib import Path
import sys

# Aggiungi il root del progetto al path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))


def test_import_structure():
    """Test che verifica l'importazione dei moduli principali."""
    try:
        import config
        import core.models
        import llm.groq_client
        import pipeline.indexer_service
        import utils.embedder
        assert True, "Tutti i moduli sono importabili"
    except ImportError as e:
        pytest.fail(f"Errore nell'importazione dei moduli: {e}")


def test_project_structure():
    """Test che verifica la struttura del progetto."""
    required_dirs = [
        "src",
        "src/core", 
        "src/llm",
        "src/pipeline",
        "src/utils",
        "streamlit_app",
        "streamlit_app/components",
        "tests",
        "tests/unit",
        "tests/integration",
        "scripts",
        "docs", 
        "data",
        "data/models",
        "data/raw",
        "data/processed"
    ]
    
    for dir_path in required_dirs:
        full_path = ROOT_DIR / dir_path
        assert full_path.exists(), f"Directory mancante: {dir_path}"
        assert full_path.is_dir(), f"Non è una directory: {dir_path}"


def test_required_files():
    """Test che verifica la presenza dei file essenziali."""
    required_files = [
        "README.md",
        "requirements.txt",
        "pyproject.toml",
        ".env.example",
        ".gitignore",
        "Dockerfile",
        "docker-compose.yml",
        "Makefile"
    ]
    
    for file_path in required_files:
        full_path = ROOT_DIR / file_path
        assert full_path.exists(), f"File mancante: {file_path}"
        assert full_path.is_file(), f"Non è un file: {file_path}"


def test_python_files_have_init():
    """Test che verifica la presenza di __init__.py nei package Python."""
    python_packages = [
        "src",
        "src/core",
        "src/llm", 
        "src/pipeline",
        "src/utils",
        "streamlit_app",
        "streamlit_app/components",
        "tests",
        "tests/unit",
        "tests/integration"
    ]
    
    for package in python_packages:
        init_file = ROOT_DIR / package / "__init__.py"
        assert init_file.exists(), f"File __init__.py mancante in: {package}"


if __name__ == "__main__":
    # Esegui i test se il file viene eseguito direttamente
    pytest.main([__file__, "-v"])
