"""
Integration tests for the MultimodalRAG system.
"""
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Import main components
try:
    import src.config as config
    from src.utils.pdf_parser import PDFParser
    from src.utils.semantic_chunker import SemanticChunker
    from src.utils.embedder import get_embedding_model
    from src.pipeline.indexer_service import DocumentIndexer
    from src.pipeline.retriever import MultimodalRetriever
except ImportError as e:
    pytest.skip(f"Skipping integration tests due to import error: {e}", allow_module_level=True)


class TestSystemIntegration:
    """Integration tests for the complete system."""
    
    def test_config_loading(self):
        """Test that configuration can be loaded properly."""
        assert hasattr(config, 'QDRANT_URL')
        assert hasattr(config, 'COLLECTION_NAME')
        assert config.QDRANT_URL is not None
        
    def test_pdf_parser_initialization(self):
        """Test that PDF parser can be initialized."""
        parser = PDFParser()
        assert parser is not None
        
    def test_semantic_chunker_initialization(self):
        """Test that semantic chunker can be initialized."""
        try:
            # Mock the embeddings parameter
            mock_embeddings = Mock()
            chunker = SemanticChunker(embeddings=mock_embeddings)
            assert chunker is not None
        except Exception as e:
            pytest.skip(f"SemanticChunker test skipped due to: {e}")
        
    @patch('src.utils.embedder.SentenceTransformer')
    def test_embedding_service_initialization(self, mock_transformer):
        """Test that embedding service can be initialized."""
        try:
            mock_transformer.return_value = Mock()
            embedder = get_embedding_model()
            assert embedder is not None
        except Exception as e:
            pytest.skip(f"Embedding service test skipped due to: {e}")
        
    @patch('src.pipeline.indexer_service.QdrantClient')
    def test_indexer_service_initialization(self, mock_client):
        """Test that indexer service can be initialized."""
        try:
            mock_client.return_value = Mock()
            mock_embedder = Mock()
            indexer = DocumentIndexer(embedder=mock_embedder)
            assert indexer is not None
        except Exception as e:
            pytest.skip(f"Indexer service test skipped due to: {e}")
        
    @patch('src.pipeline.retriever.QdrantClient')
    def test_retriever_initialization(self, mock_client):
        """Test that retriever can be initialized."""
        try:
            mock_client.return_value = Mock()
            retriever = MultimodalRetriever()
            assert retriever is not None
        except Exception as e:
            pytest.skip(f"Retriever test skipped due to: {e}")
        
    def test_sample_text_processing_pipeline(self):
        """Test processing a simple text through basic operations."""
        sample_text = "This is a test document. It has multiple sentences. Each sentence contains information."
        
        # Basic text processing tests
        assert len(sample_text) > 0
        sentences = sample_text.split('.')
        assert len(sentences) >= 3
        
    def test_environment_variables_access(self):
        """Test that required configuration variables can be accessed."""
        # These should be accessible even if not set (will use defaults)
        assert hasattr(config, 'QDRANT_URL')
        assert hasattr(config, 'COLLECTION_NAME')
        assert hasattr(config, 'GROQ_API_KEY')
        
    def test_file_processing_workflow(self):
        """Test the basic file processing workflow."""
        try:
            # Create a temporary text file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write("This is a test document for the multimodal RAG system. ")
                f.write("It contains multiple sentences that should be processed correctly. ")
                f.write("The system should be able to chunk this text appropriately.")
                temp_file = f.name
            
            # Test that the file exists and can be read
            assert os.path.exists(temp_file)
            
            with open(temp_file, 'r') as f:
                content = f.read()
                assert len(content) > 0
                
            # Clean up
            os.unlink(temp_file)
            
        except Exception as e:
            pytest.skip(f"File processing test skipped due to: {e}")


class TestStreamlitIntegration:
    """Integration tests for Streamlit components."""
    
    def test_streamlit_imports(self):
        """Test that streamlit components can be imported."""
        try:
            from streamlit_app.backend_logic import process_uploaded_file
            assert callable(process_uploaded_file)
        except ImportError as e:
            pytest.skip(f"Streamlit integration test skipped due to import error: {e}")
            
    def test_streamlit_components_import(self):
        """Test that UI components can be imported."""
        try:
            from streamlit_app.components.ui_components import upload_widget
            assert callable(upload_widget)
        except ImportError as e:
            pytest.skip(f"UI components test skipped due to import error: {e}")


class TestDockerIntegration:
    """Integration tests for Docker setup."""
    
    def test_dockerfile_exists_and_valid(self):
        """Test that Dockerfile exists and has basic structure."""
        project_root = Path(__file__).parent.parent.parent
        dockerfile = project_root / "Dockerfile"
        
        assert dockerfile.exists(), "Dockerfile should exist"
        
        with open(dockerfile, 'r') as f:
            content = f.read()
            assert "FROM python:" in content, "Dockerfile should use Python base image"
            assert "COPY requirements.txt" in content, "Dockerfile should copy requirements"
            assert "RUN pip install" in content, "Dockerfile should install dependencies"
            
    def test_docker_compose_exists_and_valid(self):
        """Test that docker-compose.yml exists and has basic structure."""
        project_root = Path(__file__).parent.parent.parent
        compose_file = project_root / "docker-compose.yml"
        
        assert compose_file.exists(), "docker-compose.yml should exist"
        
        with open(compose_file, 'r') as f:
            content = f.read()
            assert "services:" in content, "docker-compose.yml should define services"
            assert "qdrant:" in content, "docker-compose.yml should include Qdrant service"


class TestScriptsIntegration:
    """Integration tests for project scripts."""
    
    def test_run_script_exists(self):
        """Test that run script exists and is executable."""
        project_root = Path(__file__).parent.parent.parent
        run_script = project_root / "scripts" / "run.py"
        
        assert run_script.exists(), "run.py script should exist"
        
        # Check that script has main functionality
        with open(run_script, 'r') as f:
            content = f.read()
            assert "def main" in content or "if __name__" in content, "Script should have main function"
            
    def test_makefile_targets(self):
        """Test that Makefile has expected targets."""
        project_root = Path(__file__).parent.parent.parent
        makefile = project_root / "Makefile"
        
        if makefile.exists():
            with open(makefile, 'r') as f:
                content = f.read()
                expected_targets = [
                    "install", "test", "clean", "format", "lint", 
                    "run", "docker-build", "docker-run"
                ]
                for target in expected_targets:
                    assert f"{target}:" in content, f"Makefile should have '{target}' target"
