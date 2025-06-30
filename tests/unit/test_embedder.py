import pytest
from unittest.mock import Mock, patch
import sys
import os

# Setup path per import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from src.utils.embedder import AdvancedEmbedder, get_embedding_model


class TestAdvancedEmbedder:
    """Test per la classe AdvancedEmbedder"""
    
    @patch('src.utils.embedder.HuggingFaceEmbedding')
    def test_embedder_initialization(self, mock_hf_embedding):
        """Test inizializzazione embedder"""
        # Mock del modello HuggingFace
        mock_model = Mock()
        mock_model.get_text_embedding.return_value = [0.1] * 384
        mock_hf_embedding.return_value = mock_model
        
        embedder = AdvancedEmbedder()
        
        assert embedder.embedding_dim == 384
        assert embedder.model_name == "sentence-transformers/clip-ViT-B-32-multilingual-v1"
        mock_hf_embedding.assert_called_once()
    
    @patch('src.utils.embedder.HuggingFaceEmbedding')
    def test_embed_query(self, mock_hf_embedding):
        """Test embedding di una singola query"""
        mock_model = Mock()
        mock_model.get_text_embedding.return_value = [0.1] * 384
        mock_hf_embedding.return_value = mock_model
        
        embedder = AdvancedEmbedder()
        result = embedder.embed_query("test query")
        
        assert len(result) == 384
        assert all(isinstance(x, float) for x in result)
        mock_model.get_text_embedding.assert_called_with("test query")
    
    @patch('src.utils.embedder.HuggingFaceEmbedding')
    def test_embed_documents(self, mock_hf_embedding):
        """Test embedding di documenti multipli"""
        mock_model = Mock()
        mock_model.get_text_embedding_batch.return_value = [[0.1] * 384] * 3
        mock_model.get_text_embedding.return_value = [0.1] * 384
        mock_hf_embedding.return_value = mock_model
        
        embedder = AdvancedEmbedder()
        texts = ["doc 1", "doc 2", "doc 3"]
        result = embedder.embed_documents(texts)
        
        assert len(result) == 3
        assert all(len(emb) == 384 for emb in result)
        mock_model.get_text_embedding_batch.assert_called_once()
    
    @patch('src.utils.embedder.HuggingFaceEmbedding')
    def test_embed_empty_text(self, mock_hf_embedding):
        """Test gestione testo vuoto"""
        mock_model = Mock()
        mock_model.get_text_embedding.return_value = [0.1] * 384
        mock_hf_embedding.return_value = mock_model
        
        embedder = AdvancedEmbedder()
        result = embedder.embed_query("")
        
        # Dovrebbe usare FALLBACK_TEXT_FOR_EMPTY_DOC
        assert len(result) == 384
        mock_model.get_text_embedding.assert_called_with(" ")
    
    @patch('src.utils.embedder.HuggingFaceEmbedding')
    def test_embedding_error_handling(self, mock_hf_embedding):
        """Test gestione errori embedding"""
        mock_model = Mock()
        mock_model.get_text_embedding.side_effect = Exception("Embedding failed")
        mock_model.get_text_embedding.return_value = [0.1] * 384  # Per __init__
        mock_hf_embedding.return_value = mock_model
        
        embedder = AdvancedEmbedder()
        
        # Simula errore
        mock_model.get_text_embedding.side_effect = Exception("Embedding failed")
        result = embedder.embed_query("test")
        
        # Dovrebbe restituire vettore zero
        assert len(result) == 384
        assert all(x == 0.0 for x in result)
    
    @patch('src.utils.embedder.get_embedding_model')
    def test_factory_function(self, mock_get_embedding):
        """Test funzione factory"""
        mock_embedder = Mock()
        mock_get_embedding.return_value = mock_embedder
        
        result = get_embedding_model()
        assert result == mock_embedder
        mock_get_embedding.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
