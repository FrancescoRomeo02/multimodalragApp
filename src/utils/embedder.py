# file: src/utils/embedder.py

import logging
from typing import List, Optional
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_core.embeddings import Embeddings
from src.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_BATCH_SIZE, FALLBACK_TEXT_FOR_EMPTY_DOC

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedEmbedder(Embeddings):
    """Embedder avanzato per testo e descrizioni di immagini"""
    
    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: Optional[str] = None,
        trust_remote_code: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or "cpu"
        
        try:
            self.model = HuggingFaceEmbedding(
                model_name=self.model_name,
                device=self.device,
                trust_remote_code=trust_remote_code,
                embed_batch_size=self.batch_size
            )
            self._determine_embedding_dim()
            logger.info(f"Embedder inizializzato con {model_name}")
        except Exception as e:
            logger.error(f"Errore inizializzazione: {e}")
            raise

    def _determine_embedding_dim(self):
        """Determina automaticamente la dimensione degli embedding"""
        try:
            test_embedding = self.model.get_text_embedding("test")
            self.embedding_dim = len(test_embedding)
            logger.info(f"Dimensione embedding: {self.embedding_dim}")
        except Exception as e:
            logger.error("Impossibile determinare dimensione embedding")
            raise RuntimeError("Dimensioni embedding sconosciute") from e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embedding per documenti testuali"""
        if not texts:
            return []

        sanitized_texts = [text or FALLBACK_TEXT_FOR_EMPTY_DOC for text in texts]
        
        try:
            return self.model.get_text_embedding_batch(sanitized_texts, show_progress=False)
        except Exception as e:
            logger.error(f"Errore batch embedding: {e}")
            return [self.embed_query(text) for text in sanitized_texts]

    def embed_query(self, text: str) -> List[float]:
        """Embedding per singola query"""
        try:
            return self.model.get_text_embedding(text or FALLBACK_TEXT_FOR_EMPTY_DOC)
        except Exception as e:
            logger.error(f"Errore embedding query: {e}")
            return [0.0] * self.embedding_dim

def get_embedding_model() -> AdvancedEmbedder:
    """Factory function per creare un'istanza preconfigurata"""
    return AdvancedEmbedder()


# Funzione mantenuta per compatibilità MIGLIORARE QUESTA GESTIONE
def get_multimodal_embedding_model() -> AdvancedEmbedder:
    """Alias per get_embedding_model() per mantenere compatibilità"""
    logger.warning("get_multimodal_embedding_model() è deprecato, usa get_embedding_model()")
    return get_embedding_model()
