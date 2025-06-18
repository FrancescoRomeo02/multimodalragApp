from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import List, Dict, Union
from PIL import Image
from io import BytesIO
import base64
from app.config import EMBEDDING_MODEL_NAME
import logging

logger = logging.getLogger(__name__)

class QdrantCompatibleEmbedder:
    def __init__(self):
        self.model = HuggingFaceEmbedding(
            model_name=EMBEDDING_MODEL_NAME,
            device="cpu",
            trust_remote_code=True
        )
        
        # Verifica se il modello supporta embedding multimodali
        self.supports_images = hasattr(self.model, 'get_image_embedding')
        if not self.supports_images:
            logger.warning(f"Il modello {EMBEDDING_MODEL_NAME} non supporta embedding di immagini. Le immagini verranno processate usando la loro descrizione testuale.")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embedding di una lista di documenti testuali"""
        embeddings = []
        for text in texts:
            try:
                embedding = self.model.get_text_embedding(text)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Errore embedding documento: {str(e)}")
                # Fallback: embedding di testo vuoto
                embeddings.append(self.model.get_text_embedding(""))
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        try:
            return self.model.get_text_embedding(text)
        except Exception as e:
            logger.error(f"Errore embedding query: {str(e)}")
            return self.model.get_text_embedding("")
    
    def embed_images(self, image_dicts: List[Dict]) -> List[Dict]:
        results = []
        
        for img_dict in image_dicts:
            try:
                if self.supports_images:
                    # Usa embedding multimodale se disponibile
                    img_data = base64.b64decode(img_dict["image_base64"])
                    img = Image.open(BytesIO(img_data))
                    embedding = self.model.get_image_embedding(img)
                    img_dict["embedding"] = embedding
                    logger.debug("Usato embedding multimodale per immagine")
                else:
                    # Fallback: usa la descrizione testuale dell'immagine
                    description = img_dict.get("page_content", "Immagine documento")
                    
                    # Arricchisci la descrizione con metadati se disponibili
                    metadata = img_dict.get("metadata", {})
                    if "media" in metadata:
                        media_info = metadata["media"]
                        dimensions = media_info.get("dimensions", {})
                        format_info = media_info.get("format", "unknown")
                        
                        enhanced_description = f"{description}. Formato: {format_info}. Dimensioni: {dimensions.get('width', 'N/A')}x{dimensions.get('height', 'N/A')} pixel."
                    else:
                        enhanced_description = description
                    
                    embedding = self.embed_query(enhanced_description)
                    img_dict["embedding"] = embedding
                    logger.debug("Usato embedding testuale per immagine")
                
                # Verifica dimensione embedding
                if len(img_dict["embedding"]) == 0:
                    logger.error("Embedding vuoto generato per immagine")
                    continue
                    
                results.append(img_dict)
                
            except Exception as e:
                logger.error(f"Errore embedding immagine: {str(e)}")
                try:
                    fallback_text = "Immagine documento non processabile"
                    img_dict["embedding"] = self.embed_query(fallback_text)
                    results.append(img_dict)
                    logger.warning("Usato embedding fallback per immagine")
                except Exception as fallback_e:
                    logger.error(f"Anche il fallback è fallito: {str(fallback_e)}")
                    continue
        
        logger.info(f"Processate {len(results)}/{len(image_dicts)} immagini con successo")
        return results
    
    def __call__(self, input: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        if isinstance(input, str):
            return self.embed_query(input)
        return self.embed_documents(input)

def get_embedding_model() -> QdrantCompatibleEmbedder:
    return QdrantCompatibleEmbedder()

def validate_embedding_dimensions(embeddings: List[List[float]], expected_dim: int = None) -> bool:    
    if not embeddings:
        return False
    
    first_dim = len(embeddings[0])
    if expected_dim and first_dim != expected_dim:
        logger.error(f"Dimensione embedding {first_dim} non corrisponde a quella attesa {expected_dim}")
        return False
    
    for i, emb in enumerate(embeddings):
        if len(emb) != first_dim:
            logger.error(f"Embedding {i} ha dimensione diversa: {len(emb)} vs {first_dim}")
            return False
    
    return True