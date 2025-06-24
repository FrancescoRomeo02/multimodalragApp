# file: app/utils/embedder.py

from io import BytesIO
import logging
import base64
from typing import List, Optional, Dict, Any
from PIL import Image

import torch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_core.embeddings import Embeddings
from transformers import CLIPProcessor, CLIPModel

from app.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_BATCH_SIZE, FALLBACK_TEXT_FOR_EMPTY_DOC


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdvancedEmbedder(Embeddings):
    """
    Embedder avanzato conforme all'interfaccia di LangChain, con supporto
    multimodale e ottimizzazioni per il batching e il fallback.
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: Optional[str] = None,
        trust_remote_code: bool = True,
        enable_images: bool = False,  # Nuovo parametro per abilitare immagini
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_images = enable_images

        logger.info(f"Inizializzazione di HuggingFaceEmbedding con il modello: {self.model_name}")
        
        try:
            self.model = HuggingFaceEmbedding(
                model_name=self.model_name,
                device=self.device,
                trust_remote_code=trust_remote_code,
                embed_batch_size=self.batch_size
            )
            self._determine_embedding_dim()
        except Exception as e:
            logger.error(f"Errore critico durante il caricamento del modello '{self.model_name}': {e}")
            raise

        # Carica CLIP solo se abilitato
        if self.enable_images:
            self._init_clip_model()
        else:
            self.clip_model = None
            self.clip_processor = None

        self.supports_images = self.enable_images and self.clip_model is not None
        if self.enable_images and not self.supports_images:
            logger.warning("Supporto immagini richiesto ma CLIP non caricato correttamente")

    def _determine_embedding_dim(self):
        try:
            test_embedding = self.model.get_text_embedding("test")
            self.embedding_dim = len(test_embedding)
            logger.info(f"Modello caricato con successo. Dimensione embedding: {self.embedding_dim}")
        except Exception as e:
            raise RuntimeError("Impossibile determinare la dimensione dell'embedding.") from e

    def _init_clip_model(self):
        """Inizializza CLIP solo se necessario."""
        try:
            logger.info(f"Caricamento CLIP: {DEFAULT_CLIP_MODEL}")
            self.clip_model = CLIPModel.from_pretrained(DEFAULT_CLIP_MODEL)
            self.clip_processor = CLIPProcessor.from_pretrained(DEFAULT_CLIP_MODEL)
            logger.info("CLIP caricato con successo")
        except Exception as e:
            logger.error(f"Errore caricamento CLIP: {e}")
            self.clip_model = None
            self.clip_processor = None

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Metodo obbligatorio di LangChain. Genera embedding per una lista di testi.
        """
        if not texts:
            return []
        
        sanitized_texts = [text or FALLBACK_TEXT_FOR_EMPTY_DOC for text in texts]
        
        try:
            return self.model.get_text_embedding_batch(sanitized_texts, show_progress=False)
        except Exception as e:
            logger.error(f"Errore batch embedding: {e}. Fallback su singolo testo.")
            embeddings = []
            for text in sanitized_texts:
                try:
                    embeddings.append(self.model.get_text_embedding(text))
                except Exception as ind_e:
                    logger.error(f"Errore su singolo testo: '{text[:50]}...': {ind_e}. Uso zeri.")
                    embeddings.append([0.0] * self.embedding_dim)
            return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Metodo obbligatorio di LangChain. Genera l'embedding per una singola query.
        """
        try:
            return self.model.get_text_embedding(text or FALLBACK_TEXT_FOR_EMPTY_DOC)
        except Exception as e:
            logger.error(f"Errore embedding query: {e}. Uso zeri.")
            return [0.0] * self.embedding_dim

    def embed_images_batch(self, images_data: List[Dict[str, str]]) -> List[tuple[List[float], Dict[str, Any]]]:
        """
        Genera embedding per un BATCH di immagini in modo efficiente e restituisce
        una lista di tuple (embedding, metadata). QUESTA È LA VERSIONE OTTIMIZZATA.

        Args:
            images_data: Lista di dizionari, ognuno con 'base64' e 'description'.
        """
        if not self.supports_images:
            logger.error("Tentativo di usare embed_images_batch con un modello non multimodale.")
            return []

        pil_images = []
        original_data = [] # Manteniamo i dati originali per i metadati

        # 1. Pre-processa tutte le immagini e le converte in oggetti PIL
        for i, data in enumerate(images_data):
            base64_data = data.get('base64', '')
            description = data.get('description', f'Immagine {i+1}')
            
            try:
                if ',' in base64_data:
                    base64_data = base64_data.split(',')[1]
                img_data = base64.b64decode(base64_data)
                image = Image.open(BytesIO(img_data)).convert('RGB')
                pil_images.append(image)
                original_data.append({
                    'description': description,
                    'image': image,
                    'source': data.get('source')
                })
            except Exception as e:
                logger.warning(f"Immagine {i} non valida e sarà saltata: {e}")
                pil_images.append(None) # Segnaposto per immagini fallite
                original_data.append(None)

        # Filtra le immagini che sono state processate con successo
        valid_images = [img for img in pil_images if img is not None]
        if not valid_images:
            return [] # Nessuna immagine valida da processare

        # 2. ESEGUI L'EMBEDDING IN UN'UNICA CHIAMATA BATCH
        inputs = self.clip_processor(images=valid_images, return_tensors="pt", padding=True).to(self.device) # type: ignore
        
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs) # type: ignore
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        all_embeddings = image_features.cpu().numpy().tolist()

        # 3. Ricostruisci l'output finale, associando embedding e metadati
        results = []
        embedding_iter = iter(all_embeddings)

        for i, data in enumerate(original_data):
            if data is not None:
                embedding = next(embedding_iter)
                image = data['image']
                metadata = {
                    "content_type": "image",
                    "image_format": image.format or "unknown",
                    "image_size": f"{image.size[0]}x{image.size[1]}",
                    "description": data['description'],
                    "embedding_model": DEFAULT_CLIP_MODEL,
                    "source": data['source']
                }
                results.append((embedding, metadata))
            else:
                # Opzionale: puoi decidere di non includere le immagini fallite nell'output
                # o di includerle con un embedding di zeri
                embedding = [0.0] * self.embedding_dim
                metadata = {"error": "Immagine non processabile"}
                results.append((embedding, metadata))
                
        return results


def get_multimodal_embedding_model() -> AdvancedEmbedder:
    """Factory function per creare un'istanza dell'embedder con supporto immagini."""
    return AdvancedEmbedder(enable_images=True)