import os
import logging
import uuid
from typing import List, Dict, Any, Tuple
from pydantic import ValidationError
import qdrant_client
from langchain_qdrant import QdrantVectorStore
from qdrant_client import models as qdrant_models

from app.utils.embedder import AdvancedEmbedder
from app.utils.pdf_parser import parse_pdf_elements
from app.utils.image_info import get_image_description

from app.core.models import TextElement, ImageElement

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentIndexer:
    def __init__(self, embedder: AdvancedEmbedder, qdrant_url: str, collection_name: str):
        self.embedder = embedder
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self._initialize_qdrant()

    def _initialize_qdrant(self):
        try:
            self.qdrant_client = qdrant_client.QdrantClient(
                url=self.qdrant_url,
                prefer_grpc=True,
                timeout=60
            )
            self._verify_connection()
            self._ensure_collection_exists()
            
            self.vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=self.collection_name,
                embedding=self.embedder
            )
            
        except Exception as e:
            logger.error(f"Errore inizializzazione Qdrant: {e}")
            raise

    def _verify_connection(self):
        try:
            self.qdrant_client.get_collections()
            logger.info(f"Connesso a Qdrant all'URL {self.qdrant_url}")
        except Exception as e:
            logger.error(f"Connessione Qdrant fallita: {e}")
            raise

    def _ensure_collection_exists(self):
        if not self.qdrant_client.collection_exists(self.collection_name):
            self._create_collection()

    def _create_collection(self):
        try:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=self.embedder.embedding_dim,
                    distance=qdrant_models.Distance.COSINE,
                    on_disk=True
                )
            )
            logger.info(f"Creata collezione {self.collection_name}")
        except Exception as e:
            logger.error(f"Errore creazione collezione: {e}")
            raise

    def _prepare_image_data(self, image_element: ImageElement) -> Dict[str, Any]:
        """Prepara i dati dell'immagine per l'embedding"""
        try:
            metadata = image_element.metadata.model_dump()
            
            # Genera descrizione obbligatoria
            description = get_image_description(image_element.image_base64)
            if not description.strip():
                description = f"Immagine da {metadata.get('source', '')} pagina {metadata.get('page', 0)}"
            
            return {
                'text': description,
                'metadata': metadata
            }
        except Exception as e:
            logger.error(f"Errore preparazione dati immagine: {e}")
            raise
    
    def index_files(self, pdf_paths: List[str], force_recreate: bool = False):

        """Indicizza file PDF con opzione di ricreare la collezione."""
        if not pdf_paths:
            logger.warning("Nessun file da indicizzare")
            return

        logger.info(f"Inizio indicizzazione di {len(pdf_paths)} file")

        # Gestione force_recreate
        if force_recreate:
            logger.info("Modalità force_recreate attivata")
            try:
                if self.qdrant_client.collection_exists(self.collection_name):
                    self.qdrant_client.delete_collection(self.collection_name)
                    logger.info(f"Collezione {self.collection_name} eliminata")
                
                self._create_collection()
                logger.info("Collezione ricreata con successo")
            except Exception as e:
                logger.error(f"Errore force_recreate: {e}")
                raise

        # Verifica normale esistenza collezione
        self._ensure_collection_exists()

        # Processamento file
        all_texts = []
        all_text_metadatas = []
        all_image_descriptions = []

        for path in pdf_paths:
            try:
                texts, images = self._prepare_elements_from_file(path)
                
                # Processa testo
                all_texts.extend(elem.text for elem in texts)
                all_text_metadatas.extend(elem.metadata.model_dump() for elem in texts)
                
                # Processa immagini
                for elem in images:
                    try:
                        image_data = self._prepare_image_data(elem)
                        all_image_descriptions.append(image_data)
                    except Exception as e:
                        logger.error(f"Errore processamento immagine {path}: {e}")
                        
            except Exception as e:
                logger.error(f"Errore processamento file {path}: {e}")
                continue

        # Indicizzazione testo
        if all_texts:
            try:
                self.vector_store.add_texts(
                    texts=all_texts,
                    metadatas=all_text_metadatas,
                    batch_size=64
                )
                logger.info(f"Indicizzati {len(all_texts)} elementi di testo")
            except Exception as e:
                logger.error(f"Errore indicizzazione testo: {e}")

        # Indicizzazione immagini
        if all_image_descriptions:
            try:
                embedding_results = self.embedder.embed_image_descriptions(all_image_descriptions)
                
                if embedding_results:
                    points = [
                        qdrant_models.PointStruct(
                            id=str(uuid.uuid4()),
                            vector=embedding,
                            payload=payload
                        )
                        for embedding, payload in embedding_results
                    ]

                    self.qdrant_client.upsert(
                        collection_name=self.collection_name,
                        points=points,
                        wait=True
                    )
                    logger.info(f"Indicizzate {len(points)} immagini")
                else:
                    logger.warning("Nessun embedding immagine generato")
                    
            except Exception as e:
                logger.error(f"Errore indicizzazione immagini: {e}")

        logger.info("Processo di indicizzazione completato")

    def _prepare_elements_from_file(self, pdf_path: str) -> Tuple[List[TextElement], List[ImageElement]]:
        """Estrae elementi da file PDF con validazione"""
        filename = os.path.basename(pdf_path)
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File non trovato: {pdf_path}")

        try:
            text_elements_raw, image_elements_raw = parse_pdf_elements(pdf_path)
            
            validated_texts = []
            for raw in text_elements_raw:
                try:
                    validated_texts.append(TextElement(**raw))
                except ValidationError as e:
                    logger.warning(f"Elemento testo non valido: {e}")

            validated_images = []
            for raw in image_elements_raw:
                try:
                    # Assicura che i metadati abbiano la struttura minima
                    if 'media' not in raw['metadata']:
                        raw['metadata']['media'] = {'format': 'unknown'}
                    validated_images.append(ImageElement(**raw))
                except (ValidationError, KeyError) as e:
                    logger.warning(f"Elemento immagine non valido: {e}")

            return validated_texts, validated_images

        except Exception as e:
            logger.error(f"Errore parsing PDF {filename}: {e}")
            raise