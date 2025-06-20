import os
import logging
import uuid
from typing import List, Tuple, Dict, Any
import time
from pydantic import BaseModel, ValidationError
import qdrant_client
from langchain_qdrant import Qdrant
import qdrant_client.models
from qdrant_client import models as qdrant_models

from app.utils.embedder import AdvancedEmbedder
from app.utils.pdf_parser import parse_pdf_elements
from app.utils.image_caption import get_caption

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ElementMetadata(BaseModel):
    """Metadati comuni a tutti gli elementi, validati da Pydantic."""
    source: str
    page: int
    type: str

class TextElement(BaseModel):
    """Modello per un elemento testuale."""
    text: str
    metadata: ElementMetadata

class ImageElement(BaseModel):
    """Modello per un elemento immagine."""
    page_content: str
    image_base64: str
    metadata: ElementMetadata


class DocumentIndexer:
    """Gestisce l'indicizzazione di testi e immagini in Qdrant."""

    def __init__(self, embedder: AdvancedEmbedder, qdrant_url: str, collection_name: str):
        if not isinstance(embedder, AdvancedEmbedder):
            raise TypeError("L'embedder fornito non è un'istanza di AdvancedEmbedder.")
        
        self.embedder = embedder
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name

        try:
            # Inizializzazione client Qdrant
            self.qdrant_client = qdrant_client.QdrantClient(url=self.qdrant_url, prefer_grpc=True)
            
            # Verifica connessione
            _ = self.qdrant_client.get_collections()
            logger.info(f"Connessione a Qdrant stabilita all'URL {self.qdrant_url}")

            # Crea collezione se non esiste
            if not self.qdrant_client.collection_exists(self.collection_name):
                self._create_collection()
            
            # Inizializza vector store LangChain
            self.vector_store = Qdrant(
                client=self.qdrant_client,
                collection_name=self.collection_name,
                embeddings=self.embedder
            )

        except Exception as e:
            logger.error(f"Errore durante l'inizializzazione: {e}")
            raise

    def _create_collection(self):
        try:
            # Verifica più robusta dell'esistenza della collezione
            existing_collections = self.qdrant_client.get_collections()
            if any(col.name == self.collection_name for col in existing_collections.collections):
                logger.info(f"La collezione {self.collection_name} esiste già")
                return
            
            logger.info(f"Creazione collezione {self.collection_name}...")
            
            # Verifica che embedding_dim sia un intero valido
            if not isinstance(self.embedder.embedding_dim, int) or self.embedder.embedding_dim <= 0:
                raise ValueError(f"Dimensione embedding non valida: {self.embedder.embedding_dim}")
            
            # Crea la collezione con parametri più robusti
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=self.embedder.embedding_dim,
                    distance=qdrant_models.Distance.COSINE,
                    on_disk=True
                ),
                timeout=60  # Aumenta il timeout a 60 secondi
            )
            
            # Attendi e verifica la creazione
            max_retries = 5
            for _ in range(max_retries):
                time.sleep(1)  # Attendi 1 secondo tra i tentativi
                if self.qdrant_client.collection_exists(self.collection_name):
                    logger.info(f"Collezione {self.collection_name} creata con successo")
                    return
            
            raise TimeoutError(f"Timeout nella verifica della creazione della collezione {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Errore creazione collezione: {str(e)}", exc_info=True)
            raise RuntimeError(f"Impossibile creare collezione: {str(e)}") from e
    
    def ensure_collection_exists(self):
        if not self.qdrant_client.collection_exists(self.collection_name):
            self._create_collection()

    def delete_documents_by_source(self, source_name: str) -> Tuple[bool, str]:
        
        try:
            if not self.qdrant_client.collection_exists(self.collection_name):
                return False, f"Collezione '{self.collection_name}' non esiste"

            # Crea filtro per il nome del file sorgente
            from qdrant_client import models
            
            # Filtro per entrambi i tipi di documenti (testo e immagini)
            source_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.source",  # Assicurati che questo path sia corretto
                        match=models.MatchValue(value=source_name),
                    ),
                    # Aggiungi questa parte per includere sia testi che immagini
                    models.FieldCondition(
                        key="metadata.type",  # Assicurati che questo campo esista
                        match=models.MatchAny(any=["text", "image"]),  # O i valori che usi
                    ),
                ]
            )

            # Esegui eliminazione
            delete_result = self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=source_filter),
                wait=True
            )

            logger.info(f"Eliminati documenti per '{source_name}': {delete_result}")
            return True, f"Eliminati tutti i documenti relativi a '{source_name}'"
        
        except Exception as e:
            logger.error(f"Errore eliminazione documenti per '{source_name}': {e}")
            return False, f"Errore durante l'eliminazione: {str(e)}"
    

    def _prepare_elements_from_file(self, pdf_path: str) -> Tuple[List[TextElement], List[ImageElement]]:
        """
        Estrae, valida e converte gli elementi da un singolo PDF in modelli Pydantic.
        """
        filename = os.path.basename(pdf_path)
        if not os.path.exists(pdf_path):
            logger.error(f"File non trovato, saltato: {pdf_path}")
            return [], []

        logger.info(f"Estrazione elementi da: {filename}")
        try:
            text_elements_raw, image_elements_raw = parse_pdf_elements(pdf_path)
        except Exception as e:
            logger.error(f"Fallita l'estrazione da '{filename}': {e}")
            return [], []

        validated_texts, validated_images = [], []
        
        for raw_elem in text_elements_raw:
            try:
                validated_texts.append(TextElement(**raw_elem))
            except ValidationError as e:
                logger.warning(f"Elemento di testo in '{filename}' saltato per dati malformati: {e}")

        for raw_elem in image_elements_raw:
            try:
                validated_images.append(ImageElement(**raw_elem))
            except ValidationError as e:
                logger.warning(f"Elemento immagine in '{filename}' saltato per dati malformati: {e}")
        
        logger.info(f"Elementi validati da '{filename}': {len(validated_texts)} testi, {len(validated_images)} immagini.")
        return validated_texts, validated_images

    def index_files(self, pdf_paths: List[str], force_recreate: bool = False):
        """
        Indicizza una lista di file PDF, gestendo testo e immagini in batch separati.
        """
        logger.info(f"Avvio indicizzazione per {len(pdf_paths)} file.")

        if force_recreate:
            logger.warning(f"La collezione '{self.collection_name}' sarà eliminata e ricreata.")
            self.qdrant_client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=qdrant_client.models.VectorParams(
                    size=self.embedder.embedding_dim,
                    distance=qdrant_client.models.Distance.COSINE
                )
            )

        # Liste per raccogliere tutti i dati da indicizzare in batch
        all_texts: List[str] = []
        all_text_metadatas: List[Dict] = []
        all_image_base64: List[str] = []
        all_image_descriptions: List[str] = []
        all_image_metadatas: List[Dict] = []
        
        # 1. Raccogli e valida gli elementi da tutti i file
        for path in pdf_paths:
            texts, images = self._prepare_elements_from_file(path)
            for elem in texts:
                all_texts.append(elem.text)
                all_text_metadatas.append(elem.metadata.model_dump())
            
            if self.embedder.supports_images:
                for elem in images:
                    all_image_base64.append(elem.image_base64)
                    meta = elem.metadata.model_dump()
                    meta["type"] = "image"
                    meta["source"] = elem.metadata.source
                    meta["page"] = elem.metadata.page
                    meta["context"] = elem.page_content
                    all_image_descriptions.append(get_caption(elem.image_base64))
                    all_image_metadatas.append(meta)

        # 2. Indicizza il TESTO in un unico batch
        if all_texts:
            logger.info(f"Indicizzazione di {len(all_texts)} elementi di testo...")
            try:
                self.vector_store.add_texts(texts=all_texts, metadatas=all_text_metadatas, batch_size=64)
                logger.info("Elementi di testo indicizzati con successo.")
            except Exception as e:
                logger.error(f"Errore durante l'indicizzazione del testo: {e}", exc_info=True)

        # 3. Indicizza le IMMAGINI in un unico batch (usando il client Qdrant di basso livello)
        # --- NEL TUO METODO index_files DELLA CLASSE DocumentIndexer ---

        if all_image_base64: # La variabile 'all_image_base64' dovrebbe contenere le stringhe base64
            logger.info(f"Indicizzazione di {len(all_image_base64)} immagini...")
            
            try:
                # 1. Prepara l'input corretto per la funzione di batch
                # La funzione vuole una lista di dizionari.
                images_to_embed = [
                    {'base64': b64, 'description': desc}
                    for b64, desc in zip(all_image_base64, all_image_descriptions) # Assumiamo che esista all_image_descriptions
                ]

                # 2. Chiama la funzione di embedding BATCH una sola volta
                # Questa chiamata è super efficiente.
                embedding_results = self.embedder.embed_images_batch(images_to_embed)

                # 3. Prepara i punti per Qdrant iterando sui risultati
                points_to_upsert = []
                for vector, metadata in embedding_results:
                    # La tupla (vector, metadata) viene "spacchettata" direttamente nel ciclo for.
                    
                    # Controlla se l'embedding è fallito (la nostra funzione restituisce un payload con 'error')
                    if "error" in metadata:
                        logger.warning(f"Saltata un'immagine a causa di un errore di embedding: {metadata['error']}")
                        continue

                    # Il payload per Qdrant sono i metadati stessi.
                    # Non serve avvolgerli in un altro dizionario.
                    payload = metadata

                    point = qdrant_client.models.PointStruct(
                        id=str(uuid.uuid4()),  # Genera un ID unico per il punto
                        vector=vector,         # Il vettore (lista di float)
                        payload=payload        # I metadati (dizionario)
                    )
                    points_to_upsert.append(point)
                
                # 4. Carica i punti validi in Qdrant, se ce ne sono
                if points_to_upsert:
                    self.qdrant_client.upsert(
                        collection_name=self.collection_name,
                        points=points_to_upsert,
                        wait=True,
                    )
                    logger.info(f"Indicizzate con successo {len(points_to_upsert)} immagini.")
                else:
                    logger.warning("Nessuna immagine è stata indicizzata a causa di errori.")

            except Exception as e:
                logger.error(f"Errore critico durante l'indicizzazione delle immagini: {e}", exc_info=True)

        logger.info("--- Processo di indicizzazione completato. ---")
    
    
    

