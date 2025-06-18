# file: app/services/indexer_service.py

import os
import logging
import uuid
from typing import List, Tuple, Dict, Any

# --- Importazioni da librerie di terze parti ---
from pydantic import BaseModel, ValidationError
import qdrant_client
from langchain_qdrant import Qdrant
import qdrant_client.models

# --- Importazioni dai moduli della tua applicazione ---
# Si assume che questi file esistano e siano corretti
from app.utils.embedder import AdvancedEmbedder
from app.utils.pdf_parser import parse_pdf_elements
from app.utils.image_caption import get_caption
# --- Setup del Logger ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Modelli di Dati Pydantic per la Validazione ---

class ElementMetadata(BaseModel):
    """Metadati comuni a tutti gli elementi, validati da Pydantic."""
    source: str
    page: int
    type: str

class TextElement(BaseModel):
    """Modello per un elemento testuale. Garantisce che 'text' e 'metadata' esistano."""
    text: str
    metadata: ElementMetadata

class ImageElement(BaseModel):
    """Modello per un elemento immagine. Garantisce la presenza dei campi necessari."""
    page_content: str  # Descrizione testuale dell'immagine
    image_base64: str
    metadata: ElementMetadata


class DocumentIndexer:
    """
    Gestisce l'intero processo di indicizzazione di testi e immagini in Qdrant.
    Progettato per essere efficiente, robusto e tipologicamente corretto.
    """

    def __init__(self, embedder: AdvancedEmbedder, qdrant_url: str, collection_name: str):
        if not isinstance(embedder, AdvancedEmbedder):
            raise TypeError("L'embedder fornito non è un'istanza di AdvancedEmbedder.")
        
        self.embedder = embedder
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name

        # Verifica che la collezione esista, altrimenti la crea
        try:
            self.qdrant_client = qdrant_client.QdrantClient(url=self.qdrant_url, prefer_grpc=True)
            if not self.qdrant_client.collection_exists(self.collection_name):
                logger.info(f"La collezione '{self.collection_name}' non esiste. Creazione in corso...")
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qdrant_client.models.VectorParams(
                        size=self.embedder.embedding_dim,
                        distance=qdrant_client.models.Distance.COSINE
                    )
                )
                logger.info(f"Collezione '{self.collection_name}' creata con successo.")
            else:
                logger.info(f"La collezione '{self.collection_name}' esiste già.")

        except qdrant_client as e:
            logger.error(f"Errore durante la connessione a Qdrant: {e}")
            raise RuntimeError(f"Impossibile connettersi a Qdrant all'URL {self.qdrant_url}.") from e
        
        # Inizializziamo sia il client Qdrant di basso livello che il wrapper LangChain
        self.qdrant_client = qdrant_client.QdrantClient(url=self.qdrant_url, prefer_grpc=True)
        self.vector_store = Qdrant(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embeddings=self.embedder 
        )
        logger.info(f"DocumentIndexer inizializzato per la collezione '{self.collection_name}'.")

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

