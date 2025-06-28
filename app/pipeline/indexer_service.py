import os
import logging
import uuid
from typing import List, Dict, Any, Tuple
from pydantic import ValidationError
from qdrant_client.http import models as qdrant_models

from app.utils.qdrant_utils import qdrant_manager
from app.utils.embedder import AdvancedEmbedder
from app.utils.pdf_parser import parse_pdf_elements
from app.utils.image_info import get_image_description
from app.core.models import TextElement, ImageElement

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DocumentIndexer:
    """
    Gestisce l'indicizzazione di documenti PDF in Qdrant.
    Ora utilizza QdrantManager per tutte le operazioni sul database vettoriale.
    """
    
    def __init__(self, embedder: AdvancedEmbedder):
        self.embedder = embedder
        self.qdrant_manager = qdrant_manager

    def _prepare_image_data(self, image_element: ImageElement) -> Dict[str, Any]:
        """
        Prepara i dati dell'immagine per l'embedding
        
        Args:
            image_element: Elemento immagine da processare
            
        Returns:
            Dict con testo descrittivo e metadati dell'immagine
        """
        try:
            metadata = image_element.metadata.model_dump()
            
            # Genera descrizione obbligatoria
            description = get_image_description(image_element.image_base64)
            if not description.strip():
                description = f"Immagine da {metadata.get('source', '')} pagina {metadata.get('page', 0)}"
            
            return {
                'text': description,
                'metadata': metadata,
                'image_base64': image_element.image_base64  # Mantieni l'immagine per la ricerca
            }
        except Exception as e:
            logger.error(f"Errore preparazione dati immagine: {e}")
            raise

    def _create_text_points(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> Tuple[List[qdrant_models.PointStruct], bool]:
        """
        Crea punti Qdrant per elementi testuali
        
        Args:
            texts: Lista di testi da indicizzare
            metadatas: Lista di metadati corrispondenti
            
        Returns:
            Tuple[Lista punti, successo operazione]
        """
        try:
            if not texts:
                return [], True
                
            # Genera embeddings per i testi
            text_embeddings = self.embedder.embed_documents(texts)
            
            if len(text_embeddings) != len(texts):
                logger.error(f"Mismatch embeddings/testi: {len(text_embeddings)} vs {len(texts)}")
                return [], False
            
            points = []
            for i, (text, metadata, embedding) in enumerate(zip(texts, metadatas, text_embeddings)):
                # Assicura che il metadata abbia il tipo corretto
                metadata['type'] = 'text'
                metadata['content_type'] = 'text'
                
                point = qdrant_models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        'page_content': text,
                        'metadata': metadata
                    }
                )
                points.append(point)
            
            return points, True
            
        except Exception as e:
            logger.error(f"Errore creazione punti testo: {e}")
            return [], False

    def _create_image_points(self, image_descriptions: List[Dict[str, Any]]) -> Tuple[List[qdrant_models.PointStruct], bool]:
        """
        Crea punti Qdrant per elementi immagine
        
        Args:
            image_descriptions: Lista di dati immagini processati
            
        Returns:
            Tuple[Lista punti, successo operazione]
        """
        try:
            if not image_descriptions:
                return [], True
            
            # Estrai testi descrittivi per embedding
            descriptions_text = [desc['text'] for desc in image_descriptions]
            
            # Genera embeddings per le descrizioni
            image_embeddings = self.embedder.embed_documents(descriptions_text)
            
            if len(image_embeddings) != len(image_descriptions):
                logger.error(f"Mismatch embeddings/descrizioni: {len(image_embeddings)} vs {len(image_descriptions)}")
                return [], False
            
            points = []
            for desc_data, embedding in zip(image_descriptions, image_embeddings):
                metadata = desc_data['metadata'].copy()
                
                # Assicura metadati corretti per immagini
                metadata['type'] = 'image'
                metadata['content_type'] = 'image'
                metadata['image_base64'] = desc_data.get('image_base64', '')
                
                point = qdrant_models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        'page_content': desc_data['text'],
                        'metadata': metadata
                    }
                )
                points.append(point)
            
            return points, True
            
        except Exception as e:
            logger.error(f"Errore creazione punti immagine: {e}")
            return [], False

    def index_files(self, pdf_paths: List[str], force_recreate: bool = False) -> bool:
        """
        Indicizza file PDF con opzione di ricreare la collezione
        
        Args:
            pdf_paths: Lista percorsi file PDF da indicizzare
            force_recreate: Se True, ricrea la collezione eliminando quella esistente
            
        Returns:
            bool: True se l'indicizzazione è completata con successo
        """
        if not pdf_paths:
            logger.warning("Nessun file da indicizzare")
            return True

        logger.info(f"Inizio indicizzazione di {len(pdf_paths)} file")

        # Verifica connessione Qdrant
        if not self.qdrant_manager.verify_connection():
            logger.error("Impossibile connettersi a Qdrant")
            return False

        # Gestisci creazione/ricreazione collezione
        try:
            if force_recreate:
                logger.info("Forzata ricreazione della collezione")
                if not self.qdrant_manager.create_collection(
                    embedding_dim=self.embedder.embedding_dim,
                    force_recreate=True
                ):
                    logger.error("Fallita ricreazione collezione")
                    return False
            else:
                if not self.qdrant_manager.ensure_collection_exists(self.embedder.embedding_dim):
                    logger.error("Impossibile assicurare esistenza collezione")
                    return False
        except Exception as e:
            logger.error(f"Errore gestione collezione: {e}")
            return False

        # Processamento e raccolta dati da tutti i file
        all_texts = []
        all_text_metadatas = []
        all_image_descriptions = []
        processed_files = 0

        for path in pdf_paths:
            try:
                texts, images = self._prepare_elements_from_file(path)
                
                # Processa elementi testuali
                if texts:
                    all_texts.extend(elem.text for elem in texts)
                    all_text_metadatas.extend(elem.metadata.model_dump() for elem in texts)
                
                # Processa elementi immagine
                for elem in images:
                    try:
                        image_data = self._prepare_image_data(elem)
                        all_image_descriptions.append(image_data)
                    except Exception as e:
                        logger.error(f"Errore processamento immagine da {path}: {e}")
                        continue
                
                processed_files += 1
                logger.info(f"Processato file {processed_files}/{len(pdf_paths)}: {os.path.basename(path)}")
                        
            except Exception as e:
                logger.error(f"Errore processamento file {path}: {e}")
                continue

        if processed_files == 0:
            logger.error("Nessun file processato con successo")
            return False

        logger.info(f"Raccolti {len(all_texts)} testi e {len(all_image_descriptions)} immagini")

        # Indicizzazione elementi testuali
        success_text = True
        if all_texts:
            try:
                text_points, success_text = self._create_text_points(all_texts, all_text_metadatas)
                
                if success_text and text_points:
                    if self.qdrant_manager.upsert_points(text_points):
                        logger.info(f"Indicizzati {len(text_points)} elementi di testo")
                    else:
                        logger.error("Fallito inserimento punti testo")
                        success_text = False
                else:
                    logger.error("Fallita creazione punti testo")
                    success_text = False
                    
            except Exception as e:
                logger.error(f"Errore indicizzazione testo: {e}")
                success_text = False

        # Indicizzazione elementi immagine
        success_images = True
        if all_image_descriptions:
            try:
                image_points, success_images = self._create_image_points(all_image_descriptions)
                
                if success_images and image_points:
                    if self.qdrant_manager.upsert_points(image_points):
                        logger.info(f"Indicizzate {len(image_points)} immagini")
                    else:
                        logger.error("Fallito inserimento punti immagine")
                        success_images = False
                else:
                    logger.error("Fallita creazione punti immagine")
                    success_images = False
                    
            except Exception as e:
                logger.error(f"Errore indicizzazione immagini: {e}")
                success_images = False

        # Risultato finale
        overall_success = success_text and success_images
        if overall_success:
            logger.info("Processo di indicizzazione completato con successo")
        else:
            logger.warning("Processo di indicizzazione completato con alcuni errori")

        return overall_success

    def _prepare_elements_from_file(self, pdf_path: str) -> Tuple[List[TextElement], List[ImageElement]]:
        """
        Estrae elementi da file PDF con validazione
        
        Args:
            pdf_path: Percorso del file PDF
            
        Returns:
            Tuple[Lista elementi testo validati, Lista elementi immagine validati]
        """
        filename = os.path.basename(pdf_path)
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File non trovato: {pdf_path}")

        try:
            text_elements_raw, image_elements_raw = parse_pdf_elements(pdf_path)
            
            # Validazione elementi testuali
            validated_texts = []
            for raw in text_elements_raw:
                try:
                    validated_texts.append(TextElement(**raw))
                except ValidationError as e:
                    logger.warning(f"Elemento testo non valido in {filename}: {e}")

            # Validazione elementi immagine
            validated_images = []
            for raw in image_elements_raw:
                try:
                    # Assicura che i metadati abbiano la struttura minima
                    if 'media' not in raw['metadata']:
                        raw['metadata']['media'] = {'format': 'unknown'}
                    validated_images.append(ImageElement(**raw))
                except (ValidationError, KeyError) as e:
                    logger.warning(f"Elemento immagine non valido in {filename}: {e}")

            logger.info(f"Estratti da {filename}: {len(validated_texts)} testi, {len(validated_images)} immagini")
            return validated_texts, validated_images

        except Exception as e:
            logger.error(f"Errore parsing PDF {filename}: {e}")
            raise

    def delete_file_from_index(self, filename: str) -> bool:
        """
        Elimina tutti i documenti di un file specifico dall'indice
        
        Args:
            filename: Nome del file da eliminare
            
        Returns:
            bool: True se l'eliminazione è riuscita
        """
        try:
            success, message = self.qdrant_manager.delete_by_source(filename)
            if success:
                logger.info(f"Eliminato file {filename} dall'indice: {message}")
            else:
                logger.error(f"Errore eliminazione file {filename}: {message}")
            return success
        except Exception as e:
            logger.error(f"Errore eliminazione file {filename}: {e}")
            return False

    def get_index_status(self) -> Dict[str, Any]:
        """
        Restituisce informazioni sullo stato dell'indice
        
        Returns:
            Dict con informazioni sulla collezione e connessione
        """
        try:
            return self.qdrant_manager.health_check()
        except Exception as e:
            logger.error(f"Errore controllo stato indice: {e}")
            return {"error": str(e)}