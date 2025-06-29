from app.utils.pdf_parser import parse_pdf_elements
from app.utils.image_info import get_comprehensive_image_info
from app.core.models import TextElement, ImageElement

import os
import logging
import uuid
from typing import List, Dict, Any, Tuple
from pydantic import ValidationError
from qdrant_client.http import models as qdrant_models

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


from app.utils.pdf_parser import parse_pdf_elements
from app.utils.image_info import get_comprehensive_image_info
from app.core.models import TableData, TextElement, ImageElement, TableElement
from app.utils.embedder import AdvancedEmbedder
import os
import logging
import uuid
from typing import List, Dict, Any, Tuple, Optional
from pydantic import ValidationError
from qdrant_client.http import models as qdrant_models
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_elements_from_file(pdf_path: str) -> Tuple[List[TextElement], List[ImageElement], List[TableElement]]:
    """Estrae e valida elementi da PDF con gestione rinforzata delle tabelle"""
    filename = os.path.basename(pdf_path)
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File non trovato: {pdf_path}")

    try:
        text_elements_raw, image_elements_raw, table_elements_raw = parse_pdf_elements(pdf_path)

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
                if "media" not in raw["metadata"]:
                    raw["metadata"]["media"] = {"format": "unknown"}
                validated_images.append(ImageElement(**raw))
            except (ValidationError, KeyError) as e:
                logger.warning(f"Elemento immagine non valido in {filename}: {e}")

        # Validazione rinforzata elementi tabella
        validated_tables = []
        for raw in table_elements_raw:
            try:
                # Standardizzazione metadati
                if "metadata" not in raw:
                    raw["metadata"] = {}
                
                raw["metadata"].update({
                    "type": "table",
                    "source": filename,
                    "page": raw["metadata"].get("page", 0),
                    "content_type": "table",
                    "bbox": raw["metadata"].get("bbox"),
                    "table_shape": raw["table_data"].get("shape", (0, 0))
                })

                # Conversione esplicita a TableData
                if isinstance(raw["table_data"], dict):
                    raw["table_data"] = TableData(**raw["table_data"])
                
                validated_tables.append(TableElement(**raw))
            except ValidationError as e:
                logger.warning(f"Elemento tabella non valido in {filename}: {e}")
            except Exception as e:
                logger.error(f"Errore imprevisto validazione tabella in {filename}: {e}")

        logger.info(f"Estratti da {filename}: {len(validated_texts)} testi, {len(validated_images)} immagini, {len(validated_tables)} tabelle")
        return validated_texts, validated_images, validated_tables

    except Exception as e:
        logger.error(f"Errore parsing PDF {filename}: {e}")
        raise

def prepare_table_data(table_element: TableElement) -> Dict[str, Any]:
    """Prepara i dati della tabella per l'embedding con gestione robusta"""
    try:
        metadata = table_element.metadata.model_dump()
        table_data = table_element.table_data
        
        # Descrizione dettagliata della tabella
        description = (
            f"Tabella da {metadata.get('source', '')} pagina {metadata.get('page', 0)}. "
            f"Dimensione: {table_data.shape[0]} righe × {table_data.shape[1]} colonne. "
            f"Intestazioni: {', '.join(table_data.headers)}. "
            f"Contenuto: {table_element.table_markdown[:300]}..."
        )
        
        return {
            "text": description,
            "metadata": metadata,
            "table_markdown": table_element.table_markdown,
            "table_data": table_data.model_dump(),
            "content_type": "table"
        }
    except Exception as e:
        logger.error(f"Errore preparazione dati tabella: {e}")
        raise

def create_table_points(table_descriptions: List[Dict[str, Any]], embedder: AdvancedEmbedder) -> Tuple[List[qdrant_models.PointStruct], bool]:
    """Crea punti Qdrant per tabelle con validazione rinforzata"""
    points = []
    try:
        if not table_descriptions or not embedder:
            return [], True

        # Genera embeddings per le descrizioni
        texts = [desc["text"] for desc in table_descriptions]
        embeddings = embedder.embed_documents(texts)

        for desc, embedding in zip(table_descriptions, embeddings):
            try:
                metadata = desc["metadata"].copy()
                metadata.update({
                    "processing_time": datetime.now().isoformat(),
                    "element_type": "table"
                })

                point = qdrant_models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "page_content": desc["text"],
                        "metadata": metadata,
                        "table_markdown": desc["table_markdown"],
                        "table_data": desc["table_data"],
                        "content_type": "table"
                    }
                )
                points.append(point)
            except Exception as e:
                logger.warning(f"Errore creazione punto tabella: {e}")
                continue

        return points, len(points) == len(table_descriptions)

    except Exception as e:
        logger.error(f"Errore creazione punti tabella: {e}")
        return [], False


def prepare_image_data(image_element: ImageElement) -> Dict[str, Any]:
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
        description = get_comprehensive_image_info(image_element.image_base64)
        description_text = f"Immagine da {metadata.get('source', '')} pagina {metadata.get('page', 0)} - {', '.join(f'{description[key]} ,' for key in description.keys() if description[key])}"

        return {
            "text": description_text,
            "metadata": metadata,
            "image_base64": image_element.image_base64,  # Mantieni l'immagine per la ricerca
        }
    except Exception as e:
        logger.error(f"Errore preparazione dati immagine: {e}")
        raise


def create_text_points(
    texts: List[str], metadatas: List[Dict[str, Any]],
    embedder: Any = None
) -> Tuple[List[qdrant_models.PointStruct], bool]:
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
        text_embeddings = embedder.embed_documents(texts)

        if len(text_embeddings) != len(texts):
            logger.error(
                f"Mismatch embeddings/testi: {len(text_embeddings)} vs {len(texts)}"
            )
            return [], False

        points = []
        for i, (text, metadata, embedding) in enumerate(
            zip(texts, metadatas, text_embeddings)
        ):
            # Assicura che il metadata abbia il tipo corretto
            metadata["type"] = "text"
            metadata["content_type"] = "text"

            point = qdrant_models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={"page_content": text, "metadata": metadata},
            )
            points.append(point)

        return points, True

    except Exception as e:
        logger.error(f"Errore creazione punti testo: {e}")
        return [], False


def create_image_points(
    image_descriptions: List[Dict[str, Any]],
    embedder: Any = None
) -> Tuple[List[qdrant_models.PointStruct], bool]:
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
        descriptions_text = [desc["text"] for desc in image_descriptions]

        # Genera embeddings per le descrizioni
        image_embeddings = embedder.embed_documents(descriptions_text)

        if len(image_embeddings) != len(image_descriptions):
            logger.error(
                f"Mismatch embeddings/descrizioni: {len(image_embeddings)} vs {len(image_descriptions)}"
            )
            return [], False

        points = []
        for desc_data, embedding in zip(image_descriptions, image_embeddings):
            metadata = desc_data["metadata"].copy()

            # Assicura metadati corretti per immagini
            metadata["type"] = "image"
            metadata["content_type"] = "image"
            metadata["image_base64"] = desc_data.get("image_base64", "")

            point = qdrant_models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={"page_content": desc_data["text"], "metadata": metadata},
            )
            points.append(point)

        return points, True

    except Exception as e:
        logger.error(f"Errore creazione punti immagine: {e}")
        return [], False


def delete_file_from_index(filename: str, qdrant_manager: Any) -> bool:
    """
    Elimina tutti i documenti di un file specifico dall'indice

    Args:
        filename: Nome del file da eliminare

    Returns:
        bool: True se l'eliminazione è riuscita
    """
    try:
        success, message = qdrant_manager.delete_by_source(filename)
        if success:
            logger.info(f"Eliminato file {filename} dall'indice: {message}")
        else:
            logger.error(f"Errore eliminazione file {filename}: {message}")
        return success
    except Exception as e:
        logger.error(f"Errore eliminazione file {filename}: {e}")
        return False
