# streamlit_app/backend_logic.py

import os
import logging
from src.pipeline.indexer_service import DocumentIndexer
from src.utils.qdrant_utils import qdrant_manager
from src.config import RAW_DATA_PATH

logger = logging.getLogger(__name__)

def process_uploaded_file(uploaded_file, indexer: DocumentIndexer) -> tuple[bool, str]:
    """
    Salva e indicizza un file caricato. Non contiene codice UI.
    Restituisce (successo, messaggio).
    """
    try:
        file_name = uploaded_file.name
        save_path = os.path.join(RAW_DATA_PATH, file_name)

        # Salva il file
        os.makedirs(RAW_DATA_PATH, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Indicizza il file
        if uploaded_file.type == "application/pdf":
            indexer.index_files([save_path], force_recreate=False)
            msg = f"File '{file_name}' indicizzato con successo."
            logger.info(msg)
            return True, msg
        else:
            # Qui potresti estendere per gestire anche le immagini
            raise NotImplementedError("Attualmente è supportato solo il formato PDF.")

    except Exception as e:
        logger.error(f"Errore durante l'elaborazione di '{uploaded_file.name}': {e}", exc_info=True)
        return False, f"Errore durante l'indicizzazione: {e}"

def delete_source(filename: str) -> tuple[bool, str]:
    """
    Gestisce l'eliminazione di una fonte da Qdrant e dal disco. Non contiene codice UI.
    Restituisce (successo, messaggio).
    """
    try:
        # 1. Elimina da Qdrant
        success_qdrant, msg_qdrant = qdrant_manager.delete_by_source(filename)
        if not success_qdrant:
            raise Exception(f"Errore Qdrant: {msg_qdrant}")

        # 2. Elimina dal disco
        file_path = os.path.join(RAW_DATA_PATH, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        
        msg = f"Fonte '{filename}' eliminata con successo."
        logger.info(msg)
        return True, msg

    except Exception as e:
        logger.error(f"Errore durante l'eliminazione di '{filename}': {e}", exc_info=True)
        return False, f"Errore durante l'eliminazione: {e}"