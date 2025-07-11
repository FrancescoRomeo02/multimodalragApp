# streamlit_app/backend_logic.py

import os
import logging
from src.pipeline.indexer_service import DocumentIndexer
from src.utils.qdrant_utils import qdrant_manager
from src.utils.mongodb_utils import mongodb_manager
from src.config import RAW_DATA_PATH

logger = logging.getLogger(__name__)

def process_uploaded_file(uploaded_file, indexer: DocumentIndexer) -> tuple[bool, str]:
    """
    Save and index an uploaded file.
    """
    try:
        file_name = uploaded_file.name
        save_path = os.path.join(RAW_DATA_PATH, file_name)

        # Save the file
        os.makedirs(RAW_DATA_PATH, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Index the file
        if uploaded_file.type == "application/pdf":
            indexer.index_files([save_path], force_recreate=False)
            msg = f"File '{file_name}' indexed successfully."
            logger.info(msg)
            return True, msg
        else:
            # Here you could extend to also handle images
            raise NotImplementedError("Currently, only PDF format is supported.")

    except Exception as e:
        logger.error(f"Error processing '{uploaded_file.name}': {e}", exc_info=True)
        return False, f"Error indexing: {e}"

def delete_source(filename: str) -> tuple[bool, str]:
    """
    Manage the deletion of a source from both databases (Qdrant and MongoDB) and the disk.
    Returns (success, message).
    """
    try:
        # 1. Delete from Qdrant
        success_qdrant, msg_qdrant = qdrant_manager.delete_by_source(filename)
        if not success_qdrant:
            raise Exception(f"Qdrant error: {msg_qdrant}")
            
        # 2. Delete from MongoDB
        success_mongodb, msg_mongodb = mongodb_manager.delete_by_source(filename)
        if not success_mongodb:
            logger.warning(f"MongoDB error while deleting: {msg_mongodb}")
            # Continuiamo comunque per eliminare il file dal disco

        # 3. Delete from disk
        file_path = os.path.join(RAW_DATA_PATH, filename)
        if os.path.exists(file_path):
            os.remove(file_path)

        msg = f"Source '{filename}' deleted successfully from all databases."
        logger.info(msg)
        return True, msg

    except Exception as e:
        logger.error(f"Error deleting '{filename}': {e}", exc_info=True)
        return False, f"Error deleting: {e}"