# Home.py

import sys
import os
import streamlit as st
import logging
# --- Bootstrap per risolvere gli import ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from streamlit_app.styles import get_custom_css

# --- Importazioni dalla nuova architettura ---

from streamlit_app.components.ui_components import upload_widget, source_selector_widget, enhanced_chat_interface_widget
from src.pipeline.indexer_service import DocumentIndexer
from src.utils.embedder import get_multimodal_embedding_model
from src.config import QDRANT_URL, COLLECTION_NAME

# --- Configurazione ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(layout="wide", page_title="RAG per Paper Scientifici", page_icon="🔬")
st.markdown(get_custom_css(), unsafe_allow_html=True)

# --- Caricamento Servizi Critici (con caching) ---
@st.cache_resource
def load_main_services():
    """Carica i servizi pesanti una sola volta all'avvio dell'app."""
    logger.info("Inizializzazione dei servizi (embedder e indexer con chunking semantico)...")
    try:
        embedder = get_multimodal_embedding_model()
        # Usa la configurazione globale per il chunking semantico
        indexer = DocumentIndexer(embedder=embedder)
        
        # Verifica se chunking semantico è attivo
        chunking_type = "Semantico" if indexer.semantic_chunker else "Classico"
        logger.info(f"Servizi inizializzati con successo. Chunking: {chunking_type}")
        return indexer, chunking_type
    except Exception as e:
        logger.error(f"Errore critico durante il caricamento dei servizi: {e}", exc_info=True)
        return None, None

# --- Avvio dell'Applicazione ---
indexer, chunking_type = load_main_services()

if not indexer:
    st.error("Errore critico: impossibile caricare i servizi di base. L'applicazione non può continuare. Controlla i log per i dettagli.")
    st.stop()

# Mostra il tipo di chunking utilizzato
if chunking_type:
    if chunking_type == "Semantico":
        st.success(f"🧠 Sistema attivo con Chunking {chunking_type} - Segmentazione intelligente dei documenti")
    else:
        st.warning(f"📝 Sistema attivo con Chunking {chunking_type} - Fallback alla segmentazione classica")

st.title("Assistente di Ricerca per Paper Scientifici 🔬")


with st.sidebar:
    upload_widget(indexer)
    st.markdown("<hr>", unsafe_allow_html=True)
    selected_sources = source_selector_widget()


enhanced_chat_interface_widget(selected_sources=selected_sources)