# Home.py

import sys
import os
import streamlit as st
import logging
# --- Bootstrap to resolve imports ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from streamlit_app.styles import get_custom_css

# --- Imports from new architecture ---

from streamlit_app.components.ui_components import upload_widget, source_selector_widget, enhanced_chat_interface_widget
from src.pipeline.indexer_service import DocumentIndexer
from src.utils.embedder import get_embedding_model
from metrics.main_metrics import global_metrics_collector

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(layout="wide", page_title="Scientific Papers RAG", page_icon="🔬")
st.markdown(get_custom_css(), unsafe_allow_html=True)

# --- Critical Services Loading (with caching) ---
@st.cache_resource
def load_main_services():
    """Load heavy services only once at app startup."""
    logger.info("Initializing services (embedder and indexer with semantic chunking)...")
    try:
        embedder = get_embedding_model()
        # Use global configuration for semantic chunking
        indexer = DocumentIndexer(embedder=embedder)
        
        # Check if semantic chunking is active
        chunking_type = "Semantic" if indexer.semantic_chunker else "Classic"
        logger.info(f"Services initialized successfully. Chunking: {chunking_type}")
        return indexer, chunking_type
    except Exception as e:
        logger.error(f"Critical error during services loading: {e}", exc_info=True)
        return None, None

# --- Application Startup ---
indexer, chunking_type = load_main_services()

if not indexer:
    st.error("Critical error: unable to load base services. The application cannot continue. Check logs for details.")
    st.stop()

# Show the type of chunking being used
if chunking_type:
    if chunking_type == "Semantic":
        st.success(f"System active with {chunking_type} Chunking - Intelligent document segmentation")
    else:
        st.warning(f"System active with {chunking_type} Chunking - Fallback to classic segmentation")

st.title("Scientific Papers Research Assistant")

# Aggiungi sezione metriche nella sidebar
with st.sidebar:
    upload_widget(indexer)
    st.markdown("<hr>", unsafe_allow_html=True)
    selected_sources = source_selector_widget()
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Sezione Metriche
    with st.expander("📊 Metriche Query", expanded=False):
        cost_summary = global_metrics_collector.get_cost_efficiency_summary()
        
        if cost_summary.get('total_queries', 0) > 0:
            st.metric("Query Totali", cost_summary['total_queries'])
            st.metric("Token Totali", cost_summary['total_tokens'])
            st.metric("Tempo Medio (s)", f"{cost_summary['avg_response_time']:.2f}")
            st.metric("Token/s", f"{cost_summary['avg_tokens_per_second']:.1f}")
            
            # Pulsante per esportare metriche
            if st.button("💾 Esporta Metriche", key="export_metrics"):
                try:
                    export_path = global_metrics_collector.export_metrics()
                    st.success(f"Metriche esportate in: {export_path}")
                except Exception as e:
                    st.error(f"Errore nell'export: {e}")
            
            # Pulsante per pulire metriche
            if st.button("🗑️ Pulisci Metriche", key="clear_metrics"):
                global_metrics_collector.clear_metrics()
                st.success("Metriche pulite!")
                st.rerun()
        else:
            st.info("Nessuna metrica disponibile. Inizia facendo delle query!")


enhanced_chat_interface_widget(selected_sources=selected_sources)
