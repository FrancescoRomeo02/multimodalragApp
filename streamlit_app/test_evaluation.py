import streamlit as st
import logging
import time
import uuid
from typing import List, Dict, Any, Optional

# Add project root to path
import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from streamlit_app.styles import get_custom_css
from streamlit_app.components.ui_components import upload_widget, source_selector_widget
from src.pipeline.indexer_service import DocumentIndexer
from src.utils.embedder import get_embedding_model
from src.pipeline.retriever import enhanced_rag_query
from metrics.main_metrics import (
    global_metrics_collector,
    track_retrieval,
    track_generation,
    track_multimodal,
    track_cost_efficiency
)

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(layout="wide", page_title="MultimodalRAG Test & Evaluation", page_icon="🧪")
st.markdown(get_custom_css(), unsafe_allow_html=True)

# --- Critical Services Loading (with caching) ---
@st.cache_resource
def load_main_services():
    """Load heavy services only once at app startup."""
    logger.info("Initializing services (embedder and indexer with semantic chunking)...")
    try:
        embedder = get_embedding_model()
        indexer = DocumentIndexer(embedder=embedder)
        chunking_type = "Semantic" if indexer.semantic_chunker else "Classic"
        logger.info(f"Services initialized successfully - Chunking: {chunking_type}")
        return embedder, indexer
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        st.error(f"Failed to initialize services: {e}")
        return None, None

def extract_page_numbers_from_sources(source_docs) -> List[str]:
    """Extract page numbers or document IDs from source documents"""
    pages = []
    for doc in source_docs:
        # Handle both dict and object formats
        if isinstance(doc, dict):
            metadata = doc.get('metadata', {})
            content = doc.get('page_content', str(doc))
        else:
            metadata = getattr(doc, 'metadata', {})
            content = getattr(doc, 'page_content', str(doc))
        
        if metadata:
            # Try to get page number
            if 'page' in metadata:
                pages.append(f"page_{metadata['page']}")
            elif 'page_number' in metadata:
                pages.append(f"page_{metadata['page_number']}")
            elif 'source' in metadata:
                # Extract filename and use as identifier
                source = metadata['source']
                filename = os.path.basename(source)
                pages.append(filename)
            else:
                # Fallback to document ID if available
                pages.append(f"doc_{hash(content) % 10000}")
        else:
            # Fallback to document ID if no metadata
            pages.append(f"doc_{hash(content) % 10000}")
    return pages

def calculate_multimodal_metrics(query: str, source_docs, relevant_pages: List[str]) -> Dict[str, Any]:
    """Calculate multimodal metrics based on query type and retrieved content"""
    query_lower = query.lower()
    
    # Check if query asks for images
    image_query = any(word in query_lower for word in ["image", "immagine", "figura", "figure", "diagram", "grafico"])
    # Check if query asks for tables  
    table_query = any(word in query_lower for word in ["table", "tabella", "data", "dati"])
    
    # Check what was actually retrieved
    image_retrieved = False
    text_retrieved = False
    
    for doc in source_docs:
        # Handle both dict and object formats
        if isinstance(doc, dict):
            metadata = doc.get('metadata', {})
        else:
            metadata = getattr(doc, 'metadata', {})
        
        if metadata:
            content_type = metadata.get('content_type', '')
            if 'image' in content_type:
                image_retrieved = True
            if 'text' in content_type:
                text_retrieved = True
    
    # Simple heuristic for relevance (in real scenario, this would be more sophisticated)
    retrieved_pages = extract_page_numbers_from_sources(source_docs)
    relevant_set = set(relevant_pages)
    retrieved_set = set(retrieved_pages)
    
    # Image relevance: did we retrieve images when needed and are they from relevant pages?
    image_relevant = False
    if image_query:
        image_relevant = image_retrieved and len(relevant_set & retrieved_set) > 0
    else:
        image_relevant = not image_retrieved  # Correct if we didn't retrieve images when not needed
    
    # Text relevance: similar logic for text
    text_relevant = text_retrieved and len(relevant_set & retrieved_set) > 0
    
    # Overall multimodal accuracy
    multimodal_accuracy = (image_relevant + text_relevant) / 2.0
    
    return {
        "image_retrieved": image_retrieved,
        "text_retrieved": text_retrieved, 
        "image_relevant": image_relevant,
        "text_relevant": text_relevant,
        "multimodal_accuracy": multimodal_accuracy
    }

def process_test_query(query: str, expected_answer: str, relevant_pages: List[str], 
                      selected_files: Optional[List[str]] = None) -> Dict[str, Any]:
    """Process a test query and automatically calculate all metrics"""
    
    query_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    # Execute the RAG query
    result = enhanced_rag_query(query, selected_files)
    total_time = time.time() - start_time
    
    # Extract information from result
    generated_answer = result.answer
    source_docs = result.source_documents
    retrieval_time = (result.query_time_ms / 1000.0) if result.query_time_ms else total_time  # Convert to seconds or use total time
    
    # Extract page numbers from retrieved documents
    retrieved_pages = extract_page_numbers_from_sources(source_docs)
    
    # Calculate metrics
    metrics_data = {}
    
    # 1. Retrieval Metrics
    retrieval_metrics = track_retrieval(
        query_id=query_id,
        relevant_docs=relevant_pages,
        retrieved_docs=retrieved_pages,
        retrieval_time=retrieval_time
    )
    metrics_data['retrieval'] = {
        'recall_at_1': retrieval_metrics.recall_at_k(1),
        'recall_at_3': retrieval_metrics.recall_at_k(3),
        'recall_at_5': retrieval_metrics.recall_at_k(5),
        'precision_at_1': retrieval_metrics.precision_at_k(1),
        'precision_at_3': retrieval_metrics.precision_at_k(3),
        'precision_at_5': retrieval_metrics.precision_at_k(5),
        'mrr': retrieval_metrics.mean_reciprocal_rank(),
        'retrieved_pages': retrieved_pages,
        'relevant_pages': relevant_pages
    }
    
    # 2. Generation Metrics  
    # Estimate token count (rough approximation)
    token_count = len(generated_answer.split()) * 1.3  # Rough token estimation
    generation_time = total_time - retrieval_time
    
    generation_metrics = track_generation(
        query_id=query_id,
        generated_text=generated_answer,
        reference_text=expected_answer,
        generation_time=generation_time,
        token_count=int(token_count)
    )
    
    bert_scores = generation_metrics.bert_score()
    metrics_data['generation'] = {
        'exact_match': generation_metrics.exact_match(),
        'bert_score_f1': bert_scores['f1'],
        'bert_score_precision': bert_scores['precision'],
        'bert_score_recall': bert_scores['recall'],
        'generated_answer': generated_answer,
        'expected_answer': expected_answer
    }
    
    # 3. Multimodal Metrics
    multimodal_data = calculate_multimodal_metrics(query, source_docs, relevant_pages)
    multimodal_metrics = track_multimodal(
        query_id=query_id,
        **multimodal_data
    )
    metrics_data['multimodal'] = {
        'image_text_accuracy': multimodal_metrics.image_text_accuracy(),
        **multimodal_data
    }
    
    # 4. Cost & Efficiency Metrics
    # Rough estimation for demo purposes
    embedding_tokens = len(query.split()) * 1.3
    generation_tokens = token_count
    total_tokens = embedding_tokens + generation_tokens
    
    cost_metrics = track_cost_efficiency(
        query_id=query_id,
        total_tokens=int(total_tokens),
        embedding_tokens=int(embedding_tokens),
        generation_tokens=int(generation_tokens),
        response_time=total_time,
        embedding_time=retrieval_time
    )
    
    metrics_data['cost_efficiency'] = {
        'total_tokens': int(total_tokens),
        'response_time': total_time,
        'tokens_per_second': cost_metrics.tokens_per_second(),
        'embedding_tokens_per_second': cost_metrics.embedding_tokens_per_second()
    }
    
    # Add original result data
    metrics_data['result'] = {
        'answer': generated_answer,
        'source_documents': source_docs,
        'confidence_score': result.confidence_score,
        'query_time_ms': result.query_time_ms,
        'retrieved_count': result.retrieved_count
    }
    
    return metrics_data

def main():
    """Main application function"""
    
    # Initialize services
    embedder, indexer = load_main_services()
    if not embedder or not indexer:
        st.stop()
    
    # Header
    st.title("🧪 MultimodalRAG Test & Evaluation")
    st.markdown("Interfaccia per testare e valutare automaticamente le performance del sistema RAG")
    
    # Sidebar for file management
    with st.sidebar:
        st.header("📁 Document Management")
        
        # File upload
        upload_widget(indexer)
        
        # Source selection
        st.subheader("🎯 Source Selection")
        selected_files = source_selector_widget()
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔍 Test Query Setup")
        
        # Query input
        query = st.text_area(
            "Query/Domanda:",
            placeholder="Inserisci la domanda da testare...",
            height=100
        )
        
        # Expected answer input
        expected_answer = st.text_area(
            "Risposta Attesa:",
            placeholder="Inserisci la risposta corretta/attesa...",
            height=150
        )
        
        # Relevant pages input
        st.subheader("📄 Pagine/Documenti Rilevanti")
        st.markdown("Inserisci i nomi delle pagine o documenti che contengono la risposta (uno per riga)")
        
        relevant_pages_text = st.text_area(
            "Pagine rilevanti:",
            placeholder="esempio:\npage_1\npage_3\ndocumento.pdf\nfigura_2.png",
            height=100
        )
        
        relevant_pages = [page.strip() for page in relevant_pages_text.split('\n') if page.strip()]
        
        # Test button
        if st.button("🚀 Esegui Test e Calcola Metriche", type="primary", disabled=not (query and expected_answer and relevant_pages)):
            with st.spinner("Executing query and calculating metrics..."):
                try:
                    # Process the test query
                    metrics_data = process_test_query(
                        query=query,
                        expected_answer=expected_answer,
                        relevant_pages=relevant_pages,
                        selected_files=selected_files
                    )
                    
                    # Store in session state for display
                    st.session_state['last_test_result'] = metrics_data
                    st.success("✅ Test completato con successo!")
                    
                except Exception as e:
                    st.error(f"❌ Errore durante il test: {e}")
                    logger.error(f"Test error: {e}", exc_info=True)
    
    with col2:
        st.header("📊 Metrics Overview")
        
        # Display current metrics summary
        try:
            summary = global_metrics_collector.get_complete_summary()
            
            # Create metrics cards
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.metric("Total Queries", summary.get('retrieval_metrics', {}).get('total_queries', 0))
                st.metric("Avg MRR", f"{summary.get('retrieval_metrics', {}).get('mean_reciprocal_rank', 0):.3f}")
            
            with col_b:
                st.metric("Avg BERT F1", f"{summary.get('generation_metrics', {}).get('bert_score', {}).get('avg_f1', 0):.3f}")
                st.metric("EM Rate", f"{summary.get('generation_metrics', {}).get('exact_match_rate', 0):.3f}")
            
            # Export button
            if st.button("📥 Export Metrics"):
                export_path = global_metrics_collector.export_metrics("test_session_metrics.json")
                st.success(f"Metrics exported to: {export_path}")
        
        except Exception as e:
            st.info("Nessuna metrica disponibile ancora")
    
    # Display last test result
    if 'last_test_result' in st.session_state:
        st.header("📈 Risultati Ultimo Test")
        
        result_data = st.session_state['last_test_result']
        
        # Create tabs for different metric types
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Risultato", "🔍 Retrieval", "✍️ Generation", "🖼️ Multimodal", "💰 Cost & Efficiency"])
        
        with tab1:
            st.subheader("Risposta Generata")
            st.write(result_data['result']['answer'])
            
            st.subheader("Documenti Recuperati")
            for i, doc in enumerate(result_data['result']['source_documents'][:3]):  # Show first 3
                with st.expander(f"Documento {i+1}"):
                    # Handle both dict and object formats
                    if isinstance(doc, dict):
                        content = doc.get('page_content', str(doc))
                        metadata = doc.get('metadata', {})
                    else:
                        content = getattr(doc, 'page_content', str(doc))
                        metadata = getattr(doc, 'metadata', {})
                    
                    # Display content (truncated if too long)
                    if isinstance(content, str) and len(content) > 500:
                        st.write(content[:500] + "...")
                    else:
                        st.write(content)
                    
                    # Display metadata if available
                    if metadata:
                        st.json(metadata)
        
        with tab2:
            retrieval = result_data['retrieval']
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Recall@1", f"{retrieval['recall_at_1']:.3f}")
                st.metric("Recall@3", f"{retrieval['recall_at_3']:.3f}")
            
            with col2:
                st.metric("Precision@1", f"{retrieval['precision_at_1']:.3f}")
                st.metric("Precision@3", f"{retrieval['precision_at_3']:.3f}")
            
            with col3:
                st.metric("MRR", f"{retrieval['mrr']:.3f}")
            
            st.subheader("Confronto Pagine")
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("**Pagine Rilevanti:**")
                for page in retrieval['relevant_pages']:
                    st.write(f"- {page}")
            
            with col_b:
                st.write("**Pagine Recuperate:**")
                for page in retrieval['retrieved_pages']:
                    st.write(f"- {page}")
        
        with tab3:
            generation = result_data['generation']
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Exact Match", "✅" if generation['exact_match'] else "❌")
                st.metric("BERT F1", f"{generation['bert_score_f1']:.3f}")
            
            with col2:
                st.metric("BERT Precision", f"{generation['bert_score_precision']:.3f}")
                st.metric("BERT Recall", f"{generation['bert_score_recall']:.3f}")
            
            st.subheader("Confronto Testi")
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("**Risposta Attesa:**")
                st.info(generation['expected_answer'])
            
            with col_b:
                st.write("**Risposta Generata:**")
                st.info(generation['generated_answer'])
        
        with tab4:
            multimodal = result_data['multimodal']
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Image Retrieved", "✅" if multimodal['image_retrieved'] else "❌")
                st.metric("Text Retrieved", "✅" if multimodal['text_retrieved'] else "❌")
            
            with col2:
                st.metric("Image Relevant", "✅" if multimodal['image_relevant'] else "❌")
                st.metric("Text Relevant", "✅" if multimodal['text_relevant'] else "❌")
            
            st.metric("Image/Text Accuracy", f"{multimodal['image_text_accuracy']:.3f}")
        
        with tab5:
            cost = result_data['cost_efficiency']
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Tokens", cost['total_tokens'])
                st.metric("Response Time", f"{cost['response_time']:.2f}s")
            
            with col2:
                st.metric("Tokens/sec", f"{cost['tokens_per_second']:.1f}")
                st.metric("Embedding Tokens/sec", f"{cost['embedding_tokens_per_second']:.1f}")

if __name__ == "__main__":
    main()
