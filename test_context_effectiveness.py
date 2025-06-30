#!/usr/bin/env python3
"""
Test per confrontare l'efficacia della ricerca con e senza contesto
"""

import sys
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from app.pipeline.retriever import enhanced_rag_query
from app.utils.qdrant_utils import qdrant_manager
from app.config import COLLECTION_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_context_effectiveness():
    """
    Testa l'efficacia del contesto nella ricerca
    """
    # Query specifiche che dovrebbero beneficiare del contesto
    test_queries = [
        "configurazioni PEFT su GPU A100",
        "tabella sui metodi QLoRA",
        "risultati fine-tuning LLaMA",
        "tempo di inferenza quantizzato",
        "parametri LoRA rank alpha"
    ]
    
    print("=== TEST EFFICACIA CONTESTO ===\n")
    
    for i, query in enumerate(test_queries):
        print(f"Query {i+1}: {query}")
        
        try:
            # Esegui la query
            result = enhanced_rag_query(query, multimodal=True)
            
            print(f"  Documenti trovati: {len(result.source_documents)}")
            
            # Analizza i primi 3 risultati per tipo
            tables_found = 0
            images_found = 0
            text_found = 0
            context_used = 0
            
            for doc in result.source_documents[:5]:  # Primi 5 risultati
                doc_type = doc.get('type', 'unknown')
                
                if doc_type == 'table':
                    tables_found += 1
                elif doc_type == 'image':
                    images_found += 1
                elif doc_type == 'text':
                    text_found += 1
                
                # Controlla se ha contesto
                if doc.get('context_text') or doc.get('caption'):
                    context_used += 1
            
            print(f"  Tabelle: {tables_found}, Immagini: {images_found}, Testo: {text_found}")
            print(f"  Elementi con contesto: {context_used}/{min(5, len(result.source_documents))}")
            
            # Mostra un esempio di risultato con contesto
            for doc in result.source_documents[:2]:
                if doc.get('type') in ['table', 'image'] and (doc.get('context_text') or doc.get('caption')):
                    print(f"  Esempio con contesto - Tipo: {doc.get('type')}")
                    if doc.get('caption'):
                        print(f"    Caption: {doc.get('caption')[:80]}...")
                    if doc.get('context_text'):
                        print(f"    Contesto: {doc.get('context_text')[:80]}...")
                    break
            
            print()
            
        except Exception as e:
            print(f"  Errore: {e}")
            print()

def analyze_database_content():
    """
    Analizza il contenuto del database per vedere l'uso del contesto
    """
    print("=== ANALISI CONTENUTO DATABASE ===\n")
    
    client = qdrant_manager.client
    points = client.scroll(COLLECTION_NAME, limit=20, with_payload=True)
    
    total_elements = 0
    elements_with_context = 0
    elements_with_caption = 0
    
    context_examples = []
    
    for point in points[0]:
        payload = point.payload
        content_type = payload.get('content_type', 'unknown')
        page_content = payload.get('page_content', '')
        metadata = payload.get('metadata', {})
        
        total_elements += 1
        
        # Controlla se l'elemento ha contesto nel contenuto dell'embedding
        if 'Contesto' in page_content or '|' in page_content:
            elements_with_context += 1
            
            if content_type in ['table', 'image'] and len(context_examples) < 3:
                context_examples.append({
                    'type': content_type,
                    'content_preview': page_content[:150],
                    'has_caption': metadata.get('caption') is not None or metadata.get('manual_caption') is not None,
                    'has_context': metadata.get('context_text') is not None
                })
        
        # Controlla caption nei metadati
        if metadata.get('caption') or metadata.get('manual_caption'):
            elements_with_caption += 1
    
    print(f"Elementi totali: {total_elements}")
    print(f"Elementi con contesto nell'embedding: {elements_with_context} ({elements_with_context/total_elements*100:.1f}%)")
    print(f"Elementi con caption nei metadati: {elements_with_caption} ({elements_with_caption/total_elements*100:.1f}%)")
    
    print(f"\nEsempi di contenuto arricchito:")
    for i, example in enumerate(context_examples):
        print(f"\nEsempio {i+1} - Tipo: {example['type']}")
        print(f"  Ha caption: {example['has_caption']}")
        print(f"  Ha contesto: {example['has_context']}")
        print(f"  Contenuto: {example['content_preview']}...")

if __name__ == "__main__":
    print("VERIFICA EFFICACIA SISTEMA CON CONTESTO\n")
    
    analyze_database_content()
    print("\n" + "="*50 + "\n")
    test_context_effectiveness()
    
    print("Test completato!")
