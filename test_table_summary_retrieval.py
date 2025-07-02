#!/usr/bin/env python3
"""
Test per verificare che il table_summary sia disponibile nel retrieval e nei risultati RAG.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.pipeline.retriever import create_retriever
from src.utils.qdrant_utils import QdrantManager
from src.utils.embedder import get_multimodal_embedding_model
from src.config import QDRANT_URL, COLLECTION_NAME
from langchain_qdrant import Qdrant
import qdrant_client
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_table_summary_in_retrieval():
    """
    Test per verificare che il retrieval includa i table_summary nei risultati.
    """
    try:
        print("🔍 Creazione retriever...")
        
        # Crea il client Qdrant e il vector store
        client = qdrant_client.QdrantClient(url=QDRANT_URL)
        embedding_model = get_multimodal_embedding_model()
        vector_store = Qdrant(client=client,
                             collection_name=COLLECTION_NAME,
                             embeddings=embedding_model)
        
        # Crea il retriever
        retriever = create_retriever(vector_store)
        
        print("✅ Retriever creato")
        
        # Test di ricerca per tabelle
        test_queries = [
            "tabelle performance modelli",
            "confronto configurazioni GPU",
            "risultati efficienza",
            "table 2",
            "PEFT methods"
        ]
        
        for query in test_queries:
            print(f"\n--- Query: '{query}' ---")
            
            # Esegui ricerca
            results = retriever.invoke(query)
            
            if not results:
                print("❌ Nessun risultato trovato")
                continue
            
            print(f"📊 Risultati trovati: {len(results)}")
            
            # Filtra i risultati che sono tabelle
            table_results = [r for r in results if r.metadata.get('content_type') == 'table']
            
            if not table_results:
                print("⚠️  Nessuna tabella nei risultati")
                continue
                
            print(f"📋 Tabelle nei risultati: {len(table_results)}")
            
            # Verifica presenza del table_summary
            tables_with_summary = 0
            for i, result in enumerate(table_results):
                metadata = result.metadata
                table_summary = metadata.get('table_summary')
                
                print(f"\n  Tabella {i+1}:")
                print(f"    Page: {metadata.get('page', 'N/A')}")
                print(f"    Shape: {metadata.get('table_shape', 'N/A')}")
                print(f"    Score: {getattr(result, 'score', 'N/A')}")
                
                if table_summary:
                    tables_with_summary += 1
                    print(f"    ✅ Summary: {table_summary[:80]}...")
                else:
                    print(f"    ❌ Summary: NON PRESENTE")
            
            print(f"  📈 Tabelle con summary: {tables_with_summary}/{len(table_results)}")
        
        print("\n" + "="*50)
        print("🎯 Test completato con successo!")
        return True
        
    except Exception as e:
        print(f"❌ Errore durante il test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_table_summary_in_retrieval()
    sys.exit(0 if success else 1)
