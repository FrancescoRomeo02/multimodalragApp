#!/usr/bin/env python3
"""
Test semplice per verificare se i table_summary sono nel datab        print(f"📈 Tabelle c        print(f"📈 Tabelle con summary: {tables_with_summary}/{len(table_results)}")n summary: {tables_with_summary}/{len(table_results)}")se.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.qdrant_utils import QdrantManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_table_summary_in_database():
    """
    Test per verificare che le tabelle nel database abbiano il table_summary.
    """
    try:
        print("🔍 Connessione al database Qdrant...")
        
        # Crea il manager Qdrant
        qdrant_manager = QdrantManager()
        
        if not qdrant_manager.verify_connection():
            print("❌ Impossibile connettersi a Qdrant")
            return False
        
        if not qdrant_manager.collection_exists():
            print("❌ Collection non esiste")
            return False
        
        print("✅ Connessione al database riuscita")
        
        # Query per tabelle
        query = "tabelle performance"
        
        print(f"🔍 Ricerca: '{query}'")
        
        # Query direttamente per tabelle usando search_vectors
        import qdrant_client
        from qdrant_client import models
        
        query_embedding = qdrant_manager.embedder.embed_query(query)
        
        # Filtra esplicitamente per content_type = table
        table_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="content_type",
                    match=models.MatchValue(value="table")
                )
            ]
        )
        
        raw_results = qdrant_manager.client.search(
            collection_name=qdrant_manager.collection_name,
            query_vector=query_embedding,
            query_filter=table_filter,
            limit=10,
            with_payload=True,
            score_threshold=0.1
        )
        
        # Converti i risultati in formato standard
        results = []
        for result in raw_results:
            payload = result.payload or {}
            metadata = payload.get("metadata", {})
            results.append({
                "score": result.score,
                "content_type": payload.get("content_type", "unknown"),
                "metadata": metadata,
                "source": metadata.get("source", "Unknown"),
                "page": metadata.get("page", "N/A"),
                "type": metadata.get("type", "unknown")
            })
        
        if not results:
            print("❌ Nessun risultato trovato nel database")
            return False
        
        print(f"📊 Risultati trovati: {len(results)}")
        
        # Filtra solo le tabelle
        table_results = [r for r in results if r.get('content_type') == 'table']
        
        if not table_results:
            print("❌ Nessuna tabella trovata nei risultati")
            return False
        
        print(f"� Tabelle nei risultati: {len(table_results)}")
        
        # Analizza ogni risultato
        tables_with_summary = 0
        
        for i, result in enumerate(table_results):
            print(f"\n--- Tabella {i+1} ---")
            print(f"Score: {result.get('score', 'N/A')}")
            print(f"Source: {result.get('source', 'N/A')}")
            print(f"Page: {result.get('page', 'N/A')}")
            print(f"Type: {result.get('type', 'N/A')}")
            print(f"Content Type: {result.get('content_type', 'N/A')}")
            
            metadata = result.get('metadata', {})
            print(f"Shape: {metadata.get('table_shape', 'N/A')}")
            
            # Verifica presence del table_summary
            table_summary = metadata.get('table_summary')
            
            if table_summary:
                tables_with_summary += 1
                print(f"✅ Table Summary: {table_summary[:100]}...")
            else:
                print("❌ Table Summary: NON PRESENTE")
        
        print(f"\n📈 Tabelle con summary: {tables_with_summary}/{len(results)}")
        
        if tables_with_summary > 0:
            print("🎉 SUCCESSO: Alcune tabelle hanno il riassunto nel database!")
            return True
        else:
            print("⚠️  Nessuna tabella ha il riassunto nel database - serve reindicizzare")
            return False
            
    except Exception as e:
        print(f"❌ Errore durante il test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_table_summary_in_database()
    sys.exit(0 if success else 1)
