#!/usr/bin/env python3
"""
Test per verificare che image_base64 non venga salvato nel database.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.pdf_parser import parse_pdf_elements
from src.utils.qdrant_utils import QdrantManager
from qdrant_client import models
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_image_base64_removal():
    """
    Test per verificare che image_base64 non sia salvato nel database.
    """
    try:
        print("🔍 Test parsing PDF e verifica image_base64...")
        
        # Path del PDF di test
        pdf_path = "/Users/fraromeo/Documents/02_Areas/University/LM/LM_24-25/SEM2/AD/multimodalrag/data/raw/fintuning.pdf"
        
        if not os.path.exists(pdf_path):
            print(f"❌ File PDF non trovato: {pdf_path}")
            return False
        
        # Parse del PDF
        text_elements, image_elements, table_elements = parse_pdf_elements(pdf_path)
        
        print(f"✅ Parsing completato: {len(image_elements)} immagini estratte")
        
        if not image_elements:
            print("⚠️  Nessuna immagine trovata nel PDF")
            return True
        
        # Verifica che image_base64 sia presente negli elementi estratti
        images_with_base64 = 0
        for i, img in enumerate(image_elements):
            if 'image_base64' in img:
                images_with_base64 += 1
                print(f"✅ Immagine {i+1}: image_base64 presente (lunghezza: {len(img['image_base64'])} caratteri)")
            else:
                print(f"❌ Immagine {i+1}: image_base64 NON presente")
        
        print(f"📊 Immagini con image_base64: {images_with_base64}/{len(image_elements)}")
        
        # Ora testiamo se viene salvato nel database
        print("\n🔍 Test salvataggio nel database...")
        
        qdrant_manager = QdrantManager()
        
        if not qdrant_manager.verify_connection():
            print("❌ Impossibile connettersi a Qdrant")
            return False
        
        # Cerca immagini nel database
        img_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="content_type",
                    match=models.MatchValue(value="image")
                )
            ]
        )
        
        query_embedding = qdrant_manager.embedder.embed_query("immagini")
        
        raw_results = qdrant_manager.client.search(
            collection_name=qdrant_manager.collection_name,
            query_vector=query_embedding,
            query_filter=img_filter,
            limit=5,
            with_payload=True,
            score_threshold=0.1
        )
        
        if not raw_results:
            print("⚠️  Nessuna immagine trovata nel database")
            return True
        
        print(f"📊 Immagini nel database: {len(raw_results)}")
        
        # Verifica che image_base64 NON sia nel database
        images_with_base64_in_db = 0
        for i, result in enumerate(raw_results):
            payload = result.payload or {}
            
            print(f"\n--- Immagine DB {i+1} ---")
            print(f"Score: {result.score:.3f}")
            print(f"Source: {payload.get('metadata', {}).get('source', 'N/A')}")
            print(f"Page: {payload.get('metadata', {}).get('page', 'N/A')}")
            
            # Verifica image_base64 nel payload principale
            if 'image_base64' in payload:
                images_with_base64_in_db += 1
                print(f"❌ image_base64 presente nel payload principale!")
            else:
                print(f"✅ image_base64 NON presente nel payload principale")
            
            # Verifica image_base64 nei metadati
            metadata = payload.get('metadata', {})
            if 'image_base64' in metadata:
                images_with_base64_in_db += 1
                print(f"❌ image_base64 presente nei metadati!")
            else:
                print(f"✅ image_base64 NON presente nei metadati")
        
        print(f"\n📈 Immagini con image_base64 nel DB: {images_with_base64_in_db}/{len(raw_results)}")
        
        if images_with_base64_in_db == 0:
            print("🎉 SUCCESSO: image_base64 non è salvato nel database!")
            return True
        else:
            print(f"❌ FALLIMENTO: {images_with_base64_in_db} immagini hanno ancora image_base64 nel database")
            return False
            
    except Exception as e:
        print(f"❌ Errore durante il test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_image_base64_removal()
    sys.exit(0 if success else 1)
