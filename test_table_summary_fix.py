#!/usr/bin/env python3
"""
Test per verificare che il table_summary sia correttamente aggiunto ai metadati delle tabelle.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.pdf_parser import parse_pdf_elements
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_table_summary_parsing():
    """
    Test per verificare che il parsing PDF aggiunga correttamente il table_summary ai metadati.
    """
    # Path del PDF di test
    pdf_path = "/Users/fraromeo/Documents/02_Areas/University/LM/LM_24-25/SEM2/AD/multimodalrag/data/raw/fintuning.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ File PDF non trovato: {pdf_path}")
        return False
    
    print(f"🔍 Parsing PDF: {pdf_path}")
    
    try:
        # Parse del PDF
        text_elements, table_elements, image_elements = parse_pdf_elements(pdf_path)
        
        # Combina tutti gli elementi per il test
        elements = text_elements + table_elements + image_elements
        
        print(f"✅ Parsing completato. Elementi estratti: {len(elements)}")
        
        # Filtra le tabelle
        tables = [elem for elem in elements if elem.get('metadata', {}).get('content_type') == 'table']
        
        print(f"📊 Tabelle trovate: {len(tables)}")
        
        if not tables:
            print("⚠️  Nessuna tabella trovata nel PDF")
            return True
        
        # Verifica ogni tabella
        tables_with_summary = 0
        for i, table in enumerate(tables):
            metadata = table.get('metadata', {})
            table_summary = metadata.get('table_summary')
            
            print(f"\n--- Tabella {i+1} ---")
            print(f"Page: {metadata.get('page', 'N/A')}")
            print(f"Shape: {metadata.get('table_shape', 'N/A')}")
            print(f"Caption: {metadata.get('caption', 'N/A')}")
            
            if table_summary:
                tables_with_summary += 1
                print(f"✅ Table Summary: {table_summary[:100]}...")
                
                # Verifica che non sia un oggetto invece di una stringa
                if isinstance(table_summary, str):
                    print("✅ Table summary è una stringa (corretto)")
                else:
                    print(f"❌ Table summary NON è una stringa: {type(table_summary)}")
                    return False
            else:
                print("❌ Table Summary: NON PRESENTE")
        
        print(f"\n📈 Summary: {tables_with_summary}/{len(tables)} tabelle hanno il riassunto")
        
        if tables_with_summary == len(tables):
            print("🎉 SUCCESSO: Tutte le tabelle hanno il riassunto!")
            return True
        else:
            print(f"⚠️  Solo {tables_with_summary} su {len(tables)} tabelle hanno il riassunto")
            return False
            
    except Exception as e:
        print(f"❌ Errore durante il test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_table_summary_parsing()
    sys.exit(0 if success else 1)
