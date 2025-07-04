#!/usr/bin/env python3
"""
Test per verificare che la standardizzazione dei metadati sia corretta
e che non ci siano più errori di validazione Pydantic.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.core.models import (
    TextElement, TextMetadata,
    ImageElement, ImageMetadata, 
    TableElement, TableMetadata, TableData
)

def test_unified_metadata():
    """Test per verificare che tutti i modelli usino content_type uniformemente."""
    
    print("🧪 Test unificazione metadati con content_type...")
    
    # Test TextMetadata
    try:
        text_meta = TextMetadata(
            source="test.pdf",
            page=1,
            content_type="text"
        )
        print(f"✅ TextMetadata: {text_meta.content_type}")
        
        text_element = TextElement(
            text="Testo di esempio",
            metadata=text_meta
        )
        print(f"✅ TextElement creato correttamente")
        
    except Exception as e:
        print(f"❌ Errore TextMetadata: {e}")
        return False
    
    # Test ImageMetadata
    try:
        image_meta = ImageMetadata(
            source="test.pdf",
            page=2,
            content_type="image",
            image_caption="Caption test",
            context_text=None,
            manual_caption=None
        )
        print(f"✅ ImageMetadata: {image_meta.content_type}")
        
        image_element = ImageElement(
            page_content="Descrizione immagine",
            image_base64="test_base64",
            metadata=image_meta
        )
        print(f"✅ ImageElement creato correttamente")
        
    except Exception as e:
        print(f"❌ Errore ImageMetadata: {e}")
        return False
        
    # Test TableMetadata
    try:
        table_meta = TableMetadata(
            source="test.pdf",
            page=3,
            content_type="table",
            bbox=None,
            table_shape=None,
            caption=None,
            context_text=None,
            table_summary="Tabella di esempio"
        )
        print(f"✅ TableMetadata: {table_meta.content_type}")
        
        table_data = TableData(
            cells=[["A", "B"], ["1", "2"]],
            headers=["Col1", "Col2"],
            shape=(2, 2)
        )
        
        table_element = TableElement(
            table_data=table_data,
            table_markdown="| Col1 | Col2 |\n|------|------|\n| A | B |\n| 1 | 2 |",
            metadata=table_meta
        )
        print(f"✅ TableElement creato correttamente")
        
    except Exception as e:
        print(f"❌ Errore TableMetadata: {e}")
        return False
    
    # Test validazione metadati
    print("\n🔍 Verifica consistenza metadati:")
    print(f"Text content_type: {text_meta.content_type}")
    print(f"Image content_type: {image_meta.content_type}")
    print(f"Table content_type: {table_meta.content_type}")
    
    # Verifica che non ci siano più campi "type" duplicati
    text_dict = text_meta.model_dump()
    image_dict = image_meta.model_dump()
    table_dict = table_meta.model_dump()
    
    if "type" in text_dict:
        print(f"❌ Campo 'type' legacy ancora presente in TextMetadata: {text_dict['type']}")
        return False
    if "type" in image_dict:
        print(f"❌ Campo 'type' legacy ancora presente in ImageMetadata: {image_dict['type']}")
        return False
    if "type" in table_dict:
        print(f"❌ Campo 'type' legacy ancora presente in TableMetadata: {table_dict['type']}")
        return False
        
    print("✅ Nessun campo 'type' legacy trovato - solo content_type!")
    
    print("\n🎉 Tutti i test di unificazione metadati passati!")
    return True

if __name__ == "__main__":
    success = test_unified_metadata()
    sys.exit(0 if success else 1)
