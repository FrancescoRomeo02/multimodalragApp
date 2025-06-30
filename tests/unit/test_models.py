import pytest
from unittest.mock import Mock, patch
import sys
import os

# Setup path per import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from src.core.models import TextElement, ImageElement, TableElement, TextMetadata, ImageMetadata, TableMetadata, TableData


class TestModels:
    """Test per i modelli Pydantic"""
    
    def test_text_element_creation(self):
        """Test creazione TextElement"""
        metadata = TextMetadata(
            source="test.pdf",
            page=1,
            type="text"
        )
        
        element = TextElement(
            text="Testo di esempio",
            metadata=metadata
        )
        
        assert element.text == "Testo di esempio"
        assert element.metadata.source == "test.pdf"
        assert element.metadata.page == 1
        assert element.metadata.content_type == "text"
    
    def test_image_element_creation(self):
        """Test creazione ImageElement"""
        metadata = ImageMetadata(
            source="test.pdf",
            page=1,
            type="image",
            image_caption="Una bella immagine",
            context_text="Contesto dell'immagine"
        )
        
        element = ImageElement(
            page_content="Descrizione immagine",
            image_base64="base64_data_here",
            metadata=metadata
        )
        
        assert element.page_content == "Descrizione immagine"
        assert element.image_base64 == "base64_data_here"
        assert element.metadata.image_caption == "Una bella immagine"
        assert element.metadata.content_type == "image"
    
    def test_table_element_creation(self):
        """Test creazione TableElement"""
        table_data = TableData(
            cells=[["A", "B"], ["1", "2"]],
            headers=["Col1", "Col2"],
            shape=(2, 2)
        )
        
        metadata = TableMetadata(
            source="test.pdf",
            page=1,
            type="table",
            caption="Tabella di esempio"
        )
        
        element = TableElement(
            table_data=table_data,
            table_markdown="| Col1 | Col2 |\n|------|------|\n| A | B |\n| 1 | 2 |",
            metadata=metadata
        )
        
        assert element.table_data.shape == (2, 2)
        assert len(element.table_data.headers) == 2
        assert element.metadata.caption == "Tabella di esempio"
        assert element.metadata.content_type == "table"
    
    def test_metadata_validation(self):
        """Test validazione metadati"""
        # Test metadati validi
        metadata = TextMetadata(
            source="test.pdf",
            page=1,
            type="text"
        )
        assert metadata.source == "test.pdf"
        
        # Test page negativa dovrebbe sollevare errore
        with pytest.raises(Exception):  # Pydantic ValidationError
            TextMetadata(
                source="test.pdf",
                page=-1,  # Page non può essere negativa
                type="text"
            )
    
    def test_table_data_validation(self):
        """Test validazione TableData"""
        # Dati validi
        table_data = TableData(
            cells=[["A", "B"], ["1", "2"]],
            headers=["Col1", "Col2"],
            shape=(2, 2)
        )
        
        assert table_data.cells == [["A", "B"], ["1", "2"]]
        assert table_data.headers == ["Col1", "Col2"]
        assert table_data.shape == (2, 2)
    
    def test_optional_fields(self):
        """Test campi opzionali"""
        # TableMetadata con campi opzionali
        metadata = TableMetadata(
            source="test.pdf",
            page=1,
            type="table"
            # bbox, table_shape, caption, context_text sono opzionali
        )
        
        assert metadata.bbox is None
        assert metadata.caption is None
        assert metadata.context_text is None
        
        # ImageMetadata con campi opzionali
        img_metadata = ImageMetadata(
            source="test.pdf",
            page=1,
            type="image"
        )
        
        assert img_metadata.image_caption is None
        assert img_metadata.context_text is None
        assert img_metadata.manual_caption is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
