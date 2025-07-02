from enum import Enum
from typing import List, Optional, Dict, Tuple, Any
from pydantic import BaseModel, Field


class ImageResult(BaseModel):
    image_base64: str
    metadata: dict
    score: float
    page_content: str

class TableResult(BaseModel):
    table_markdown: str
    metadata: dict
    score: float


#METADATI COMUNI
class ElementMetadata(BaseModel):
    """Metadati comuni a tutti gli elementi, validati da Pydantic."""
    source: str
    page: int
    type: str

class QdrantPayload(BaseModel):
    """Payload unificato per Qdrant - solo metadati essenziali in JSON"""
    metadata: Dict[str, Any] = Field(..., description="Metadati essenziali in formato JSON")
    
    @classmethod
    def create_text_payload(cls, source: str, page: int, **extra_fields) -> "QdrantPayload":
        """Crea payload per elementi di testo"""
        metadata = {
            "type": "text",
            "source": source,
            "page": page,
            **{k: v for k, v in extra_fields.items() if v is not None}
        }
        return cls(metadata=metadata)
    
    @classmethod
    def create_table_payload(cls, source: str, page: int, table_shape: Optional[Tuple[int, int]] = None, 
                           caption: Optional[str] = None, **extra_fields) -> "QdrantPayload":
        """Crea payload per elementi tabella"""
        metadata = {
            "type": "table",
            "source": source,
            "page": page
        }
        if table_shape:
            metadata["table_shape"] = {"rows": table_shape[0], "cols": table_shape[1]}
        if caption:
            metadata["caption"] = caption
        
        # Aggiungi altri campi opzionali
        metadata.update({k: v for k, v in extra_fields.items() if v is not None})
        return cls(metadata=metadata)
    
    @classmethod  
    def create_image_payload(cls, source: str, page: int, image_caption: Optional[str] = None,
                           manual_caption: Optional[str] = None, **extra_fields) -> "QdrantPayload":
        """Crea payload per elementi immagine"""
        metadata = {
            "type": "image",
            "source": source,
            "page": page
        }
        if image_caption:
            metadata["image_caption"] = image_caption
        if manual_caption:
            metadata["manual_caption"] = manual_caption
            
        # Aggiungi altri campi opzionali
        metadata.update({k: v for k, v in extra_fields.items() if v is not None})
        return cls(metadata=metadata)
    
    @classmethod
    def from_text_element(cls, text_element: "TextElement") -> "QdrantPayload":
        """Converte un TextElement in payload Qdrant con metadati essenziali"""
        return cls.create_text_payload(
            source=text_element.metadata.source,
            page=text_element.metadata.page
        )
    
    @classmethod
    def from_table_element(cls, table_element: "TableElement") -> "QdrantPayload":
        """Converte un TableElement in payload Qdrant con metadati essenziali"""
        return cls.create_table_payload(
            source=table_element.metadata.source,
            page=table_element.metadata.page,
            table_shape=table_element.metadata.table_shape,
            caption=table_element.metadata.caption
        )
    
    @classmethod
    def from_image_element(cls, image_element: "ImageElement") -> "QdrantPayload":
        """Converte un ImageElement in payload Qdrant con metadati essenziali"""
        return cls.create_image_payload(
            source=image_element.metadata.source,
            page=image_element.metadata.page,
            image_caption=image_element.metadata.image_caption,
            manual_caption=image_element.metadata.manual_caption
        )


#TABELLA
class TableMetadata(BaseModel):
    """Metadati standard per gli elementi tabella"""
    type: str = Field(default="table", description="Tipo di elemento (sempre 'table')")
    source: str = Field(..., description="Nome del file sorgente")
    page: int = Field(..., description="Numero di pagina della tabella")
    content_type: str = Field(default="table", description="Tipo di contenuto (sempre 'table')")
    bbox: Optional[List[float]] = Field(None, description="Bounding box della tabella [x0, y0, x1, y1]")
    table_shape: Optional[Tuple[int, int]] = Field(None, description="Forma della tabella (righe, colonne)")
    caption: Optional[str] = Field(None, description="Caption o testo circostante della tabella")
    context_text: Optional[str] = Field(None, description="Testo di contesto prima e dopo la tabella")

class TableData(BaseModel):
    """Modello per i dati strutturati della tabella"""
    cells: List[List[Optional[str]]]
    headers: List[str]
    shape: Tuple[int, int]

class TableElement(BaseModel):
    """Modello per gli elementi tabella estratti dai PDF"""
    table_data: TableData = Field(..., description="Dati strutturati della tabella")
    table_markdown: str = Field(..., description="Rappresentazione markdown della tabella")
    metadata: TableMetadata = Field(..., description="Metadati standardizzati della tabella")

#TESTO
class TextMetadata(ElementMetadata):
    content_type: str = Field(default="text", description="Tipo di contenuto (sempre 'text')")
    page: int

class TextElement(BaseModel):
    """Modello per un elemento testuale."""
    text: str
    metadata: TextMetadata

#IMMAGINI
class ImageMetadata(ElementMetadata):
    #content_type: str = Field(default="image", description="Tipo di contenuto (sempre 'image')")
    image_caption: Optional[str] = Field(None, description="Caption generata automaticamente dall'immagine")
    context_text: Optional[str] = Field(None, description="Testo di contesto prima e dopo l'immagine")
    manual_caption: Optional[str] = Field(None, description="Caption estratta dal PDF (se presente)")

class ImageElement(BaseModel):
    page_content: str  # Può contenere didascalia o testo associato all'immagine
    image_base64: str  # Contenuto immagine codificato base64
    metadata: ImageMetadata


#RISULTATO RETRIEVER
class RetrievalResult(BaseModel):
    answer: str
    source_documents: List[Dict]
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Score di confidenza della risposta")
    query_time_ms: Optional[int] = Field(None, description="Tempo di esecuzione query in millisecondi")
    retrieved_count: Optional[int] = Field(None, description="Numero di documenti recuperati")
    filters_applied: Optional[Dict] = Field(None, description="Filtri applicati alla ricerca")

class ColorSpace(Enum):
    GRAY = 1
    RGB = 2
    CMYK = 3
    UNKNOWN = 4

    @classmethod
    def from_fitz(cls, cs_num: int):
        try:
            return cls(cs_num).name.lower()
        except ValueError:
            return cls.UNKNOWN.name.lower()