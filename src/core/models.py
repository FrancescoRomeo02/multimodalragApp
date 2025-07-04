from enum import Enum
from typing import List, Optional, Dict, Tuple
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
    content_type: str


#TABELLA
class TableMetadata(BaseModel):
    """Metadati essenziali per gli elementi tabella"""
    content_type: str = Field(default="table", description="Tipo di contenuto (sempre 'table')")
    source: str = Field(..., description="Nome del file sorgente")
    page: int = Field(..., description="Numero di pagina della tabella")
    table_id: str = Field(..., description="Identificatore univoco della tabella (es. table_1)")
    table_summary: Optional[str] = Field(None, description="Riassunto AI generato della tabella")

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
    pass  # Eredita tutti i campi necessari da ElementMetadata

class TextElement(BaseModel):
    """Modello per un elemento testuale."""
    text: str
    metadata: TextMetadata

#IMMAGINI
class ImageMetadata(ElementMetadata):
    content_type: str = Field(default="image", description="Tipo di contenuto (sempre 'image')")
    image_id: str = Field(..., description="Identificatore univoco dell'immagine (es. image_1)")
    image_caption: Optional[str] = Field(None, description="Caption generata dall'AI combinata con contesto")

class ImageElement(BaseModel):
    page_content: str  # Può contenere didascalia o testo associato all'immagine
    image_base64: str  # Contenuto immagine codificato base64
    metadata: ImageMetadata


#RISULTATO RETRIEVER
class RetrievalResult(BaseModel):
    answer: str
    source_documents: List[Dict]
    confidence_score:float
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