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
    
class RetrievalResult(BaseModel):
    answer: str
    source_documents: List[Dict]
    images: Optional[List[ImageResult]] = None
    confidence_score: Optional[float] = None

class ElementMetadata(BaseModel):
    """Metadati comuni a tutti gli elementi, validati da Pydantic."""
    source: str
    page: int
    type: str

class TableMetadata(BaseModel):
    """Metadati standard per gli elementi tabella"""
    type: str = Field(default="table", description="Tipo di elemento (sempre 'table')")
    source: str = Field(..., description="Nome del file sorgente")
    page: int = Field(..., description="Numero di pagina della tabella")
    content_type: str = Field(default="table", description="Tipo di contenuto (sempre 'table')")
    bbox: Optional[List[float]] = Field(None, description="Bounding box della tabella [x0, y0, x1, y1]")
    table_shape: Optional[Tuple[int, int]] = Field(None, description="Forma della tabella (righe, colonne)")

class TableData(BaseModel):
    """Modello per i dati strutturati della tabella"""
    cells: List[List[str]]
    headers: List[str]
    shape: Tuple[int, int]

class TableElement(BaseModel):
    """Modello per gli elementi tabella estratti dai PDF"""
    table_data: TableData = Field(..., description="Dati strutturati della tabella")
    table_markdown: str = Field(..., description="Rappresentazione markdown della tabella")
    metadata: TableMetadata = Field(..., description="Metadati standardizzati della tabella")


class TextElement(BaseModel):
    """Modello per un elemento testuale."""
    text: str
    metadata: ElementMetadata

class ImageElement(BaseModel):
    """Modello per un elemento immagine."""
    page_content: str
    image_base64: str
    metadata: ElementMetadata

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