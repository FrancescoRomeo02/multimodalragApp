from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class TextChunk(BaseModel):
    """
    Modello standardizzato per tutti i tipi di contenuto.
    Ora tutto viene convertito in formato testuale con metadati uniformi.
    Include riferimenti ai dati originali in MongoDB.
    """
    text: str = Field(..., description="Contenuto testuale del chunk")
    metadata: Dict[str, Any] = Field(..., description="Metadati standardizzati (almeno source e page)")
    mongo_id: Optional[str] = Field(None, description="ID del documento originale in MongoDB")


#RISULTATO RETRIEVER
class RetrievalResult(BaseModel):
    """
    Risultato di una query di retrieval.
    Ora tutti i document sono nel formato standard {page_content, metadata}.
    """
    answer: str
    source_documents: List[Dict]
    confidence_score: float
    query_time_ms: Optional[int] = Field(None, description="Tempo di esecuzione query in millisecondi")
    retrieved_count: Optional[int] = Field(None, description="Numero di documenti recuperati")
    filters_applied: Optional[Dict] = Field(None, description="Filtri applicati alla ricerca")