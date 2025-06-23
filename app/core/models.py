# app/core/models.py
from typing import List, Optional, Dict
from pydantic import BaseModel

class ImageResult(BaseModel):
    image_base64: str
    metadata: dict
    score: float
    page_content: str

class RetrievalResult(BaseModel):
    answer: str
    source_documents: List[Dict]
    images: Optional[List[ImageResult]] = None
    confidence_score: Optional[float] = None