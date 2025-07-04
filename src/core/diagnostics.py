from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_retrieval_quality(retrieved_docs: List, min_docs: int = 1, min_score: float = 0.6) -> float:
    """
    Valida la qualità del retrieval
    """
    if not retrieved_docs or len(retrieved_docs) < min_docs:
        return 0.0
    
    # Calcola score medio (se disponibile)
    scores = []
    for doc in retrieved_docs:
        if hasattr(doc, 'metadata') and 'score' in doc.metadata:
            scores.append(doc.metadata['score'])
    
    avg_score = sum(scores) / len(scores) if scores else 0.5
    is_quality = avg_score >= min_score
    
    return avg_score
