"""
Sistema di metriche e monitoring delle performance per MultimodalRAG
"""
import time
import psutil
import logging
from typing import Dict, Any, List, Optional
from functools import wraps
from dataclasses import dataclass, asdict
from datetime import datetime
import json

logger = logging.getLogger(__name__)

@dataclass
class QueryMetrics:
    """Metriche per una singola query"""
    query_id: str
    query_text: str
    query_type: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[int] = None
    documents_retrieved: int = 0
    confidence_score: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converte le metriche in dizionario per logging/storage"""
        return asdict(self)

class PerformanceMonitor:
    """Monitor delle performance del sistema RAG"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.metrics_history: List[QueryMetrics] = []
        self.start_time = datetime.now()
        
    def start_query_tracking(self, query_id: str, query_text: str, query_type: str) -> QueryMetrics:
        """Inizia il tracking di una query"""
        if not self.enabled:
            return None
            
        metrics = QueryMetrics(
            query_id=query_id,
            query_text=query_text[:100],  # Limita lunghezza per privacy
            query_type=query_type,
            start_time=datetime.now(),
            memory_usage_mb=psutil.Process().memory_info().rss / 1024 / 1024,
            cpu_usage_percent=psutil.cpu_percent()
        )
        
        return metrics
    
    def end_query_tracking(self, metrics: QueryMetrics, 
                          documents_retrieved: int = 0,
                          confidence_score: float = 0.0,
                          success: bool = True,
                          error_message: Optional[str] = None) -> None:
        """Termina il tracking di una query"""
        if not self.enabled or not metrics:
            return
            
        metrics.end_time = datetime.now()
        metrics.duration_ms = int((metrics.end_time - metrics.start_time).total_seconds() * 1000)
        metrics.documents_retrieved = documents_retrieved
        metrics.confidence_score = confidence_score
        metrics.success = success
        metrics.error_message = error_message
        
        self.metrics_history.append(metrics)
        
        # Log delle metriche
        logger.info(f"Query completed: {metrics.query_id} - "
                   f"Duration: {metrics.duration_ms}ms - "
                   f"Success: {metrics.success} - "
                   f"Docs: {metrics.documents_retrieved}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Restituisce un riassunto delle performance"""
        if not self.metrics_history:
            return {"message": "Nessuna metrica disponibile"}
        
        successful_queries = [m for m in self.metrics_history if m.success]
        failed_queries = [m for m in self.metrics_history if not m.success]
        
        durations = [m.duration_ms for m in successful_queries if m.duration_ms]
        confidence_scores = [m.confidence_score for m in successful_queries]
        
        return {
            "uptime_hours": (datetime.now() - self.start_time).total_seconds() / 3600,
            "total_queries": len(self.metrics_history),
            "successful_queries": len(successful_queries),
            "failed_queries": len(failed_queries),
            "success_rate": len(successful_queries) / len(self.metrics_history) if self.metrics_history else 0,
            "avg_response_time_ms": sum(durations) / len(durations) if durations else 0,
            "min_response_time_ms": min(durations) if durations else 0,
            "max_response_time_ms": max(durations) if durations else 0,
            "avg_confidence_score": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
            "current_memory_mb": psutil.Process().memory_info().rss / 1024 / 1024,
            "current_cpu_percent": psutil.cpu_percent(),
            "query_types": self._get_query_type_stats()
        }
    
    def _get_query_type_stats(self) -> Dict[str, int]:
        """Statistiche sui tipi di query"""
        stats = {}
        for metrics in self.metrics_history:
            query_type = metrics.query_type
            stats[query_type] = stats.get(query_type, 0) + 1
        return stats
    
    def export_metrics(self, filepath: str) -> None:
        """Esporta le metriche in un file JSON"""
        data = {
            "summary": self.get_performance_summary(),
            "detailed_metrics": [m.to_dict() for m in self.metrics_history]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

# Singleton per l'uso globale
performance_monitor = PerformanceMonitor()

def track_performance(query_type: str = "unknown"):
    """Decorator per tracciare automaticamente le performance di una funzione"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not performance_monitor.enabled:
                return func(*args, **kwargs)
            
            import uuid
            query_id = str(uuid.uuid4())[:8]
            
            # Prova a estrarre il testo della query dai parametri
            query_text = ""
            if args and isinstance(args[0], str):
                query_text = args[0]
            elif "query" in kwargs:
                query_text = kwargs["query"]
            
            metrics = performance_monitor.start_query_tracking(
                query_id=query_id,
                query_text=query_text,
                query_type=query_type
            )
            
            try:
                result = func(*args, **kwargs)
                
                # Prova a estrarre metriche dal risultato
                docs_count = 0
                confidence = 0.0
                
                if hasattr(result, 'source_documents'):
                    docs_count = len(result.source_documents)
                if hasattr(result, 'confidence_score'):
                    confidence = result.confidence_score
                
                performance_monitor.end_query_tracking(
                    metrics=metrics,
                    documents_retrieved=docs_count,
                    confidence_score=confidence,
                    success=True
                )
                
                return result
                
            except Exception as e:
                performance_monitor.end_query_tracking(
                    metrics=metrics,
                    success=False,
                    error_message=str(e)
                )
                raise
        
        return wrapper
    return decorator
