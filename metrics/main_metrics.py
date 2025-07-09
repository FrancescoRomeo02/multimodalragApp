"""
Sistema di metriche per MultimodalRAG
Implementa metriche di retrieval, generazione, multimodali, costi ed efficienza
"""

import time
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from collections import defaultdict, Counter

# Try to import BERT score dependencies
try:
    from bert_score import score as bert_score_func
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False
    print("Warning: bert_score not available. Install with: pip install bert_score")

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    """Metriche di retrieval"""
    query_id: str
    relevant_docs: List[str]  # IDs dei documenti rilevanti (ground truth)
    retrieved_docs: List[str]  # IDs dei documenti recuperati
    retrieval_time: float
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def recall_at_k(self, k: int) -> float:
        """Calcola Recall@k"""
        if not self.relevant_docs:
            return 0.0
        
        retrieved_k = set(self.retrieved_docs[:k])
        relevant_set = set(self.relevant_docs)
        
        intersection = len(retrieved_k & relevant_set)
        return intersection / len(relevant_set)

    def precision_at_k(self, k: int) -> float:
        """Calcola Precision@k"""
        if k == 0:
            return 0.0
        
        retrieved_k = set(self.retrieved_docs[:k])
        relevant_set = set(self.relevant_docs)
        
        intersection = len(retrieved_k & relevant_set)
        return intersection / k

    def mean_reciprocal_rank(self) -> float:
        """Calcola Mean Reciprocal Rank (MRR)"""
        relevant_set = set(self.relevant_docs)
        
        for rank, doc_id in enumerate(self.retrieved_docs, 1):
            if doc_id in relevant_set:
                return 1.0 / rank
        
        return 0.0


@dataclass
class GenerationMetrics:
    """Metriche di generazione"""
    query_id: str
    generated_text: str
    reference_text: str
    generation_time: float
    token_count: int
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def exact_match(self) -> bool:
        """Calcola Exact Match (EM)"""
        return self.generated_text.strip().lower() == self.reference_text.strip().lower()

    def bert_score(self) -> Dict[str, float]:
        """Calcola BERTScore"""
        if not BERT_SCORE_AVAILABLE:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        try:
            from bert_score import score as bert_score_func
            P, R, F1 = bert_score_func([self.generated_text], [self.reference_text], lang="it")
            return {
                "precision": float(P[0]),
                "recall": float(R[0]), 
                "f1": float(F1[0])
            }
        except Exception as e:
            logger.error(f"Error calculating BERT score: {e}")
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}


@dataclass
class MultimodalMetrics:
    """Metriche multimodali"""
    query_id: str
    image_retrieved: bool
    text_retrieved: bool
    image_relevant: bool
    text_relevant: bool
    multimodal_accuracy: float
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def image_text_accuracy(self) -> float:
        """Calcola l'accuratezza combinata di immagini e testo"""
        # Accuratezza basata su se entrambi i tipi sono stati recuperati correttamente
        image_correct = (self.image_retrieved and self.image_relevant) or (not self.image_retrieved and not self.image_relevant)
        text_correct = (self.text_retrieved and self.text_relevant) or (not self.text_retrieved and not self.text_relevant)
        
        return float(image_correct and text_correct)


@dataclass  
class CostEfficiencyMetrics:
    """Metriche di costo ed efficienza"""
    query_id: str
    total_tokens: int
    embedding_tokens: int
    generation_tokens: int
    response_time: float
    embedding_time: float
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def tokens_per_second(self) -> float:
        """Calcola token per secondo per la generazione"""
        if self.response_time == 0:
            return 0.0
        return self.generation_tokens / self.response_time

    def embedding_tokens_per_second(self) -> float:
        """Calcola token per secondo per gli embedding"""
        if self.embedding_time == 0:
            return 0.0
        return self.embedding_tokens / self.embedding_time


class MetricsCollector:
    """Collettore centrale per tutte le metriche"""
    
    def __init__(self, storage_path: str = "metrics_data"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Storage per le metriche
        self.retrieval_metrics: List[RetrievalMetrics] = []
        self.generation_metrics: List[GenerationMetrics] = []
        self.multimodal_metrics: List[MultimodalMetrics] = []
        self.cost_efficiency_metrics: List[CostEfficiencyMetrics] = []
        
        logger.info(f"MetricsCollector initialized with storage path: {self.storage_path}")

    def add_retrieval_metrics(self, metrics: RetrievalMetrics) -> None:
        """Aggiunge metriche di retrieval"""
        self.retrieval_metrics.append(metrics)
        logger.debug(f"Added retrieval metrics for query: {metrics.query_id}")

    def add_generation_metrics(self, metrics: GenerationMetrics) -> None:
        """Aggiunge metriche di generazione"""
        self.generation_metrics.append(metrics)
        logger.debug(f"Added generation metrics for query: {metrics.query_id}")

    def add_multimodal_metrics(self, metrics: MultimodalMetrics) -> None:
        """Aggiunge metriche multimodali"""
        self.multimodal_metrics.append(metrics)
        logger.debug(f"Added multimodal metrics for query: {metrics.query_id}")

    def add_cost_efficiency_metrics(self, metrics: CostEfficiencyMetrics) -> None:
        """Aggiunge metriche di costo ed efficienza"""
        self.cost_efficiency_metrics.append(metrics)
        logger.debug(f"Added cost/efficiency metrics for query: {metrics.query_id}")

    def get_retrieval_summary(self, k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
        """Restituisce un riassunto delle metriche di retrieval"""
        if not self.retrieval_metrics:
            return {"error": "No retrieval metrics available"}

        summary = {
            "total_queries": len(self.retrieval_metrics),
            "avg_retrieval_time": np.mean([m.retrieval_time for m in self.retrieval_metrics]),
            "recall_at_k": {},
            "precision_at_k": {},
            "mean_reciprocal_rank": np.mean([m.mean_reciprocal_rank() for m in self.retrieval_metrics])
        }

        for k in k_values:
            recalls = [m.recall_at_k(k) for m in self.retrieval_metrics]
            precisions = [m.precision_at_k(k) for m in self.retrieval_metrics]
            
            summary["recall_at_k"][f"recall@{k}"] = np.mean(recalls)
            summary["precision_at_k"][f"precision@{k}"] = np.mean(precisions)

        return summary

    def get_generation_summary(self) -> Dict[str, Any]:
        """Restituisce un riassunto delle metriche di generazione"""
        if not self.generation_metrics:
            return {"error": "No generation metrics available"}

        bert_scores = [m.bert_score() for m in self.generation_metrics]
        exact_matches = [m.exact_match() for m in self.generation_metrics]

        summary = {
            "total_generations": len(self.generation_metrics),
            "avg_generation_time": np.mean([m.generation_time for m in self.generation_metrics]),
            "avg_token_count": np.mean([m.token_count for m in self.generation_metrics]),
            "exact_match_rate": np.mean(exact_matches),
            "bert_score": {
                "avg_precision": np.mean([bs["precision"] for bs in bert_scores]),
                "avg_recall": np.mean([bs["recall"] for bs in bert_scores]),
                "avg_f1": np.mean([bs["f1"] for bs in bert_scores])
            }
        }

        return summary

    def get_multimodal_summary(self) -> Dict[str, Any]:
        """Restituisce un riassunto delle metriche multimodali"""
        if not self.multimodal_metrics:
            return {"error": "No multimodal metrics available"}

        image_text_accuracies = [m.image_text_accuracy() for m in self.multimodal_metrics]

        summary = {
            "total_multimodal_queries": len(self.multimodal_metrics),
            "avg_image_text_accuracy": np.mean(image_text_accuracies),
            "image_retrieval_rate": np.mean([m.image_retrieved for m in self.multimodal_metrics]),
            "text_retrieval_rate": np.mean([m.text_retrieved for m in self.multimodal_metrics]),
            "image_relevance_rate": np.mean([m.image_relevant for m in self.multimodal_metrics]),
            "text_relevance_rate": np.mean([m.text_relevant for m in self.multimodal_metrics])
        }

        return summary

    def get_cost_efficiency_summary(self) -> Dict[str, Any]:
        """Restituisce un riassunto delle metriche di costo ed efficienza"""
        if not self.cost_efficiency_metrics:
            return {"error": "No cost/efficiency metrics available"}

        summary = {
            "total_queries": len(self.cost_efficiency_metrics),
            "total_tokens": sum([m.total_tokens for m in self.cost_efficiency_metrics]),
            "avg_tokens_per_query": np.mean([m.total_tokens for m in self.cost_efficiency_metrics]),
            "avg_embedding_tokens": np.mean([m.embedding_tokens for m in self.cost_efficiency_metrics]),
            "avg_generation_tokens": np.mean([m.generation_tokens for m in self.cost_efficiency_metrics]),
            "avg_response_time": np.mean([m.response_time for m in self.cost_efficiency_metrics]),
            "avg_embedding_time": np.mean([m.embedding_time for m in self.cost_efficiency_metrics]),
            "avg_tokens_per_second": np.mean([m.tokens_per_second() for m in self.cost_efficiency_metrics]),
            "avg_embedding_tokens_per_second": np.mean([m.embedding_tokens_per_second() for m in self.cost_efficiency_metrics])
        }

        return summary

    def get_complete_summary(self) -> Dict[str, Any]:
        """Restituisce un riassunto completo di tutte le metriche"""
        return {
            "timestamp": datetime.now().isoformat(),
            "retrieval_metrics": self.get_retrieval_summary(),
            "generation_metrics": self.get_generation_summary(),
            "multimodal_metrics": self.get_multimodal_summary(),
            "cost_efficiency_metrics": self.get_cost_efficiency_summary()
        }

    def export_metrics(self, filename: Optional[str] = None) -> str:
        """Esporta tutte le metriche in formato JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_export_{timestamp}.json"
        
        filepath = self.storage_path / filename
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "summary": self.get_complete_summary(),
            "raw_data": {
                "retrieval": [asdict(m) for m in self.retrieval_metrics],
                "generation": [asdict(m) for m in self.generation_metrics],
                "multimodal": [asdict(m) for m in self.multimodal_metrics],
                "cost_efficiency": [asdict(m) for m in self.cost_efficiency_metrics]
            }
        }
        
        # Convert datetime objects to ISO format strings
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_datetime(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetime(item) for item in obj]
            return obj
        
        export_data = convert_datetime(export_data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Metrics exported to: {filepath}")
        return str(filepath)

    def clear_metrics(self) -> None:
        """Pulisce tutte le metriche memorizzate"""
        self.retrieval_metrics.clear()
        self.generation_metrics.clear()
        self.multimodal_metrics.clear()
        self.cost_efficiency_metrics.clear()
        logger.info("All metrics cleared")

    def save_session(self, session_name: str) -> str:
        """Salva la sessione corrente di metriche"""
        session_file = f"session_{session_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        return self.export_metrics(session_file)


# Singleton globale per l'uso nell'applicazione
global_metrics_collector = MetricsCollector()


def track_retrieval(query_id: str, relevant_docs: List[str], retrieved_docs: List[str], 
                   retrieval_time: float) -> RetrievalMetrics:
    """Utility function per tracciare metriche di retrieval"""
    metrics = RetrievalMetrics(
        query_id=query_id,
        relevant_docs=relevant_docs,
        retrieved_docs=retrieved_docs,
        retrieval_time=retrieval_time
    )
    global_metrics_collector.add_retrieval_metrics(metrics)
    return metrics


def track_generation(query_id: str, generated_text: str, reference_text: str,
                    generation_time: float, token_count: int) -> GenerationMetrics:
    """Utility function per tracciare metriche di generazione"""
    metrics = GenerationMetrics(
        query_id=query_id,
        generated_text=generated_text,
        reference_text=reference_text,
        generation_time=generation_time,
        token_count=token_count
    )
    global_metrics_collector.add_generation_metrics(metrics)
    return metrics


def track_multimodal(query_id: str, image_retrieved: bool, text_retrieved: bool,
                    image_relevant: bool, text_relevant: bool, 
                    multimodal_accuracy: float) -> MultimodalMetrics:
    """Utility function per tracciare metriche multimodali"""
    metrics = MultimodalMetrics(
        query_id=query_id,
        image_retrieved=image_retrieved,
        text_retrieved=text_retrieved,
        image_relevant=image_relevant,
        text_relevant=text_relevant,
        multimodal_accuracy=multimodal_accuracy
    )
    global_metrics_collector.add_multimodal_metrics(metrics)
    return metrics


def track_cost_efficiency(query_id: str, total_tokens: int, embedding_tokens: int,
                         generation_tokens: int, response_time: float, 
                         embedding_time: float) -> CostEfficiencyMetrics:
    """Utility function per tracciare metriche di costo ed efficienza"""
    metrics = CostEfficiencyMetrics(
        query_id=query_id,
        total_tokens=total_tokens,
        embedding_tokens=embedding_tokens,
        generation_tokens=generation_tokens,
        response_time=response_time,
        embedding_time=embedding_time
    )
    global_metrics_collector.add_cost_efficiency_metrics(metrics)
    return metrics
