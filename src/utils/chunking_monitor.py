"""
Utilità per monitorare e analizzare le performance del chunking semantico
"""

import logging
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChunkingPerformanceMonitor:
    """Monitor per analizzare le performance del chunking semantico"""
    
    def __init__(self):
        self.chunking_metrics = []
        
    def record_chunking_session(
        self,
        session_id: str,
        chunk_type: str,
        original_elements: int,
        final_chunks: int,
        processing_time: float,
        avg_chunk_size: float,
        semantic_chunks: int = 0,
        fallback_chunks: int = 0
    ):
        """Registra una sessione di chunking"""
        
        metrics = {
            "session_id": session_id,
            "chunk_type": chunk_type,
            "timestamp": time.time(),
            "original_elements": original_elements,
            "final_chunks": final_chunks,
            "processing_time": processing_time,
            "avg_chunk_size": avg_chunk_size,
            "semantic_chunks": semantic_chunks,
            "fallback_chunks": fallback_chunks,
            "compression_ratio": final_chunks / original_elements if original_elements > 0 else 0,
            "chunks_per_second": final_chunks / processing_time if processing_time > 0 else 0
        }
        
        self.chunking_metrics.append(metrics)
        logger.info(f"Registrata sessione {session_id}: {final_chunks} chunk in {processing_time:.2f}s")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Restituisce un riepilogo delle performance"""
        
        if not self.chunking_metrics:
            return {"error": "Nessuna metrica disponibile"}
        
        df = pd.DataFrame(self.chunking_metrics)
        
        summary = {
            "total_sessions": len(df),
            "avg_processing_time": df["processing_time"].mean(),
            "avg_chunks_per_session": df["final_chunks"].mean(),
            "avg_compression_ratio": df["compression_ratio"].mean(),
            "semantic_chunk_percentage": (df["semantic_chunks"].sum() / df["final_chunks"].sum()) * 100,
            "fallback_chunk_percentage": (df["fallback_chunks"].sum() / df["final_chunks"].sum()) * 100,
            "total_chunks_processed": df["final_chunks"].sum(),
            "fastest_session": df.loc[df["processing_time"].idxmin()]["session_id"],
            "slowest_session": df.loc[df["processing_time"].idxmax()]["session_id"]
        }
        
        return summary
    
    def generate_performance_plots(self, output_dir: str = "performance_plots"):
        """Genera grafici di performance"""
        
        if not self.chunking_metrics:
            logger.warning("Nessuna metrica per generare grafici")
            return
        
        Path(output_dir).mkdir(exist_ok=True)
        df = pd.DataFrame(self.chunking_metrics)
        
        # Configura lo stile
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Grafico 1: Tempo di processing vs numero di chunk
        plt.figure(figsize=(10, 6))
        plt.scatter(df["final_chunks"], df["processing_time"], alpha=0.7)
        plt.xlabel("Numero di Chunk Finali")
        plt.ylabel("Tempo di Processing (s)")
        plt.title("Tempo di Processing vs Numero di Chunk")
        plt.savefig(f"{output_dir}/processing_time_vs_chunks.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Grafico 2: Distribuzione delle dimensioni dei chunk
        plt.figure(figsize=(10, 6))
        plt.hist(df["avg_chunk_size"], bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel("Dimensione Media Chunk (caratteri)")
        plt.ylabel("Frequenza")
        plt.title("Distribuzione Dimensioni Chunk")
        plt.savefig(f"{output_dir}/chunk_size_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Grafico 3: Semantic vs Fallback chunks
        chunk_types = ["Semantic", "Fallback"]
        chunk_counts = [df["semantic_chunks"].sum(), df["fallback_chunks"].sum()]
        
        plt.figure(figsize=(8, 8))
        plt.pie(chunk_counts, labels=chunk_types, autopct='%1.1f%%', startangle=90)
        plt.title("Distribuzione Tipi di Chunk")
        plt.savefig(f"{output_dir}/chunk_type_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Grafico 4: Timeline delle performance
        if len(df) > 1:
            df_sorted = df.sort_values('timestamp')
            plt.figure(figsize=(12, 6))
            plt.plot(range(len(df_sorted)), df_sorted["processing_time"], marker='o')
            plt.xlabel("Sessione")
            plt.ylabel("Tempo di Processing (s)")
            plt.title("Timeline Performance Chunking")
            plt.xticks(range(len(df_sorted)), [f"S{i+1}" for i in range(len(df_sorted))], rotation=45)
            plt.savefig(f"{output_dir}/performance_timeline.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Grafici salvati in {output_dir}")
    
    def compare_chunking_methods(self, semantic_metrics: List[Dict], classic_metrics: List[Dict]):
        """Confronta chunking semantico vs classico"""
        
        if not semantic_metrics or not classic_metrics:
            logger.warning("Dati insufficienti per il confronto")
            return
        
        semantic_df = pd.DataFrame(semantic_metrics)
        classic_df = pd.DataFrame(classic_metrics)
        
        comparison = {
            "semantic": {
                "avg_processing_time": semantic_df["processing_time"].mean(),
                "avg_chunk_count": semantic_df["final_chunks"].mean(),
                "avg_chunk_size": semantic_df["avg_chunk_size"].mean(),
                "compression_ratio": semantic_df["compression_ratio"].mean()
            },
            "classic": {
                "avg_processing_time": classic_df["processing_time"].mean(),
                "avg_chunk_count": classic_df["final_chunks"].mean(),
                "avg_chunk_size": classic_df["avg_chunk_size"].mean(),
                "compression_ratio": classic_df["compression_ratio"].mean()
            }
        }
        
        # Calcola miglioramenti percentuali
        improvements = {}
        for metric in ["avg_processing_time", "avg_chunk_count", "avg_chunk_size", "compression_ratio"]:
            semantic_val = comparison["semantic"][metric]
            classic_val = comparison["classic"][metric]
            
            if classic_val != 0:
                improvement = ((semantic_val - classic_val) / classic_val) * 100
                improvements[metric] = improvement
        
        logger.info("Confronto Chunking Semantico vs Classico:")
        logger.info(f"  Tempo processing: {improvements.get('avg_processing_time', 0):.1f}% di differenza")
        logger.info(f"  Numero chunk: {improvements.get('avg_chunk_count', 0):.1f}% di differenza")
        logger.info(f"  Dimensione chunk: {improvements.get('avg_chunk_size', 0):.1f}% di differenza")
        
        return comparison, improvements
    
    def export_metrics(self, filename: str = "chunking_metrics.csv"):
        """Esporta le metriche in CSV"""
        
        if not self.chunking_metrics:
            logger.warning("Nessuna metrica da esportare")
            return
        
        df = pd.DataFrame(self.chunking_metrics)
        df.to_csv(filename, index=False)
        logger.info(f"Metriche esportate in {filename}")
    
    def clear_metrics(self):
        """Pulisce le metriche registrate"""
        self.chunking_metrics.clear()
        logger.info("Metriche cancellate")


def benchmark_chunking_performance(
    text_elements: List[Dict[str, Any]], 
    chunker,
    iterations: int = 3
) -> Dict[str, Any]:
    """
    Esegue un benchmark delle performance di chunking
    
    Args:
        text_elements: Elementi testuali da processare
        chunker: Istanza del chunker da testare
        iterations: Numero di iterazioni per il benchmark
        
    Returns:
        Dizionario con i risultati del benchmark
    """
    
    logger.info(f"Avvio benchmark chunking con {iterations} iterazioni")
    
    times = []
    chunk_counts = []
    
    for i in range(iterations):
        start_time = time.time()
        
        try:
            chunks = chunker.chunk_text_elements(text_elements)
            processing_time = time.time() - start_time
            
            times.append(processing_time)
            chunk_counts.append(len(chunks))
            
            logger.info(f"Iterazione {i+1}: {len(chunks)} chunk in {processing_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Errore iterazione {i+1}: {e}")
            continue
    
    if not times:
        return {"error": "Tutte le iterazioni sono fallite"}
    
    benchmark_results = {
        "iterations": len(times),
        "avg_processing_time": sum(times) / len(times),
        "min_processing_time": min(times),
        "max_processing_time": max(times),
        "avg_chunk_count": sum(chunk_counts) / len(chunk_counts),
        "min_chunk_count": min(chunk_counts),
        "max_chunk_count": max(chunk_counts),
        "total_input_elements": len(text_elements),
        "chunks_per_second": (sum(chunk_counts) / len(chunk_counts)) / (sum(times) / len(times))
    }
    
    logger.info(f"Benchmark completato: {benchmark_results['avg_chunk_count']:.1f} chunk/iterazione "
               f"in {benchmark_results['avg_processing_time']:.3f}s")
    
    return benchmark_results
