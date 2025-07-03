"""
Modulo per il chunking semantico dei documenti.
Preserva i metadati e gestisce testo, immagini e tabelle in modo intelligente.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
import re

from src.config import DEFAULT_EMBEDDING_MODEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalSemanticChunker:
    """
    Chunker semantico avanzato per documenti multimodali.
    
    Gestisce:
    - Chunking semantico del testo preservando i metadati
    - Associazione intelligente di immagini e tabelle ai chunk di testo
    - Mantenimento del contesto e delle relazioni semantiche
    """
    
    def __init__(
        self,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        semantic_threshold: float = 0.75,
        min_chunk_size: int = 100
    ):
        """
        Inizializza il chunker semantico.
        
        Args:
            embedding_model: Modello per gli embeddings
            chunk_size: Dimensione target dei chunk
            chunk_overlap: Sovrapposizione tra chunk
            semantic_threshold: Soglia di similarità semantica
            min_chunk_size: Dimensione minima dei chunk
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.semantic_threshold = semantic_threshold
        self.min_chunk_size = min_chunk_size
        
        # Inizializza il modello di embedding
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={'device': 'cpu'}
            )
            self.sentence_model = SentenceTransformer(embedding_model)
            logger.info(f"Inizializzato modello embedding: {embedding_model}")
        except Exception as e:
            logger.error(f"Errore inizializzazione embedding model: {e}")
            raise
            
        # Inizializza il text splitter di fallback
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Inizializza il semantic chunker di LangChain
        try:
            self.semantic_splitter = SemanticChunker(
                embeddings=self.embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=80
            )
            logger.info("Inizializzato SemanticChunker di LangChain")
        except Exception as e:
            logger.warning(f"Fallback a chunking classico: {e}")
            self.semantic_splitter = None

    def chunk_text_elements(self, text_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Applica chunking semantico agli elementi testuali.
        
        Args:
            text_elements: Lista di elementi testuali da chunkare
            
        Returns:
            Lista di chunk semantici con metadati preservati
        """
        logger.info(f"Inizio chunking semantico di {len(text_elements)} elementi testuali")
        
        chunked_elements = []
        
        for element in text_elements:
            try:
                text = element.get("text", "")
                metadata = element.get("metadata", {})
                
                if not text or len(text.strip()) < self.min_chunk_size:
                    continue
                
                # Applica chunking semantico
                chunks = self._semantic_chunk_text(text)
                
                # Crea elementi chunk con metadati preservati
                for i, chunk_text in enumerate(chunks):
                    if len(chunk_text.strip()) >= self.min_chunk_size:
                        chunk_metadata = metadata.copy()
                        chunk_metadata.update({
                            "chunk_id": f"{metadata.get('page', 0)}_{i}",
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "chunk_type": "semantic",
                            "original_length": len(text),
                            "chunk_length": len(chunk_text)
                        })
                        
                        chunked_elements.append({
                            "text": chunk_text,
                            "metadata": chunk_metadata
                        })
                        
            except Exception as e:
                logger.error(f"Errore chunking elemento: {e}")
                # Fallback a chunking classico
                fallback_chunks = self.fallback_splitter.split_text(element.get("text", ""))
                metadata = element.get("metadata", {})
                for i, chunk_text in enumerate(fallback_chunks):
                    if len(chunk_text.strip()) >= self.min_chunk_size:
                        chunk_metadata = metadata.copy()
                        chunk_metadata.update({
                            "chunk_id": f"{metadata.get('page', 0)}_fallback_{i}",
                            "chunk_index": i,
                            "chunk_type": "fallback",
                            "chunk_length": len(chunk_text)
                        })
                        
                        chunked_elements.append({
                            "text": chunk_text,
                            "metadata": chunk_metadata
                        })
        
        logger.info(f"Generati {len(chunked_elements)} chunk semantici")
        return chunked_elements

    def _semantic_chunk_text(self, text: str) -> List[str]:
        """
        Applica chunking semantico a un testo.
        
        Args:
            text: Testo da chunkare
            
        Returns:
            Lista di chunk semantici
        """
        try:
            # Usa SemanticChunker di LangChain se disponibile
            if self.semantic_splitter:
                docs = self.semantic_splitter.split_text(text)
                return [doc for doc in docs if len(doc.strip()) >= self.min_chunk_size]
            
            # Fallback a chunking custom semantico
            return self._custom_semantic_chunk(text)
            
        except Exception as e:
            logger.warning(f"Fallback a chunking classico: {e}")
            return self.fallback_splitter.split_text(text)

    def _custom_semantic_chunk(self, text: str) -> List[str]:
        """
        Implementazione custom del chunking semantico.
        
        Args:
            text: Testo da chunkare
            
        Returns:
            Lista di chunk semantici
        """
        # Dividi in frasi
        sentences = self._split_into_sentences(text)
        
        if len(sentences) <= 2:
            return [text]
        
        # Calcola embeddings delle frasi
        try:
            sentence_embeddings = self.sentence_model.encode(sentences)
        except Exception as e:
            logger.warning(f"Errore calcolo embeddings: {e}")
            return self.fallback_splitter.split_text(text)
        
        # Calcola similarità tra frasi consecutive
        similarities = []
        for i in range(len(sentence_embeddings) - 1):
            sim = np.dot(sentence_embeddings[i], sentence_embeddings[i + 1]) / (
                np.linalg.norm(sentence_embeddings[i]) * np.linalg.norm(sentence_embeddings[i + 1])
            )
            similarities.append(sim)
        
        # Trova punti di split basati sulla similarità
        split_points = [0]
        current_chunk_size = 0
        
        for i, sim in enumerate(similarities):
            current_chunk_size += len(sentences[i])
            
            # Split se:
            # 1. Similarità sotto soglia E chunk abbastanza grande
            # 2. Chunk troppo grande
            if ((sim < self.semantic_threshold and current_chunk_size >= self.min_chunk_size) or 
                current_chunk_size >= self.chunk_size):
                split_points.append(i + 1)
                current_chunk_size = 0
        
        split_points.append(len(sentences))
        
        # Crea chunk
        chunks = []
        for i in range(len(split_points) - 1):
            start_idx = split_points[i]
            end_idx = split_points[i + 1]
            chunk = " ".join(sentences[start_idx:end_idx])
            
            if len(chunk.strip()) >= self.min_chunk_size:
                chunks.append(chunk)
        
        return chunks if chunks else [text]

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Divide il testo in frasi usando regex avanzate.
        
        Args:
            text: Testo da dividere
            
        Returns:
            Lista di frasi
        """
        # Pattern per dividere in frasi
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        
        # Pulisci e filtra frasi vuote
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences

    def associate_media_to_chunks(
        self,
        text_chunks: List[Dict[str, Any]],
        image_elements: List[Dict[str, Any]],
        table_elements: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Associa elementi multimediali ai chunk di testo più rilevanti.
        
        Args:
            text_chunks: Chunk di testo
            image_elements: Elementi immagine
            table_elements: Elementi tabella
            
        Returns:
            Tuple di (text_chunks_enriched, image_elements_enriched, table_elements_enriched)
        """
        logger.info("Associazione elementi multimediali ai chunk semantici")
        
        # Arricchisci text chunks con riferimenti a media
        enriched_text_chunks = []
        for chunk in text_chunks:
            chunk_copy = chunk.copy()
            
            # Trova immagini e tabelle della stessa pagina
            page_num = chunk.get("metadata", {}).get("page", 0)
            
            related_images = [
                img for img in image_elements 
                if img.get("metadata", {}).get("page") == page_num
            ]
            
            related_tables = [
                table for table in table_elements 
                if table.get("metadata", {}).get("page") == page_num
            ]
            
            # Aggiungi riferimenti ai metadati
            chunk_copy["metadata"]["related_images"] = len(related_images)
            chunk_copy["metadata"]["related_tables"] = len(related_tables)
            
            enriched_text_chunks.append(chunk_copy)
        
        # Arricchisci elementi media con riferimenti ai chunk
        enriched_images = []
        for img in image_elements:
            img_copy = img.copy()
            page_num = img.get("metadata", {}).get("page", 0)
            
            # Trova chunk della stessa pagina
            related_chunks = [
                chunk for chunk in text_chunks
                if chunk.get("metadata", {}).get("page") == page_num
            ]
            
            img_copy["metadata"]["related_text_chunks"] = len(related_chunks)
            enriched_images.append(img_copy)
        
        enriched_tables = []
        for table in table_elements:
            table_copy = table.copy()
            page_num = table.get("metadata", {}).get("page", 0)
            
            # Trova chunk della stessa pagina
            related_chunks = [
                chunk for chunk in text_chunks
                if chunk.get("metadata", {}).get("page") == page_num
            ]
            
            table_copy["metadata"]["related_text_chunks"] = len(related_chunks)
            enriched_tables.append(table_copy)
        
        logger.info(f"Associazione completata: {len(enriched_text_chunks)} chunk, "
                   f"{len(enriched_images)} immagini, {len(enriched_tables)} tabelle")
        
        return enriched_text_chunks, enriched_images, enriched_tables

    def get_chunking_stats(self, elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calcola statistiche sui chunk generati.
        
        Args:
            elements: Lista di elementi chunkati
            
        Returns:
            Dizionario con statistiche
        """
        if not elements:
            return {
                "total_chunks": 0,
                "avg_chunk_length": 0,
                "min_chunk_length": 0,
                "max_chunk_length": 0,
                "median_chunk_length": 0,
                "semantic_chunks": 0,
                "fallback_chunks": 0
            }
        
        chunk_lengths = [len(el.get("text", "")) for el in elements]
        
        # Gestione caso con chunk_lengths vuoti o solo con valori 0
        if not chunk_lengths or all(length == 0 for length in chunk_lengths):
            return {
                "total_chunks": len(elements),
                "avg_chunk_length": 0,
                "min_chunk_length": 0,
                "max_chunk_length": 0,
                "median_chunk_length": 0,
                "semantic_chunks": len([el for el in elements 
                                      if el.get("metadata", {}).get("chunk_type") == "semantic"]),
                "fallback_chunks": len([el for el in elements 
                                      if el.get("metadata", {}).get("chunk_type") == "fallback"])
            }
        
        return {
            "total_chunks": len(elements),
            "avg_chunk_length": np.mean(chunk_lengths),
            "min_chunk_length": min(chunk_lengths),
            "max_chunk_length": max(chunk_lengths),
            "median_chunk_length": np.median(chunk_lengths),
            "semantic_chunks": len([el for el in elements 
                                  if el.get("metadata", {}).get("chunk_type") == "semantic"]),
            "fallback_chunks": len([el for el in elements 
                                  if el.get("metadata", {}).get("chunk_type") == "fallback"])
        }
