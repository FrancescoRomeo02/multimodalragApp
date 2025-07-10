from typing import List, Optional, Dict, Any, Tuple
import logging
import qdrant_client
from qdrant_client.http import models
from src.config import QDRANT_URL, COLLECTION_NAME
from src.utils.embedder import get_embedding_model
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QdrantManager:
    """
    Gestisce tutte le operazioni con il database vettoriale Qdrant.
    Centralizza connessione, creazione collezioni, inserimento e ricerca.
    """
    
    def __init__(self, url: str = QDRANT_URL, collection_name: str = COLLECTION_NAME):
        self.url = url
        self.collection_name = collection_name
        self._client = None
        self._embedder = None
    
    @property
    def client(self) -> qdrant_client.QdrantClient:
        if self._client is None:
            self._client = qdrant_client.QdrantClient(
                url=self.url,
                prefer_grpc=True,
                timeout=60
            )
        return self._client
    
    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = get_embedding_model()
        return self._embedder
    
    # === Funzioni helper per convertire i modelli in punti Qdrant ===
    
    def convert_text_chunks_to_points(
        self,
        text_chunks: List[Any],
        vectors: List[List[float]]
    ) -> List[models.PointStruct]:
        """
        Converte una lista di TextChunk e i loro vettori in punti Qdrant.
        Tutti i tipi di contenuto sono gestiti in modo standardizzato.
        """
        points = []
        for chunk, vector in zip(text_chunks, vectors):
            # Estrai metadati dal chunk e assicurati che siano in formato corretto
            metadata = chunk.metadata.copy() if chunk.metadata else {}
            
            # Assicurati che almeno source e page siano presenti nei metadati
            if "source" not in metadata:
                metadata["source"] = "Sconosciuto"
                
            # Crea payload completamente standardizzato
            payload = {
                "page_content": chunk.text,  # Contenuto testuale
                "metadata": metadata         # Solo metadati standardizzati
            }
            
            points.append(models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload
            ))
        
        return points
    
    # === GESTIONE COLLEZIONE E CONNESSIONE ===
    
    def verify_connection(self) -> bool:
        try:
            self.client.get_collections()
            logger.info(f"Connesso a Qdrant all'URL {self.url}")
            return True
        except Exception as e:
            logger.error(f"Connessione Qdrant fallita: {e}")
            return False
    
    def collection_exists(self) -> bool:
        try:
            return self.client.collection_exists(self.collection_name)
        except Exception as e:
            logger.error(f"Errore verifica collezione: {e}")
            return False
    
    def create_collection(self, embedding_dim: int, force_recreate: bool = False) -> bool:
        try:
            if force_recreate and self.collection_exists():
                self.delete_collection()
                logger.info(f"Collezione {self.collection_name} eliminata per ricreazione")
            if not self.collection_exists():
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=embedding_dim,
                        distance=models.Distance.COSINE,
                        on_disk=True
                    )
                )
                logger.info(f"Collezione {self.collection_name} creata con successo")
                return True
            else:
                logger.info(f"Collezione {self.collection_name} esiste già")
                return True
        except Exception as e:
            logger.error(f"Errore creazione collezione: {e}")
            return False
    
    def delete_collection(self) -> bool:
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Collezione {self.collection_name} eliminata")
            return True
        except Exception as e:
            logger.error(f"Errore eliminazione collezione: {e}")
            return False
    
    def ensure_collection_exists(self, embedding_dim: int) -> bool:
        if not self.collection_exists():
            return self.create_collection(embedding_dim)
        return True
    
    # === OPERAZIONI CRUD ===
    
    def upsert_points(self, points: List[models.PointStruct], batch_size: int = 64) -> bool:
        try:
            # Assicura che la collezione esista prima dell'inserimento
            if not self.collection_exists():
                embedding_dim = 1024  # Fallback
                if points and points[0].vector and isinstance(points[0].vector, list):
                    embedding_dim = len(points[0].vector)
                self.create_collection(embedding_dim)

            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                    wait=True
                )
            logger.info(f"Inseriti {len(points)} punti nella collezione")
            return True
        except Exception as e:
            logger.error(f"Errore inserimento punti: {e}")
            return False
    
    def delete_by_source(self, filename: str) -> Tuple[bool, str]:
        logger.info(f"Eliminazione documenti per source='{filename}'")
        try:
            qdrant_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.source",
                        match=models.MatchValue(value=filename),
                    )
                ],
            )
            response = self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=qdrant_filter),
                wait=True
            )
            logger.info(f"Risultato eliminazione: {response}")
            return True, "Punti eliminati con successo da Qdrant"
        except Exception as e:
            error_message = f"Errore durante l'eliminazione da Qdrant: {e}"
            logger.error(error_message)
            return False, error_message
    
    # === FILTRI ===

    def create_file_filter(self, selected_files: List[str]) -> Optional[models.Filter]:
        """
        Crea un filtro per selezionare documenti in base ai nomi dei file.
        """
        if not selected_files:
            return None
        
        # Crea condizioni di filtro per ogni file selezionato
        file_conditions = []
        for filename in selected_files:
            file_conditions.append(
                models.FieldCondition(
                    key="metadata.source",
                    match=models.MatchValue(value=filename)
                )
            )
        
        return models.Filter(should=file_conditions)
  
    def build_combined_filter(self, 
                            selected_files: List[str] = []) -> Optional[models.Filter]:
        """
        Crea un filtro combinato per la ricerca di documenti.
        Ora supporta solo filtro per file in formato standardizzato.
        """
        if selected_files:
            return self.create_file_filter(selected_files)
        return None
    
    # === RICERCA ===
    
    def search_vectors(self, 
                      query_embedding: List[float],
                      top_k: int = 500,
                      selected_files: List[str] = [],
                      score_threshold: Optional[float] = 0.80) -> List[models.ScoredPoint]:
        """
        Ricerca vettoriale nella collezione.
        Ora supporta solo filtro per file in formato standardizzato.
        """
        try:
            qdrant_filter = self.build_combined_filter(selected_files)
            
            # Log del filtro per debug
            if qdrant_filter:
                logger.debug(f"Filtro applicato: {qdrant_filter}")
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
                score_threshold=score_threshold
            )
            logger.info(f"Ricerca vettoriale: trovati {len(results)} risultati, files={selected_files}, score_threshold={score_threshold}")
            return results
        except Exception as e:
            logger.error(f"Errore ricerca vettoriale: {e}")
            return []

    def get_documents_by_source(self, 
                               source_file: str, 
                               page: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Recupera documenti in base al file di origine e opzionalmente alla pagina.
        Restituisce risultati standardizzati con page_content e metadata.
        """
        try:
            # Crea il filtro per metadata.source
            source_filter = models.FieldCondition(
                key="metadata.source",
                match=models.MatchValue(value=source_file)
            )
            
            if page is not None:
                # Crea il filtro per metadata.page
                page_filter = models.FieldCondition(
                    key="metadata.page", 
                    match=models.MatchValue(value=page)
                )
                
                # Combina filtri: (source_filter) AND (page_filter)
                scroll_filter = models.Filter(
                    must=[source_filter, page_filter]
                )
            else:
                scroll_filter = models.Filter(must=[source_filter])
            
            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=1000,
                with_payload=True,
                with_vectors=False
            )
            
            documents = []
            for result in results:
                payload = result.payload or {}
                page_content = payload.get("page_content", "")
                metadata = payload.get("metadata", {})
                
                doc_info = {
                    "content": page_content,
                    "metadata": metadata,
                    "id": result.id
                }
                documents.append(doc_info)
            
            logger.info(f"Trovati {len(documents)} documenti per fonte {source_file}")
            return documents
        except Exception as e:
            logger.error(f"Errore recupero documenti per fonte {source_file}: {e}")
            return []
    
    def debug_collection_content(self, limit: int = 10) -> Dict[str, Any]:
        """
        Metodo di debug per vedere il contenuto della collezione.
        Ora mostra informazioni sulla struttura standardizzata.
        """
        try:
            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            debug_info = {
                "total_points_sampled": len(results),
                "sources": set(),
                "sample_points": []
            }
            
            for result in results:
                payload = result.payload or {}
                metadata = payload.get("metadata", {})
                page_content = payload.get("page_content", "")
                
                # Raccoglie le fonti
                source = metadata.get("source", "unknown")
                debug_info["sources"].add(source)
                
                # Campione di punti
                if len(debug_info["sample_points"]) < 5:
                    debug_info["sample_points"].append({
                        "id": result.id,
                        "source": source,
                        "page": metadata.get("page", "N/A"),
                        "content_preview": page_content[:100] + "..." if page_content else "No content"
                    })
            
            debug_info["sources"] = list(debug_info["sources"])
            logger.info(f"Debug collezione: trovati {len(results)} punti")
            return debug_info
            
        except Exception as e:
            logger.error(f"Errore debug collezione: {e}")
            return {"error": str(e)}
    
    def get_collection_info(self) -> Dict[str, Any]:
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": collection_info.vectors_count,
                "points_count": collection_info.points_count,
                "status": collection_info.status,
                "config": collection_info.config
            }
        except Exception as e:
            logger.error(f"Errore recupero info collezione: {e}")
            return {}

    def health_check(self) -> Dict[str, Any]:
        try:
            connection_ok = self.verify_connection()
            collection_exists = self.collection_exists()
            collection_info = self.get_collection_info() if collection_exists else {}
            debug_info = self.debug_collection_content(5) if collection_exists else {}
            
            return {
                "connection": connection_ok,
                "collection_exists": collection_exists,
                "collection_info": collection_info,
                "debug_info": debug_info,
                "url": self.url,
                "collection_name": self.collection_name
            }
        except Exception as e:
            logger.error(f"Errore health check: {e}")
            return {"error": str(e)}

# Singleton per uso globale
qdrant_manager = QdrantManager()