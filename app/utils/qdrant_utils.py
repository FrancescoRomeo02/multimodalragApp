from typing import List, Optional, Dict, Any, Tuple
import logging
import qdrant_client
from qdrant_client.http import models
from app.config import QDRANT_URL, COLLECTION_NAME
from app.core.models import ImageResult
from app.utils.embedder import get_multimodal_embedding_model

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
        """Lazy loading del client Qdrant"""
        if self._client is None:
            self._client = qdrant_client.QdrantClient(
                url=self.url,
                prefer_grpc=True,
                timeout=60
            )
        return self._client
    
    @property
    def embedder(self):
        """Lazy loading dell'embedder"""
        if self._embedder is None:
            self._embedder = get_multimodal_embedding_model()
        return self._embedder
    
    # === GESTIONE CONNESSIONE E COLLEZIONI ===
    
    def verify_connection(self) -> bool:
        """Verifica la connessione a Qdrant"""
        try:
            self.client.get_collections()
            logger.info(f"Connesso a Qdrant all'URL {self.url}")
            return True
        except Exception as e:
            logger.error(f"Connessione Qdrant fallita: {e}")
            return False
    
    def collection_exists(self) -> bool:
        """Verifica se la collezione esiste"""
        try:
            return self.client.collection_exists(self.collection_name)
        except Exception as e:
            logger.error(f"Errore verifica collezione: {e}")
            return False
    
    def create_collection(self, embedding_dim: int, force_recreate: bool = False) -> bool:
        """
        Crea la collezione Qdrant
        
        Args:
            embedding_dim: Dimensione dei vettori
            force_recreate: Se True, elimina e ricrea la collezione esistente
        """
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
        """Elimina la collezione"""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Collezione {self.collection_name} eliminata")
            return True
        except Exception as e:
            logger.error(f"Errore eliminazione collezione: {e}")
            return False
    
    def ensure_collection_exists(self, embedding_dim: int) -> bool:
        """Assicura che la collezione esista, altrimenti la crea"""
        if not self.collection_exists():
            return self.create_collection(embedding_dim)
        return True
    
    # === OPERAZIONI CRUD ===
    
    def upsert_points(self, points: List[models.PointStruct], batch_size: int = 64) -> bool:
        """Inserisce o aggiorna punti nella collezione"""
        try:
            # Inserimento in batch per performance
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
        """
        Elimina tutti i punti che hanno 'source' nel payload uguale al nome file
        
        Args:
            filename: Nome del file da eliminare
            
        Returns:
            Tuple[bool, str]: (successo, messaggio)
        """
        logger.info(f"Eliminazione documenti per source='{filename}'")
        
        try:
            qdrant_filter = models.Filter(
                should=[
                    models.FieldCondition(
                        key="metadata.source",
                        match=models.MatchValue(value=filename),
                    ),
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value=filename),
                    ),
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
    
    # === COSTRUZIONE FILTRI ===
    
    def create_content_filter(self, query_type: Optional[str] = None) -> Optional[models.Filter]:
        """Crea filtri per tipo di contenuto"""
        if query_type == "image":
            return models.Filter(
                must=[
                    models.FieldCondition(
                        key="content_type",
                        match=models.MatchValue(value="image"),
                    )
                ]
            )
        elif query_type == "text":
            return models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.type",
                        match=models.MatchValue(value="text"),
                    )
                ]
            )
        return None
    
    def create_file_filter(self, selected_files: List[str]) -> Optional[models.Filter]:
        """Crea filtri per file specifici"""
        if not selected_files:
            return None
            
        return models.Filter(
            should=[
                models.FieldCondition(
                    key="metadata.source",
                    match=models.MatchValue(value=filename),
                )
                for filename in selected_files
            ]
        )
    
    def create_page_filter(self, pages: List[int]) -> Optional[models.Filter]:
        """Crea filtri per pagine specifiche"""
        if not pages:
            return None
            
        return models.Filter(
            should=[
                models.FieldCondition(
                    key="metadata.page",
                    match=models.MatchValue(value=page),
                )
                for page in pages
            ]
        )
    
    def build_combined_filter(self, 
                            selected_files: List[str] = None,
                            query_type: Optional[str] = None,
                            pages: List[int] = None) -> Optional[models.Filter]:
        """Combina filtri multipli"""
        filters = []
        
        if selected_files:
            file_filter = self.create_file_filter(selected_files)
            if file_filter:
                filters.append(file_filter)
        
        if query_type:
            content_filter = self.create_content_filter(query_type)  
            if content_filter:
                filters.append(content_filter)
        
        if pages:
            page_filter = self.create_page_filter(pages)
            if page_filter:
                filters.append(page_filter)
        
        if not filters:
            return None
        
        if len(filters) == 1:
            return filters[0]
        
        # Combina tutti i filtri con AND
        return models.Filter(must=filters)
    
    # === OPERAZIONI DI RICERCA ===
    
    def search_vectors(self, 
                      query_embedding: List[float],
                      top_k: int = 10,
                      selected_files: List[str] = None,
                      query_type: Optional[str] = None,
                      pages: List[int] = None,
                      score_threshold: Optional[float] = None) -> List[models.ScoredPoint]:
        """
        Ricerca vettoriale generica
        
        Args:
            query_embedding: Vettore di ricerca
            top_k: Numero massimo di risultati
            selected_files: Lista file da filtrare
            query_type: Tipo di contenuto ('text', 'image')
            pages: Pagine specifiche
            score_threshold: Soglia minima di similarità
            
        Returns:
            Lista di risultati ordinati per score
        """
        try:
            qdrant_filter = self.build_combined_filter(selected_files, query_type, pages)
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
                score_threshold=score_threshold
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Errore ricerca vettoriale: {e}")
            return []
    
    def query_images(self, 
                    query: str, 
                    selected_files: List[str] = None,
                    top_k: int = 3) -> List[ImageResult]:
        """
        Ricerca specifica per immagini con risultati formattati
        
        Args:
            query: Query di ricerca
            selected_files: File da includere nella ricerca
            top_k: Numero massimo di risultati
            
        Returns:
            Lista di ImageResult
        """
        logger.info(f"Query immagini: '{query}' con top_k={top_k}, file: {selected_files}")
        
        try:
            query_embedding = self.embedder.embed_query(query)
            results = self.search_vectors(
                query_embedding=query_embedding,
                top_k=top_k,
                selected_files=selected_files,
                query_type="image"
            )
            
            image_results = []
            for result in results:
                try:
                    metadata = result.payload.get("metadata", {})
                    image_base64 = metadata.get("image_base64", "")
                    
                    if image_base64:
                        image_results.append(ImageResult(
                            image_base64=image_base64,
                            metadata=metadata,
                            score=result.score,
                            page_content=result.payload.get("page_content", "")
                        ))
                except Exception as e:
                    logger.warning(f"Errore processamento risultato immagine: {e}")
                    continue
            
            logger.info(f"Trovate {len(image_results)} immagini per query '{query}'")
            return image_results
            
        except Exception as e:
            logger.error(f"Errore query immagini: {e}")
            return []
    def query_tables(self, 
                 query: str, 
                 selected_files: List[str] = None,
                 top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Ricerca specifica per tabelle, restituendo contenuto e metadati.

        Args:
            query: Testo della query
            selected_files: Filtri opzionali sui file
            top_k: Numero massimo di risultati

        Returns:
            Lista di dizionari contenenti tabella (base64), metadati, score e pagina
        """
        logger.info(f"Query tabelle: '{query}' con top_k={top_k}, file: {selected_files}")

        try:
            # 1. Embedding della query testuale
            query_embedding = self.embedder.embed_query(query)

            # 2. Ricerca vettoriale con filtro per tabelle
            results = self.search_vectors(
                query_embedding=query_embedding,
                top_k=top_k,
                selected_files=selected_files,
                query_type="table"
            )

            # 3. Parsing risultati
            table_results = []
            for result in results:
                try:
                    metadata = result.payload.get("metadata", {})
                    table_base64 = metadata.get("table_base64", "")

                    if table_base64:
                        table_results.append({
                            "table_base64": table_base64,
                            "metadata": metadata,
                            "score": result.score,
                            "page_content": result.payload.get("page_content", "")
                        })
                except Exception as e:
                    logger.warning(f"Errore processamento risultato tabella: {e}")
                    continue

            logger.info(f"Trovate {len(table_results)} tabelle per query '{query}'")
            return table_results

        except Exception as e:
            logger.error(f"Errore query tabelle: {e}")
            return []

    
    def search_similar_documents(self, 
                                query: str,
                                selected_files: List[str] = None,
                                similarity_threshold: float = 0.7,
                                max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Ricerca documenti simili testuali
        
        Args:
            query: Query di ricerca
            selected_files: File da includere
            similarity_threshold: Soglia di similarità
            max_results: Numero massimo risultati
            
        Returns:
            Lista di documenti con metadati
        """
        try:
            query_embedding = self.embedder.embed_query(query)
            results = self.search_vectors(
                query_embedding=query_embedding,
                top_k=max_results,
                selected_files=selected_files,
                query_type="text",
                score_threshold=similarity_threshold
            )
            
            similar_docs = []
            for result in results:
                payload = result.payload or {}
                metadata = payload.get("metadata", {})
                doc_info = {
                    "content": payload.get("page_content", ""),
                    "metadata": metadata,
                    "score": result.score,
                    "source": metadata.get("source", "Unknown"),
                    "page": metadata.get("page", "N/A")
                }
                similar_docs.append(doc_info)
            
            return similar_docs
            
        except Exception as e:
            logger.error(f"Errore ricerca documenti simili: {e}")
            return []
    
    def get_documents_by_source(self, 
                               source_file: str, 
                               page: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Recupera tutti i documenti da un file specifico
        
        Args:
            source_file: Nome del file sorgente
            page: Pagina specifica (opzionale)
            
        Returns:
            Lista di documenti dal file
        """
        try:
            filters = [
                models.FieldCondition(
                    key="metadata.source",
                    match=models.MatchValue(value=source_file)
                )
            ]
            
            if page is not None:
                filters.append(
                    models.FieldCondition(
                        key="metadata.page",
                        match=models.MatchValue(value=page)
                    )
                )
            
            scroll_filter = models.Filter(must=filters)
            
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
                metadata = payload.get("metadata", {})
                doc_info = {
                    "content": payload.get("page_content", ""),
                    "metadata": metadata,
                    "id": result.id,
                    "source": metadata.get("source", "Unknown"),
                    "page": metadata.get("page", "N/A"),
                    "type": metadata.get("type", "unknown")
                }
                documents.append(doc_info)
            
            return documents
            
        except Exception as e:
            logger.error(f"Errore recupero documenti per fonte {source_file}: {e}")
            return []
    
    # === METODI DI UTILITÀ ===
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Restituisce informazioni sulla collezione"""
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
        """Controllo dello stato di salute del sistema Qdrant"""
        try:
            connection_ok = self.verify_connection()
            collection_exists = self.collection_exists()
            collection_info = self.get_collection_info() if collection_exists else {}
            
            return {
                "connection": connection_ok,
                "collection_exists": collection_exists,
                "collection_info": collection_info,
                "url": self.url,
                "collection_name": self.collection_name
            }
        except Exception as e:
            logger.error(f"Errore health check: {e}")
            return {"error": str(e)}


# Istanza singleton per uso globale
qdrant_manager = QdrantManager()