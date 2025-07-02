from typing import List, Optional, Dict, Any, Tuple, Union
import logging
import qdrant_client
from qdrant_client.http import models
from src.config import QDRANT_URL, COLLECTION_NAME
from src.core.models import ImageResult, TextElement, ImageElement, TableElement, QdrantPayload
from src.utils.embedder import get_multimodal_embedding_model
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
            self._embedder = get_multimodal_embedding_model()
        return self._embedder
    
    # === Funzioni helper per convertire i modelli in punti Qdrant con payload unificato ===
    
    def _text_element_to_point(self, element: TextElement, vector: List[float]) -> models.PointStruct:
        """Converte TextElement in punto Qdrant con payload unificato contenente solo metadati JSON"""
        payload = QdrantPayload.from_text_element(element)
        return models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload=payload.model_dump()  # Solo metadati JSON essenziali
        )
    
    def _image_element_to_point(self, element: ImageElement, vector: List[float], image_caption: str = "") -> models.PointStruct:
        """Converte ImageElement in punto Qdrant con payload unificato contenente solo metadati JSON"""
        payload = QdrantPayload.from_image_element(element)
        return models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload=payload.model_dump()  # Solo metadati JSON essenziali
        )
    
    def _table_element_to_point(self, element: TableElement, vector: List[float]) -> models.PointStruct:
        """Converte TableElement in punto Qdrant con payload unificato contenente solo metadati JSON"""
        payload = QdrantPayload.from_table_element(element)
        return models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload=payload.model_dump()  # Solo metadati JSON essenziali
        )
    
    def convert_elements_to_points(
        self,
        elements: List[Union[TextElement, ImageElement, TableElement]],
        vectors: List[List[float]]
    ) -> List[models.PointStruct]:
        """
        Converte una lista di elementi e vettori in punti Qdrant da inserire.
        Assumiamo che elements[i] corrisponda a vectors[i].
        """
        points = []
        for element, vector in zip(elements, vectors):
            if isinstance(element, TextElement):
                points.append(self._text_element_to_point(element, vector))
            elif isinstance(element, ImageElement):
                points.append(self._image_element_to_point(element, vector))
            elif isinstance(element, TableElement):
                points.append(self._table_element_to_point(element, vector))
            else:
                logger.warning(f"Elemento non riconosciuto per indicizzazione: {element}")
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
    
    # === FILTRI ===

    def create_file_filter(self, selected_files: List[str]) -> Optional[models.Filter]:
        if not selected_files:
            return None
        
        # Supporta sia metadata.source che source per compatibilità
        file_conditions = []
        for filename in selected_files:
            file_conditions.extend([
                models.FieldCondition(
                    key="metadata.source",
                    match=models.MatchValue(value=filename),
                ),
                models.FieldCondition(
                    key="source",
                    match=models.MatchValue(value=filename),
                )
            ])
        
        return models.Filter(should=file_conditions)
    
    def create_page_filter(self, pages: List[int]) -> Optional[models.Filter]:
        if not pages:
            return None
        
        # Supporta sia metadata.page che page per compatibilità
        page_conditions = []
        for page in pages:
            page_conditions.extend([
                models.FieldCondition(
                    key="metadata.page",
                    match=models.MatchValue(value=page),
                ),
                models.FieldCondition(
                    key="page",
                    match=models.MatchValue(value=page),
                )
            ])
        
        return models.Filter(should=page_conditions)
    
    def build_combined_filter(self, 

                            selected_files: List[str] = [],
                            query_type: Optional[str] = None,
                            pages: List[int] = []) -> Optional[models.Filter]:
        filters = []
        
        if selected_files:
            file_filter = self.create_file_filter(selected_files)
            if file_filter:
                filters.append(file_filter)
        
        if pages:
            page_filter = self.create_page_filter(pages)
            if page_filter:
                filters.append(page_filter)
        
        if not filters:
            return None
        if len(filters) == 1:
            return filters[0]
        return models.Filter(must=filters)
    
    # === RICERCA ===
    
    def search_vectors(self, 
                      query_embedding: List[float],
                      top_k: int = 10,
                      selected_files: List[str] = [],
                      query_type: Optional[str] = None,
                      pages: List[int] = [],
                      score_threshold: Optional[float] = None) -> List[models.ScoredPoint]:
        try:
            qdrant_filter = self.build_combined_filter(selected_files, pages)
            
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
            logger.info(f"Ricerca vettoriale: trovati {len(results)} risultati per files={selected_files}")
            return results
        except Exception as e:
            logger.error(f"Errore ricerca vettoriale: {e}")
            return []
    
    def query_text(self, 
                   query: str, 
                   selected_files: List[str] = [],
                   top_k: int = 10,
                   score_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Cerca documenti di testo simili alla query.
        """
        logger.info(f"Query testo: '{query}' con top_k={top_k}, file: {selected_files}, threshold: {score_threshold}")
        try:
            query_embedding = self.embedder.embed_query(query)
            results = self.search_vectors(
                query_embedding=query_embedding,
                top_k=top_k,
                selected_files=selected_files,
                score_threshold=score_threshold
            )
            
            text_results = []
            for result in results:
                try:
                    payload = result.payload or {}
                    metadata = payload.get("metadata", {})
                    content = payload.get("page_content", "")
                    
                    # Verifica che sia effettivamente contenuto testuale
                    if not content:
                        logger.debug(f"Saltato risultato senza contenuto testuale: {result.id}")
                        continue
                    
                    text_results.append({
                        "content": content,
                        "metadata": metadata,
                        "score": result.score,
                        "source": metadata.get("source", "Unknown"),
                        "page": metadata.get("page", "N/A"),
                        "type": metadata.get("type", "text"),
                        "content_type": payload.get("content_type", "text")
                    })
                except Exception as e:
                    logger.warning(f"Errore processamento risultato testo: {e}")
                    continue
            
            logger.info(f"Trovati {len(text_results)} documenti di testo per query '{query}'")
            return text_results
        except Exception as e:
            logger.error(f"Errore query testo: {e}")
            return []
    
    def query_images(self, 
                    query: str, 
                    selected_files: List[str] = [],
                    top_k: int = 3) -> List[ImageResult]:
        logger.info(f"Query immagini: '{query}' con top_k={top_k}, file: {selected_files}")
        try:
            query_embedding = self.embedder.embed_query(query)
            results = self.search_vectors(
                query_embedding=query_embedding,
                top_k=top_k,
                selected_files=selected_files
            )
            
            image_results = []
            for result in results:
                try:
                    payload = result.payload or {}
                    metadata = payload.get("metadata", {})
                    image_base64 = payload.get("image_base64", "")
                    
                    if not image_base64:
                        logger.debug(f"Saltato risultato senza immagine base64: {result.id}")
                        continue
                    
                    image_results.append(ImageResult(
                        image_base64=image_base64,
                        metadata=metadata,
                        score=result.score,
                        page_content=payload.get("page_content", "")
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
                     selected_files: List[str] = [],

                     top_k: int = 3) -> List[Dict[str, Any]]:
        logger.info(f"Query tabelle: '{query}' con top_k={top_k}, file: {selected_files}")
        try:
            query_embedding = self.embedder.embed_query(query)
            results = self.search_vectors(
                query_embedding=query_embedding,
                top_k=top_k,
                selected_files=selected_files
            )
            
            table_results = []
            for result in results:
                try:
                    payload = result.payload or {}
                    metadata = payload.get("metadata", {})
                    table_markdown = payload.get("page_content", "")
                    
                    if not table_markdown:
                        logger.debug(f"Saltato risultato senza contenuto tabella: {result.id}")
                        continue
                    
                    table_results.append({
                        "table_markdown": table_markdown,
                        "metadata": metadata,
                        "score": result.score,
                        "page_content": table_markdown,
                        "table_data": payload.get("table_data", {})
                    })
                except Exception as e:
                    logger.warning(f"Errore processamento risultato tabella: {e}")
                    continue
            
            logger.info(f"Trovate {len(table_results)} tabelle per query '{query}'")
            return table_results
        except Exception as e:
            logger.error(f"Errore query tabelle: {e}")
            return []
    
    def query_all_content(self, 
                         query: str, 
                         selected_files: List[str] = [],
                         top_k_per_type: int = 5,
                         score_threshold: float = 0.3) -> Dict[str, Any]:
        """
        Cerca contenuti di tutti i tipi (testo, immagini, tabelle) per una query.
        """
        logger.info(f"Query contenuti misti: '{query}' con top_k_per_type={top_k_per_type}, file: {selected_files}")
        
        results = {
            "text": [],
            "images": [],
            "tables": []
        }
        
        try:
            # Cerca testo
            text_results = self.query_text(query, selected_files, top_k_per_type, score_threshold)
            results["text"] = text_results
            
            # Cerca immagini
            image_results = self.query_images(query, selected_files, top_k_per_type)
            results["images"] = image_results
            
            # Cerca tabelle
            table_results = self.query_tables(query, selected_files, top_k_per_type)
            results["tables"] = table_results
            
            total_results = len(text_results) + len(image_results) + len(table_results)
            logger.info(f"Query completa: trovati {total_results} risultati totali "
                       f"(testo: {len(text_results)}, immagini: {len(image_results)}, tabelle: {len(table_results)})")
            
        except Exception as e:
            logger.error(f"Errore nella query_all_content: {e}")
        
        return results
    
    def search_similar_documents(self, 
                                query: str,
                                selected_files: List[str] = [],
                                similarity_threshold: float = 0.7,
                                max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Alias per query_text con parametri diversi per retrocompatibilità.
        """
        logger.info(f"search_similar_documents chiamato per query: '{query}'")
        return self.query_text(
            query=query,
            selected_files=selected_files,
            top_k=max_results,
            score_threshold=similarity_threshold
        )
    
    def get_documents_by_source(self, 
                               source_file: str, 
                               page: Optional[int] = None) -> List[Dict[str, Any]]:
        try:
            # Supporta sia metadata.source che source
            filters = [
                models.FieldCondition(
                    key="metadata.source",
                    match=models.MatchValue(value=source_file)
                ),
                models.FieldCondition(
                    key="source",
                    match=models.MatchValue(value=source_file)
                )
            ]
            
            if page is not None:
                page_filters = [
                    models.FieldCondition(
                        key="metadata.page",
                        match=models.MatchValue(value=page)
                    ),
                    models.FieldCondition(
                        key="page",
                        match=models.MatchValue(value=page)
                    )
                ]
                # Combina filtri: (source_filter) AND (page_filter)
                scroll_filter = models.Filter(
                    must=[
                        models.Filter(should=filters),  # source
                        models.Filter(should=page_filters)  # page
                    ]
                )
            else:
                scroll_filter = models.Filter(should=filters)
            
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
                    "source": metadata.get("source", payload.get("source", "Unknown")),
                    "page": metadata.get("page", payload.get("page", "N/A")),
                    "type": metadata.get("type", "unknown"),
                    "content_type": payload.get("content_type", "unknown")
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
                "content_types": {},
                "sources": set(),
                "sample_points": []
            }
            
            for result in results:
                payload = result.payload or {}
                content_type = payload.get("content_type", "unknown")
                
                # Conta i tipi di contenuto
                if content_type not in debug_info["content_types"]:
                    debug_info["content_types"][content_type] = 0
                debug_info["content_types"][content_type] += 1
                
                # Raccoglie le fonti
                metadata = payload.get("metadata", {})
                source = metadata.get("source", payload.get("source", "unknown"))
                debug_info["sources"].add(source)
                
                # Campione di punti
                if len(debug_info["sample_points"]) < 5:
                    debug_info["sample_points"].append({
                        "id": result.id,
                        "content_type": content_type,
                        "source": source,
                        "page": metadata.get("page", payload.get("page", "N/A")),
                        "content_preview": payload.get("page_content", "")[:100] + "..." if payload.get("page_content") else "No content"
                    })
            
            debug_info["sources"] = list(debug_info["sources"])
            logger.info(f"Debug collezione: {debug_info['content_types']}")
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
    
    def test_text_query_direct(self, query: str, filename: str) -> Dict[str, Any]:
        """
        Test diretto per verificare se il testo viene trovato correttamente
        """
        logger.info(f"🧪 Test query testo diretto: '{query}' per file: {filename}")
        
        try:
            # Genera embedding
            query_embedding = self.embedder.embed_query(query)
            
            # Test 1: Query senza filtri
            results_no_filter = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=5,
                with_payload=True,
                with_vectors=False
            )
            
            # Test 2: Query solo con filtro file
            file_filter = self.create_file_filter([filename])
            results_file_filter = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=file_filter,
                limit=5,
                with_payload=True,
                with_vectors=False
            )
            
            # Test 3: Query con filtro per file
            combined_filter = self.build_combined_filter([filename])
            results_combined = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=combined_filter,
                limit=5,
                with_payload=True,
                with_vectors=False
            )
            
            def format_results(results, test_name):
                formatted = []
                for r in results:
                    payload = r.payload or {}
                    formatted.append({
                        "score": r.score,
                        "content_type": payload.get("content_type", "unknown"),
                        "source": payload.get("metadata", {}).get("source", payload.get("source", "unknown")),
                        "page": payload.get("metadata", {}).get("page", payload.get("page", "N/A")),
                        "content_preview": payload.get("page_content", "")[:100] + "..." if len(payload.get("page_content", "")) > 100 else payload.get("page_content", "")
                    })
                return formatted
            
            return {
                "query": query,
                "filename": filename,
                "tests": {
                    "no_filter": {
                        "count": len(results_no_filter),
                        "results": format_results(results_no_filter, "no_filter")
                    },
                    "file_filter": {
                        "count": len(results_file_filter),
                        "results": format_results(results_file_filter, "file_filter")
                    },
                    "combined_filter": {
                        "count": len(results_combined),
                        "results": format_results(results_combined, "combined_filter")
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Errore nel test diretto: {e}")
            return {"error": str(e)}

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