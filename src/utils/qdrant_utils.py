from typing import List, Optional, Dict, Any, Tuple, Union
import logging
import qdrant_client
from qdrant_client.http import models
from src.config import QDRANT_URL, COLLECTION_NAME
from src.core.models import ImageResult, TextElement, ImageElement, TableElement
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
    
    def _text_element_to_point(self, 
                               element: TextElement, 
                               vector: List[float]) -> models.PointStruct:
        return models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "page_content": element.text,
                "metadata": element.metadata.model_dump(),
            }
        )
    
    def _image_element_to_point(self,
                                element: Union[ImageElement, Dict[str, Any]], 
                                vector: List[float]) -> models.PointStruct:
        # Gestisce sia oggetti Pydantic che dizionari dal parser
        if isinstance(element, dict):
            # Caso dizionario dal parser legacy
            metadata = element.get("metadata", {})
            # Rimuovi image_base64 dai metadati se presente per evitare duplicazione pesante
            if "image_base64" in metadata:
                metadata = metadata.copy()
                del metadata["image_base64"]
            
            return models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "page_content": element.get("page_content", ""),
                    # image_base64 rimosso dal database - troppo pesante e inutile per la ricerca
                    "content_type": "image",
                    "metadata": metadata
                }
            )
        else:
            # Caso oggetto Pydantic
            return models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "page": element.metadata.page,
                    "content_type": "image",
                    "metadata": element.metadata.model_dump()
                }
            )
    
    def _table_element_to_point(self, 
                                element: Union[TableElement, Dict[str, Any]], 
                                vector: List[float]) -> models.PointStruct:
        # Gestisce sia oggetti Pydantic che dizionari dal parser
        if isinstance(element, dict):
            # Caso dizionario dal parser legacy
            return models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "page_content": element.get("table_html", ""),
                    "metadata": element.get("metadata", {}),
                    "content_type": "table",
                }
            )
        else:
            # Caso oggetto Pydantic
            return models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "page_content": element.table_html,
                    "metadata": element.metadata.model_dump(),
                    "content_type": "table",
                }
            )
    
    def convert_elements_to_points(
        self,
        elements: List[Any],
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
            elif isinstance(element, (ImageElement, dict)) and (
                hasattr(element, 'image_base64') or 
                (isinstance(element, dict) and 'image_base64' in element)
            ):
                points.append(self._image_element_to_point(element, vector))
            elif isinstance(element, (TableElement, dict)) and (
                hasattr(element, 'table_html') or 
                (isinstance(element, dict) and 'table_html' in element)
            ):
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
    
    def create_content_filter(self, 
                              query_type: Optional[str] = None) -> Optional[models.Filter]:
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
                        key="content_type",
                        match=models.MatchValue(value="text"),
                    )
                ]
            )
        elif query_type == "table":
            return models.Filter(
                must=[
                    models.FieldCondition(
                        key="content_type",
                        match=models.MatchValue(value="table"),
                    )
                ]
            )
        return None
    
    def create_file_filter(self, 
                           selected_files: List[str]) -> Optional[models.Filter]:
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

    def create_page_filter(self, 
                            pages: List[int]) -> Optional[models.Filter]:
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
                            query_type: Optional[str] = None) -> Optional[models.Filter]:
        filters = []
        
        if selected_files:
            file_filter = self.create_file_filter(selected_files)
            if file_filter:
                filters.append(file_filter)
        
        if query_type:
            content_filter = self.create_content_filter(query_type)
            if content_filter:
                filters.append(content_filter)
        
        if not filters:
            return None
        if len(filters) == 1:
            return filters[0]
        return models.Filter(must=filters)
    
    # === RICERCA ===
    
    def search_vectors(self, 
                      query_embedding: List[float],
                      top_k: int = 500,
                      selected_files: List[str] = [],
                      query_type: Optional[str] = None,
                      score_threshold: Optional[float] = 0.80) -> List[models.ScoredPoint]:
        try:
            qdrant_filter = self.build_combined_filter(selected_files, query_type)

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
            logger.info(f"Ricerca vettoriale: trovati {len(results)} risultati per query_type='{query_type}', files={selected_files}")
            return results
        except Exception as e:
            logger.error(f"Errore ricerca vettoriale: {e}")
            return []
    
    def query_text(self, 
                   query: str, 
                   selected_files: List[str] = [],
                   top_k: int = 500,
                   score_threshold: float = 0.80) -> List[Dict[str, Any]]:
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
                query_type="text",
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
                    top_k: int = 500) -> List[ImageResult]:
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
                     top_k: int = 500) -> List[Dict[str, Any]]:
        logger.info(f"Query tabelle: '{query}' con top_k={top_k}, file: {selected_files}")
        try:
            query_embedding = self.embedder.embed_query(query)
            results = self.search_vectors(
                query_embedding=query_embedding,
                top_k=top_k,
                selected_files=selected_files,
                query_type="table"
            )
            
            table_results = []
            for result in results:
                try:
                    payload = result.payload or {}
                    metadata = payload.get("metadata", {})
                    table_html = payload.get("page_content", "")
                    
                    if not table_html:
                        logger.debug(f"Saltato risultato senza contenuto tabella: {result.id}")
                        continue
                    
                    table_results.append({
                        "table_html": table_html,
                        "metadata": metadata,
                        "score": result.score,
                        "page_content": table_html,
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
                         top_k_per_type: int = 500,
                         score_threshold: float = 0.80) -> Dict[str, Any]:
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

    def debug_collection_content(self, 
                                 limit: int = 10) -> Dict[str, Any]:
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