"""
Modulo per la gestione del database MongoDB per i contenuti originali.
Questo database memorizza i dati completi (testo, immagini, tabelle) 
mentre Qdrant contiene solo gli embedding per la ricerca.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import pymongo
from pymongo import MongoClient
from pymongo.collection import Collection
from bson.objectid import ObjectId
import base64
import json

from src.config import MONGO_URI, MONGO_DB_NAME, MONGO_COLLECTION

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MongoManager:
    """
    Gestisce tutte le operazioni con il database MongoDB.
    Memorizza i dati originali completi (testo, immagini, tabelle).
    """
    
    def __init__(self, uri: str = MONGO_URI, db_name: str = MONGO_DB_NAME, collection_name: str = MONGO_COLLECTION):
        self.uri = uri
        self.db_name = db_name
        self.collection_name = collection_name
        self._client = None
        self._db = None
        self._collection = None
    
    @property
    def client(self) -> MongoClient:
        if self._client is None:
            self._client = MongoClient(self.uri)
        return self._client
    
    @property
    def db(self):
        if self._db is None:
            self._db = self.client[self.db_name]
        return self._db
    
    @property
    def collection(self) -> Collection:
        if self._collection is None:
            self._collection = self.db[self.collection_name]
        return self._collection
    
    def verify_connection(self) -> bool:
        """Verifica la connessione a MongoDB."""
        try:
            info = self.client.server_info()
            logger.info(f"Connesso a MongoDB v{info.get('version')}")
            return True
        except Exception as e:
            logger.error(f"Connessione MongoDB fallita: {e}")
            return False
    
    def ensure_indexes(self) -> bool:
        """Crea gli indici necessari per ottimizzare le query."""
        try:
            # Indice per ricercare documenti per source
            self.collection.create_index("source")
            # Indice per ricercare documenti per page
            self.collection.create_index("page")
            # Indice per ricercare documenti per tipo di contenuto
            self.collection.create_index("content_type")
            # Indice composito per source + page
            self.collection.create_index([("source", pymongo.ASCENDING), ("page", pymongo.ASCENDING)])
            logger.info("Indici MongoDB creati/verificati con successo")
            return True
        except Exception as e:
            logger.error(f"Errore creazione indici MongoDB: {e}")
            return False
    
    def store_document(self, document: Dict[str, Any]) -> Optional[str]:
        """
        Memorizza un documento in MongoDB e restituisce l'ID generato.
        
        Args:
            document: Il documento da memorizzare
            
        Returns:
            L'ID del documento inserito o None in caso di errore
        """
        try:
            # Verifica che ci sia almeno il tipo di contenuto
            if "content_type" not in document:
                document["content_type"] = "unknown"
                
            result = self.collection.insert_one(document)
            doc_id = str(result.inserted_id)
            logger.debug(f"Documento inserito in MongoDB con ID: {doc_id}")
            return doc_id
        except Exception as e:
            logger.error(f"Errore inserimento documento in MongoDB: {e}")
            return None
    
    def store_text_document(self, text: str, source: str, page: int, 
                           additional_metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Memorizza un documento testuale in MongoDB.
        
        Args:
            text: Il contenuto testuale
            source: La fonte del documento
            page: Il numero di pagina
            additional_metadata: Metadati aggiuntivi
            
        Returns:
            L'ID del documento inserito o None in caso di errore
        """
        try:
            metadata = {"source": source, "page": page}
            if additional_metadata:
                metadata.update(additional_metadata)
                
            document = {
                "content_type": "text",
                "content": text,
                **metadata
            }
            
            return self.store_document(document)
        except Exception as e:
            logger.error(f"Errore inserimento documento testuale in MongoDB: {e}")
            return None
    
    def store_image_document(self, image_base64: str, caption: str, source: str, page: int, 
                            additional_metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Memorizza un'immagine in MongoDB.
        
        Args:
            image_base64: L'immagine codificata in base64
            caption: Didascalia o descrizione dell'immagine
            source: La fonte del documento
            page: Il numero di pagina
            additional_metadata: Metadati aggiuntivi
            
        Returns:
            L'ID del documento inserito o None in caso di errore
        """
        try:
            metadata = {"source": source, "page": page}
            if additional_metadata:
                metadata.update(additional_metadata)
                
            document = {
                "content_type": "image",
                "image_data": image_base64,
                "caption": caption,
                **metadata
            }
            
            return self.store_document(document)
        except Exception as e:
            logger.error(f"Errore inserimento immagine in MongoDB: {e}")
            return None
    
    def store_table_document(self, table_markdown: str, table_data: Dict[str, Any], 
                            source: str, page: int, 
                            additional_metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Memorizza una tabella in MongoDB.
        
        Args:
            table_markdown: La tabella in formato markdown
            table_data: I dati strutturati della tabella
            source: La fonte del documento
            page: Il numero di pagina
            additional_metadata: Metadati aggiuntivi
            
        Returns:
            L'ID del documento inserito o None in caso di errore
        """
        try:
            metadata = {"source": source, "page": page}
            if additional_metadata:
                metadata.update(additional_metadata)
                
            document = {
                "content_type": "table",
                "table_markdown": table_markdown,
                "table_data": table_data,
                **metadata
            }
            
            return self.store_document(document)
        except Exception as e:
            logger.error(f"Errore inserimento tabella in MongoDB: {e}")
            return None
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Recupera un documento da MongoDB dato il suo ID.
        
        Args:
            doc_id: L'ID del documento
            
        Returns:
            Il documento recuperato o None in caso di errore
        """
        try:
            document = self.collection.find_one({"_id": ObjectId(doc_id)})
            if document:
                document["_id"] = str(document["_id"])  # Converti ObjectId in stringa
            return document
        except Exception as e:
            logger.error(f"Errore recupero documento da MongoDB: {e}")
            return None
    
    def get_documents_by_source(self, source: str, page: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Recupera tutti i documenti relativi a una fonte e opzionalmente a una pagina.
        
        Args:
            source: La fonte dei documenti
            page: Il numero di pagina (opzionale)
            
        Returns:
            Lista di documenti trovati
        """
        try:
            query = {"source": source}
            if page is not None:
                query["page"] = page
                
            documents = list(self.collection.find(query))
            
            # Converti ObjectId in stringa per ogni documento
            for doc in documents:
                doc["_id"] = str(doc["_id"])
                
            logger.info(f"Recuperati {len(documents)} documenti da MongoDB per {source}")
            return documents
        except Exception as e:
            logger.error(f"Errore recupero documenti da MongoDB: {e}")
            return []
    
    def delete_by_source(self, source: str) -> Tuple[bool, str]:
        """
        Elimina tutti i documenti relativi a una fonte.
        
        Args:
            source: La fonte dei documenti da eliminare
            
        Returns:
            Tupla (successo, messaggio)
        """
        try:
            result = self.collection.delete_many({"source": source})
            deleted_count = result.deleted_count
            
            logger.info(f"Eliminati {deleted_count} documenti da MongoDB per {source}")
            return True, f"Eliminati {deleted_count} documenti da MongoDB"
        except Exception as e:
            error_message = f"Errore eliminazione documenti da MongoDB: {e}"
            logger.error(error_message)
            return False, error_message
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Restituisce informazioni sulla collezione MongoDB.
        
        Returns:
            Dizionario con informazioni sulla collezione
        """
        try:
            count = self.collection.count_documents({})
            content_types = {}
            
            # Conteggio per tipo di contenuto
            for content_type in ["text", "image", "table"]:
                content_types[content_type] = self.collection.count_documents({"content_type": content_type})
            
            # Recupera le fonti uniche
            sources = list(self.collection.distinct("source"))
            
            return {
                "total_documents": count,
                "content_types": content_types,
                "sources": sources,
                "collection_name": self.collection_name,
                "db_name": self.db_name
            }
        except Exception as e:
            logger.error(f"Errore recupero info collezione MongoDB: {e}")
            return {"error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """
        Verifica lo stato di salute della connessione a MongoDB.
        
        Returns:
            Dizionario con informazioni sullo stato di salute
        """
        try:
            connection_ok = self.verify_connection()
            
            if connection_ok:
                collection_info = self.get_collection_info()
                sample_docs = []
                
                # Recupera un campione di documenti per tipo
                for content_type in ["text", "image", "table"]:
                    sample = list(self.collection.find({"content_type": content_type}).limit(1))
                    if sample:
                        doc = sample[0]
                        doc["_id"] = str(doc["_id"])
                        
                        # Per evitare di restituire dati binari enormi
                        if content_type == "image" and "image_data" in doc:
                            doc["image_data"] = f"{doc['image_data'][:30]}... (truncated)"
                        
                        sample_docs.append(doc)
                
                return {
                    "connection": True,
                    "collection_info": collection_info,
                    "sample_documents": sample_docs,
                    "uri": self.uri.split("@")[-1]  # Nascondi credenziali
                }
            else:
                return {"connection": False}
        except Exception as e:
            logger.error(f"Errore health check MongoDB: {e}")
            return {"connection": False, "error": str(e)}

# Singleton per uso globale
mongodb_manager = MongoManager()
