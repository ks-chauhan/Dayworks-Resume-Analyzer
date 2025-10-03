import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Optional, Tuple
import logging
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages ChromaDB vector store operations."""
    
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "resume_embeddings"):
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Create persistence directory if it doesn't exist
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client with persistence
            self.client = chromadb.PersistentClient(path=str(self.persist_directory))
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"Loaded existing collection: {self.collection_name}")
            except:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Resume embeddings for screening"}
                )
                logger.info(f"Created new collection: {self.collection_name}")
        
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def add_documents(self, documents: List[str], embeddings: List[List[float]], 
                     metadata: List[Dict], ids: Optional[List[str]] = None) -> List[str]:
        """Add documents with embeddings to the vector store."""
        try:
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in documents]
            
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadata,
                ids=ids
            )
            
            logger.info(f"Added {len(documents)} documents to vector store")
            return ids
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise
    
    def similarity_search(self, query_embedding: List[float], 
                         top_k: int = 5, 
                         metadata_filter: Optional[Dict] = None) -> Tuple[List[str], List[float], List[Dict]]:
        """Perform similarity search and return documents with scores."""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=metadata_filter
            )
            
            documents = results['documents'][0] if results['documents'] else []
            distances = results['distances'][0] if results['distances'] else []
            metadatas = results['metadatas'][0] if results['metadatas'] else []
            
            # Convert distances to similarity scores (ChromaDB returns L2 distances)
            similarity_scores = [1 / (1 + distance) for distance in distances]
            
            return documents, similarity_scores, metadatas
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            raise
    
    def batch_similarity_search(self, query_embeddings: List[List[float]], 
                               top_k: int = 5) -> List[Tuple[List[str], List[float], List[Dict]]]:
        """Perform batch similarity search for multiple queries."""
        try:
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=top_k
            )
            
            batch_results = []
            for i in range(len(query_embeddings)):
                documents = results['documents'][i] if i < len(results['documents']) else []
                distances = results['distances'][i] if i < len(results['distances']) else []
                metadatas = results['metadatas'][i] if i < len(results['metadatas']) else []
                
                similarity_scores = [1 / (1 + distance) for distance in distances]
                batch_results.append((documents, similarity_scores, metadatas))
            
            return batch_results
            
        except Exception as e:
            logger.error(f"Error performing batch similarity search: {e}")
            raise
    
    def get_collection_count(self) -> int:
        """Get the number of documents in the collection."""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Error getting collection count: {e}")
            return 0
    
    def clear_collection(self):
        """Clear all documents from the collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Resume embeddings for screening"}
            )
            logger.info("Collection cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise
