import os
import logging
import re
from typing import List, Dict, Any, Optional
from chromadb import HttpClient
from openai import OpenAI
from .base import BaseRetriever

logger = logging.getLogger("flare_proxy")

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))

class ChromaRetriever(BaseRetriever):
    """
    Retriever implementation for ChromaDB, which is used by AnythingLLM.
    This implementation directly accesses stored documents without performing
    embedding-based similarity search, as the documents are already processed
    and stored by AnythingLLM.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ChromaDB retriever.
        
        Args:
            config: Configuration dictionary containing ChromaDB settings
        """
        super().__init__(config)
        self.client = None
        self.endpoint = os.getenv("CHROMA_ENDPOINT", "http://localhost:1523")
        self.embedding_model = os.getenv("EMBEDDING_MODEL_PREF", "text-embedding-3-large")
        
    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text using OpenAI's API.
        
        Args:
            text: Text to get embedding for
            
        Returns:
            List of embedding values
        """
        try:
            response = openai_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            raise
        
    async def initialize(self) -> None:
        """
        Initialize the ChromaDB client connection.
        """
        try:
            self.client = HttpClient(host=self.endpoint)
            # Test the connection
            self.client.heartbeat()
        except Exception as e:
            raise Exception(f"Failed to initialize ChromaDB client: {str(e)}")
    
    def _clean_html(self, text: str) -> str:
        """
        Clean HTML/XML content from text.
        
        Args:
            text: Text that may contain HTML/XML tags
            
        Returns:
            Cleaned text with tags removed
        """
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        # Convert to lowercase and normalize German characters
        text = text.lower()
        text = text.replace('ß', 'ss')
        text = text.replace('ä', 'a')
        text = text.replace('ö', 'o')
        text = text.replace('ü', 'u')
        return text
    
    async def retrieve(self, query: str) -> List[str]:
        """
        Retrieve relevant documents from ChromaDB based on the query.
        
        Args:
            query: The query to retrieve context for
            
        Returns:
            List of relevant document contents
        """
        if not self.client:
            await self.initialize()
            
        try:
            # Get all collection names
            collection_names = self.client.list_collections()
            logger.debug(f"Found collections: {collection_names}")
            
            all_results = []
            for name in collection_names:
                try:
                    # Get the collection by name
                    collection = self.client.get_collection(name=name)
                    logger.debug(f"Querying collection: {name}")
                    
                    # Generate query embedding
                    query_embedding = self._get_embedding(query)
                    logger.debug(f"Generated embedding for query (dim: {len(query_embedding)})")
                    
                    # Use embedding for similarity search with configured parameters
                    n_results = int(os.getenv("N_RESULTS", "3"))
                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=n_results,
                        include=['documents', 'distances']
                    )
                    
                    # Process documents based on similarity
                    filtered_docs = []
                    if results.get('documents') and results['documents'][0]:
                        logger.debug(f"Found {len(results['documents'][0])} matches")
                        
                        # Process each document
                        for doc, dist in zip(results['documents'][0], results['distances'][0]):
                            # Skip documents that are too dissimilar based on configured threshold
                            distance_threshold = float(os.getenv("DISTANCE_THRESHOLD", "0.5"))
                            if dist > distance_threshold:
                                logger.debug(f"Skipping document with distance {dist:.3f} (threshold: {distance_threshold})")
                                continue
                            
                            # Extract content
                            content = None
                            
                            # Try to parse as JSON
                            if isinstance(doc, str):
                                try:
                                    doc_json = json.loads(doc)
                                    if 'pageContent' in doc_json:
                                        content = doc_json['pageContent'].strip()
                                except:
                                    pass
                            
                            # Fallback to raw content
                            if not content:
                                content = self._clean_html(str(doc)).strip()
                            
                            # Add if we got valid content
                            if content:
                                logger.debug(f"Found relevant document (distance: {dist:.3f})")
                                filtered_docs.append(content)
                    
                    if filtered_docs:
                        all_results.extend(filtered_docs)
                        logger.debug(f"Found {len(filtered_docs)} relevant results in collection {name}")
                    else:
                        logger.debug(f"No relevant documents found in collection {name}")
                        
                except Exception as e:
                    logger.error(f"Error querying collection {name}: {str(e)}")
                    continue
            
            # Return top N most relevant results across all collections
            max_results = int(os.getenv("MAX_RESULTS", "5"))
            logger.debug(f"Returning top {max_results} results from {len(all_results)} total results")
            return all_results[:max_results] if all_results else []
            
        except Exception as e:
            raise Exception(f"ChromaDB retrieval error: {str(e)}")
    
    async def cleanup(self) -> None:
        """
        Clean up ChromaDB client resources.
        """
        if self.client:
            # ChromaDB HTTP client doesn't require explicit cleanup
            self.client = None
    
    def validate_config(self) -> bool:
        """
        Validate the ChromaDB configuration.
        
        Returns:
            bool: True if configuration is valid
        """
        return bool(self.endpoint)
