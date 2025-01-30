from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

class BaseRetriever(ABC):
    """
    Abstract base class for all retrievers.
    Retrievers are responsible for fetching relevant context based on queries.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the retriever with optional configuration.
        
        Args:
            config: Dictionary containing retriever-specific configuration
        """
        self.config = config or {}
    
    @abstractmethod
    async def retrieve(self, query: str) -> List[str]:
        """
        Retrieve relevant context based on the query.
        
        Args:
            query: The query to retrieve context for
            
        Returns:
            List of relevant context strings
            
        Raises:
            NotImplementedError: Must be implemented by concrete retrievers
        """
        pass
    
    async def initialize(self) -> None:
        """
        Optional initialization method for retrievers that need setup.
        For example, establishing database connections or loading models.
        """
        pass
    
    async def cleanup(self) -> None:
        """
        Optional cleanup method for retrievers to release resources.
        For example, closing database connections.
        """
        pass
    
    def validate_config(self) -> bool:
        """
        Validate the retriever configuration.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        return True
