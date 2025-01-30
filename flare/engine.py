import logging
from typing import List, Dict, Any, Optional, Tuple
import asyncio
from llm.anthropic_client import AnthropicClient
from retrievers.chroma import ChromaRetriever

logger = logging.getLogger("flare_proxy")

class FlareProcessor:
    def __init__(self, confidence_threshold: float = 0.7, max_retrieval_rounds: int = 3, traffic_logger = None):
        """
        Initialize the FLARE processor.
        
        Args:
            confidence_threshold: Threshold below which additional context is retrieved
            max_retrieval_rounds: Maximum number of retrieval rounds to perform
            traffic_logger: Optional logger for complete traffic logging
        """
        self.confidence_threshold = confidence_threshold
        self.max_retrieval_rounds = max_retrieval_rounds
        self.llm_client = AnthropicClient()
        self.retriever = ChromaRetriever()
        self.traffic_logger = traffic_logger
        
    async def process_message(self, messages: List[Dict[str, str]], parameters: Dict[str, Any], traffic_logger = None) -> Dict[str, Any]:
        """
        Process a message through the FLARE algorithm.
        
        Args:
            messages: List of conversation messages
            parameters: Additional parameters for the LLM
            
        Returns:
            Dict containing the response and metadata
        """
        try:
            current_round = 0
            last_message = messages[-1]["content"]
            additional_context = []
            
            # Add confidence scoring instruction once
            scoring_message = {
                "role": "system",
                "content": "Please provide your response followed by your confidence level (0-1) on a new line at the end. Format your confidence as: [confidence: X.XX]"
            }
            messages.insert(-1, scoring_message)
            
            # Process message with multiple retrieval rounds if needed
            for round in range(self.max_retrieval_rounds):
                # Get prediction and confidence score
                prediction, confidence = await self.llm_client.get_completion(messages, parameters)
                logger.debug(f"Round {round + 1}: confidence {confidence}")
                
                # Log complete LLM response in debug mode
                if self.traffic_logger:
                    response_text = prediction.content[0].text if prediction.content else ""
                    self.traffic_logger.log_response("Claude", f"Confidence: {confidence}\nResponse:\n{response_text}")
                
                # If confidence is high enough, return the response
                if confidence >= self.confidence_threshold:
                    logger.debug(f"Confidence {confidence} meets threshold {self.confidence_threshold}")
                    return prediction
                
                # Try to get additional context
                try:
                    logger.debug(f"Confidence {confidence} below threshold {self.confidence_threshold}, retrieving context")
                    new_context = await self.retriever.retrieve(last_message)
                    
                    # Log ChromaDB response in debug mode
                    if self.traffic_logger:
                        context_text = "\n".join(new_context) if new_context else "No results"
                        self.traffic_logger.log_response("ChromaDB", f"Query: {last_message}\nResults:\n{context_text}")
                    
                    # Check if we got any new context
                    if not new_context:
                        logger.debug("No new context found, returning current prediction")
                        return prediction
                    
                    # Filter out already used contexts
                    unique_contexts = [ctx for ctx in new_context if ctx not in additional_context]
                    if not unique_contexts:
                        logger.debug("No new unique context found, returning current prediction")
                        return prediction
                    
                    # Add context to messages
                    additional_context.extend(unique_contexts)
                    context_text = "\n".join(unique_contexts)
                    context_message = {
                        "role": "system",
                        "content": f"Additional context:\n{context_text}"
                    }
                    messages.insert(-1, context_message)
                    logger.debug(f"Added {len(unique_contexts)} new context items")
                    
                except Exception as e:
                    logger.warning(f"Failed to retrieve additional context: {str(e)}")
                    return prediction
            
            # Max rounds reached
            logger.debug(f"Max rounds ({self.max_retrieval_rounds}) reached")
            return prediction
            
        except Exception as e:
            raise Exception(f"FLARE processing error: {str(e)}")
