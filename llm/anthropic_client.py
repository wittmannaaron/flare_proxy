import os
from typing import List, Dict, Any, Tuple
from anthropic import Anthropic

class AnthropicClient:
    """
    Client for interacting with Anthropic's Claude API.
    Handles message formatting, API calls, and confidence scoring.
    """
    
    def __init__(self):
        """
        Initialize the Anthropic client with API key and model configuration.
        """
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
            
        self.model = os.getenv("MODEL_NAME")
        if not self.model:
            raise ValueError("MODEL_NAME environment variable is required")
        self.client = Anthropic(api_key=self.api_key)
        
    async def get_completion(
        self,
        messages: List[Dict[str, str]],
        parameters: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float]:
        """
        Get a completion from Claude with confidence scoring.
        
        Args:
            messages: List of conversation messages
            parameters: Additional parameters for the API call
            
        Returns:
            Tuple of (completion_response, confidence_score)
        """
        try:
            # Convert messages to Anthropic format
            formatted_messages = self._format_messages(messages)
            
            # Make API call
            response = self.client.messages.create(
                model=self.model,
                messages=formatted_messages,
                temperature=parameters.get("temperature"),  # Use value from parameters without default
                max_tokens=parameters.get("max_tokens")     # Use value from parameters without default
            )
            
            # TODO: Implement confidence scoring
            # For now, return a placeholder confidence score
            confidence_score = 0.5
            
            return response, confidence_score
            
        except Exception as e:
            raise Exception(f"Anthropic API error: {str(e)}")
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Format messages for the Anthropic API.
        
        Args:
            messages: List of messages in OpenAI format
            
        Returns:
            List of messages in Anthropic format
        """
        formatted = []
        for msg in messages:
            # Convert OpenAI message format to Anthropic format
            formatted.append({
                "role": "assistant" if msg["role"] == "assistant" else "user",
                "content": msg["content"]
            })
        return formatted
