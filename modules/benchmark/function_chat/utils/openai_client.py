"""
OpenAI-compatible client for local model endpoints.

This module provides a unified interface for accessing both OpenAI API
and local model endpoints using OpenAI-compatible format.
"""

import json
from typing import Dict, List, Optional, Any
import openai
from openai import OpenAI


class UniversalOpenAIClient:
    """
    Universal OpenAI client that supports both OpenAI API and local endpoints.
    
    This client maintains OpenAI object interface while allowing flexibility
    to connect to various local serving models by changing only the endpoint.
    """
    
    def __init__(self, 
                 model_name: str,
                 api_key: str,
                 endpoint: str = "https://api.openai.com/v1",
                 temperature: float = 0.0,
                 max_tokens: int = 1024):
        """
        Initialize the universal OpenAI client.
        
        Args:
            model_name: Name of the model (e.g., "gpt-4", "local-llama-7b")
            api_key: API key for authentication
            endpoint: API endpoint URL (default: OpenAI, can be local endpoint)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.model_name = model_name
        self.api_key = api_key
        self.endpoint = endpoint
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize OpenAI client with custom endpoint
        self.client = OpenAI(
            api_key=api_key,
            base_url=endpoint
        )
    
    def predict(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction request compatible with original FunctionChat-Bench format.
        
        Args:
            request_data: Dictionary containing 'messages', 'temperature', etc.
            
        Returns:
            Response in FunctionChat-Bench expected format
        """
        try:
            messages = request_data.get('messages', [])
            temperature = request_data.get('temperature', self.temperature)
            tools = request_data.get('tools', None)
            tool_choice = request_data.get('tool_choice', None)
            
            # Prepare request parameters
            params = {
                'model': self.model_name,
                'messages': messages,
                'temperature': temperature,
                'max_tokens': self.max_tokens
            }
            
            # Add tools if provided
            if tools:
                params['tools'] = tools
            if tool_choice:
                params['tool_choice'] = tool_choice
            
            # Make API call
            response = self.client.chat.completions.create(**params)
            
            # Convert to FunctionChat-Bench expected format
            return self._format_response(response)
            
        except Exception as e:
            # Return error response in expected format
            return {
                "id": "error",
                "choices": [{
                    "finish_reason": "error",
                    "index": 0,
                    "message": {
                        "content": f"Error: {str(e)}",
                        "role": "assistant"
                    },
                    "function_call": None,
                    "tool_calls": None,
                }],
                "error": str(e)
            }
    
    def _format_response(self, response) -> Dict[str, Any]:
        """
        Format OpenAI response to FunctionChat-Bench expected format.
        
        Args:
            response: OpenAI API response
            
        Returns:
            Formatted response dictionary
        """
        choice = response.choices[0]
        message = choice.message
        
        # Extract tool calls if present
        tool_calls = None
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_calls = []
            for tool_call in message.tool_calls:
                tool_calls.append({
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                })
        
        return {
            "id": response.id,
            "choices": [{
                "finish_reason": choice.finish_reason,
                "index": choice.index,
                "message": {
                    "content": message.content,
                    "role": message.role,
                },
                "function_call": None,  # Legacy support
                "tool_calls": tool_calls,
            }]
        }


class LocalModelClient(UniversalOpenAIClient):
    """
    Specialized client for local model endpoints.
    
    This class can be extended for specific local model configurations
    or custom authentication methods.
    """
    
    def __init__(self, 
                 model_name: str,
                 endpoint: str,
                 api_key: str = "local-key",
                 **kwargs):
        """
        Initialize local model client.
        
        Args:
            model_name: Local model name
            endpoint: Local endpoint URL (e.g., "http://localhost:8000/v1")
            api_key: API key (can be dummy for local models)
            **kwargs: Additional parameters
        """
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            endpoint=endpoint,
            **kwargs
        )


def create_openai_client(config: Dict[str, Any]) -> UniversalOpenAIClient:
    """
    Factory function to create OpenAI client from configuration.
    
    Args:
        config: Configuration dictionary with llm_model_name, llm_api_key, llm_endpoint
        
    Returns:
        Configured OpenAI client
    """
    model_name = config['llm_model_name']
    api_key = config['llm_api_key']
    endpoint = config.get('llm_endpoint', 'https://api.openai.com/v1')
    
    # Determine if it's a local endpoint
    if endpoint.startswith('http://localhost') or endpoint.startswith('http://127.0.0.1'):
        return LocalModelClient(
            model_name=model_name,
            endpoint=endpoint,
            api_key=api_key
        )
    else:
        return UniversalOpenAIClient(
            model_name=model_name,
            api_key=api_key,
            endpoint=endpoint
        )