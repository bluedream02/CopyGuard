"""
API-based Model Implementation (GPT, Claude, Gemini, etc.)
"""

import os
import base64
import time
from typing import Optional
from .model_loader import BaseModel


class APIModel(BaseModel):
    """
    Wrapper for API-based models (OpenAI GPT, Claude, Gemini, etc.)
    """
    
    def __init__(self, model_name: str, api_key: Optional[str] = None,
                 api_base: Optional[str] = None, **kwargs):
        """
        Initialize API model.
        
        Args:
            model_name: Name of the model (e.g., 'gpt-4o', 'claude-3-opus', 'gemini-pro')
            api_key: API key (if not provided, reads from environment)
            api_base: API base URL (optional)
        """
        super().__init__(model_name, device="api", **kwargs)
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.load_model()
    
    def load_model(self):
        """Initialize API client"""
        try:
            from openai import OpenAI
            
            # Determine API key from environment if not provided
            if not self.api_key:
                if 'gpt' in self.model_name.lower():
                    self.api_key = os.getenv("OPENAI_API_KEY")
                elif 'claude' in self.model_name.lower():
                    self.api_key = os.getenv("ANTHROPIC_API_KEY")
                elif 'gemini' in self.model_name.lower():
                    self.api_key = os.getenv("GOOGLE_API_KEY")
            
            if not self.api_key:
                raise ValueError(
                    f"API key not found for {self.model_name}. "
                    "Please provide api_key parameter or set environment variable."
                )
            
            # Initialize OpenAI-compatible client
            # (works for GPT, and can be adapted for Claude/Gemini via compatible endpoints)
            client_kwargs = {"api_key": self.api_key}
            if self.api_base:
                client_kwargs["base_url"] = self.api_base
            
            self.client = OpenAI(**client_kwargs)
            
            print(f"API client initialized for {self.model_name}")
            
        except ImportError:
            raise ImportError(
                "OpenAI package not found. Please install: pip install openai"
            )
    
    def generate(self, image_path: Optional[str], query: str,
                temperature: float = 1.0, max_tokens: int = 512,
                retry_delay: float = 1.0, **kwargs) -> str:
        """
        Generate response using API model.
        
        Args:
            image_path: Path to image file (can be None for text-only)
            query: Text query
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            retry_delay: Delay between retries (seconds)
            
        Returns:
            Generated text
        """
        # Prepare messages
        content = [{"type": "text", "text": query}]
        
        # Add image if provided
        if image_path:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
        
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        # Make API call with retry logic
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            # Add delay to avoid rate limiting
            time.sleep(retry_delay)
            
            if response.choices:
                return response.choices[0].message.content
            else:
                return ""
                
        except Exception as e:
            print(f"Error calling API model {self.model_name}: {e}")
            raise

