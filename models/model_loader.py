"""
Unified Model Loader for different LVLMs

This module provides a unified interface for loading and using different Vision-Language Models.
"""

import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseModel(ABC):
    """
    Abstract base class for all Vision-Language Models
    """
    
    def __init__(self, model_path: str, device: str = "cuda", **kwargs):
        """
        Initialize the model.
        
        Args:
            model_path: Path to model checkpoint or model name
            device: Device to run model on (cuda/cpu)
            **kwargs: Additional model-specific arguments
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None
        
    @abstractmethod
    def load_model(self):
        """Load the model and processor"""
        pass
    
    @abstractmethod
    def generate(self, image_path: Optional[str], query: str, **kwargs) -> str:
        """
        Generate response for given image and query.
        
        Args:
            image_path: Path to image file (can be None for text-only)
            query: Text query/prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        pass
    
    def __call__(self, image_path: Optional[str], query: str, **kwargs) -> str:
        """Shorthand for generate()"""
        return self.generate(image_path, query, **kwargs)


class ModelLoader:
    """
    Factory class for loading different models
    """
    
    @staticmethod
    def load_model(model_type: str, model_path: str, **kwargs) -> BaseModel:
        """
        Load a model based on model type.
        
        Args:
            model_type: Type of model ('qwen', 'deepseek', 'llava', 'glm', 'gpt', 'claude', etc.)
            model_path: Path to model or API endpoint
            **kwargs: Additional model-specific arguments
            
        Returns:
            Loaded model instance
        """
        model_type = model_type.lower()
        
        if model_type in ['qwen', 'qwen2.5-vl', 'qwen2_5_vl']:
            from .qwen_model import QwenModel
            return QwenModel(model_path, **kwargs)
        
        elif model_type in ['deepseek', 'deepseek-vl']:
            from .deepseek_model import DeepSeekModel
            return DeepSeekModel(model_path, **kwargs)
        
        elif model_type in ['llava', 'llava-1.5', 'llava-next']:
            from .llava_model import LLaVAModel
            return LLaVAModel(model_path, **kwargs)
        
        elif model_type in ['glm', 'glm-4v', 'glm4v']:
            from .glm_model import GLMModel
            return GLMModel(model_path, **kwargs)
        
        elif model_type in ['instructblip', 'instructblip-vicuna']:
            from .instructblip_model import InstructBLIPModel
            return InstructBLIPModel(model_path, **kwargs)
        
        elif model_type in ['janus', 'janus-pro']:
            from .janus_model import JanusModel
            return JanusModel(model_path, **kwargs)
        
        elif model_type in ['gpt', 'gpt-4', 'gpt-4o', 'gpt-4o-mini', 'claude', 'gemini']:
            from .api_model import APIModel
            return APIModel(model_type, **kwargs)
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    @staticmethod
    def list_supported_models():
        """List all supported model types"""
        return [
            'qwen', 'qwen2.5-vl',
            'deepseek', 'deepseek-vl',
            'llava', 'llava-1.5', 'llava-next',
            'glm', 'glm-4v',
            'instructblip', 'instructblip-vicuna',
            'janus', 'janus-pro',
            'gpt-4', 'gpt-4o', 'gpt-4o-mini',
            'claude', 'gemini'
        ]

