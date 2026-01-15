"""
Models Module for CopyGuard

This module provides unified interfaces for various LVLMs (Large Vision-Language Models).
Supports both local models and API-based models.
"""

from .model_loader import ModelLoader, BaseModel
from .generate_responses import generate_responses
from .generate_responses_with_defense import generate_responses_with_defense

__all__ = [
    'ModelLoader',
    'BaseModel',
    'generate_responses',
    'generate_responses_with_defense'
]

