"""
Janus-Pro Model Implementation
"""

import sys
import os
import torch
from typing import Optional
from PIL import Image
from .model_loader import BaseModel


class JanusModel(BaseModel):
    """
    Janus-Pro Model wrapper
    """
    
    def __init__(self, model_path: str, device: str = "cuda",
                 janus_repo_path: Optional[str] = None, **kwargs):
        """
        Initialize Janus-Pro model.
        
        Args:
            model_path: Path to Janus-Pro model
            device: Device to run on
            janus_repo_path: Path to Janus repository (if needed)
        """
        super().__init__(model_path, device, **kwargs)
        self.janus_repo_path = janus_repo_path
        if janus_repo_path and janus_repo_path not in sys.path:
            sys.path.append(janus_repo_path)
        self.load_model()
    
    def load_model(self):
        """Load Janus-Pro model"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            print(f"Loading Janus-Pro model from {self.model_path}...")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            ).to(self.device).eval()
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            print("Janus-Pro model loaded successfully!")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Janus-Pro model: {e}")
    
    def generate(self, image_path: Optional[str], query: str,
                max_new_tokens: int = 512, **kwargs) -> str:
        """
        Generate response using Janus-Pro.
        
        Args:
            image_path: Path to image file
            query: Text query
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        if not image_path:
            raise ValueError("Janus-Pro requires an image input")
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Prepare conversation
        conversation = [
            {
                "role": "user",
                "content": f"<image>\n{query}"
            }
        ]
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Process inputs (this may vary based on Janus implementation)
        inputs = self.tokenizer(
            text,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
        
        # Decode
        generated_text = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )
        
        return generated_text

