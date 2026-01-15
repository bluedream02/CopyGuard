"""
InstructBLIP Model Implementation
"""

import torch
from typing import Optional
from PIL import Image
from .model_loader import BaseModel


class InstructBLIPModel(BaseModel):
    """
    InstructBLIP Model wrapper
    """
    
    def __init__(self, model_path: str, device: str = "cuda", **kwargs):
        """
        Initialize InstructBLIP model.
        
        Args:
            model_path: Path to InstructBLIP model
            device: Device to run on
        """
        super().__init__(model_path, device, **kwargs)
        self.load_model()
    
    def load_model(self):
        """Load InstructBLIP model and processor"""
        try:
            from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
            
            print(f"Loading InstructBLIP model from {self.model_path}...")
            
            self.model = InstructBlipForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16
            ).to(self.device)
            
            self.processor = InstructBlipProcessor.from_pretrained(self.model_path)
            
            print("InstructBLIP model loaded successfully!")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load InstructBLIP model: {e}")
    
    def generate(self, image_path: Optional[str], query: str,
                max_new_tokens: int = 512, **kwargs) -> str:
        """
        Generate response using InstructBLIP.
        
        Args:
            image_path: Path to image file
            query: Text query/prompt
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        if not image_path:
            raise ValueError("InstructBLIP requires an image input")
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Process inputs
        inputs = self.processor(
            images=image,
            text=query,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
        
        # Decode
        generated_text = self.processor.batch_decode(
            outputs,
            skip_special_tokens=True
        )[0].strip()
        
        return generated_text

