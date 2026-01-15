"""
GLM-4V Model Implementation
"""

import os
import torch
from typing import Optional
from PIL import Image
from .model_loader import BaseModel


class GLMModel(BaseModel):
    """
    GLM-4V Model wrapper
    """
    
    def __init__(self, model_path: str, device: str = "cuda", **kwargs):
        """
        Initialize GLM-4V model.
        
        Args:
            model_path: Path to GLM-4V model
            device: Device to run on
        """
        super().__init__(model_path, device, **kwargs)
        self.load_model()
    
    def load_model(self):
        """Load GLM-4V model and tokenizer"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            print(f"Loading GLM-4V model from {self.model_path}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).to(self.device).eval()
            
            print("GLM-4V model loaded successfully!")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load GLM-4V model: {e}")
    
    def generate(self, image_path: Optional[str], query: str,
                max_length: int = 2048, do_sample: bool = True,
                top_k: int = 1, **kwargs) -> str:
        """
        Generate response using GLM-4V.
        
        Args:
            image_path: Path to image file (can be None)
            query: Text query
            max_length: Maximum sequence length
            do_sample: Whether to use sampling
            top_k: Top-k sampling parameter
            
        Returns:
            Generated text
        """
        # Load image if provided
        image = None
        if image_path and os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
        
        # Construct messages
        messages = [{"role": "user", "image": image, "content": query}]
        
        # Apply chat template
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        ).to(self.device)
        
        # Generate
        gen_kwargs = {
            "max_length": max_length,
            "do_sample": do_sample,
            "top_k": top_k,
            **kwargs
        }
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            # Only get newly generated tokens
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return output_text

