"""
Qwen2.5-VL Model Implementation
"""

import os
import torch
from typing import Optional
from .model_loader import BaseModel


class QwenModel(BaseModel):
    """
    Qwen2.5-VL Model wrapper
    """
    
    def __init__(self, model_path: str, device: str = "cuda", 
                 min_pixels: int = 256*28*28, max_pixels: int = 1280*28*28, **kwargs):
        """
        Initialize Qwen2.5-VL model.
        
        Args:
            model_path: Path to Qwen2.5-VL model
            device: Device to run on
            min_pixels: Minimum image pixels
            max_pixels: Maximum image pixels
        """
        super().__init__(model_path, device, **kwargs)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.load_model()
    
    def load_model(self):
        """Load Qwen2.5-VL model and processor"""
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            from qwen_vl_utils import process_vision_info
            
            print(f"Loading Qwen2.5-VL model from {self.model_path}...")
            
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype="auto",
                device_map="auto"
            )
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels
            )
            
            self.process_vision_info = process_vision_info
            
            print("Qwen2.5-VL model loaded successfully!")
            
        except ImportError as e:
            raise ImportError(
                "Failed to import Qwen dependencies. Please install: "
                "pip install qwen-vl-utils\n"
                f"Error: {e}"
            )
    
    def generate(self, image_path: Optional[str], query: str, 
                max_new_tokens: int = 512, **kwargs) -> str:
        """
        Generate response using Qwen2.5-VL.
        
        Args:
            image_path: Path to image file (can be None)
            query: Text query
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        # Construct messages
        content = []
        if image_path:
            content.append({"type": "image", "image": image_path})
        content.append({"type": "text", "text": query})
        
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        # Apply chat template
        text_input = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Process vision info
        image_inputs, video_inputs = self.process_vision_info(messages)
        
        # Prepare inputs
        inputs = self.processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        # Generate
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
        
        # Decode output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return output_text

