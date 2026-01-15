"""
DeepSeek-VL Model Implementation
"""

import torch
from typing import Optional
from .model_loader import BaseModel


class DeepSeekModel(BaseModel):
    """
    DeepSeek-VL Model wrapper
    """
    
    def __init__(self, model_path: str, device: str = "cuda", **kwargs):
        """
        Initialize DeepSeek-VL model.
        
        Args:
            model_path: Path to DeepSeek-VL model
            device: Device to run on
        """
        super().__init__(model_path, device, **kwargs)
        self.load_model()
    
    def load_model(self):
        """Load DeepSeek-VL model and processor"""
        try:
            from transformers import AutoModelForCausalLM
            from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
            from deepseek_vl.utils.io import load_pil_images
            
            print(f"Loading DeepSeek-VL model from {self.model_path}...")
            
            self.vl_chat_processor = VLChatProcessor.from_pretrained(self.model_path)
            self.tokenizer = self.vl_chat_processor.tokenizer
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            self.model = self.model.to(torch.bfloat16).to(self.device).eval()
            
            self.load_pil_images = load_pil_images
            
            print("DeepSeek-VL model loaded successfully!")
            
        except ImportError as e:
            raise ImportError(
                "Failed to import DeepSeek dependencies. Please install DeepSeek-VL:\n"
                "git clone https://github.com/deepseek-ai/DeepSeek-VL.git\n"
                "cd DeepSeek-VL && pip install -e .\n"
                f"Error: {e}"
            )
    
    def generate(self, image_path: Optional[str], query: str,
                max_new_tokens: int = 512, **kwargs) -> str:
        """
        Generate response using DeepSeek-VL.
        
        Args:
            image_path: Path to image file (can be None)
            query: Text query
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        # Construct conversation
        conversation = [
            {
                "role": "User",
                "content": f"<image_placeholder>{query}" if image_path else query,
                "images": [image_path] if image_path else []
            },
            {
                "role": "Assistant",
                "content": ""
            }
        ]
        
        # Load images
        pil_images = self.load_pil_images(conversation)
        
        # Prepare inputs
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True
        ).to(self.device)
        
        # Prepare embeddings
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
        
        # Generate
        with torch.inference_mode():
            outputs = self.model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                **kwargs
            )
        
        # Decode
        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        
        return answer

