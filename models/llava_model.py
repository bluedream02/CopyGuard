"""
LLaVA Model Implementation
"""

import sys
import os
import re
import torch
from pathlib import Path
from typing import Optional
from PIL import Image
from .model_loader import BaseModel


class LLaVAModel(BaseModel):
    """
    LLaVA Model wrapper (supports LLaVA-1.5 and LLaVA-Next)
    """
    
    def __init__(self, model_path: str, device: str = "cuda",
                 llava_repo_path: Optional[str] = None,
                 conv_mode: Optional[str] = None,
                 temperature: float = 0.2,
                 **kwargs):
        """
        Initialize LLaVA model.
        
        Args:
            model_path: Path to LLaVA model
            device: Device to run on
            llava_repo_path: Path to LLaVA repository (required for import)
            conv_mode: Conversation mode (auto-detected if None)
            temperature: Sampling temperature
        """
        super().__init__(model_path, device, **kwargs)
        self.llava_repo_path = llava_repo_path
        self.conv_mode = conv_mode
        self.temperature = temperature
        self._setup_llava_path()
        self.load_model()
    
    def _setup_llava_path(self):
        """Add LLaVA repository to Python path"""
        if self.llava_repo_path:
            if self.llava_repo_path not in sys.path:
                sys.path.append(self.llava_repo_path)
        else:
            # Try to find LLaVA in common locations
            possible_paths = [
                os.path.expanduser("~/LLaVA"),
                os.path.expanduser("~/VLMs/LLaVA"),
                "/workspace/LLaVA"
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    sys.path.append(path)
                    self.llava_repo_path = path
                    break
            
            if not self.llava_repo_path:
                print("Warning: LLaVA repository path not found. "
                      "Please set PYTHONPATH or provide llava_repo_path parameter.")
    
    def load_model(self):
        """Load LLaVA model"""
        try:
            from llava.constants import (
                IMAGE_TOKEN_INDEX,
                DEFAULT_IMAGE_TOKEN,
                DEFAULT_IM_START_TOKEN,
                DEFAULT_IM_END_TOKEN,
                IMAGE_PLACEHOLDER,
            )
            from llava.conversation import conv_templates, SeparatorStyle
            from llava.model.builder import load_pretrained_model
            from llava.utils import disable_torch_init
            from llava.mm_utils import (
                process_images,
                tokenizer_image_token,
                get_model_name_from_path,
            )
            
            print(f"Loading LLaVA model from {self.model_path}...")
            
            # Store constants
            self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
            self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
            self.DEFAULT_IM_START_TOKEN = DEFAULT_IM_START_TOKEN
            self.DEFAULT_IM_END_TOKEN = DEFAULT_IM_END_TOKEN
            self.IMAGE_PLACEHOLDER = IMAGE_PLACEHOLDER
            self.conv_templates = conv_templates
            self.tokenizer_image_token = tokenizer_image_token
            self.process_images_fn = process_images
            
            # Load model
            disable_torch_init()
            
            model_name = get_model_name_from_path(self.model_path)
            self.tokenizer, self.model, self.image_processor, self.context_len = \
                load_pretrained_model(self.model_path, None, model_name)
            
            # Auto-detect conversation mode if not provided
            if not self.conv_mode:
                if "llama-2" in model_name.lower():
                    self.conv_mode = "llava_llama_2"
                elif "mistral" in model_name.lower():
                    self.conv_mode = "mistral_instruct"
                elif "v1.6-34b" in model_name.lower():
                    self.conv_mode = "chatml_direct"
                elif "v1" in model_name.lower():
                    self.conv_mode = "llava_v1"
                elif "mpt" in model_name.lower():
                    self.conv_mode = "mpt"
                else:
                    self.conv_mode = "llava_v0"
            
            print(f"LLaVA model loaded successfully! Using conv_mode: {self.conv_mode}")
            
        except ImportError as e:
            raise ImportError(
                "Failed to import LLaVA dependencies. Please install LLaVA:\n"
                "git clone https://github.com/haotian-liu/LLaVA.git\n"
                "export PYTHONPATH=/path/to/LLaVA:$PYTHONPATH\n"
                f"Error: {e}"
            )
    
    def generate(self, image_path: Optional[str], query: str,
                max_new_tokens: int = 512, num_beams: int = 1,
                top_p: Optional[float] = None, **kwargs) -> str:
        """
        Generate response using LLaVA.
        
        Args:
            image_path: Path to image file
            query: Text query
            max_new_tokens: Maximum tokens to generate
            num_beams: Number of beams for beam search
            top_p: Top-p sampling parameter
            
        Returns:
            Generated text
        """
        if not image_path:
            raise ValueError("LLaVA requires an image input")
        
        # Prepare query with image tokens
        qs = query
        image_token_se = self.DEFAULT_IM_START_TOKEN + self.DEFAULT_IMAGE_TOKEN + self.DEFAULT_IM_END_TOKEN
        
        if self.IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(self.IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(self.IMAGE_PLACEHOLDER, self.DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = self.DEFAULT_IMAGE_TOKEN + "\n" + qs
        
        # Prepare conversation
        conv = self.conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # Load and process image
        image = Image.open(image_path).convert("RGB")
        images_tensor = self.process_images_fn(
            [image],
            self.image_processor,
            self.model.config
        ).to(self.model.device, dtype=torch.float16)
        
        # Tokenize
        input_ids = (
            self.tokenizer_image_token(prompt, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(self.device)
        )
        
        # Generate
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=[image.size],
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                **kwargs
            )
        
        # Decode
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        return outputs

