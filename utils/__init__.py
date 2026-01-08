"""
Utility functions for the copyright compliance framework.
"""

from .config_loader import load_config, get_config_value
from .path_helper import (
    get_text_file_path,
    get_image_file_path,
    get_all_image_paths,
    verify_file_paths
)

__all__ = [
    'load_config', 
    'get_config_value',
    'get_text_file_path',
    'get_image_file_path',
    'get_all_image_paths',
    'verify_file_paths'
]



