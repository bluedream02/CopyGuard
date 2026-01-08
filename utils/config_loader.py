"""
Configuration loader utility.

Loads configuration from YAML file or environment variables.
"""

import os
import yaml
from typing import Dict, Optional


def load_config(config_path: Optional[str] = None) -> Dict:
    """
    Load configuration from YAML file or use defaults.
    Environment variables take precedence over config file values.
    
    Args:
        config_path: Path to config YAML file
        
    Returns:
        Dictionary containing configuration
    """
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "config",
            "config.yaml"
        )
    
    default_config = {
        "paths": {
            "dataset_dir": "dataset",
            "results_dir": "results",
            "models_dir": "models"
        },
        "api_keys": {
            "openai_api_key": "",
            "openai_api_base": "",
            "archives_api_key": "",
            "serper_api_key": "",
            "deepseek_api_key": "",
            "deepseek_api_base": ""
        },
        "models": {
            "embedding_model": "all-MiniLM-L6-v2",
            "bert_model": "roberta-large"
        }
    }
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                if user_config:
                    # Deep merge with defaults
                    _deep_merge(default_config, user_config)
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            print("Using default configuration")
    
    # Environment variables override config file values
    env_key_mapping = {
        "OPENAI_API_KEY": ("api_keys", "openai_api_key"),
        "OPENAI_API_BASE": ("api_keys", "openai_api_base"),
        "ARCHIVES_API_KEY": ("api_keys", "archives_api_key"),
        "SERPER_API_KEY": ("api_keys", "serper_api_key"),
        "DEEPSEEK_API_KEY": ("api_keys", "deepseek_api_key"),
        "DEEPSEEK_API_BASE": ("api_keys", "deepseek_api_base"),
    }
    
    for env_var, (section, key) in env_key_mapping.items():
        env_value = os.getenv(env_var)
        if env_value:
            default_config[section][key] = env_value
    
    return default_config


def _deep_merge(base_dict: Dict, update_dict: Dict) -> None:
    """
    Deep merge update_dict into base_dict.
    
    Args:
        base_dict: Base dictionary to update
        update_dict: Dictionary with updates
    """
    for key, value in update_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            _deep_merge(base_dict[key], value)
        else:
            base_dict[key] = value


def get_config_value(config: Dict, key_path: str, default=None):
    """
    Get configuration value using dot-separated key path.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path (e.g., "paths.dataset_dir")
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    keys = key_path.split('.')
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value



