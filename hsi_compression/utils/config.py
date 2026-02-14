"""Configuration management for HSI compression training and evaluation."""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field, asdict
from omegaconf import OmegaConf, DictConfig


def load_config(config_path: str, overrides: Optional[list] = None) -> DictConfig:
    """
    Load configuration from YAML file with optional CLI overrides.
    
    Args:
        config_path: Path to config YAML file
        overrides: List of override strings (e.g., ["training.lr=0.001", "epochs=50"])
    
    Returns:
        OmegaConf DictConfig object
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load the main config
    config = OmegaConf.load(config_path)
    
    # If config has defaults, load and merge the base config
    if 'defaults' in config:
        defaults = config.get('defaults', []) or []
        # Remove defaults from config after processing
        if 'defaults' in config:
            del config['defaults']
        
        for default in defaults:
            if isinstance(default, str) and default.startswith('/'):
                # Try to load from configs directory
                # default is like '/default', so remove the leading /
                default_name = default.lstrip('/')
                # Get the configs directory (parent of models/, parent of tcn_lossless.yaml's parent)
                config_dir = os.path.dirname(os.path.abspath(config_path))  # .../configs/models
                configs_dir = os.path.dirname(config_dir)  # .../configs
                base_path = os.path.join(configs_dir, default_name + '.yaml')
                
                if os.path.exists(base_path):
                    base_config = OmegaConf.load(base_path)
                    config = OmegaConf.merge(base_config, config)
    
    # Apply CLI overrides if provided
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
    
    return config


def save_config(config: DictConfig, save_path: str) -> None:
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    OmegaConf.save(config, save_path)


def config_to_dict(config: DictConfig) -> Dict[str, Any]:
    """Convert OmegaConf DictConfig to plain dictionary."""
    return OmegaConf.to_container(config, resolve=True)


def print_config(config: DictConfig) -> None:
    """Pretty print configuration."""
    print(OmegaConf.to_yaml(config))
