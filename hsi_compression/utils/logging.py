"""Weights & Biases (WandB) integration for experiment tracking."""

import os
from typing import Dict, Optional, Any
from omegaconf import DictConfig

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class WandBLogger:
    """Logger for tracking experiments with Weights & Biases."""
    
    def __init__(self, config: DictConfig, enabled: bool = True):
        """
        Initialize WandB logger.
        
        Args:
            config: Training configuration
            enabled: Whether to enable WandB logging
        """
        self.config = config
        self.enabled = enabled and WANDB_AVAILABLE
        self.run = None
        
        if self.enabled:
            self._init_wandb()
    
    def _init_wandb(self):
        """Initialize WandB run."""
        try:
            # Extract WandB configuration
            wandb_config = self.config.get('wandb', {})
            project = wandb_config.get('project', 'hsi-compression')
            entity = wandb_config.get('entity', None)
            name = wandb_config.get('name', None)
            tags = wandb_config.get('tags', [])
            notes = wandb_config.get('notes', '')
            
            # Initialize wandb run
            self.run = wandb.init(
                project=project,
                entity=entity,
                name=name,
                tags=tags,
                notes=notes,
                config=dict(self.config),
                mode='online' if wandb_config.get('enabled', True) else 'disabled'
            )
            print(f"WandB initialized: {self.run.get_url()}")
        except Exception as e:
            print(f"Warning: Failed to initialize WandB: {e}")
            self.enabled = False
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """
        Log metrics to WandB.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Current step/epoch
        """
        if not self.enabled or self.run is None:
            return
        
        try:
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)
        except Exception as e:
            print(f"Warning: Failed to log metrics to WandB: {e}")
    
    def log_config(self, config: Dict[str, Any]):
        """
        Log configuration to WandB.
        
        Args:
            config: Configuration dictionary
        """
        if not self.enabled or self.run is None:
            return
        
        try:
            wandb.config.update(config)
        except Exception as e:
            print(f"Warning: Failed to log config to WandB: {e}")
    
    def log_artifact(self, artifact_path: str, artifact_type: str = 'model'):
        """
        Log artifact (e.g., model checkpoint) to WandB.
        
        Args:
            artifact_path: Path to artifact
            artifact_type: Type of artifact (e.g., 'model', 'dataset')
        """
        if not self.enabled or self.run is None:
            return
        
        try:
            artifact = wandb.Artifact(
                name=os.path.basename(artifact_path),
                type=artifact_type
            )
            artifact.add_file(artifact_path)
            wandb.log_artifact(artifact)
        except Exception as e:
            print(f"Warning: Failed to log artifact to WandB: {e}")
    
    def finish(self):
        """Finish WandB run."""
        if self.enabled and self.run is not None:
            try:
                wandb.finish()
            except Exception as e:
                print(f"Warning: Failed to finish WandB run: {e}")
