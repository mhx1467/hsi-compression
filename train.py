"""Training script for HSI compression models."""

import argparse
import torch
import sys
from pathlib import Path
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from hsi_compression.utils.config import load_config, print_config
from hsi_compression.datasets import get_dataset, get_default_transforms
from hsi_compression.models import get_model
from hsi_compression.engine import Trainer
import numpy as np


def setup_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description='Train HSI compression model')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file'
    )
    parser.add_argument(
        '--overrides',
        nargs='*',
        default=[],
        help='Config overrides (e.g., training.lr=0.001)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading config from {args.config}")
    config = load_config(args.config, args.overrides)
    
    print("\nConfiguration:")
    print_config(config)
    
    # Setup seed
    setup_seed(config.seed)
    
    # Setup device
    device = torch.device(
        f"{config.device.type}:{config.device.device_id}"
        if config.device.type == 'cuda' and torch.cuda.is_available()
        else 'cpu'
    )
    print(f"\nUsing device: {device}")
    
    # Load dataset
    print(f"\nLoading dataset: {config.data.dataset}")
    train_dataset = get_dataset(
        config.data.dataset,
        root_dir=config.data.root_dir,
        mode=config.data.mode,
        split='train',
        transform=get_default_transforms(
            'lossless' if 'tcn' in config.model.name else 'lossy',
            normalize_input=config.model.normalize_input
        )
    )
    
    val_dataset = None
    if config.get('validation') and config.validation.get('data_root'):
        val_dataset = get_dataset(
            config.data.dataset,
            root_dir=config.validation.data_root,
            mode=config.data.mode,
            split='val',
            transform=get_default_transforms(
                'lossless' if 'tcn' in config.model.name else 'lossy',
                normalize_input=config.model.normalize_input
            )
        )
    
    print(f"Training samples: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=config.data.shuffle,
        num_workers=config.data.num_workers,
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )
    
    # Load model
    print(f"\nLoading model: {config.model.name}")
    model_config = dict(config.model)
    model_name = model_config.pop('name')  # Remove 'name' from kwargs
    model = get_model(model_name, **model_config)
    print(f"Model: {model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )
    
    # Train
    print("\nStarting training...")
    try:
        trainer.train()
        print("\nTraining completed successfully!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise


if __name__ == '__main__':
    main()
