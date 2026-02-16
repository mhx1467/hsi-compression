"""Training script for HSI compression models."""

import argparse
import torch
import sys
from pathlib import Path
from torch.utils.data import DataLoader, Subset

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
    parser.add_argument(
        '--quick-check',
        action='store_true',
        help='Run quick sanity check on small subset (32 samples, 2 epochs)'
    )
    parser.add_argument(
        '--quick-check-samples',
        type=int,
        default=32,
        help='Number of samples to use in quick-check mode (default: 32)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading config from {args.config}")
    config = load_config(args.config, args.overrides)
    
    # Apply quick-check overrides if enabled
    if args.quick_check:
        print("\n" + "="*60)
        print("QUICK-CHECK MODE ENABLED")
        print("="*60)
        print(f"Using {args.quick_check_samples} samples with 2 epochs for sanity check")
        config.training.epochs = 2
    
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
    
    # Apply quick-check subset if enabled
    if args.quick_check:
        indices = list(range(min(args.quick_check_samples, len(train_dataset))))
        train_dataset = Subset(train_dataset, indices)
        print(f"  [QUICK-CHECK] Using subset of {len(train_dataset)} samples")
    
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
        
        # Apply quick-check subset to validation if enabled
        if args.quick_check:
            indices = list(range(min(args.quick_check_samples // 4, len(val_dataset))))
            val_dataset = Subset(val_dataset, indices)
            print(f"  [QUICK-CHECK] Using subset of {len(val_dataset)} validation samples")
    
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
