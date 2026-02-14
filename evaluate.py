"""Evaluation script for HSI compression models."""

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
from hsi_compression.engine import load_checkpoint, Evaluator
import numpy as np
import json


def main():
    parser = argparse.ArgumentParser(description='Evaluate HSI compression model')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./results',
        help='Directory to save results'
    )
    parser.add_argument(
        '--overrides',
        nargs='*',
        default=[],
        help='Config overrides'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading config from {args.config}")
    config = load_config(args.config, args.overrides)
    
    print("\nConfiguration:")
    print_config(config)
    
    # Setup device
    device = torch.device(
        f"{config.device.type}:{config.device.device_id}"
        if config.device.type == 'cuda' and torch.cuda.is_available()
        else 'cpu'
    )
    print(f"\nUsing device: {device}")
    
    # Load model
    print(f"\nLoading model: {config.model.name}")
    model_config = dict(config.model)
    model_name = model_config.pop('name')
    model = get_model(model_name, **model_config)
    
    # Load checkpoint
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load test dataset
    print(f"\nLoading dataset: {config.data.dataset}")
    test_dataset = get_dataset(
        config.data.dataset,
        root_dir=config.data.root_dir,
        mode=config.data.mode,
        split='test',
        transform=get_default_transforms(
            'lossless' if 'tcn' in config.model.name else 'lossy',
            normalize_input=config.model.normalize_input
        )
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )
    
    # Evaluate
    print("\nEvaluating model...")
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        config=config,
        device=device,
    )
    
    results = evaluator.evaluate()
    
    # Print results
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    for metric, value in results.items():
        print(f"{metric:20s}: {value:.6f}")
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    main()
