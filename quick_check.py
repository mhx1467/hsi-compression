#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description='Quick sanity check on small dataset subset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['tcn', 'mamba'],
        required=True,
        help='Model to test (tcn or mamba)'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=32,
        help='Number of samples to use (default: 32)'
    )
    parser.add_argument(
        '--overrides',
        nargs='*',
        default=[],
        help='Config overrides (e.g., training.lr=0.001)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print full training output'
    )
    
    args = parser.parse_args()
    
    # Map model names to config files
    config_map = {
        'tcn': 'hsi_compression/configs/models/tcn_lossless.yaml',
        'mamba': 'hsi_compression/configs/models/mamba_lossy.yaml',
    }
    
    config_path = config_map[args.model]
    
    # Build training command
    cmd = [
        sys.executable, 'train.py',
        '--config', config_path,
        '--quick-check',
        '--quick-check-samples', str(args.samples),
    ]
    
    if args.overrides:
        cmd.extend(['--overrides'] + args.overrides)
    
    print(f"\n{'='*70}")
    print(f"QUICK-CHECK: {args.model.upper()} Model")
    print(f"{'='*70}")
    print(f"Config: {config_path}")
    print(f"Samples: {args.samples} (train) + {args.samples//4} (val)")
    print(f"Epochs: 2 (sanity check only)")
    print(f"{'='*70}\n")
    
    # Run training
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent)
        if result.returncode == 0:
            print(f"\n{'='*70}")
            print("✓ QUICK-CHECK PASSED")
            print(f"{'='*70}\n")
            return 0
        else:
            print(f"\n{'='*70}")
            print("✗ QUICK-CHECK FAILED")
            print(f"{'='*70}\n")
            return 1
    except KeyboardInterrupt:
        print("\n\nQuick-check interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\nQuick-check failed with error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
