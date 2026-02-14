"""Generate dummy HSI dataset for testing - Pure Python version."""

import os
import struct
from pathlib import Path
import csv
import argparse
import random
from typing import Tuple


def generate_dummy_hsi_pure(
    num_bands: int = 224,
    height: int = 32,
    width: int = 32,
    seed: int = None,
) -> bytes:
    """
    Generate a dummy HSI patch and return as bytes (numpy-less version).
    Will be loaded as float32 by numpy.load()
    
    Args:
        num_bands: Number of spectral bands
        height: Spatial height
        width: Spatial width
        seed: Random seed for reproducibility
    
    Returns:
        Binary data in numpy .npy format
    """
    if seed is not None:
        random.seed(seed)
    
    # Create 3D array in memory: (num_bands, height, width)
    shape = (num_bands, height, width)
    total_elements = num_bands * height * width
    
    # Generate synthetic data
    data = []
    for c in range(num_bands):
        wavelength_factor = (c / num_bands) * 2 * 3.14159
        
        for y in range(height):
            for x in range(width):
                # Spatial pattern
                spatial_pattern = (
                    (2 * 3.14159 * (x + y) / 32) % (2 * 3.14159) - 3.14159
                )
                spatial_val = (spatial_pattern / 3.14159 + 1) / 2
                
                # Spectral variation
                spectral_var = (wavelength_factor % (2 * 3.14159)) / 3.14159
                
                # Combine
                val = spatial_val * (0.5 + 0.5 * spectral_var)
                
                # Add small noise
                noise = (random.random() - 0.5) * 0.1
                val = max(0, min(1, val + noise))
                
                data.append(struct.pack('<f', float(val)))
    
    # Create numpy .npy format (v1.0)
    # Format: magic number (6 bytes) + version (2 bytes) + header_length (2 bytes) + header + data
    
    magic = b'\x93NUMPY'
    version = struct.pack('BB', 1, 0)  # version 1.0
    
    # Create header
    dtype_str = "'<f4'"  # little-endian float32
    fortran_order = 'False'
    shape_str = str(shape)
    
    header = f"{{'descr': {dtype_str}, 'fortran_order': {fortran_order}, 'shape': {shape_str}, }}"
    # Pad header to multiple of 16
    header_bytes = header.encode('ascii')
    padding_needed = 16 - (len(header_bytes) + 1) % 16
    if padding_needed < 0:
        padding_needed += 16
    header_bytes += b' ' * padding_needed + b'\n'
    
    header_len = len(header_bytes)
    header_len_bytes = struct.pack('<H', header_len)
    
    # Combine all
    npy_data = magic + version + header_len_bytes + header_bytes + b''.join(data)
    
    return npy_data


def create_dummy_dataset(
    output_dir: str,
    num_scenes: int = 5,
    patches_per_scene: int = 10,
    num_bands: int = 224,
    patch_size: Tuple[int, int] = (32, 32),
    modes: list = None,
    splits: list = None,
):
    """
    Create a dummy dataset conforming to HySpecNet-11k structure.
    
    Args:
        output_dir: Output directory
        num_scenes: Number of scenes to generate
        patches_per_scene: Number of patches per scene
        num_bands: Number of spectral bands
        patch_size: Spatial size (H, W)
        modes: List of modes ('easy', 'hard', etc.)
        splits: List of splits ('train', 'val', 'test')
    """
    if modes is None:
        modes = ['easy']
    if splits is None:
        splits = ['train', 'val', 'test']
    
    output_dir = Path(output_dir)
    patches_dir = output_dir / 'patches'
    splits_dir = output_dir / 'splits'
    
    # Create directories
    patches_dir.mkdir(parents=True, exist_ok=True)
    
    for mode in modes:
        mode_splits_dir = splits_dir / mode
        mode_splits_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_scenes} scenes with {patches_per_scene} patches each...")
    print(f"Patch format: {num_bands} bands × {patch_size[0]} × {patch_size[1]} pixels")
    
    # Generate patches and track file paths
    split_data = {mode: {split: [] for split in splits} for mode in modes}
    
    patch_counter = 0
    for scene_idx in range(num_scenes):
        for patch_idx in range(patches_per_scene):
            # Generate dummy HSI
            seed = scene_idx * patches_per_scene + patch_idx
            hsi_bytes = generate_dummy_hsi_pure(num_bands, patch_size[0], patch_size[1], seed)
            
            # Save as .npy file
            patch_name = f"scene{scene_idx:02d}_{patch_idx:02d}"
            patch_path = patches_dir / f"{patch_name}.npy"
            
            with open(patch_path, 'wb') as f:
                f.write(hsi_bytes)
            
            # Assign to splits (70% train, 15% val, 15% test)
            rand_val = random.random()
            if rand_val < 0.7:
                split = 'train'
            elif rand_val < 0.85:
                split = 'val'
            else:
                split = 'test'
            
            # Add to all modes
            for mode in modes:
                split_data[mode][split].append(f"{patch_name}.npy")
            
            patch_counter += 1
            if (patch_counter + 1) % 5 == 0:
                print(f"  Generated {patch_counter} patches...")
    
    # Write CSV files for each mode and split
    for mode in modes:
        for split in splits:
            csv_path = splits_dir / mode / f"{split}.csv"
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                for item in split_data[mode][split]:
                    writer.writerow([item])
            
            num_items = len(split_data[mode][split])
            print(f"  {mode}/{split}: {num_items} patches")
    
    print(f"\nDataset created successfully!")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description='Generate dummy HSI dataset for testing'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./dummy_hyspecnet11k',
        help='Output directory'
    )
    parser.add_argument(
        '--num_scenes',
        type=int,
        default=5,
        help='Number of scenes'
    )
    parser.add_argument(
        '--patches_per_scene',
        type=int,
        default=10,
        help='Patches per scene'
    )
    parser.add_argument(
        '--num_bands',
        type=int,
        default=224,
        help='Number of spectral bands'
    )
    parser.add_argument(
        '--patch_size',
        type=int,
        default=32,
        help='Patch size (H=W)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    create_dummy_dataset(
        args.output,
        num_scenes=args.num_scenes,
        patches_per_scene=args.patches_per_scene,
        num_bands=args.num_bands,
        patch_size=(args.patch_size, args.patch_size),
    )


if __name__ == '__main__':
    main()
