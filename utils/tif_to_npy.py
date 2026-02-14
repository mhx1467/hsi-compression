import argparse
import glob
import multiprocessing
import numpy as np
import rasterio

invalid_channels = [126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 160, 161, 162, 163, 164, 165, 166]
valid_channels_ids = [c+1 for c in range(224) if c not in invalid_channels]
minimum_value = 0
maximum_value = 10000

def convert(patch_path):
    dataset = rasterio.open(patch_path)
    src = dataset.read(valid_channels_ids)
    clipped = np.clip(src, a_min=minimum_value, a_max=maximum_value)
    out_data = (clipped - minimum_value) / (maximum_value - minimum_value)
    out_data = out_data.astype(np.float32)
    out_path = patch_path.replace("SPECTRAL_IMAGE", "DATA").replace("TIF", "npy")
    np.save(out_path, out_data)

def main(in_directory):
    in_patches = glob.glob(f"{in_directory}/**/**/*SPECTRAL_IMAGE.TIF", recursive=True)
    with multiprocessing.Pool(64) as pool:
        pool.map(convert, in_patches)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TIF hyperspectral images to normalized NPY format.")
    parser.add_argument("in_directory", help="Input directory containing patch folders")
    args = parser.parse_args()
    main(args.in_directory)