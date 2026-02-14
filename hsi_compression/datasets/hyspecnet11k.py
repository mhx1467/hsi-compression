import csv
import os

import numpy as np
import torch

from torch.utils.data import Dataset


class HySpecNet11k(Dataset):
    def __init__(self, root_dir, mode="easy", split="train", transform=None):
        isExperimental = False
        if mode == "experimental":
            isExperimental = True
            mode = "easy"

        self.root_dir = root_dir

        self.csv_path = os.path.join(self.root_dir, "splits", mode, f"{split}.csv")
        with open(self.csv_path, newline='') as f:
            csv_reader = csv.reader(f)
            csv_data = list(csv_reader)
            self.npy_paths = sum(csv_data, [])
        self.npy_paths = [os.path.join(self.root_dir, "patches", x) for x in self.npy_paths]

        if isExperimental:
            self.npy_paths = self.npy_paths[:10]

        self.transform = transform

    def __len__(self):
        return len(self.npy_paths)

    def __getitem__(self, index):
        npy_path = self.npy_paths[index]
        img = np.load(npy_path)
        img = torch.from_numpy(img)
        if self.transform:
            img = self.transform(img)
        return img