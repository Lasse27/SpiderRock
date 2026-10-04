import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch.utils.data import DataLoader, Dataset


class WakeupWordDataset(Dataset):
    def __init__(self, base_dir: Path, split: str) -> None:
        self.split: str = split
        self.base_dir: Path = base_dir
        self.split_dir: Path = Path(os.path.join(base_dir, split))
        self.computed_dir: Path = Path(os.path.join(self.split_dir, "Computed"))
        self.anno_file: Path = Path(os.path.join(self.computed_dir, ".annotations.csv"))
        self.annos: DataFrame = pd.read_csv(self.anno_file, encoding="UTF-8")
        self.annos_len: int = len(self.annos)

    def __len__(self):
        return self.annos_len

    def __getitem__(self, index: int) -> tuple:
        annotation = self.annos.iloc[index]
        mel = np.load(os.path.join(self.computed_dir, annotation["mel_spec"]))
        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(annotation["label"], dtype=torch.float32)
        return mel, label


if __name__ == "__main__":
    try:
        BASE_FOLDER = Path(os.path.dirname(os.path.abspath(__file__)))
        PARENT_FOLDER = Path(os.path.dirname(BASE_FOLDER))
        DATASET_DIR = Path(os.path.join(PARENT_FOLDER, "Datasets"))
        train_dataset = WakeupWordDataset(DATASET_DIR, "Train")
        print("Dataset:              ", train_dataset.split)
        print("Dataset-Directory:    ", train_dataset.split_dir)
        print("Dataset-Anno-File:    ", train_dataset.anno_file)
        print("Entries in Dataset:   ", len(train_dataset))

        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        features, labels = next(iter(train_dataloader))
        print("Feature-Shape (Batch):", features.shape)
        print("Label-Shape (Batch):  ", labels.shape)

    except Exception as ex:  # noqa: BLE001
        print(ex)
