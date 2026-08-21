from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class WakeupWordDataset(Dataset):
    def __init__(self, directory, split) -> None:
        self.annotations_file = Path(f"{directory}/{split}/Compute/annotations.csv")
        self.annotations = pd.read_csv(self.annotations_file, encoding="UTF-8")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index: int) -> tuple:
        annotation = self.annotations.iloc[index]
        mel = np.load(annotation["mel_spec"])
        mel = torch.tensor(mel, dtype=torch.float32)
        mel = mel.unsqueeze(0)
        label = torch.tensor(annotation["label"], dtype=torch.float32)
        return mel, label


if __name__ == "__main__":
    try:
        THIS_FOLDER = Path(__file__).parent
        WORKING_DIR = Path(f"{THIS_FOLDER}/Datasets")
        data = WakeupWordDataset(WORKING_DIR, "Test")
        print(len(data))

        train_dataloader = DataLoader(data, batch_size=64, shuffle=True)
        features, labels = next(iter(train_dataloader))
        print(features.shape)
        print(labels.shape)
    except Exception as ex:
        print(ex)
