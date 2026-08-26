import os
import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*")
from pathlib import Path

import numpy as np
import torch
from console_utils import write
from dataset import WakeupWordDataset
from dataset_utils import generate_annotations_file
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from model import WakeupWordModel


@torch.inference_mode()
def evaluate(model: WakeupWordModel, dataloader: DataLoader, device: torch.device):
    MODEL.eval()
    for delimiter in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        write(f"Delimiter: {delimiter}")
        actual_labels, predicted_labels, probabilities = [], [], []
        for features, labels in dataloader:
            features = features.to(device, non_blocking=True)
            logits = model(features).squeeze(1)
            probability = torch.sigmoid(logits)
            prediction = (probability >= delimiter).int()

            probabilities.extend(probability.cpu().numpy().astype(float))
            actual_labels.extend(labels.cpu().numpy().astype(int))
            predicted_labels.extend(prediction.cpu().numpy())

        write(f"Probabilities:  {np.average(probabilities)}")
        write(classification_report(actual_labels, predicted_labels, zero_division=0))
        write(confusion_matrix(actual_labels, predicted_labels))


if __name__ == "__main__":
    THIS_FOLDER = os.path.dirname(__file__)
    WORKING_DIR = Path(f"{THIS_FOLDER}/Products")
    MODEL_FILE = Path(f"{WORKING_DIR}/best_model_26_08_26.pt")

    WAVS = Path(f"{THIS_FOLDER}/Datasets/Test/Negative/Similar")
    OUT_SIMILAR_WAVS = Path(f"{THIS_FOLDER}/Products/Test/Computed")
    generate_annotations_file(WAVS, OUT_SIMILAR_WAVS)

    BATCH_SIZE = 64
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL = WakeupWordModel().to(DEVICE)
    MODEL.load_state_dict(torch.load(MODEL_FILE, weights_only=True))

    WORKING_DIR = Path(f"{THIS_FOLDER}/Products")
    TEST_DATASET = WakeupWordDataset(WORKING_DIR, "Test")
    TEST_LOADER = DataLoader(
        TEST_DATASET,
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    evaluate(MODEL, TEST_LOADER, DEVICE)
