import os
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

print("=" * 60)
print("Config:")

# Base path for all generated files
THIS_FOLDER = os.path.dirname(__file__)
print("THIS_FOLDER", THIS_FOLDER)

DATASET_FOLDER = rf"{THIS_FOLDER}\Datasets"
print("DATASET_FOLDER", DATASET_FOLDER)

COMPUTED_FOLDER = rf"{DATASET_FOLDER}\Train\Computed"
Path(COMPUTED_FOLDER).mkdir(parents=True, exist_ok=True)
print("COMPUTED_FOLDER", COMPUTED_FOLDER)

NEGATIVE_FOLDER = rf"{DATASET_FOLDER}\Train\Negative"
print("NEGATIVE_FOLDER", NEGATIVE_FOLDER)

POSITIVE_FOLDER = rf"{DATASET_FOLDER}\Train\Positive"
print("POSITIVE_FOLDER", POSITIVE_FOLDER)

ANNO_FILE = rf"{COMPUTED_FOLDER}\annotations.csv"
print("ANNO_FILE", ANNO_FILE)

SUBFOLDERS = ["Clean", "Noise", "Custom"]
print("SUBFOLDERS", SUBFOLDERS)

index: int = 0
annotations = []


# Collect negative samples
def train_data_compute_folder(basefolder, label):
    global index
    for folder in SUBFOLDERS:
        folder_fullpath = Path(rf"{basefolder}\{folder}")
        print(folder_fullpath)
        if not folder_fullpath.exists():
            print(folder_fullpath, "does not exist.")
            continue

        wav_files = list(folder_fullpath.rglob("*.wav"))
        for file in wav_files:
            # Load waveform and sample_rate
            waveform, sample_rate = librosa.load(file, sr=None, mono=True)
            mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sample_rate)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            Path(rf"{COMPUTED_FOLDER}\Train").mkdir(parents=True, exist_ok=True)
            output_path = rf"{COMPUTED_FOLDER}\Train\{index}_{folder}.npy"
            np.save(output_path, mel_spec_db)

            annotations.append({"id": index, "mel_file": output_path, "label": label})
            index += 1


print("=" * 60)
print("Computing positive samples")

train_data_compute_folder(POSITIVE_FOLDER, 1)

print("=" * 60)
print("Computing negative samples")

train_data_compute_folder(NEGATIVE_FOLDER, 0)

print("=" * 60)
print("Writing annotations")

annotation_df = pd.DataFrame(annotations)
annotation_df.to_csv(ANNO_FILE, index=False, mode="a")
