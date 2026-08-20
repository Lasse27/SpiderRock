# For each clean data sample we create:
#   For each different background noise:
#       3 different noised samples:
#           1. where the wakeword is at the start.
#           2. where the wakeword is central.
#           3. where the wakeword is at the end.

import enum
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


class Position(enum.Enum):
    START = (1,)
    MIDDLE = (2,)
    END = 3


def shifted_speech(speech, sample_rate, full_duration, position):
    out = np.zeros(sample_rate * full_duration)
    if speech.size > out.size:
        speech = speech[0 : out.size]
    if position == Position.START:
        out[0 : len(speech)] = speech
    elif position == Position.MIDDLE:
        start = (out.size - speech.size) // 2
        out[start : start + len(speech)] = speech
    elif position == Position.END:
        start = out.size - speech.size
        out[start : start + len(speech)] = speech

    return out


# Setup environment and model
####################################################

print("=" * 60)
print("Config:")

# Base path for all generated files
THIS_FOLDER = os.path.dirname(__file__)
print("Base folder:", THIS_FOLDER)

CLEAN_IN_PATH = rf"{THIS_FOLDER}\Datasets\Train\Positive\Clean"
print("In folder:", CLEAN_IN_PATH)

NOISE_OUT_PATH = rf"{THIS_FOLDER}\Datasets\Train\Positive\Noise"
Path(NOISE_OUT_PATH).mkdir(parents=True, exist_ok=True)
print("Out folder:", NOISE_OUT_PATH)

NOISE_PATH = rf"{THIS_FOLDER}\Noise"
print("Data folder:", NOISE_OUT_PATH)

# Load all noises into the script
####################################################
print("=" * 60)
print("Loading noise wav files")

noise_files = {}
for file in os.listdir(NOISE_PATH):
    if file.endswith(".wav"):
        filepath = os.path.join(NOISE_PATH, file)
        print(filepath)
        noise_files[file.split(".")[0]] = librosa.load(filepath, mono=True, sr=None)


# Load all noises into the script
####################################################
print("=" * 60)
print("Generating noised wakeup word files")
for key, (orig_noise, orig_noise_sr) in noise_files.items():

    # Iterate throught speakers
    for speaker in os.listdir(CLEAN_IN_PATH):
        full_folder_path = os.path.join(CLEAN_IN_PATH, speaker)

        # Iterate through clean speaker samples
        for file in os.listdir(full_folder_path):
            full_file_path = os.path.join(full_folder_path, file)
            filename = file.split(".")[0]
            OUT_SPEAKER_DIR = rf"{NOISE_OUT_PATH}\{speaker}"
            Path(OUT_SPEAKER_DIR).mkdir(parents=True, exist_ok=True)

            speech, speech_sr = librosa.load(
                full_file_path,
                sr=None,
                mono=True,
            )

            # Immer frisch vom Original resamplen, statt die Schleifenvariable zu überschreiben
            noise = orig_noise.copy()
            if orig_noise_sr != speech_sr:
                # print("Adjusting noise sample rate: ", orig_noise_sr, "to", speech_sr)
                noise = librosa.resample(
                    orig_noise, orig_sr=orig_noise_sr, target_sr=speech_sr
                )

            # Extract 3 Seconds of noise
            noise = noise[0 : speech_sr * 3]  # 3s

            # Make sample where speaker speaks at the start
            OUT_SPEAKER_FILE = rf"{OUT_SPEAKER_DIR}\{filename}_{key}_start.wav"
            if not os.path.exists(OUT_SPEAKER_FILE):
                speech_start = shifted_speech(speech, speech_sr, 3, Position.START)
                mixed_start = speech_start + noise
                sf.write(
                    OUT_SPEAKER_FILE,
                    mixed_start,
                    int(speech_sr),
                )
            else:
                print("Skipping:", OUT_SPEAKER_FILE)

            # Make sample where speaker speaks in the middle
            OUT_SPEAKER_FILE = rf"{OUT_SPEAKER_DIR}\{filename}_{key}_middle.wav"
            if not os.path.exists(OUT_SPEAKER_FILE):
                speech_mid = shifted_speech(speech, speech_sr, 3, Position.MIDDLE)
                mixed_mid = speech_mid + noise
                sf.write(
                    OUT_SPEAKER_FILE,
                    mixed_mid,
                    int(speech_sr),
                )
            else:
                print("Skipping:", OUT_SPEAKER_FILE)

            # Make sample where speaker speaks at the end
            OUT_SPEAKER_FILE = rf"{OUT_SPEAKER_DIR}\{filename}_{key}_end.wav"
            if not os.path.exists(OUT_SPEAKER_FILE):
                speech_end = shifted_speech(speech, speech_sr, 3, Position.END)
                mixed_end = speech_end + noise
                sf.write(
                    OUT_SPEAKER_FILE,
                    mixed_end,
                    int(speech_sr),
                )
            else:
                print("Skipping:", OUT_SPEAKER_FILE)
