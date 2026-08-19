import os

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

# Setup environment and model
####################################################

print("=" * 60)
print("Config:")

# PyTorch
torch.set_num_threads(8)
torch.set_num_interop_threads(1)
print("PyTorch threads:", torch.get_num_threads())
cuda_available = torch.cuda.is_available()
print("CUDA available:", cuda_available)
if cuda_available:
    print("GPU:", torch.cuda.get_device_name(0))

# Base path for all generated files
THIS_FOLDER = os.path.dirname(__file__)
DATA_PATH = rf"{THIS_FOLDER}\Datasets"
WORDS_PATH = rf"{THIS_FOLDER}\Words"

print("Base folder:", THIS_FOLDER)
print("Data folder:", DATA_PATH)
print("Words folder:", WORDS_PATH)

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    device_map="cuda:0" if cuda_available else "cpu",
    dtype=torch.float32,
)

print("Model:", model)
speakers = model.get_supported_speakers()
if speakers is None:
    speakers = []

print("Speakers:", speakers)

instructs = [
    ("normal", ""),
    (
        "calm",
        "Speak naturally, calmly, and clearly. Use a neutral speaking style.",
    ),
    (
        "professional",
        "Speak professionally, calmly, and confidently.",
    ),
    (
        "expressive",
        "Speak like a professional audiobook narrator. Use expressive but natural intonation.",
    ),
    (
        "excited",
        "Speak with genuine excitement and enthusiasm, while remaining natural.",
    ),
    (
        "angry",
        "Speak clearly and firmly with controlled anger.",
    ),
]

# Load all words into the script
####################################################
print("=" * 60)
print("Loading words")
random_words = []
with open(rf"{WORDS_PATH}\1000_random_words.txt", mode="r") as file:
    random_words = [line.rstrip() for line in file]

print("=" * 60)
print("Generating wav files")

# Generate .wav for each word for each speaker
####################################################

if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)

with torch.inference_mode():
    # Iterate over each speaker
    for speaker in speakers:
        SPEAKER_DIR = rf"{DATA_PATH}\{speaker}"
        if not os.path.exists(SPEAKER_DIR):
            os.mkdir(SPEAKER_DIR)

        print("=" * 60)
        print("Speaker:", speaker)

        # Iterate over each word
        for word in random_words:
            print("Word:", word)

            for key, value in instructs:
                WAV_FILE = rf"{SPEAKER_DIR}\{speaker}_{key}_{word}.wav"

                # Skip existing
                if os.path.exists(WAV_FILE):
                    print("Skipping:", WAV_FILE)
                    continue

                wavs, sr = model.generate_custom_voice(
                    text=word, language="English", speaker=speaker, instruct=value
                )

                sf.write(WAV_FILE, wavs[0], sr)
