import os
from pathlib import Path

import librosa
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
DATA_PATH = rf"{THIS_FOLDER}\Datasets\Train\Negative\Clean"
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
    (
        "normal",
        "Speak normally. Speak English.",
    ),
    (
        "calm",
        "Speak naturally, calmly, and clearly. Use a neutral speaking style. Speak English.",
    ),
    (
        "professional",
        "Speak professionally, calmly, and confidently. Speak English.",
    ),
    (
        "expressive",
        "Speak like a professional audiobook narrator. Use expressive but natural intonation. Speak English.",
    ),
    (
        "excited",
        "Speak with genuine excitement and enthusiasm, while remaining natural. Speak English.",
    ),
    (
        "angry",
        "Speak clearly and firmly with controlled anger. Speak English.",
    ),
    (
        "whisper",
        "Whisper softly and quietly, as if trying not to wake someone up nearby. Speak English.",
    ),
    (
        "tired",
        "Speak slowly and tiredly, as if you just woke up or are exhausted. Speak English.",
    ),
    (
        "fast",
        "Speak quickly and casually, as if you're in a hurry. Speak English.",
    ),
    (
        "shouting",
        "Speak loudly, as if calling out to someone from across a room. Speak English.",
    ),
    (
        "sad",
        "Speak softly with a sad, subdued tone. Speak English.",
    ),
    (
        "questioning",
        "Speak with a rising, uncertain, questioning intonation, as if unsure. Speak English.",
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

Path(DATA_PATH).mkdir(parents=True, exist_ok=True)

with torch.inference_mode():
    # Iterate over each speaker
    for speaker in speakers:
        SPEAKER_DIR = rf"{DATA_PATH}\{speaker}"
        Path(SPEAKER_DIR).mkdir(parents=True, exist_ok=True)

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

                # Find first occurrence of sound and last occurrence
                # Cut off all outside silence
                # Silence is added later manually
                non_silent_intervalls = librosa.effects.split(wavs[0], top_db=70)
                if len(non_silent_intervalls) == 0:
                    print(f"No audio detected: {WAV_FILE}")
                    sf.write(WAV_FILE, wavs[0], sr)

                start = non_silent_intervalls[0][0]
                end = non_silent_intervalls[-1][1]

                # Add a small amount of silence around the speech
                margin = int(sr * 100 / 1000)
                start = max(0, start - margin)
                end = min(len(wavs[0]), end + margin)
                trimmed = wavs[0][start:end]

                if len(trimmed) / sr > 1.5:
                    print(WAV_FILE, "longer than 1,5 seconds")

                sf.write(WAV_FILE, trimmed, sr)
