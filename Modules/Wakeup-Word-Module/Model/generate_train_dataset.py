# This file uses the other script files to generate the training dataset
# Voices/speakers used in the training dataset shouldn't be used in the test dataset to
# validate that the model also works with unknown voices/speakers.

import logging
import os
from pathlib import Path

import librosa
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel
from tqdm import tqdm

logging.disable(logging.WARNING)  # blendet DEBUG, INFO, WARNING aus
print("Disclaimer: The execution of this script may take a while.\n")

# ======================================================================
# Common Helper methods
# ======================================================================


# Generates a header in the console
def header(txt: object):
    print("+" + "-" * 100)
    print("|", txt.__str__().upper())
    print("+" + "-" * 100)


# Generates a header in the console
def subheader(*args):
    print(">>>", *args)


# Generates a message in the console
def log(*args):
    print("   ", *args)


# Creates a directory and logs
def mkdir(path):
    subheader(path)
    path.mkdir(parents=True, exist_ok=True)
    log("Done")
    return path


def trim_waveform(waveform, sample_rate):
    # Find first occurrence of sound and last occurrence
    # Cut off all outside silence
    # Silence is added later manually
    non_silent_intervalls = librosa.effects.split(waveform[0], top_db=70)
    start = non_silent_intervalls[0][0]
    end = non_silent_intervalls[-1][1]

    # Add a small amount of silence around the speech
    margin = int(sample_rate * 100 / 1000)
    start = max(0, start - margin)
    end = min(len(waveform[0]), end + margin)
    return waveform[0][start:end]


# ======================================================================
# Constants/Setup
# ======================================================================

header("Setup")

subheader("PyTorch")
TORCH_THREADS = torch.get_num_threads()
TORCH_CUDA_ACTIVE = torch.cuda.is_available()
TORCH_DEVICE = torch.cuda.get_device_name(0) if TORCH_CUDA_ACTIVE else "cpu"

log("Threads:", TORCH_THREADS)
log("CUDA available:", TORCH_CUDA_ACTIVE)
log("Device:", TORCH_DEVICE)

subheader("Working Paths")
THIS_DIRECTORY = Path(os.path.dirname(__file__))
NOISE_DIRECTORY = Path(f"{THIS_DIRECTORY}/Noise")
WORDS_DIRECTORY = Path(f"{THIS_DIRECTORY}/Words")
CUSTOM_DIRECTORY = Path(f"{THIS_DIRECTORY}/Custom")

log("Working directory:", THIS_DIRECTORY)
log("Noise directory:", NOISE_DIRECTORY)
log("Words directory:", CUSTOM_DIRECTORY)
log("Custom directory:", CUSTOM_DIRECTORY)

# ======================================================================
# Generating directories
# ======================================================================

header("Generating output paths")
DATASET_DIRECTORY = mkdir(Path(f"{THIS_DIRECTORY}/Datasets"))
TRAIN_DIRECTORY = mkdir(Path(f"{DATASET_DIRECTORY}/Train"))
TRAIN_POSITIVE_DIR = mkdir(Path(f"{TRAIN_DIRECTORY}/Positive"))
TRAIN_NEGATIVE_DIR = mkdir(Path(f"{TRAIN_DIRECTORY}/Negative"))
TRAIN_COMPUTE_DIR = mkdir(Path(f"{TRAIN_DIRECTORY}/Compute"))
ANNOTATION_FILE = Path(f"{TRAIN_COMPUTE_DIR}/annotations.csv")
subheader("Annotation file:", TRAIN_COMPUTE_DIR)


# ======================================================================
# Loading the model from the web
# ======================================================================

header("Loading text-to-speach model")
MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
subheader(MODEL_NAME)

MODEL_DTYPE = torch.bfloat16
TTS_MODEL = Qwen3TTSModel.from_pretrained(
    MODEL_NAME,
    device_map=TORCH_DEVICE,
    dtype=MODEL_DTYPE,
)

TTS_SPEAKERS = TTS_MODEL.get_supported_speakers()
if TTS_SPEAKERS is None:
    TTS_SPEAKERS = []

TTS_INSTRUCTS = [
    (
        "normal",
        "Speak normally. Speak English.",
    ),
    (
        "calm",
        "Speak naturally, calmly. Use a neutral speaking style. Speak English.",
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

log("Model:", MODEL_NAME)
log("Instance", TTS_MODEL)
log("Datatype:", MODEL_DTYPE)
log("Speakers:", TTS_SPEAKERS)
log("Instructs (x):", len(TTS_INSTRUCTS))


# ======================================================================
# Generating negative samples from random words with TTS model
# ======================================================================


# Generates .wav files from the supplied file that contains random sentences.
@torch.inference_mode(True)
def generate_sentences(sentence_file, out_directory):
    subheader("Loading random words from: ", sentence_file)
    with open(sentence_file, encoding="utf-8", mode="r") as file:
        random_sentences = [line.rstrip() for line in file]
    log("Found (x):", len(random_sentences))

    # Äußerster Balken: bleibt stehen (leave=True, Standard)
    for speaker in tqdm(TTS_SPEAKERS, desc="Speakers", unit="speaker"):
        SPEAKER_DIR = rf"{out_directory}/{speaker}"
        Path(SPEAKER_DIR).mkdir(parents=True, exist_ok=True)

        # tqdm.write statt print/log, damit der Balken nicht zerstört wird
        tqdm.write(f">>> Current speaker: {speaker} -> {SPEAKER_DIR}")

        # total angeben, da enumerate() keine Länge hat
        for sentence_index, sentence in tqdm(
            enumerate(random_sentences),
            total=len(random_sentences),
            desc="Sentences",
            unit="sentence",
            leave=False,  # verschwindet nach Abschluss, kein Stapeln
        ):
            for instruct_name, instruct_value in tqdm(
                TTS_INSTRUCTS,
                desc="Instructs",
                unit="instruct",
                leave=False,
            ):
                WAV_FILE = f"{SPEAKER_DIR}/{instruct_name}_{sentence_index}.wav"
                if Path(WAV_FILE).exists():
                    continue

                waveform, sample_rate = TTS_MODEL.generate_custom_voice(
                    text=sentence,
                    language="English",
                    speaker=speaker,
                    instruct=instruct_value,
                )
                trimmed = trim_waveform(waveform, sample_rate)

                split_size = sample_rate * 3
                count = trimmed.size // split_size

                if count == 0:
                    sf.write(WAV_FILE, trimmed, sample_rate)
                    continue

                # Innerster Balken nur zeigen, wenn er sich lohnt (mehrere Splits)
                for i in range(count):
                    WAV_FILE = f"{SPEAKER_DIR}/{instruct_name}_{sentence_index}_{i}.wav"
                    start_index = i * split_size
                    element = trimmed[start_index : start_index + split_size]
                    sf.write(WAV_FILE, element, sample_rate)


header("Generating clean positive samples with TTS")
SENTENCES_FILE = Path(f"{WORDS_DIRECTORY}/wakeup_word.txt")
TRAIN_POSITIVE_CLEAN = Path(f"{TRAIN_POSITIVE_DIR}/Clean")
generate_sentences(SENTENCES_FILE, TRAIN_POSITIVE_CLEAN)

header("Apply noise to positive samples")


header("Generating clean negative samples with TTS")
SENTENCES_FILE = Path(f"{WORDS_DIRECTORY}/1000_random_sentences.txt")
TRAIN_NEGATIVE_CLEAN = Path(f"{TRAIN_NEGATIVE_DIR}/Clean")
generate_sentences(SENTENCES_FILE, TRAIN_NEGATIVE_CLEAN)

header("Apply noise to negative samples")
