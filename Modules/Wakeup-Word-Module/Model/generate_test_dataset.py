# This file uses the other script files to generate the test dataset
# Voices/speakers used in the test dataset shouldn't be used in the training dataset to
# validate that the model also works with unknown voices/speakers.

import os
from pathlib import Path

import torch
from console_utils import *
from dataset_utils import *
from tqdm import tqdm

write("Disclaimer: The execution of this script may take a while.\n")

# ======================================================================
# Constants/Setup
# ======================================================================

header("Setup")

subheader("PyTorch")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

TORCH_THREADS = torch.get_num_threads()
TORCH_CUDA_ACTIVE = torch.cuda.is_available()
TORCH_DEVICE = torch.cuda.get_device_name(0) if TORCH_CUDA_ACTIVE else "cpu"

log(f"Threads: {TORCH_THREADS}")
log(f"CUDA available: {TORCH_CUDA_ACTIVE}")
log(f"Device: {TORCH_DEVICE}")

subheader("Working Paths")
THIS_DIRECTORY = Path(os.path.dirname(__file__))
NOISE_DIRECTORY = Path(f"{THIS_DIRECTORY}/Noise/Test")
WORDS_DIRECTORY = Path(f"{THIS_DIRECTORY}/Words")
CLONING_DIRECTORY = Path(f"{THIS_DIRECTORY}/VoiceCloning")

log(f"Working directory: {THIS_DIRECTORY}")
log(f"Noise directory: {NOISE_DIRECTORY}")
log(f"Words directory: {WORDS_DIRECTORY}")

# ======================================================================
# Generating directories
# ======================================================================

header("Generating output paths")
DATASET_DIRECTORY = mkdir(Path(f"{THIS_DIRECTORY}/Datasets"))
TEST_DIRECTORY = mkdir(Path(f"{DATASET_DIRECTORY}/Test"))
TEST_POSITIVE_DIR = mkdir(Path(f"{TEST_DIRECTORY}/Positive"))
TEST_NEGATIVE_DIR = mkdir(Path(f"{TEST_DIRECTORY}/Negative"))
TEST_COMPUTE_DIR = mkdir(Path(f"{TEST_DIRECTORY}/Computed"))
ANNOTATION_FILE = Path(f"{TEST_COMPUTE_DIR}/annotations.csv")
subheader(f"Annotation file: {TEST_COMPUTE_DIR}")

# ======================================================================
# Helper methods
# ======================================================================


@torch.inference_mode(True)
def generate_clean_sentences_kokoro(sentence_file, out_directory, limit=50):
    # Loading the words
    subheader(f"Loading random words from: {sentence_file}")
    with open(sentence_file, encoding="utf-8", mode="r") as file:
        random_sentences = [line.rstrip() for line in file]
        random_sentences = random_sentences[0:limit]

    log(f"Found (x): {len(random_sentences)}")

    # Generating wav files
    model = KPipeline(lang_code="b", repo_id="hexgrad/Kokoro-82M")
    for voice in tqdm(KOKORO_B_VOICES, desc="Voices", unit="voice"):
        generate_wav_files_kokoro(model, random_sentences, voice, out_directory)


@torch.inference_mode(True)
def generate_clean_sentences_chatterbox(
    sentence_file,
    source_directory: Path,
    out_directory,
    limit=50,
):
    # Loading the words
    subheader(f"Loading random words from: {sentence_file}")
    with open(sentence_file, encoding="utf-8", mode="r") as file:
        random_sentences = [line.rstrip() for line in file]
        random_sentences = random_sentences[0:limit]
    log(f"Found (x): {len(random_sentences)}")

    # Generating wav files
    model = ChatterboxTTS.from_pretrained(device="cuda" if TORCH_CUDA_ACTIVE else "cpu")
    for glob in source_directory.rglob("*.wav"):
        generate_wav_files_with_cloning(model, random_sentences, glob, out_directory)


def generate_noise_sentences(wavs_dir: Path, noise_dir: Path, out_directory: Path):
    subheader("Loading noises from: {noise_dir}")
    for noise_file in noise_dir.rglob("*.wav"):
        write(f">>> Noise file: {noise_file}")
        for wav_file in wavs_dir.rglob("*.wav"):
            create_files_from_noise(wav_file, noise_file, out_directory)


# ======================================================================
# Generations of positive samples
# ======================================================================

header("Generating clean positive samples with TTS")
WAKEWORD_FILE = Path(f"{WORDS_DIRECTORY}/100_wakeup_word.txt")
CLEAN_POSITIVE = mkdir(Path(f"{TEST_POSITIVE_DIR}/Clean"))
generate_clean_sentences_kokoro(WAKEWORD_FILE, CLEAN_POSITIVE, limit=100)

header("Generating clean positive samples with Voice cloning")
generate_clean_sentences_chatterbox(
    WAKEWORD_FILE, CLONING_DIRECTORY, CLEAN_POSITIVE, limit=100
)

header("Generating noised positive samples with librosa")
WAVS_DIRECTORY = CLEAN_POSITIVE
NOISE_POSITIVE = mkdir(Path(f"{TEST_POSITIVE_DIR}/Noise"))
generate_noise_sentences(WAVS_DIRECTORY, NOISE_DIRECTORY, NOISE_POSITIVE)

# ======================================================================
# Generations of negative samples
# ======================================================================

header("Generating clean negative samples with TTS")
RANDOM_SENTENCES = Path(f"{WORDS_DIRECTORY}/100_random_sentences.txt")
CLEAN_NEGATIVE = mkdir(Path(f"{TEST_NEGATIVE_DIR}/Clean"))
generate_clean_sentences_kokoro(RANDOM_SENTENCES, CLEAN_NEGATIVE)

header("Generating clean negative sentences with Voice Cloning")
generate_clean_sentences_chatterbox(RANDOM_SENTENCES, CLONING_DIRECTORY, CLEAN_NEGATIVE)

header("Generating noised negative sentences with TTS")
WAVS_DIRECTORY = CLEAN_NEGATIVE
NOISE_NEGATIVE = mkdir(Path(f"{TEST_NEGATIVE_DIR}/SentenceN"))
generate_noise_sentences(WAVS_DIRECTORY, NOISE_DIRECTORY, NOISE_NEGATIVE)

# ======================================================================
# Generations of negative similar samples
# ======================================================================

header("Generating clean similar words with TTS")
RANDOM_SENTENCES = Path(f"{WORDS_DIRECTORY}/100_similar_words.txt")
SIMILAR_NEGATIVE = mkdir(Path(f"{TEST_NEGATIVE_DIR}/Similar"))
generate_clean_sentences_kokoro(RANDOM_SENTENCES, SIMILAR_NEGATIVE)

header("Generating clean similar words with Voice Cloning")
generate_clean_sentences_chatterbox(
    RANDOM_SENTENCES, CLONING_DIRECTORY, SIMILAR_NEGATIVE
)

header("Generating noised similar words with TTS")
WAVS_DIRECTORY = SIMILAR_NEGATIVE
NOISE_SIMILAR = mkdir(Path(f"{TEST_NEGATIVE_DIR}/SimilarN"))
generate_noise_sentences(WAVS_DIRECTORY, NOISE_DIRECTORY, NOISE_SIMILAR)

# ======================================================================
# Only noise
# ======================================================================

header("Generating only noise")
NOISE_ONLY = mkdir(Path(f"{TEST_NEGATIVE_DIR}/OnlyNoise"))
generate_noise_splits(NOISE_DIRECTORY, NOISE_ONLY)

# ======================================================================
# Precomputation
# ======================================================================

header("Precomputing mel spectograms and generating anno file")
generate_annotations_file(TEST_DIRECTORY, TEST_COMPUTE_DIR)
