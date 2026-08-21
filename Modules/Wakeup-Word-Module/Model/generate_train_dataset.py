# This file uses the other script files to generate the test dataset
# Voices/speakers used in the test dataset shouldn't be used in the training dataset to
# validate that the model also works with unknown voices/speakers.

import os
from pathlib import Path

import torch
from console_utils import *
from dataset_utils import *
from tqdm import tqdm

print("Disclaimer: The execution of this script may take a while.\n")

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

log("Threads:", TORCH_THREADS)
log("CUDA available:", TORCH_CUDA_ACTIVE)
log("Device:", TORCH_DEVICE)

subheader("Working Paths")
THIS_DIRECTORY = Path(os.path.dirname(__file__))
NOISE_DIRECTORY = Path(f"{THIS_DIRECTORY}/Noise/Train")
WORDS_DIRECTORY = Path(f"{THIS_DIRECTORY}/Words")

log("Working directory:", THIS_DIRECTORY)
log("Noise directory:", NOISE_DIRECTORY)
log("Words directory:", WORDS_DIRECTORY)

# ======================================================================
# Generating directories
# ======================================================================

header("Generating output paths")
DATASET_DIRECTORY = mkdir(Path(f"{THIS_DIRECTORY}/Datasets"))
TEST_DIRECTORY = mkdir(Path(f"{DATASET_DIRECTORY}/Test"))
TEST_POSITIVE_DIR = mkdir(Path(f"{TEST_DIRECTORY}/Positive"))
TEST_NEGATIVE_DIR = mkdir(Path(f"{TEST_DIRECTORY}/Negative"))
TEST_COMPUTE_DIR = mkdir(Path(f"{TEST_DIRECTORY}/Compute"))
ANNOTATION_FILE = Path(f"{TEST_COMPUTE_DIR}/annotations.csv")
subheader("Annotation file:", TEST_COMPUTE_DIR)

# ======================================================================
# Helper methods
# ======================================================================


@torch.inference_mode(True)
def generate_clean_sentences(sentence_file, out_directory):
    # Loading the words
    subheader("Loading random words from: ", sentence_file)
    with open(sentence_file, encoding="utf-8", mode="r") as file:
        random_sentences = [line.rstrip() for line in file]
        random_sentences = random_sentences[0:25]
    log("Found (x):", len(random_sentences))

    # Generating wav files
    model = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    for voice in tqdm(KOKORO_A_VOICES, desc="Voices", unit="voice"):
        generate_wav_files_kokoro(model, random_sentences, voice, out_directory)


def generate_noise_sentences(wavs_dir: Path, noise_dir: Path, out_directory: Path):
    subheader("Loading noises from: ", noise_dir)
    for noise_file in noise_dir.rglob("*.wav"):
        tqdm.write(f">>> Noise file: {noise_file}")
        for wav_file in wavs_dir.rglob("*.wav"):
            create_files_from_noise(wav_file, noise_file, out_directory)


# ======================================================================
# Generations
# ======================================================================

header("Generating clean positive samples with TTS")
WAKEWORD_FILE = Path(f"{WORDS_DIRECTORY}/wakeup_word.txt")
CLEAN_POSITIVE = mkdir(Path(f"{TEST_POSITIVE_DIR}/Single"))
generate_clean_sentences(WAKEWORD_FILE, CLEAN_POSITIVE)

header("Generating noised positive samples with librosa")
WAVS_DIRECTORY = CLEAN_POSITIVE
NOISE_POSITIVE = mkdir(Path(f"{TEST_POSITIVE_DIR}/SingleN"))
generate_noise_sentences(WAVS_DIRECTORY, NOISE_DIRECTORY, NOISE_POSITIVE)

header("Generating clean negative sentences with TTS")
RANDOM_SENTENCES = Path(f"{WORDS_DIRECTORY}/100_other_sentences.txt")
CLEAN_NEGATIVE = mkdir(Path(f"{TEST_NEGATIVE_DIR}/Sentence"))
generate_clean_sentences(RANDOM_SENTENCES, CLEAN_NEGATIVE)

header("Generating noised negative sentences with TTS")
WAVS_DIRECTORY = CLEAN_NEGATIVE
NOISE_NEGATIVE = mkdir(Path(f"{TEST_POSITIVE_DIR}/SentenceN"))
generate_noise_sentences(WAVS_DIRECTORY, NOISE_DIRECTORY, NOISE_NEGATIVE)

header("Generating clean similar words with TTS")
RANDOM_SENTENCES = Path(f"{WORDS_DIRECTORY}/100_similar_words.txt")
SIMILAR_NEGATIVE = mkdir(Path(f"{TEST_NEGATIVE_DIR}/Similar"))
generate_clean_sentences(RANDOM_SENTENCES, CLEAN_NEGATIVE)

header("Generating noised similar words with TTS")
WAVS_DIRECTORY = CLEAN_NEGATIVE
NOISE_SIMILAR = mkdir(Path(f"{TEST_POSITIVE_DIR}/SimilarN"))
generate_noise_sentences(WAVS_DIRECTORY, NOISE_DIRECTORY, NOISE_SIMILAR)

header("Precomputing mel spectograms and generating anno file")
generate_annotations_file(TEST_DIRECTORY, TEST_COMPUTE_DIR)

"""
Train with positive labels:
Wakeword-Clean: 20 x 5
Wakeword-Noise: 20 x 5 x 3 x 4
= 1300
+ Personal recordings


Train with negative labels:
Sentence-Clean: 20 x 25 x 5
Sentence-Noise: 20 x 25 x 5 x 3 x 4
Similar-Clean: 20 x 25 x 5
Similar-Noise: 20 x 25 x 5 x 3 x 4
"""
