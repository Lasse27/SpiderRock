"""
This file provides helper methods used by the other generation skripts.
"""

import logging
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from chatterbox.tts import ChatterboxTTS
from kokoro import KPipeline

BASE_FOLDER = Path(os.path.dirname(os.path.abspath(__file__)))
TEST_FOLDER = Path(os.path.join(BASE_FOLDER, "Test"))
TRAIN_FOLDER = Path(os.path.join(BASE_FOLDER, "Train"))
VAL_FOLDER = Path(os.path.join(BASE_FOLDER, "Validate"))

PARENT_FOLDER = Path(os.path.dirname(BASE_FOLDER))
WORDS_FOLDER = Path(os.path.join(PARENT_FOLDER, "Words"))
NOISE_FOLDER = Path(os.path.join(PARENT_FOLDER, "Noise"))
VOICE_FOLDER = Path(os.path.join(PARENT_FOLDER, "Voices"))

LABEL_POSITIVE: str = "Positive"
LABEL_NEGATIVE: str = "Negative"

_COMPUTED: str = "Computed"
_CONTEXT_RAW: str = "Raw"
_CONTEXT_NOISE: str = "Noise"
_CONTEXT_VOLUME: str = "Volume"


# KOKORO Configuration
# ====================================================

_KOKORO_SPEED = 1.0
_KOKORO_SAMPLE_RATE = 24000
_KOKORO_AMERICAN_VOICES = [
    "af_heart",
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santa",
]

_KOKORO_BRITISH_VOICES = [
    "bf_emma",
    "bf_isabella",
    "bf_alice",
    "bf_lily",
    "bm_george",
    "bm_fable",
    "bm_lewis",
    "bm_daniel",
]
_AMERICAN_MODEL = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
_BRITISH_MODEL = KPipeline(lang_code="b", repo_id="hexgrad/Kokoro-82M")

# Chatterbox configuration
# ====================================================

CHATTERBOX: ChatterboxTTS = ChatterboxTTS.from_pretrained(
    device="cuda" if torch.cuda.is_available() else "cpu"
)


# Mel spectogram configuration
# ====================================================

N_MELS = 32
N_FFT = 512
HOP_LENGTH = 256


# Exported functions
# ====================================================


def mkdir(logger: logging.Logger, directory: Path):
    if directory.exists():
        return
    logger.info("Creating directory: %s", {directory})
    directory.mkdir(parents=True, exist_ok=True)


def initialize_logger(split: str) -> logging.Logger:
    # Ensure log folder exists
    logging_folder = Path(os.path.join(BASE_FOLDER, "Logs"))
    logging_folder.mkdir(parents=True, exist_ok=True)

    # Setting up logger
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    logging_file = Path(os.path.join(logging_folder, f"{split}_{timestamp}.log"))

    # Formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s,%(msecs)03d [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(split)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        # File-Handler
        file_handler = logging.FileHandler(logging_file, mode="a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Konsolen-Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info("Initialized logger: %s", logging_file)
    return logging.getLogger(split)


# Initializes the dataset environment
def initialize_environment(logger: logging.Logger):
    initialize_sub_environment(logger, TEST_FOLDER)
    initialize_sub_environment(logger, TRAIN_FOLDER)
    initialize_sub_environment(logger, VAL_FOLDER)


# Initializes the sub environment for a split
def initialize_sub_environment(logger: logging.Logger, directory: Path):
    first_layer = ["Positive", "Negative"]
    second_layer = ["Raw", "Volume", "Noise"]

    # Create base directory for subenvironment
    mkdir(logger, directory)

    # Create directory for precomputed output
    computed = Path(os.path.join(directory, "Computed"))
    mkdir(logger, computed)

    # Create directories for layers
    for fl_name in first_layer:
        fl_dir = Path(os.path.join(directory, fl_name))
        mkdir(logger, fl_dir)

        for sl_name in second_layer:
            sl_dir = Path(os.path.join(fl_dir, sl_name))
            mkdir(logger, sl_dir)


# Generates a voice sample with the corresponding speakers
def generate_voice_samples(
    logger: logging.Logger,
    split: str,
    label: str,
    text_file: Path,
    max_sec: int = 3,
    limit: int = 50,
):
    logger.info("Generating voice samples with kokoro")
    logger.info("> Split: %s", split)
    logger.info("> Label: %s", label)
    logger.info("> Words: %s", text_file)

    # Determine output folder from split and label
    output_dir = Path(os.path.join(BASE_FOLDER, split, label, _CONTEXT_RAW))

    # Get sentences from text file
    sentences: list = []
    with open(text_file, encoding="utf-8", mode="r") as file:
        sentences = [line.strip() for line in file]
        sentences = sentences[0:limit]
    logger.info("%s sentences read from file.", len(sentences))

    # Iterate over each sentence and speaker
    logger.info("Generating voice samples...")
    for i, sentence in enumerate(sentences):
        for speaker in _KOKORO_AMERICAN_VOICES:

            # Skip file if it already exists
            filename = f"{i:03d}_{speaker}_{label}_{_CONTEXT_RAW}.wav"
            file = Path(os.path.join(output_dir, filename))
            if file.exists():
                continue

            # Generate audio.wav file
            generator = _AMERICAN_MODEL(sentence, voice=speaker, speed=_KOKORO_SPEED)
            for _, (_, _, audio) in enumerate(generator):
                if audio is None:
                    continue

                max_samples = _KOKORO_SAMPLE_RATE * max_sec  # 3 seconds
                foundation = np.zeros(max_samples)
                trimmed_audio = audio[0 : min(max_samples, len(audio))]
                foundation[0 : len(trimmed_audio)] = trimmed_audio
                sf.write(file, foundation, _KOKORO_SAMPLE_RATE)  # type: ignore
                logger.debug("Created %s", file.name)
        logger.info("%s of %s sentences done.", i, len(sentences))


def generate_voice_samples_with_cloning(
    logger: logging.Logger,
    split: str,
    label: str,
    text_file: Path,
    max_sec: int = 3,
    limit: int = 50,
):
    logger.info("Generating voice samples with kokoro")
    logger.info("> Split: %s", split)
    logger.info("> Label: %s", label)
    logger.info("> Words: %s", text_file)

    # Determine output folder from split and label
    output_dir = Path(os.path.join(BASE_FOLDER, split, label, _CONTEXT_RAW))

    # Determine voice folder from split
    voice_dir = Path(os.path.join(VOICE_FOLDER, split))

    # Get sentences from text file
    sentences: list = []
    with open(text_file, encoding="utf-8", mode="r") as out_file:
        sentences = [line.strip() for line in out_file]
        sentences = sentences[0:limit]
    logger.info("%s sentences read from file.", len(sentences))

    # Iterate over each sentence and speaker
    max_samples = CHATTERBOX.sr * max_sec  # 3 seconds normally
    logger.info("Generating voice samples...")
    for i, sentence in enumerate(sentences):
        for voice_file in voice_dir.glob("*.wav"):
            # Skip file if it already exists
            filename = f"{i:03d}_{voice_file.stem}_{label}_{_CONTEXT_RAW}.wav"
            out_file = Path(os.path.join(output_dir, filename))
            if out_file.exists():
                continue

            # Generate voice cloned file with chatterbox
            wav = CHATTERBOX.generate(sentence, audio_prompt_path=voice_file)
            wav_np = wav.squeeze().cpu().numpy()
            trimmed_audio = wav_np[0 : min(max_samples, len(wav_np))]
            sf.write(out_file, trimmed_audio, CHATTERBOX.sr)
            logger.debug("Created %s", out_file.name)

        logger.info("%s of %s sentences done.", i, len(sentences))


# Takes each sample in the source directory and generates 3 volumed samples at different levels
def generate_volumed_samples(
    logger: logging.Logger,
    split: str,
    label: str,
):
    logger.info("Generating samples with different volumes")
    logger.info("> Split: %s", split)
    logger.info("> Label: %s", label)

    # Determine folder from where to pull the files from label and split
    source: Path = Path(os.path.join(BASE_FOLDER, split, label, _CONTEXT_RAW))

    # Determine output folder from label and split
    target: Path = Path(os.path.join(BASE_FOLDER, split, label, _CONTEXT_VOLUME))

    # Calculate gain to apply
    factor_db: float = 6.0
    gain_increased = 10 ** (factor_db / 20)
    gain_reduced = 10 ** (-factor_db / 20)

    # Iterate through all files in source directory
    for file in Path.rglob(source, "*.wav"):
        ldB_exists: bool = False
        mdB_exists: bool = False
        hdB_exists: bool = False

        # Check if files exist
        filename = f"{file.stem}_ldB.wav"
        out_file_ldB = Path(os.path.join(target, filename))
        if out_file_ldB.exists():
            ldB_exists = True

        filename = f"{file.stem}_mdB.wav"
        out_file_mdB = Path(os.path.join(target, filename))
        if out_file_mdB.exists():
            mdB_exists = True

        filename = f"{file.stem}_hdB.wav"
        out_file_hdB = Path(os.path.join(target, filename))
        if out_file_hdB.exists():
            hdB_exists = True

        if ldB_exists and mdB_exists and hdB_exists:
            continue

        # Load file and generate two more waveforms with gains
        wavs, sr = librosa.load(file, sr=None, mono=True)
        sr = int(sr)

        if not ldB_exists:
            wavs_lower = np.clip(wavs * gain_reduced, -1.0, 1.0)
            sf.write(out_file_ldB, wavs_lower, samplerate=sr)
            logger.debug("Created %s", out_file_ldB.name)

        if not mdB_exists:
            sf.write(out_file_mdB, wavs, samplerate=sr)
            logger.debug("Created %s", out_file_mdB.name)

        if not hdB_exists:
            wavs_higher = np.clip(wavs * gain_increased, -1.0, 1.0)
            sf.write(out_file_hdB, wavs_higher, samplerate=sr)
            logger.debug("Created %s", out_file_hdB.name)


# Takes each sample in the volume directory and generates a sample for each noise in the noise folder
def generate_noised_samples(
    logger: logging.Logger,
    split: str,
    label: str,
    noise_dir: Path,
):
    random.seed(42)
    logger.info("Generating samples with different noises")
    logger.info("> Split: %s", split)
    logger.info("> Label: %s", label)

    # Determine folder from where to pull the files from label and split
    source: Path = Path(os.path.join(BASE_FOLDER, split, label, _CONTEXT_VOLUME))

    # Determine output folder from label and split
    target: Path = Path(os.path.join(BASE_FOLDER, split, label, _CONTEXT_NOISE))

    # Ongoing variables
    max_samples: int = _KOKORO_SAMPLE_RATE * 3
    audiocache: dict[str, tuple] = {}

    # Iterate through files from noise folder
    noise_file_len = len(os.listdir(noise_dir))
    for i, noise_file in enumerate(noise_dir.glob("*.wav"), start=1):
        logger.debug("Noise: %s", noise_file.stem)
        noise_wavs, _ = librosa.load(noise_file, sr=None, mono=True)
        noise_len: int = len(noise_wavs)

        for soundfile in source.glob("*.wav"):
            none_exists: bool = False
            noise_exists: bool = False

            # Check if files exist and skip if yes
            filename = f"{soundfile.stem}_none.wav"
            no_noise_file = Path(os.path.join(target, filename))
            if no_noise_file.exists():
                none_exists = True

            filename = f"{soundfile.stem}_{noise_file.stem}.wav"
            out_noise_file = Path(os.path.join(target, filename))
            if out_noise_file.exists():
                noise_exists = True

            if none_exists and noise_exists:
                continue

            # Load wavs via librosa
            cached = audiocache.get(soundfile.name)
            if cached is not None:
                soundfile, wavs, sr = cached
            else:
                wavs, sr = librosa.load(soundfile, sr=None, mono=True)
                sr = int(sr)
                audiocache[soundfile.name] = (soundfile, wavs, sr)

            # Write a file with no noise
            if not none_exists:
                sf.write(no_noise_file, wavs, samplerate=sr)
                logger.debug("Created %s", no_noise_file.name)

            # Select random starting point in noise file
            random_max_idx: int = max(0, noise_len - max_samples)
            start_idx: int = random.randint(0, random_max_idx)
            random_noise = noise_wavs[start_idx : start_idx + max_samples]

            # Create files noise
            if not noise_exists:
                sf.write(out_noise_file, wavs + random_noise, samplerate=sr)  # type: ignore
                logger.debug("Created %s", out_noise_file.name)

        logger.info("%s of %s noise files done.", i, noise_file_len)


def generate_just_noise(
    logger: logging.Logger,
    split: str,
    label: str,
    noise_dir: Path,
    samples_per_noise: int = 10,
):
    random.seed(42)
    logger.info("Generating noise to output")
    logger.info("> Split: %s", split)
    logger.info("> Label: %s", label)

    # Determine output folder from label and split
    target: Path = Path(os.path.join(BASE_FOLDER, split, label, _CONTEXT_NOISE))

    # Ongoing variables
    max_samples: int = _KOKORO_SAMPLE_RATE * 3

    # Iterate through files from noise folder
    for noise_file in noise_dir.rglob("*.wav"):
        logger.debug("Noise: %s", noise_file.stem)
        noise_wavs, noise_sr = librosa.load(
            noise_file, sr=_KOKORO_SAMPLE_RATE, mono=True
        )
        noise_sr = int(noise_sr)
        noise_len: int = len(noise_wavs)

        for i in range(samples_per_noise):

            # Check if files exist and skip if yes
            filename = f"{noise_file.stem}_{i}_Noise.wav"
            out_file = Path(os.path.join(target, filename))
            if out_file.exists():
                continue

            # Select random starting point in noise file
            random_max_idx: int = max(0, noise_len - max_samples)
            start_idx: int = random.randint(0, random_max_idx)
            random_noise = noise_wavs[start_idx : start_idx + max_samples]

            # Create files noise
            sf.write(out_file, random_noise, samplerate=noise_sr)
            logger.debug("Created %s", out_file.name)


# Generates mel spectograms from the audio and creates the annotation file
def generate_computed(
    logger: logging.Logger,
    split: str,
):
    logger.info("Generating annotations and precomputations")
    logger.info("> Split: %s", split)

    # Determine output folder from split
    logger.info("Clearing annotation directory...")
    target: Path = Path(os.path.join(BASE_FOLDER, split, _COMPUTED))
    for file in target.glob("*"):
        os.remove(file)

    # Structure to hold all observed datasamples
    records = []

    # Collect all positive files
    # ========================================================
    logger.info("Collecting positive files...")
    positive_filepath = os.path.join(BASE_FOLDER, split, LABEL_POSITIVE, _CONTEXT_NOISE)
    positive_files_len = len(os.listdir(Path(positive_filepath)))
    for i, file in enumerate(Path(positive_filepath).glob("*.wav")):

        # Calculate mel and save as npy
        try:
            out_filename = f"{file.stem}.npy"
            out_file = Path(os.path.join(target, out_filename))
            calculate_mel_save_to_npy(file, out_file)
            records.append({"mel_spec": out_file.name, "label": 1})
        except Exception as ex:  # noqa: BLE001
            logger.warning("%s: %s", out_file.name, ex)
        if i % 1000 == 0:
            logger.info("%s of %s positive files done.", i, positive_files_len)

    # Collect all negative files
    # ========================================================
    logger.info("Collecting negative files...")
    negative_filepath = os.path.join(BASE_FOLDER, split, LABEL_NEGATIVE, _CONTEXT_NOISE)
    negative_files_len = len(os.listdir(Path(negative_filepath)))
    for i, file in enumerate(Path(negative_filepath).glob("*.wav")):

        # Calculate mel and save as npy
        try:
            out_filename = f"{file.stem}.npy"
            out_file = Path(os.path.join(target, out_filename))
            calculate_mel_save_to_npy(file, out_file)
            records.append({"mel_spec": out_file.name, "label": 0})
        except Exception as ex:  # noqa: BLE001
            logger.warning("%s: %s", out_file.name, ex)
        if i % 1000 == 0:
            logger.info("%s of %s negative files done.", i, negative_files_len)

    # Create annotation file
    # ========================================================
    logger.info("Creating annotation file...")
    annotation_file = Path(os.path.join(target, ".annotations.csv"))
    df = pd.DataFrame(records)
    df.to_csv(annotation_file, index=False, mode="w")
    logger.debug(df["label"].value_counts())


# Calculates the mel spectogram from a wav file and writes it into a npy file
def calculate_mel_save_to_npy(file, out_file):
    # Load waveform and sample_rate
    wavs, sr = librosa.load(file, sr=_KOKORO_SAMPLE_RATE, mono=True)

    # Calculate mel spectogram
    mel_spec = librosa.feature.melspectrogram(
        y=wavs,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Save file and record
    np.save(out_file, mel_spec_db)
