import enum
import logging
import os
import warnings
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from chatterbox.tts import ChatterboxTTS
from console_utils import *
from kokoro import KPipeline
from tqdm import tqdm

logging.disable(logging.WARNING)  # blendet DEBUG, INFO, WARNING aus
warnings.filterwarnings("ignore", message="dropout option adds dropout")
warnings.filterwarnings("ignore", message=".*weight_norm.*is deprecated")

# ======================================================================
# Constants/Setup
# ======================================================================
KOKORO_SPEED = 1.0
KOKORO_SAMPLE_RATE = 24000
KOKORO_LANG_CODE = "a"  # American english
KOKORO_A_VOICES = [
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

KOKORO_B_VOICES = [
    "bf_emma",
    "bf_isabella",
    "bf_alice",
    "bf_lily",
    "bm_george",
    "bm_fable",
    "bm_lewis",
    "bm_daniel",
]

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512

# ======================================================================
# KOKORO GENERATION
# ======================================================================


def generate_wav_file_kokoro(
    model: KPipeline,
    sentence: str,
    voice: str,
    out_file: Path,
    max_sec: int = 3,
    overwrite: bool = False,
):
    if not overwrite and out_file.exists():
        write(f"    {out_file.name} already exists. Skipping.")
        return

    generator = model(sentence, voice=voice, speed=KOKORO_SPEED)
    for _, (_, _, audio) in enumerate(generator):
        max_samples = KOKORO_SAMPLE_RATE * max_sec  # 3 seconds normally
        if audio is None:
            write(f"    {voice}: {sentence}: No audio found!")
            sf.write(out_file, torch.zeros(max_samples), KOKORO_SAMPLE_RATE)  # type: ignore
            continue

        trimmed_audio = audio[0 : min(max_samples, len(audio))]
        sf.write(out_file, trimmed_audio, KOKORO_SAMPLE_RATE)  # type: ignore
        write(f"    {out_file.name} created.")


def generate_wav_files_kokoro(
    model: KPipeline,
    sentences: list,
    voice: str,
    out_directory: Path,
    max_sec: int = 3,
    overwrite: bool = False,
):
    out_directory.mkdir(parents=True, exist_ok=True)
    write(f">>> Creating wav files for: {voice}")
    for index, sentence in tqdm(
        enumerate(sentences),
        total=len(sentences),
        desc="Sentences",
        unit="sentence",
        leave=False,
    ):
        file = Path(f"{out_directory}/{voice}_{index}.wav")
        generate_wav_file_kokoro(model, sentence, voice, file, max_sec, overwrite)


# ======================================================================
# CHATTERBOX GENERATION
# ======================================================================


def generate_wav_file_with_cloning(
    model: ChatterboxTTS,
    sentence: str,
    voice_path: Path,
    out_file: Path,
    max_sec: int = 3,
    overwrite: bool = False,
):
    if not overwrite and out_file.exists():
        write(f"    {out_file.name} already exists. Skipping.")
        return

    wav = model.generate(sentence, audio_prompt_path=voice_path)
    max_samples = model.sr * max_sec  # 3 seconds normally
    if wav is None:
        write(f"    {voice_path}: {sentence}: No audio found!")
        sf.write(out_file, torch.zeros(max_samples), model.sr)  # type: ignore

    wav_np = (
        wav.squeeze().cpu().numpy()
    )  # Tensor -> NumPy, von GPU auf CPU holen, Kanal-Dimension entfernen
    trimmed_audio = wav_np[0 : min(max_samples, len(wav_np))]
    sf.write(out_file, trimmed_audio, model.sr)


def generate_wav_files_with_cloning(
    model: ChatterboxTTS,
    sentences: list,
    voice_path: Path,
    out_directory: Path,
    max_sec: int = 3,
    overwrite: bool = False,
):
    out_directory.mkdir(parents=True, exist_ok=True)
    write(f">>> Creating wav files with cloning for: {voice_path}")
    for index, sentence in tqdm(
        enumerate(sentences),
        total=len(sentences),
        desc="Sentences",
        unit="sentence",
        leave=False,
    ):
        file = Path(f"{out_directory}/{voice_path.stem}_{index}.wav")
        generate_wav_file_with_cloning(
            model, sentence, voice_path, file, max_sec, overwrite
        )


# ======================================================================
# AUGMENTATION
# ======================================================================


class Position(enum.Enum):
    START = (1,)
    MIDDLE = (2,)
    END = 3


def shift_speech(speech, sample_rate, full_duration, position):
    out = np.zeros(sample_rate * full_duration)
    speech_len = len(speech)
    if speech_len > out.size:
        speech = speech[0 : out.size]
    if position == Position.START:
        out[0:speech_len] = speech
    elif position == Position.MIDDLE:
        start = (out.size - speech_len) // 2
        out[start : start + speech_len] = speech
    elif position == Position.END:
        start = out.size - speech_len
        out[start : start + speech_len] = speech

    return out


def resize_waveform(waveform, samplerate, duration, position):
    out = np.zeros(samplerate * duration)
    wave_len = len(waveform)

    if wave_len > out.size:
        waveform = waveform[0 : out.size]

    if position == Position.START:
        out[0:wave_len] = waveform

    elif position == Position.MIDDLE:
        start = (out.size - wave_len) // 2
        out[start : start + wave_len] = waveform

    elif position == Position.END:
        start = out.size - wave_len
        out[start : start + wave_len] = waveform

    return out


def create_files_from_noise(
    wavs_file: Path,
    noise_file: Path,
    out_dir: Path,
    volume=0.5,
    duration=3,
    override=False,
):
    # Load file as waveform
    wavs, sr = librosa.load(wavs_file, sr=None, mono=True)
    noise, noise_sr = librosa.load(noise_file, sr=None, mono=True)

    if sr != noise_sr:
        noise = librosa.resample(noise, orig_sr=noise_sr, target_sr=sr)
    noise = resize_waveform(noise, sr, duration, Position.START)

    out_dir.mkdir(parents=True, exist_ok=True)

    # First file
    out_file = Path(f"{out_dir}/{noise_file.stem}_start_{wavs_file.name}")
    if not override and out_file.exists():
        write(f"    {out_file.name} already exists. Skipping.")
    else:
        waveform_start = resize_waveform(wavs, sr, duration, Position.START)
        noised = waveform_start + (noise * volume)
        sf.write(out_file, noised, int(sr))
        write(f"    {out_file.name} created.")

    # Second file
    out_file = Path(f"{out_dir}/{noise_file.stem}_middle_{wavs_file.name}")
    if not override and out_file.exists():
        write(f"    {out_file.name} already exists. Skipping.")
    else:
        waveform_middle = resize_waveform(wavs, sr, duration, Position.MIDDLE)
        noised = waveform_middle + (noise * volume)
        sf.write(out_file, noised, int(sr))
        write(f"    {out_file.name} created.")

    # third file
    out_file = Path(f"{out_dir}/{noise_file.stem}_end_{wavs_file.name}")
    if not override and out_file.exists():
        write(f"    {out_file.name} already exists. Skipping.")
    else:
        waveform_end = resize_waveform(wavs, sr, duration, Position.END)
        noised = waveform_end + (noise * volume)
        sf.write(out_file, noised, int(sr))
        write(f"    {out_file.name} created.")


def generate_annotations_file(base_dir: Path, out_dir: Path):
    write(">>> Clearing output directory.")
    out_dir.mkdir(parents=True, exist_ok=True)
    for file in out_dir.glob("*"):
        os.remove(file)

    write(">>> Creating npy files.")
    records = []
    for index, file in enumerate(base_dir.rglob("*.wav")):
        label = 1 if "positive" in [p.name.lower() for p in file.parents] else 0

        try:
            # Load waveform and sample_rate
            wavs, sr = librosa.load(file, sr=KOKORO_SAMPLE_RATE, mono=True)
            samples_3sec = int(sr * 3)
            if len(wavs) != samples_3sec:
                zeros = np.zeros(samples_3sec)
                to_add = min(len(wavs), samples_3sec)
                zeros[0:to_add] = wavs[0:to_add]
                wavs = zeros

            mel_spec = librosa.feature.melspectrogram(
                y=wavs,
                sr=sr,
                n_mels=N_MELS,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # type: ignore
            mel_file = Path(f"{out_dir}/{index}.npy")
            np.save(mel_file, mel_spec_db)
            write(f"    {mel_file.name} created.")
            records.append(
                {
                    "index": index,
                    "mel_spec": mel_file,
                    "label": label,
                }
            )

        except Exception as ex:  # noqa: BLE001
            write(f"    {mel_file.name}: {ex}.")

    annotation_file = Path(f"{out_dir}/annotations.csv")
    df = pd.DataFrame(records)
    df.to_csv(annotation_file, index=False, mode="w")
    write(f"    {annotation_file.name} created.")
    write(df["label"].value_counts())
