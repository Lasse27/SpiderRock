import io

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from datasets import Audio, Dataset, load_dataset
from scipy.signal import spectrogram

dataset: Dataset = load_dataset(
    "m-aliabbas/idrak_timit",
    split="train",
)

# Datasetaufbau
print("Description:", dataset.description)
print("Version:", dataset.version)
print("Features:", dataset.features)
print("Columns:", dataset.num_columns)
print("Rows:", dataset.num_rows)

dataset = dataset.cast_column("audio", Audio(decode=False))
sample: dict = next(iter(dataset))  # type: ignore
waveform, sample_rate = sf.read(io.BytesIO(sample["audio"]["bytes"]))
f, t, Sxx = spectrogram(waveform, fs=sample_rate)

import librosa

mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=80)
log_mel = librosa.power_to_db(mel_spec)

fig, ax = plt.subplots()
img = librosa.display.specshow(
    log_mel, x_axis="time", y_axis="mel", sr=sample_rate, fmax=8000, ax=ax
)
librosa.display.colorbar_db(img)
ax.set(title="Mel-frequency spectrogram")

plt.figure()
plt.plot(waveform)
plt.grid()
plt.show()

plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10))
plt.ylabel("Frequenz (Hz)")
plt.xlabel("Zeit (s)")
plt.colorbar(label="dB")
plt.show()
