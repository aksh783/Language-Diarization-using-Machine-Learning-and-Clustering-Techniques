import matplotlib.pyplot as plt
import pandas as pd
import librosa
import librosa.display
import numpy as np

audio_path = r"clips\mixed_audio.mp3"
y, sr = librosa.load(audio_path, sr=None)
duration = librosa.get_duration(y=y, sr=sr)

segments_csv = "predicted_language_segments_strict.csv"  # use your actual CSV
df = pd.read_csv(segments_csv)

plt.figure(figsize=(12, 5))
librosa.display.waveshow(y, sr=sr, alpha=0.5)
plt.title("Language Diarization Timeline", fontsize=14)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

colors = {"English": "skyblue", "Hindi": "orange"}
for _, row in df.iterrows():
    plt.axvspan(row["start_s"], row["end_s"], color=colors.get(row["language"], "gray"), alpha=0.4, label=row["language"])

handles, labels = plt.gca().get_legend_handles_labels()
unique = dict(zip(labels, handles))
plt.legend(unique.values(), unique.keys(), loc="upper right")

plt.tight_layout()
plt.show()
