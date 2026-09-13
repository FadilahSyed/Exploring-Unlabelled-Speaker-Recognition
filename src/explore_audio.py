# Python file to extract properties of the audio files 

from pathlib import Path
import librosa

AUDIO_DIR = Path("data/raw")

for file in AUDIO_DIR.glob("*.wav"):
    try:
        audio, sr = librosa.load(file, sr=None, mono=False)

        channels = 1 if audio.ndim == 1 else audio.shape[0]
        duration = librosa.get_duration(y=audio, sr=sr)

        print(
            f"{file.name} | "
            f"{duration:.2f}s | "
            f"{sr} Hz | "
            f"{channels} channel(s)"
        )

    except Exception as error:
        print(f"{file.name} | ERROR: {error}")
