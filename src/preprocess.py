#python file that standardises the audio to mono, 16kHz, segements speech and performs windowing
from pathlib import Path
import librosa
import soundfile as sf

TARGET_SR = 16000
WINDOW_SECONDS = 3
HOP_SECONDS = 1.5


def preprocess(file, output_dir="data/processed"):
    #Load as mono and resample to 16 kHz
    audio, _ = librosa.load(file, sr=TARGET_SR, mono=True)

    #Simple silence removal placeholder.
    #a proper VAD can replace this when the real data is available.
    speech_regions = librosa.effects.split(audio, top_db=30)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    window = int(WINDOW_SECONDS * TARGET_SR)
    hop = int(HOP_SECONDS * TARGET_SR)

    segment_id = 0

    for start, end in speech_regions:
        speech = audio[start:end]

        for i in range(0, len(speech), hop):
            segment = speech[i:i + window]

            if len(segment) < 2 * TARGET_SR:
                continue

            output_file = output_dir / f"{Path(file).stem}_{segment_id}.wav"
            sf.write(output_file, segment, TARGET_SR)

            segment_id += 1
