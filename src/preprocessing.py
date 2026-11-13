"""Audio decoding and normalization for the analysis pipeline.

We deliberately do not loudness-normalize the waveform: doing so would destroy
part of the song-level and section-level dynamics that the pipeline measures.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass
class AudioData:
    y: np.ndarray
    sr: int
    duration: float
    wav_path: Path
    source_path: Path


def preprocess_audio(
    input_path: str | Path,
    work_dir: str | Path,
    sample_rate: int = 22_050,
) -> AudioData:
    """Decode an audio file to mono float32 WAV using FFmpeg.

    Args:
        input_path: Source MP3/WAV/FLAC/etc.
        work_dir: Directory in which the analysis WAV is written.
        sample_rate: Target sample rate. 22.05 kHz is sufficient for the
            current beat/chroma/structure analysis and keeps CPU use moderate.

    Returns:
        AudioData containing the decoded mono waveform and metadata.
    """
    source = Path(input_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input audio does not exist: {source}")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("FFmpeg was not found. Install it or use the provided Dockerfile.")

    work = Path(work_dir).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    wav_path = work / f"{source.stem}.analysis.wav"

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(source),
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        str(wav_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "FFmpeg failed to decode the source audio: "
            + (completed.stderr.strip() or f"exit code {completed.returncode}")
        )

    y, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = np.mean(y, axis=1, dtype=np.float32)
    if y.size == 0:
        raise ValueError("Decoded audio is empty")
    if not np.all(np.isfinite(y)):
        raise ValueError("Decoded audio contains NaN or infinite samples")

    # Keep the source amplitude relationship intact. Clip only pathological
    # decoder over-range values, which should not occur for normal PCM output.
    y = np.clip(y, -1.0, 1.0).astype(np.float32, copy=False)
    duration = float(len(y) / sr)

    return AudioData(
        y=y,
        sr=int(sr),
        duration=duration,
        wav_path=wav_path,
        source_path=source,
    )
