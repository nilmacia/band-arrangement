"""CLI entry point for the CPU-first band arrangement analysis pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from dynamics import analyze_dynamics
from harmony import analyze_harmony
from postprocessing import build_arrangement
from preprocessing import preprocess_audio
from rhythm import analyze_rhythm
from structure import analyze_structure


def analyze_song(
    input_path: str | Path,
    output_path: str | Path,
    use_beat_this: bool = True,
    use_essentia: bool = True,
    sample_rate: int = 22_050,
) -> dict[str, Any]:
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    work_dir = output.parent / "_work"

    print("[1/6] Preprocessing audio...", flush=True)
    audio = preprocess_audio(input_path, work_dir=work_dir, sample_rate=sample_rate)
    source = {
        "filename": audio.source_path.name,
        "duration_seconds": round(audio.duration, 3),
        "sample_rate": audio.sr,
        "channels_analyzed": 1,
    }

    print("[2/6] Rhythm analysis...", flush=True)
    rhythm = analyze_rhythm(
        audio.wav_path, audio.y, audio.sr,
        use_beat_this=use_beat_this,
        use_essentia=use_essentia,
    )

    print("[3/6] Harmony analysis...", flush=True)
    harmony = analyze_harmony(
        audio.y, audio.sr,
        beats=rhythm.get("beats", []),
        use_essentia=use_essentia,
    )

    print("[4/6] Structure analysis...", flush=True)
    structure = analyze_structure(audio.y, audio.sr)

    print("[5/6] Dynamics + arrangement fusion...", flush=True)
    dynamics = analyze_dynamics(
        audio.y, audio.sr,
        sections=structure.get("sections", []),
        use_essentia=use_essentia,
    )
    arrangement = build_arrangement(source, rhythm, harmony, structure, dynamics)

    print("[6/6] Writing JSON...", flush=True)
    output.write_text(json.dumps(arrangement, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote arrangement: {output}", flush=True)
    return arrangement


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a first-pass musical arrangement from an MP3/audio file."
    )
    parser.add_argument("--input", required=True, help="Source MP3/WAV/FLAC/etc.")
    parser.add_argument("--output", required=True, help="Output arrangement JSON path")
    parser.add_argument("--sample-rate", type=int, default=22_050)
    parser.add_argument("--no-beat-this", action="store_true", help="Skip Beat This! and use Essentia/librosa")
    parser.add_argument("--no-essentia", action="store_true", help="Disable Essentia cross-checks/fallbacks")
    args = parser.parse_args()

    try:
        analyze_song(
            input_path=args.input,
            output_path=args.output,
            use_beat_this=not args.no_beat_this,
            use_essentia=not args.no_essentia,
            sample_rate=args.sample_rate,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
