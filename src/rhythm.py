"""Beat, downbeat, tempo and coarse meter estimation.

Beat This! is the primary beat/downbeat detector. Essentia RhythmExtractor2013
is an independent tempo/beat fallback and cross-check; librosa is the final
fallback. Meter is deliberately reported as a candidate, not ground truth.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import librosa
import numpy as np


@lru_cache(maxsize=1)
def _load_beat_this():
    from beat_this.inference import File2Beats

    return File2Beats(checkpoint_path="final0", device="cpu", dbn=False)


def _load_essentia():
    import essentia.standard as es

    return es


def _as_times(values: Any) -> list[float]:
    if values is None:
        return []
    array = np.asarray(values, dtype=float).reshape(-1)
    return [float(x) for x in array if np.isfinite(x) and x >= 0.0]


def _essentia_rhythm(y: np.ndarray, sr: int) -> tuple[list[float], float | None, float | None]:
    """Essentia RhythmExtractor2013 requires 44.1 kHz input."""
    es = _load_essentia()
    target_sr = 44_100
    signal = y
    if sr != target_sr:
        signal = librosa.resample(
            np.asarray(y, dtype=np.float32), orig_sr=sr, target_sr=target_sr
        )
    extractor = es.RhythmExtractor2013(
        method="multifeature", minTempo=40, maxTempo=250
    )
    bpm, ticks, confidence, _estimates, _intervals = extractor(
        np.asarray(signal, dtype=np.float32)
    )
    bpm_value = float(bpm) if np.isfinite(bpm) and bpm > 0 else None
    confidence_value = float(confidence) if np.isfinite(confidence) else None
    return _as_times(ticks), bpm_value, confidence_value


def _librosa_fallback(y: np.ndarray, sr: int) -> tuple[list[float], float | None]:
    tempo, beat_times = librosa.beat.beat_track(
        y=y, sr=sr, units="time", trim=False
    )
    tempo_array = np.asarray(tempo, dtype=float).reshape(-1)
    bpm = float(tempo_array[0]) if tempo_array.size else None
    if bpm is not None and (not np.isfinite(bpm) or bpm <= 0):
        bpm = None
    return _as_times(beat_times), bpm


def _estimate_meter(beats: list[float], downbeats: list[float]) -> dict[str, Any]:
    """Infer a numerator candidate from downbeat spacing.

    This cannot reliably distinguish quarter-note and compound-meter
    interpretations. It is a useful starting guess for an editable grid.
    """
    if len(beats) >= 2 and len(downbeats) >= 2:
        beat_array = np.asarray(beats, dtype=float)
        indices = [int(np.argmin(np.abs(beat_array - t))) for t in downbeats]
        gaps = np.diff(indices)
        gaps = gaps[(gaps >= 2) & (gaps <= 8)]
        if gaps.size:
            candidates = np.arange(2, 9)
            counts = np.array([np.sum(gaps == c) for c in candidates])
            best_idx = int(np.argmax(counts))
            numerator = int(candidates[best_idx])
            consistency = float(counts[best_idx] / len(gaps))
            return {
                "numerator": numerator,
                "denominator": 4,
                "label": f"{numerator}/4",
                "confidence_score": round(consistency, 3),
                "method": "beat_this_downbeat_spacing_heuristic",
                "note": "Working meter candidate; denominator and compound-meter interpretation are not reliably inferred.",
            }

    return {
        "numerator": 4,
        "denominator": 4,
        "label": "4/4",
        "confidence_score": 0.2,
        "method": "fallback_default",
        "note": "Provisional 4/4 grid; verify manually.",
    }


def _tempo_summary(
    beats: list[float],
    fallback_bpm: float | None,
    essentia_bpm: float | None,
    essentia_confidence: float | None,
) -> dict[str, Any]:
    intervals = np.diff(np.asarray(beats, dtype=float)) if len(beats) >= 2 else np.asarray([])
    intervals = intervals[(intervals > 0.2) & (intervals < 2.0)]
    bpm = 60.0 / float(np.median(intervals)) if intervals.size else fallback_bpm
    if bpm is not None and (not np.isfinite(bpm) or bpm <= 0):
        bpm = None

    tempo_map: list[dict[str, float]] = []
    if len(beats) >= 10:
        b = np.asarray(beats, dtype=float)
        local: list[tuple[float, float]] = []
        for start in range(0, len(b) - 8, 4):
            diffs = np.diff(b[start : start + 9])
            diffs = diffs[(diffs > 0.2) & (diffs < 2.0)]
            if diffs.size:
                local.append((float(b[start]), 60.0 / float(np.median(diffs))))
        if len(local) >= 2:
            local_bpms = [item[1] for item in local]
            if (max(local_bpms) - min(local_bpms)) / max(float(np.median(local_bpms)), 1.0) > 0.025:
                tempo_map = [
                    {"time": round(t, 3), "bpm": round(value, 2)}
                    for t, value in local
                ]

    alternatives: list[float] = []
    if bpm is not None:
        alternatives = [round(bpm / 2.0, 2), round(bpm * 2.0, 2)]

    discrepancy = None
    if bpm is not None and essentia_bpm is not None:
        # Account for the common half/double tempo ambiguity before comparing.
        variants = [essentia_bpm / 2.0, essentia_bpm, essentia_bpm * 2.0]
        closest = min(variants, key=lambda value: abs(value - bpm))
        discrepancy = round(abs(closest - bpm), 2)

    return {
        "bpm": round(float(bpm), 2) if bpm is not None else None,
        "half_time_alternative": alternatives[0] if alternatives else None,
        "double_time_alternative": alternatives[1] if alternatives else None,
        "tempo_map": tempo_map,
        "essentia_bpm": round(essentia_bpm, 2) if essentia_bpm is not None else None,
        "essentia_confidence_raw": round(essentia_confidence, 3) if essentia_confidence is not None else None,
        "essentia_difference_after_half_double_match": discrepancy,
        "method": "beat_interval_median" if intervals.size else "fallback_estimator",
    }


def analyze_rhythm(
    wav_path: str | Path,
    y: np.ndarray,
    sr: int,
    use_beat_this: bool = True,
    use_essentia: bool = True,
) -> dict[str, Any]:
    warnings: list[str] = []
    beats: list[float] = []
    downbeats: list[float] = []
    fallback_bpm: float | None = None
    essentia_bpm: float | None = None
    essentia_confidence: float | None = None
    method = "beat_this"

    if use_beat_this:
        try:
            tracker = _load_beat_this()
            # File2Beats returns (beats, downbeats). The File2File convenience
            # wrapper uses swapped local names, so use File2Beats directly.
            beats_raw, downbeats_raw = tracker(str(wav_path))
            beats = _as_times(beats_raw)
            downbeats = _as_times(downbeats_raw)
            if len(beats) < 2:
                raise RuntimeError("Beat This! returned fewer than two beat positions")
        except Exception as exc:
            warnings.append(
                f"Beat This! unavailable; trying fallback trackers ({type(exc).__name__}: {exc})."
            )
            method = "fallback"

    if len(beats) < 2 and use_essentia:
        try:
            beats, essentia_bpm, essentia_confidence = _essentia_rhythm(y, sr)
            if len(beats) >= 2:
                method = "essentia_rhythm_extractor_2013"
            else:
                raise RuntimeError("Essentia returned fewer than two beat positions")
        except Exception as exc:
            warnings.append(
                f"Essentia rhythm fallback unavailable ({type(exc).__name__}: {exc})."
            )

    # Run Essentia independently as a tempo cross-check even when Beat This!
    # supplied timestamps. Do not replace better timestamp data with its ticks.
    if use_essentia and essentia_bpm is None:
        try:
            _ess_beats, essentia_bpm, essentia_confidence = _essentia_rhythm(y, sr)
        except Exception as exc:
            warnings.append(
                f"Essentia tempo cross-check unavailable ({type(exc).__name__}: {exc})."
            )

    if len(beats) < 2:
        beats, fallback_bpm = _librosa_fallback(y, sr)
        method = "librosa_beat_tracker"

    beats = sorted({round(t, 4) for t in beats if t <= len(y) / sr})
    downbeats = sorted({round(t, 4) for t in downbeats if t <= len(y) / sr})

    if not beats:
        warnings.append("No reliable beat sequence was found; bar grid will be provisional.")

    tempo = _tempo_summary(beats, fallback_bpm, essentia_bpm, essentia_confidence)
    meter = _estimate_meter(beats, downbeats)

    if tempo["essentia_difference_after_half_double_match"] is not None and tempo["essentia_difference_after_half_double_match"] > 8:
        warnings.append(
            "Beat This! and Essentia tempo estimates disagree by more than 8 BPM after half/double-time matching. Verify the tempo."
        )

    if not downbeats:
        warnings.append(
            "No downbeats were detected; bar starts are provisionally grouped from the beat sequence."
        )

    return {
        "tempo": tempo,
        "meter": meter,
        "beats": beats,
        "downbeats": downbeats,
        "beat_tracker": method,
        "warnings": warnings,
    }
