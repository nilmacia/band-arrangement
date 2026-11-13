"""Key and chord estimation from CQT chroma with optional Essentia cross-checks.

The chord template matcher is intentionally transparent and lightweight.
Essentia adds an independent key estimate and triad chord proposal; it is not
blindly treated as ground truth because its documented chord detector is
experimental and limited to major/minor triads.
"""

from __future__ import annotations

from typing import Any

import librosa
import numpy as np

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_QUALITIES = {
    "": [0, 4, 7],
    "m": [0, 3, 7],
    "dim": [0, 3, 6],
    "aug": [0, 4, 8],
    "7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "m7": [0, 3, 7, 10],
    "sus4": [0, 5, 7],
}


def _load_essentia():
    import essentia.standard as es

    return es


def _normalize(x: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(x))
    return x / norm if norm > 1e-9 else np.zeros_like(x)


def _estimate_key(chroma_mean: np.ndarray) -> dict[str, Any]:
    profile = _normalize(chroma_mean.astype(float))
    candidates: list[tuple[float, str, str]] = []
    for root in range(12):
        major = np.roll(_MAJOR_PROFILE, root)
        minor = np.roll(_MINOR_PROFILE, root)
        major_corr = float(np.corrcoef(profile, major)[0, 1]) if np.std(profile) > 0 else 0.0
        minor_corr = float(np.corrcoef(profile, minor)[0, 1]) if np.std(profile) > 0 else 0.0
        candidates.extend([(major_corr, PITCH_NAMES[root], "major"), (minor_corr, PITCH_NAMES[root], "minor")])
    candidates.sort(key=lambda item: item[0], reverse=True)
    best = candidates[0]
    return {
        "key": f"{best[1]} {best[2]}",
        "tonic": best[1],
        "mode": best[2],
        "score": round(float(best[0]), 4),
        "runner_up": f"{candidates[1][1]} {candidates[1][2]}",
        "method": "krumhansl_kessler_chroma_correlation",
        "note": "Score is a profile-correlation score, not a calibrated probability.",
    }


def _normalize_pitch_name(note: str) -> str:
    flats = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}
    return flats.get(note, note)


def _essentia_key(y: np.ndarray, sr: int) -> dict[str, Any] | None:
    es = _load_essentia()
    signal = np.asarray(y, dtype=np.float32)
    key, scale, strength = es.KeyExtractor(sampleRate=float(sr))(signal)
    key = str(key)
    scale = str(scale).lower()
    if key.lower() in {"none", "undefined", ""}:
        return None
    return {
        "key": f"{key} {scale}",
        "tonic": key,
        "mode": scale,
        "strength_raw": round(float(strength), 4),
        "method": "essentia_key_extractor",
    }


def _build_chord_templates() -> tuple[list[str], np.ndarray]:
    labels: list[str] = []
    templates: list[np.ndarray] = []
    for root in range(12):
        for quality, intervals in _QUALITIES.items():
            template = np.zeros(12, dtype=float)
            for interval in intervals:
                template[(root + interval) % 12] = 1.0
            labels.append(PITCH_NAMES[root] + quality)
            templates.append(_normalize(template))
    return labels, np.vstack(templates)


_CHORD_LABELS, _CHORD_TEMPLATES = _build_chord_templates()


def _beat_windows(beats: list[float], duration: float) -> list[tuple[float, float, int]]:
    if len(beats) >= 2:
        starts = list(beats)
        median_interval = float(np.median(np.diff(beats)))
        windows = []
        for i, start in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else min(duration, start + median_interval)
            if end > start:
                windows.append((float(start), float(end), i))
        return windows

    windows = []
    start = 0.0
    index = 0
    while start < duration:
        windows.append((start, min(duration, start + 1.0), index))
        start += 1.0
        index += 1
    return windows


def _essentia_chords(
    chroma: np.ndarray, beats: list[float], sr: int, hop_length: int
) -> list[str] | None:
    if len(beats) < 2:
        return None
    es = _load_essentia()
    detector = es.ChordsDetectionBeats(sampleRate=float(sr), hopSize=int(hop_length))
    # Essentia HPCP convention starts at A; librosa chroma starts at C.
    essentia_pcp = np.roll(chroma, shift=-9, axis=0).T
    chords, _strength = detector(
        np.asarray(essentia_pcp, dtype=np.float32),
        np.asarray(beats, dtype=np.float32),
    )
    return [str(chord) for chord in chords]


def _normalize_chord_name(chord: str) -> str:
    if chord in {"N", "NC", "N.C.", ""}:
        return "N.C."
    flats = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}
    for flat, sharp in flats.items():
        if chord.startswith(flat):
            return sharp + chord[len(flat):]
    return chord


def analyze_harmony(
    y: np.ndarray,
    sr: int,
    beats: list[float] | None = None,
    use_essentia: bool = True,
) -> dict[str, Any]:
    hop_length = 512
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    if chroma.size == 0:
        return {"key": None, "chords": [], "method": "chroma_template_baseline", "warnings": ["Could not calculate chroma features."]}

    chroma_mean = np.mean(chroma, axis=1)
    key = _estimate_key(chroma_mean)
    warnings: list[str] = []
    essentia_key = None
    essentia_chords = None
    if use_essentia:
        try:
            essentia_key = _essentia_key(y, sr)
        except Exception as exc:
            warnings.append(f"Essentia key estimate unavailable ({type(exc).__name__}: {exc}).")
        try:
            essentia_chords = _essentia_chords(chroma, beats or [], sr, hop_length)
        except Exception as exc:
            warnings.append(f"Essentia chord cross-check unavailable ({type(exc).__name__}: {exc}).")

    if essentia_key:
        key["essentia_alternative"] = essentia_key
        key["models_agree"] = (_normalize_pitch_name(key["tonic"]) == _normalize_pitch_name(essentia_key["tonic"]) and key["mode"] == essentia_key["mode"])
        if not key["models_agree"]:
            key["note"] += " Essentia returned a different key candidate; review both."

    frame_times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length)
    duration = float(len(y) / sr)
    beat_list = beats or []
    windows = _beat_windows(beat_list, duration)

    per_beat: list[dict[str, Any]] = []
    for start, end, beat_index0 in windows:
        mask = (frame_times >= start) & (frame_times < end)
        if not np.any(mask):
            continue
        vector = _normalize(np.mean(chroma[:, mask], axis=1).astype(float))
        scores = _CHORD_TEMPLATES @ vector
        order = np.argsort(scores)[::-1]
        best_idx = int(order[0])
        second_idx = int(order[1]) if len(order) > 1 else best_idx
        best_score = float(scores[best_idx])
        margin = best_score - float(scores[second_idx])
        item = {
            "beat_index": int(beat_index0 + 1),
            "start_time": round(start, 3),
            "end_time": round(end, 3),
            "chord": _CHORD_LABELS[best_idx] if best_score >= 0.15 else "N.C.",
            "similarity": round(best_score, 4),
            "margin": round(float(margin), 4),
            "uncertain": bool(best_score < 0.55 or margin < 0.025),
        }
        if essentia_chords is not None and beat_index0 < len(essentia_chords):
            alternative = _normalize_chord_name(essentia_chords[beat_index0])
            item["essentia_chord"] = alternative
            item["models_agree"] = (alternative == item["chord"])
            if not item["models_agree"] and alternative != "N.C.":
                item["alternative_chord"] = alternative
        per_beat.append(item)

    # Conservative one-beat glitch smoothing.
    for i in range(1, len(per_beat) - 1):
        if (
            per_beat[i - 1]["chord"] == per_beat[i + 1]["chord"]
            and per_beat[i]["chord"] != per_beat[i - 1]["chord"]
            and per_beat[i]["similarity"] < min(per_beat[i - 1]["similarity"], per_beat[i + 1]["similarity"])
        ):
            per_beat[i]["chord"] = per_beat[i - 1]["chord"]
            per_beat[i]["smoothed"] = True

    spans: list[dict[str, Any]] = []
    for item in per_beat:
        if spans and spans[-1]["chord"] == item["chord"] and abs(spans[-1]["end_time"] - item["start_time"]) < 0.08:
            spans[-1]["end_time"] = item["end_time"]
            spans[-1]["end_beat_index"] = item["beat_index"]
            spans[-1]["similarity"] = round((spans[-1]["similarity"] + item["similarity"]) / 2.0, 4)
            spans[-1]["uncertain"] = bool(spans[-1]["uncertain"] or item["uncertain"])
            if item.get("alternative_chord"):
                spans[-1].setdefault("alternative_chords", []).append(item["alternative_chord"])
        else:
            spans.append({
                "start_time": item["start_time"],
                "end_time": item["end_time"],
                "start_beat_index": item["beat_index"],
                "end_beat_index": item["beat_index"],
                "chord": item["chord"],
                "similarity": item["similarity"],
                "uncertain": item["uncertain"],
                **({"alternative_chord": item["alternative_chord"]} if item.get("alternative_chord") else {}),
            })

    return {
        "key": key,
        "chords": spans,
        "beat_level_chords": per_beat,
        "method": "librosa_cqt_chroma_templates" + ("_with_essentia_crosscheck" if essentia_key or essentia_chords is not None else ""),
        "chord_vocabulary": list(_QUALITIES.keys()),
        "warnings": warnings + [
            "Chord labels are a chroma-template baseline; inversions, extended jazz voicings and dense mixes may be misidentified.",
            "Chord similarity and key scores are heuristic, not calibrated probabilities.",
            "When enabled, Essentia chord proposals are triad-only and are treated as alternatives, not ground truth.",
        ],
    }
