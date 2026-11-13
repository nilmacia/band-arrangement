"""Section-level dynamics and neutral acoustic-character descriptors."""

from __future__ import annotations

from typing import Any

import librosa
import numpy as np


def _essentia_dynamic_complexity(signal: np.ndarray, sr: int) -> dict[str, float] | None:
    import essentia.standard as es

    complexity, loudness = es.DynamicComplexity(sampleRate=float(sr))(
        np.asarray(signal, dtype=np.float32)
    )
    return {
        "dynamic_complexity_raw": round(float(complexity), 4),
        "essentia_loudness_db": round(float(loudness), 2),
    }


def analyze_dynamics(
    y: np.ndarray,
    sr: int,
    sections: list[dict[str, Any]] | None = None,
    use_essentia: bool = True,
) -> dict[str, Any]:
    hop_length = 512
    n_fft = 2048
    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    rms = librosa.feature.rms(S=stft, frame_length=n_fft)[0]
    centroid = librosa.feature.spectral_centroid(S=stft, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(S=stft)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    frame_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    min_len = min(len(rms), len(centroid), len(flatness), len(onset))
    rms, centroid, flatness, onset = (
        rms[:min_len], centroid[:min_len], flatness[:min_len], onset[:min_len]
    )
    frame_times = frame_times[:min_len]
    duration = float(len(y) / sr)

    curves: list[dict[str, float]] = []
    for second in range(int(np.ceil(duration))):
        mask = (frame_times >= second) & (frame_times < second + 1.0)
        if not np.any(mask):
            continue
        local_rms = float(np.mean(rms[mask]))
        curves.append({
            "time": float(second),
            "rms_dbfs": round(20.0 * np.log10(max(local_rms, 1e-8)), 2),
            "spectral_centroid_hz": round(float(np.mean(centroid[mask])), 1),
            "spectral_flatness": round(float(np.mean(flatness[mask])), 4),
            "onset_strength": round(float(np.mean(onset[mask])), 4),
        })

    source_sections = sections or [{"start_time": 0.0, "end_time": duration, "label": "Whole song"}]
    section_summaries: list[dict[str, Any]] = []
    for section in source_sections:
        start = max(0.0, float(section["start_time"]))
        end = min(duration, float(section["end_time"]))
        mask = (frame_times >= start) & (frame_times < end)
        if not np.any(mask):
            continue
        mean_rms = float(np.mean(rms[mask]))
        summary: dict[str, Any] = {
            "label": section.get("label", "Section"),
            "start_time": round(start, 3),
            "end_time": round(end, 3),
            "rms_dbfs_mean": round(20.0 * np.log10(max(mean_rms, 1e-8)), 2),
            "rms_dbfs_p10": round(float(np.percentile(20.0 * np.log10(np.maximum(rms[mask], 1e-8)), 10)), 2),
            "rms_dbfs_p90": round(float(np.percentile(20.0 * np.log10(np.maximum(rms[mask], 1e-8)), 90)), 2),
            "spectral_centroid_hz_mean": round(float(np.mean(centroid[mask])), 1),
            "spectral_flatness_mean": round(float(np.mean(flatness[mask])), 4),
            "onset_strength_mean": round(float(np.mean(onset[mask])), 4),
        }
        if use_essentia and end - start >= 0.5:
            try:
                left = int(start * sr)
                right = min(len(y), max(left + 1, int(end * sr)))
                extra = _essentia_dynamic_complexity(y[left:right], sr)
                if extra:
                    summary.update(extra)
            except Exception:
                # The song-level try below records the install/API warning once.
                pass
        section_summaries.append(summary)

    if section_summaries:
        energy_values = np.array([s["rms_dbfs_mean"] for s in section_summaries])
        density_values = np.array([s["onset_strength_mean"] for s in section_summaries])
        centroid_values = np.array([s["spectral_centroid_hz_mean"] for s in section_summaries])
        for item in section_summaries:
            e = item["rms_dbfs_mean"]
            d = item["onset_strength_mean"]
            item["energy"] = "high" if e >= np.percentile(energy_values, 67) else ("low" if e <= np.percentile(energy_values, 33) else "medium")
            item["onset_density"] = "high" if d >= np.percentile(density_values, 67) else ("low" if d <= np.percentile(density_values, 33) else "medium")
            item["brightness"] = "bright" if item["spectral_centroid_hz_mean"] >= np.median(centroid_values) else "dark"

    global_essentia: dict[str, float] | None = None
    warnings: list[str] = []
    if use_essentia:
        try:
            global_essentia = _essentia_dynamic_complexity(y, sr)
        except Exception as exc:
            warnings.append(f"Essentia dynamic complexity unavailable ({type(exc).__name__}: {exc}).")

    return {
        "curves_1s": curves,
        "sections": section_summaries,
        "global_essentia": global_essentia,
        "method": "librosa_acoustic_curves" + ("_plus_essentia_dynamic_complexity" if global_essentia else ""),
        "warnings": warnings + [
            "Character descriptors are relative acoustic summaries, not mood or genre predictions.",
            "RMS is measured on the mastered mix; compression and mastering affect its interpretation.",
        ],
    }
