"""Lightweight section segmentation and repeated-section grouping.

This first version uses chroma + MFCC + RMS self-similarity features. It does
not use a large pretrained music embedding model, and it deliberately returns
neutral section labels (A/B/C) instead of pretending to know verse/chorus.
"""

from __future__ import annotations

from typing import Any

import librosa
import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.distance import cosine


def _safe_cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) < 1e-9 or np.linalg.norm(b) < 1e-9:
        return 1.0
    return float(cosine(a, b))


def _extract_window_features(y: np.ndarray, sr: int, frame_seconds: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    hop_length = 512
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0][None, :]

    # Align all feature streams to the shortest frame count.
    n_frames = min(chroma.shape[1], mfcc.shape[1], rms.shape[1])
    features = np.vstack([chroma[:, :n_frames], mfcc[:, :n_frames], rms[:, :n_frames]])
    frame_times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)

    # Pool into approximately one-second feature vectors to reduce computation.
    pooled: list[np.ndarray] = []
    pooled_times: list[float] = []
    n_bins = int(np.ceil(float(len(y) / sr) / frame_seconds))
    for i in range(n_bins):
        left = i * frame_seconds
        right = (i + 1) * frame_seconds
        mask = (frame_times >= left) & (frame_times < right)
        if np.any(mask):
            pooled.append(np.mean(features[:, mask], axis=1))
            pooled_times.append((left + min(right, len(y) / sr)) / 2.0)

    if not pooled:
        return np.zeros((1, features.shape[0])), np.array([0.0])

    matrix = np.vstack(pooled).astype(float)
    # Standardize each feature dimension so chroma, MFCC and RMS contribute
    # on comparable scales to the cosine-based novelty measure.
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    matrix = (matrix - mean) / std
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-9)
    return matrix, np.asarray(pooled_times)


def _novelty_curve(features: np.ndarray, context_seconds: int = 8) -> np.ndarray:
    n = len(features)
    novelty = np.zeros(n, dtype=float)
    for i in range(context_seconds, n - context_seconds):
        left = np.mean(features[i - context_seconds : i], axis=0)
        right = np.mean(features[i : i + context_seconds], axis=0)
        novelty[i] = _safe_cosine_distance(left, right)
    # Light smoothing helps suppress one-frame spikes.
    if n >= 3:
        novelty[1:-1] = (novelty[:-2] + 2 * novelty[1:-1] + novelty[2:]) / 4.0
    return novelty


def analyze_structure(y: np.ndarray, sr: int) -> dict[str, Any]:
    duration = float(len(y) / sr)
    if duration < 12.0:
        return {
            "sections": [{"start_time": 0.0, "end_time": round(duration, 3), "group": "A", "occurrence": 1, "label": "Section A"}],
            "method": "short_audio_single_section",
            "warnings": ["Audio is too short for reliable automatic section segmentation."],
        }

    features, times = _extract_window_features(y, sr)
    novelty = _novelty_curve(features, context_seconds=8)
    # Require a meaningful separation between detected boundaries.
    peaks, properties = find_peaks(
        novelty,
        distance=8,
        prominence=max(0.035, float(np.std(novelty) * 0.65)),
    )
    boundaries = [0.0]
    for peak in peaks.tolist():
        if 8.0 <= times[peak] <= duration - 8.0:
            boundaries.append(float(times[peak]))
    boundaries.append(duration)
    boundaries = sorted(boundaries)

    # Merge tiny segments into their neighbour; these are rarely useful as
    # separate editable arrangement sections in the first pass.
    min_segment_seconds = 8.0
    merged = [boundaries[0]]
    for boundary in boundaries[1:-1]:
        if boundary - merged[-1] >= min_segment_seconds:
            merged.append(boundary)
    if duration - merged[-1] < min_segment_seconds and len(merged) > 1:
        merged.pop()
    merged.append(duration)
    boundaries = merged

    raw_sections: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        mask = (times >= start) & (times < end)
        if not np.any(mask):
            continue
        vector = np.mean(features[mask], axis=0)
        raw_sections.append({"start_time": round(float(start), 3), "end_time": round(float(end), 3)})
        vectors.append(vector)

    # Greedy recurrence grouping: similar sections share A/B/C identifiers.
    # This is intentionally conservative and should be treated as a hint.
    group_vectors: list[np.ndarray] = []
    group_names: list[str] = []
    group_occurrences: dict[str, int] = {}
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for section, vector in zip(raw_sections, vectors):
        best_group = None
        best_similarity = -1.0
        for name, reference in zip(group_names, group_vectors):
            similarity = 1.0 - _safe_cosine_distance(vector, reference)
            if similarity > best_similarity:
                best_similarity = similarity
                best_group = name
        if best_group is not None and best_similarity >= 0.88:
            group = best_group
        else:
            group = alphabet[len(group_names) % len(alphabet)]
            group_names.append(group)
            group_vectors.append(vector)
        group_occurrences[group] = group_occurrences.get(group, 0) + 1
        section["group"] = group
        section["occurrence"] = group_occurrences[group]
        section["label"] = f"Section {group}"
        section["repeat_similarity"] = round(float(best_similarity), 3) if best_similarity >= 0 else None

    return {
        "sections": raw_sections,
        "method": "chroma_mfcc_rms_novelty_and_recurrence",
        "warnings": [
            "Section labels are neutral similarity groups (A/B/C), not semantic verse/chorus predictions.",
            "Boundary detection is a lightweight baseline; verify boundaries by listening.",
        ],
    }
