"""Fuse module outputs into a bar-oriented arrangement representation."""

from __future__ import annotations

from typing import Any

import numpy as np


def _bar_index_for_time(bars: list[dict[str, Any]], timestamp: float) -> int | None:
    for bar in bars:
        if float(bar["start_time"]) <= timestamp < float(bar["end_time"]):
            return int(bar["number"])
    if bars and timestamp >= float(bars[-1]["end_time"]):
        return int(bars[-1]["number"])
    return None


def _build_bars(rhythm: dict[str, Any], duration: float) -> list[dict[str, Any]]:
    meter = rhythm.get("meter") or {}
    numerator = max(1, int(meter.get("numerator", 4)))
    beats = [float(t) for t in rhythm.get("beats", []) if 0 <= float(t) < duration]
    downbeats = [float(t) for t in rhythm.get("downbeats", []) if 0 <= float(t) < duration]

    # Prefer model downbeats. Do not inject a fake bar 1 at t=0 if the track has
    # an unmetered/pickup intro before its first detected downbeat.
    if len(downbeats) >= 2:
        starts = sorted(set(downbeats))
        source = "detected_downbeats"
    elif len(beats) >= 2:
        # Group the actual beat timestamps into bars; this tracks tempo drift
        # better than extrapolating one global median interval.
        starts = sorted(set(beats[::numerator]))
        source = "grouped_detected_beats"
    else:
        bpm = rhythm.get("tempo", {}).get("bpm") or 120.0
        bar_duration = 60.0 / float(bpm) * numerator
        starts = list(np.arange(0.0, duration, bar_duration, dtype=float))
        source = "provisional_global_bpm"

    bars: list[dict[str, Any]] = []
    for start, next_start in zip(starts, starts[1:] + [duration]):
        end = min(float(next_start), duration)
        if end - start > 0.05:
            bars.append({
                "number": len(bars) + 1,
                "start_time": round(float(start), 3),
                "end_time": round(end, 3),
                "source": source,
            })
    return bars


def _build_beat_grid(
    beats: list[float], bars: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Annotate beat timestamps with one-based bar and beat-in-bar indices."""
    if not beats:
        return []
    beat_array = np.asarray(beats, dtype=float)
    bar_start_indices: list[int] = []
    for bar in bars:
        idx = int(np.argmin(np.abs(beat_array - float(bar["start_time"]))))
        if abs(float(beat_array[idx]) - float(bar["start_time"])) < 0.25:
            bar_start_indices.append(idx)
        else:
            bar_start_indices.append(-1)

    grid: list[dict[str, Any]] = []
    for i, timestamp in enumerate(beats):
        bar_number = _bar_index_for_time(bars, float(timestamp))
        beat_in_bar = None
        if bar_number is not None:
            start_idx = bar_start_indices[bar_number - 1]
            if start_idx >= 0:
                beat_in_bar = i - start_idx + 1
            else:
                # If bar timestamps were interpolated, infer a local beat count
                # from the bar's first detected beat.
                bar = bars[bar_number - 1]
                preceding = np.where(
                    (beat_array >= float(bar["start_time"])) & (beat_array <= timestamp)
                )[0]
                beat_in_bar = int(len(preceding)) or None
        grid.append({
            "index": i + 1,
            "time": round(float(timestamp), 4),
            "bar_number": bar_number,
            "beat_in_bar": beat_in_bar,
        })
    return grid


def _snap_section_boundaries(
    sections: list[dict[str, Any]],
    bars: list[dict[str, Any]],
    tolerance: float = 1.25,
) -> list[dict[str, Any]]:
    if not bars:
        return sections
    bar_starts = [float(bar["start_time"]) for bar in bars]
    endpoints = bar_starts + [float(bars[-1]["end_time"])]
    result: list[dict[str, Any]] = []
    for section in sections:
        item = dict(section)
        start = float(item["start_time"])
        end = float(item["end_time"])

        # Keep a genuine pre-grid intro before the first detected bar intact.
        if start >= bar_starts[0]:
            nearest_start = min(bar_starts, key=lambda t: abs(t - start))
            if abs(nearest_start - start) <= tolerance:
                item["start_time"] = round(nearest_start, 3)
                item["boundary_start_snapped"] = True
            else:
                item["boundary_start_snapped"] = False
        else:
            item["boundary_start_snapped"] = False

        nearest_end = min(endpoints, key=lambda t: abs(t - end))
        if abs(nearest_end - end) <= tolerance:
            item["end_time"] = round(nearest_end, 3)
            item["boundary_end_snapped"] = True
        else:
            item["boundary_end_snapped"] = False

        item["start_bar"] = _bar_index_for_time(bars, float(item["start_time"]))
        if item["start_bar"] is None and float(item["start_time"]) < float(bars[0]["start_time"]) < float(item["end_time"]):
            item["start_bar"] = int(bars[0]["number"])
        item["end_bar"] = _bar_index_for_time(
            bars, max(float(item["start_time"]), float(item["end_time"]) - 1e-3)
        )
        result.append(item)
    return result


def _enrich_chords(
    chords: list[dict[str, Any]], bars: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not bars:
        return [dict(chord, start_bar=None, end_bar=None) for chord in chords]

    result = []
    for chord in chords:
        item = dict(chord)
        item["start_bar"] = _bar_index_for_time(bars, float(item["start_time"]))
        if item["start_bar"] is None and float(item["start_time"]) < float(bars[0]["start_time"]) < float(item["end_time"]):
            item["start_bar"] = int(bars[0]["number"])
        item["end_bar"] = _bar_index_for_time(
            bars, max(float(item["start_time"]), float(item["end_time"]) - 1e-3)
        )
        result.append(item)
    return result


def build_arrangement(
    source: dict[str, Any],
    rhythm: dict[str, Any],
    harmony: dict[str, Any],
    structure: dict[str, Any],
    dynamics: dict[str, Any],
) -> dict[str, Any]:
    duration = float(source["duration_seconds"])
    bars = _build_bars(rhythm, duration)
    sections = _snap_section_boundaries(structure.get("sections", []), bars)
    chords = _enrich_chords(harmony.get("chords", []), bars)
    beat_grid = _build_beat_grid(rhythm.get("beats", []), bars)

    dynamics_sections = dynamics.get("sections", [])
    for section in sections:
        summary = min(
            dynamics_sections,
            key=lambda item: abs(float(item.get("start_time", 0.0)) - float(section["start_time"])),
            default=None,
        )
        if summary and abs(float(summary.get("start_time", 0.0)) - float(section["start_time"])) <= 2.0:
            section["dynamics"] = {
                key: value for key, value in summary.items()
                if key not in {"label", "start_time", "end_time"}
            }

    warnings: list[str] = []
    for module in (rhythm, harmony, structure, dynamics):
        warnings.extend(module.get("warnings", []))

    first_bar_start = float(bars[0]["start_time"]) if bars else 0.0
    return {
        "schema_version": "1.1",
        "source": source,
        "song": {
            "tempo": rhythm.get("tempo"),
            "meter": rhythm.get("meter"),
            "key": harmony.get("key"),
            "grid_start_time": round(first_bar_start, 3),
            "pre_grid_intro_seconds": round(first_bar_start, 3),
        },
        "rhythm": {
            "beat_tracker": rhythm.get("beat_tracker"),
            "beats": rhythm.get("beats", []),
            "beat_grid": beat_grid,
            "downbeats": rhythm.get("downbeats", []),
        },
        "bars": bars,
        "chords": chords,
        "sections": sections,
        "dynamics": {
            "curves_1s": dynamics.get("curves_1s", []),
            "global_essentia": dynamics.get("global_essentia"),
            "method": dynamics.get("method"),
        },
        "analysis": {
            "harmony_method": harmony.get("method"),
            "structure_method": structure.get("method"),
            "dynamics_method": dynamics.get("method"),
            "warnings": sorted(set(warnings)),
        },
    }
