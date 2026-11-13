from postprocessing import _build_bars, _build_beat_grid, build_arrangement


def test_detected_downbeats_build_bar_grid():
    rhythm = {
        "meter": {"numerator": 4},
        "beats": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        "downbeats": [0.0, 2.0],
        "tempo": {"bpm": 120.0},
    }
    bars = _build_bars(rhythm, 4.0)
    assert len(bars) == 2
    assert bars[0]["start_time"] == 0.0
    assert bars[1]["start_time"] == 2.0
    grid = _build_beat_grid(rhythm["beats"], bars)
    assert [beat["beat_in_bar"] for beat in grid] == [1, 2, 3, 4, 1, 2, 3, 4]


def test_arrangement_has_editable_bar_and_beat_coordinates():
    source = {"duration_seconds": 4.0, "filename": "test.wav"}
    rhythm = {
        "tempo": {"bpm": 120.0},
        "meter": {"numerator": 4, "denominator": 4, "label": "4/4"},
        "beats": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        "downbeats": [0.0, 2.0],
        "beat_tracker": "test",
        "warnings": [],
    }
    harmony = {"key": {"key": "C major"}, "chords": [{"start_time": 0.0, "end_time": 2.0, "chord": "C"}], "method": "test", "warnings": []}
    structure = {"sections": [{"start_time": 0.0, "end_time": 4.0, "label": "Section A"}], "method": "test", "warnings": []}
    dynamics = {"sections": [], "curves_1s": [], "method": "test", "warnings": []}

    output = build_arrangement(source, rhythm, harmony, structure, dynamics)
    assert len(output["bars"]) == 2
    assert output["rhythm"]["beat_grid"][4]["bar_number"] == 2
    assert output["chords"][0]["start_bar"] == 1
    assert output["sections"][0]["start_bar"] == 1


def test_chords_remain_valid_when_no_bar_grid_is_available():
    from postprocessing import _enrich_chords

    chords = [{"start_time": 0.2, "end_time": 1.0, "chord": "Am"}]
    result = _enrich_chords(chords, [])
    assert result[0]["chord"] == "Am"
    assert result[0]["start_bar"] is None
    assert result[0]["end_bar"] is None
