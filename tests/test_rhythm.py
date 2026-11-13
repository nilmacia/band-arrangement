from pathlib import Path

import numpy as np

import rhythm


class FakeBeatTracker:
    def __call__(self, _path):
        # Upstream File2Beats tuple contract: beats first, downbeats second.
        return [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], [0.0, 2.0, 4.0]


def test_beat_this_tuple_order_is_beats_then_downbeats(monkeypatch):
    monkeypatch.setattr(rhythm, "_load_beat_this", lambda: FakeBeatTracker())
    result = rhythm.analyze_rhythm(
        Path("unused.wav"),
        np.zeros(5 * 22_050, dtype=np.float32),
        22_050,
        use_beat_this=True,
        use_essentia=False,
    )

    assert len(result["beats"]) == 9
    assert result["downbeats"] == [0.0, 2.0, 4.0]
    assert result["meter"]["numerator"] == 4
