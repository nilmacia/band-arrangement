import numpy as np

import harmony


class FakeEssentiaDetector:
    def __init__(self, sampleRate, hopSize):
        self.sample_rate = sampleRate
        self.hop_size = hopSize

    def __call__(self, pcp, ticks):
        # Essentia HPCP bin 0 is A; librosa chroma bin 9 is A.
        assert pcp.shape[1] == 12
        assert np.allclose(pcp[0], np.roll(np.arange(12, dtype=np.float32), -9))
        assert list(ticks) == [0.0, 0.5]
        return ["A"], [0.8]


class FakeEssentia:
    ChordsDetectionBeats = FakeEssentiaDetector


def test_librosa_chroma_is_rotated_to_essentia_hpcp_order(monkeypatch):
    monkeypatch.setattr(harmony, "_load_essentia", lambda: FakeEssentia())
    chroma = np.arange(12, dtype=np.float32).reshape(12, 1)
    result = harmony._essentia_chords(chroma, [0.0, 0.5], 22_050, 512)
    assert result == ["A"]
