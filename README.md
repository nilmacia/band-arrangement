# Band arrangement analysis (CPU-first starter)

A small offline pipeline that turns an official stereo MP3 into a first-pass, editable musical map:

- tempo, beat and downbeat timestamps;
- a provisional time signature and bar grid;
- global key and timestamped chord guesses;
- neutral section groups (A/B/C) based on musical similarity;
- section-level dynamics and acoustic-character descriptors.

No editor, source separation, instrument transcription, MIDI generation, or GPU is included.

## Project layout

```text
.
├── Dockerfile
├── requirements.txt
└── src
    ├── preprocessing.py
    ├── rhythm.py
    ├── harmony.py
    ├── structure.py
    ├── dynamics.py
    ├── postprocessing.py
    └── pipeline.py
```

## Run with Docker

Build:

```bash
docker build -t band-arrangement .
```

Analyze a song:

```bash
mkdir -p data/out data/models

docker run --rm \
  -v "$PWD/data:/data" \
  -v "$PWD/data/models:/models" \
  band-arrangement \
  --input /data/song.mp3 \
  --output /data/out/song.json
```

The first run downloads the Beat This! checkpoint. The `TORCH_HOME=/models/torch` cache is persisted by the `/models` mount. For a fallback-only run without Beat This!:

```bash
docker run --rm \
  -v "$PWD/data:/data" \
  -v "$PWD/data/models:/models" \
  band-arrangement \
  --input /data/song.mp3 \
  --output /data/out/song.json \
  --no-beat-this
```

To disable Essentia cross-checks as well:

```bash
docker run --rm \
  -v "$PWD/data:/data" \
  -v "$PWD/data/models:/models" \
  band-arrangement \
  --input /data/song.mp3 \
  --output /data/out/song-no-essentia.json \
  --no-essentia
```

## Run without Docker

Requires Python 3.11, FFmpeg, and a CPU build of PyTorch:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio
pip install -r requirements.txt
python pipeline.py --input song.mp3 --output out/song.json
```

For a quick no-model smoke test (uses librosa fallbacks and omits Essentia):

```bash
python pipeline.py --input testdata/synthetic.wav --output testdata/smoke.json --no-beat-this --no-essentia
```

Run the unit tests with `pip install pytest && pytest -q`.

## Module responsibilities

- `preprocessing.py`: FFmpeg decode to mono 22.05 kHz WAV. Does not loudness-normalize because the dynamics analysis needs the original level relationships.
- `rhythm.py`: Beat This! beat/downbeat detector; Essentia RhythmExtractor2013 for tempo cross-check/fallback; librosa as the final fallback. Meter is a cautious downbeat-spacing heuristic.
- `harmony.py`: librosa CQT chroma, Krumhansl-style key estimate and chord-template matching; Essentia key/chord outputs are independent cross-checks.
- `structure.py`: chroma/MFCC/RMS novelty segmentation and coarse repeated-section grouping. Labels remain neutral (A/B/C).
- `dynamics.py`: one-second RMS, spectral centroid, spectral flatness and onset-strength curves; Essentia's DynamicComplexity adds a global and per-section dynamic descriptor.
- `postprocessing.py`: constructs bars and beat-to-bar indices, maps chords/sections to bars, and attaches section dynamics.
- `pipeline.py`: CLI orchestration and JSON serialization.

## Important limitations

1. **Time signature is heuristic.** Beat/downbeat tracking does not robustly resolve denominator or compound-meter interpretation (for example, 3/4 versus 6/8). Verify it manually.
2. **Chords are a first-pass estimate.** The transparent chroma-template matcher covers major/minor, diminished, augmented, dominant-7, major-7, minor-7 and sus4 templates. It will struggle with inversions, slash chords, extended jazz voicings, key changes and dense mixes. Essentia's built-in chord detector is triad-only and documented as experimental, so its result is retained as a cross-check/alternative rather than silently overriding the main estimate.
3. **Structure labels are neutral.** The algorithm finds novelty boundaries and repeated similarity groups (`Section A`, `Section B`); it does not claim to know which is a verse or chorus.
4. **Dynamics/character are acoustic descriptors.** Energy, onset density, brightness and dynamic complexity are measurements, not mood or genre predictions.
5. **Mono analysis is a simplification.** FFmpeg downmixes to mono for this first version. Some stereo-specific mix cues are therefore discarded.
6. **No performance claim yet.** This is an executable starting point, not an assertion of 80% accuracy. Evaluate it on representative songs you can annotate.

## Output conventions

- All timestamps are seconds from the start of the decoded audio.
- Beat `index`, `bar_number`, and `beat_in_bar` are one-based; `bar_number` can be null for a pickup/pre-grid intro.
- Chord `start_bar`/`end_bar` and section `start_bar`/`end_bar` are one-based and inclusive; they may be null when a span falls before the first detected bar.
- Time spans use exclusive `end_time` semantics.
- Fields named `score`, `similarity`, `strength_raw`, or `confidence_score` are heuristic/raw model outputs, not calibrated probabilities.
- `analysis.warnings` contains known limitations and fallback notices.

## Licensing note

Beat This! code and released model weights are MIT-licensed in its upstream repository. Essentia is AGPL-3.0-only; since this is a private, non-distributed app, that may be acceptable for your use, but the AGPL network-use clause can matter if you later expose the app as a service. Reassess the license before making it accessible to other users or distributing it. Sources: [Beat This! repository](https://github.com/CPJKU/beat_this), [Essentia documentation](https://essentia.upf.edu/documentation.html), [Essentia PyPI metadata](https://pypi.org/project/essentia/).

## Next step

Build a small ground-truth set across rock, soul, jazz, funk and reggae. Measure beat F-measure, downbeat accuracy, chord segment accuracy, section-boundary F-measure and—most importantly—average correction time per song.
