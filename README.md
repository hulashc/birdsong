# Birdsong — Spatiotemporal Acoustic Manifold

An interactive 3D visualisation of birdsong using MFCC feature extraction, PCA dimensionality reduction, and Three.js rendering. Each bird species is represented as a 3D trajectory in acoustic space, animated in real-time and synced to the species' audio.

## What it does

- Extracts 57 audio features per frame (MFCC × 40, Chroma × 12, Centroid, Bandwidth, Rolloff, ZCR, Onset Strength)
- Reduces the feature space to 3 principal components (PC1 = Timbre, PC2 = Texture, PC3 = Spectral)
- Renders the trajectory as a 3D comet-trail animation in the browser
- Dot colour maps to energy: **blue** (silence) → **teal** (mid) → **orange** (peak call)
- Dot size maps to amplitude (larger = louder)

## Project Structure

```
birdsong/
├── index.html            # Main app entry point
├── src/
│   └── main.js           # Three.js visualisation logic
├── birdsong_data.json    # Pre-computed MFCC/PCA trajectory data
├── process_birds.py      # Python script to generate birdsong_data.json
├── birds/                # Audio files (.mp3) — one per species
│   └── *.mp3
└── requirements.txt      # Python dependencies for process_birds.py
```

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Add audio files

Place your `.mp3`, `.wav`, `.ogg`, or `.flac` bird audio files in the `birds/` folder. File names become the species identifier (e.g. `robin.mp3` → species `robin`).

### 3. Generate the JSON data

```bash
python process_birds.py
```

This processes all audio files in `birds/` and writes `birdsong_data.json`. It extracts 57 audio features per frame, scales them with `StandardScaler`, reduces to 3D with PCA, and computes normalised RMS energy.

### 4. Run the app

Serve the repo from a local HTTP server (required for `fetch()` to load the JSON):

```bash
# Python
python -m http.server 8080

# Node.js
npx serve .
```

Then open [http://localhost:8080](http://localhost:8080) in your browser.

## Adding a new species

1. Drop the audio file into `birds/` (e.g. `nightingale.mp3`)
2. Re-run `python process_birds.py`
3. The new species will be added to `birdsong_data.json` automatically
4. Access it directly via `?sound=nightingale` in the URL

## URL Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `?sound=` | Select a specific species to display | `?sound=robin` |

If no `?sound=` is provided, the first species in the JSON is shown.

## Controls

- **Click** the overlay to start audio and animation
- **Drag** to orbit the 3D view
- **Scroll** to zoom in/out
- **Pause / Resume** button at the bottom

## Dependencies

| Layer | Library | Version |
|-------|---------|--------|
| 3D rendering | [Three.js](https://threejs.org) | 0.160.0 (CDN) |
| Python audio | librosa | ≥0.10 |
| Python ML | scikit-learn | ≥1.3 |
| Python numerics | numpy | ≥1.24 |

## License

MIT
