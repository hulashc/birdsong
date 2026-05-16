# 🎵 Birdsong — Spatiotemporal Acoustic Manifold

> An interactive 3D visualisation of bird vocalisations using MFCC feature extraction, PCA dimensionality reduction, and kNN species classification — with a conversational AI assistant.

**[Live Demo →](https://hulashc.github.io/birdsong)**

![GitHub deployments](https://img.shields.io/github/deployments/hulashc/birdsong/github-pages?label=frontend&style=flat-square)
![Python](https://img.shields.io/badge/python-3.11-blue?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square)
![Three.js](https://img.shields.io/badge/Three.js-r160-black?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

---

## What is this?

Birdsong processes audio recordings into sequences of 57-dimensional acoustic feature vectors (MFCCs + deltas + spectral features), reduces them to 3 principal components via PCA, and renders the result as an animated 3D trajectory — the **Acoustic Manifold**.

Each point in the graph is one 23ms frame of audio. Its position encodes:

| Axis | Name | What it means |
|------|------|---------------|
| X | **Timbre** | The tonal colour of the voice — what makes a blackbird sound like a blackbird |
| Y | **Texture** | How rapidly the sound is changing — smooth whistle vs. rapid trill |
| Z | **Spectral** | Where in the frequency range — deep boom vs. high-pitched call |

Points are **coloured by energy** (dark → silence, amber → peak) and **sized by amplitude**. Species that sound similar trace similar paths through the manifold — which is how the classifier works.

---

## Features

- 🌐 **3D Acoustic Manifold** — real-time Three.js trajectory animation per species
- 🎤 **Upload & Compare** — drag-and-drop MP3/WAV/OGG/FLAC to visualise your own recording
- 🔍 **kNN Classification** — identifies the closest matching species (cosine similarity, k=5)
- 🤖 **AI Assistant** — conversational ornithology assistant powered by Llama 3.1 (Groq)
- 🌙 **Dark / Light mode** — system preference detection with manual toggle
- 📱 **Mobile responsive** — bottom-sheet drawer keeps the 3D canvas fully visible

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    GitHub Pages                      │
│  index.html + src/main.js (Three.js, vanilla JS)    │
│  ← fetches manifold data and chat from API →        │
└──────────────────────┬──────────────────────────────┘
                       │ HTTPS
┌──────────────────────▼──────────────────────────────┐
│                 Render (FastAPI)                     │
│                                                      │
│  /api/species        → list species in database      │
│  /api/manifold/{sp}  → PCA trajectory JSON           │
│  /api/upload         → process uploaded audio        │
│  /api/chat           → LLM conversation turn         │
│                                                      │
│  librosa → 57-feature extraction                     │
│  scikit-learn → PCA fit + transform                  │
│  scikit-learn → kNN cosine similarity                │
│  Groq API → Llama 3.1 8B Instant                    │
└─────────────────────────────────────────────────────┘
```

### Feature Vector (57 dimensions)

```
x_t = [ MFCC(1–13) | Δ MFCC(1–13) | ΔΔ MFCC(1–13) | centroid | bandwidth | rolloff | ZCR | RMS | mel(1–13) ]
```

---

## Project Structure

```
birdsong/
├── api/
│   ├── main.py            # FastAPI app, all endpoints
│   ├── features.py        # MFCC + spectral extraction (librosa)
│   ├── manifold.py        # PCA fit/transform, kNN classifier
│   └── llm.py             # Groq LLM integration
├── src/
│   └── main.js            # Three.js 3D renderer, upload handler
├── birds/                 # Reference audio files per species
├── scripts/               # Data preprocessing utilities
├── process_birds.py       # Build birdsong_data.json from /birds
├── birdsong_data.json     # Pre-extracted PCA manifold data
├── index.html             # Single-page frontend
├── requirements.txt
├── pyproject.toml
├── render.yaml            # Render deployment config
└── .env.example
```

---

## Local Development

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- A [Groq API key](https://console.groq.com) (free)

### 1. Clone and install

```bash
git clone https://github.com/hulashc/birdsong.git
cd birdsong

# with uv (recommended)
uv sync

# or with pip
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
cp .env.example .env
# edit .env and add your key:
# GROQ_API_KEY=gsk_...
```

### 3. Run the backend

```bash
# PowerShell
./run.ps1

# or directly
uvicorn api.main:app --reload --port 8000
```

API docs available at `http://localhost:8000/docs`

### 4. Serve the frontend

Open `index.html` directly in your browser, or use any static server:

```bash
python -m http.server 3000
# then visit http://localhost:3000
```

---

## Adding New Species

1. Add audio files to `birds/<species_name>/` (any common format)
2. Run the processing script:
   ```bash
   python process_birds.py
   ```
3. This rebuilds `birdsong_data.json` with updated PCA manifold data
4. Commit and push — the frontend picks up the new species automatically

---

## API Reference

### `GET /api/species`
Returns list of available species.
```json
["blackbird", "robin", "swift", "wren"]
```

### `GET /api/manifold/{species}`
Returns the PCA trajectory for a species.
```json
{
  "species": "blackbird",
  "points": [
    { "x": 0.42, "y": -1.1, "z": 0.87, "energy": 0.76, "amplitude": 0.54 }
  ]
}
```

### `POST /api/upload`
Upload an audio file for classification.
```
Content-Type: multipart/form-data
Body: file=<audio file>
```
Returns:
```json
{
  "species": "blackbird",
  "matches": [
    { "species": "blackbird", "similarity_pct": 91.2 },
    { "species": "robin",     "similarity_pct": 74.5 }
  ],
  "points": [ ... ]
}
```

### `POST /api/chat`
Send a conversational message to the AI assistant.
```json
{
  "history": [
    { "role": "user",      "content": "Tell me about the blackbird" },
    { "role": "assistant", "content": "The blackbird is..." },
    { "role": "user",      "content": "Where does it live?" }
  ]
}
```
Returns:
```json
{ "role": "assistant", "content": "The blackbird is found across..." }
```

---

## Deployment

### Backend (Render)

Configured via `render.yaml`. On push to `main`, Render automatically redeploys.

Required environment variable in Render dashboard:
```
GROQ_API_KEY = gsk_...
```

### Frontend (GitHub Pages)

Configured in repository **Settings → Pages → Deploy from branch `main`**.  
No build step required — `index.html` is served directly.

> **CORS:** The FastAPI backend has CORS configured to allow requests from the GitHub Pages origin. Update `api/main.py` if your frontend URL changes.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Vanilla JS, Three.js r160, CSS custom properties |
| Backend | Python 3.11, FastAPI, uvicorn |
| Audio processing | librosa, numpy, soundfile |
| ML | scikit-learn (PCA, kNN) |
| LLM | Groq API — Llama 3.1 8B Instant |
| Frontend hosting | GitHub Pages |
| Backend hosting | Render |

---

## Roadmap

- [ ] Live microphone input via Web Audio API
- [ ] xeno-canto API integration (700k+ recordings)
- [ ] BirdNET deep learning classifier backend
- [ ] t-SNE / UMAP manifold comparison mode
- [ ] Shareable manifold URLs
- [ ] Species audio playback in-browser

---

## References

- Davis & Mermelstein (1980) — Original MFCC paper
- McFee et al. (2015) — librosa
- Briggs et al. (2012) — MFCC + kNN bird classification
- Kahl et al. (2021) — [BirdNET](https://github.com/kahst/BirdNET-Analyzer)
- Jolliffe (2002) — Principal Component Analysis
- van der Maaten & Hinton (2008) — t-SNE
- McInnes et al. (2018) — UMAP

---

## Contributing

Pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT © [Hulash Chand](https://github.com/hulashc)
