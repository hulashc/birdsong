# Birdsong Acoustic Manifold

[![CI](https://github.com/hulashc/birdsong/actions/workflows/ci.yml/badge.svg)](https://github.com/hulashc/birdsong/actions/workflows/ci.yml)
[![Live Demo](https://img.shields.io/badge/live%20demo-hulashc.github.io-brightgreen)](https://hulashc.github.io/birdsong/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Three.js](https://img.shields.io/badge/Three.js-r165-black?logo=threedotjs)](https://threejs.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

An interactive 3D acoustic manifold visualization of bird vocalisations. Extracts 57-dimensional MFCC feature vectors per frame, reduces them to 3 principal components via PCA, classifies species using kNN (cosine similarity, k=5), and serves it all through a FastAPI backend with a Three.js frontend and a conversational AI assistant powered by Llama 3.1.

**[Try the live demo →](https://hulashc.github.io/birdsong/)**

---

## How It Works

```
Raw audio (.ogg/.mp3/.wav)
    │
    ▼
[librosa] Extract 57-dim features per 23ms frame
    │  MFCCs (40) + Chroma (12) + Centroid + Bandwidth + Rolloff + ZCR + Onset
    ▼
[sklearn] StandardScaler → PCA (57D → 3D)
    │  PC1 = Timbre  |  PC2 = Texture  |  PC3 = Spectral Range
    ▼
[Three.js] Render acoustic manifold as animated 3D trajectory
    │  Points coloured by energy (dark → silence, amber → peak)
    ▼
[kNN] Cosine similarity search → top-3 nearest species
    ▼
[Groq / Llama 3.1] Conversational ornithology assistant
```

---

## Architecture

```
birdssong/
├── .github/workflows/ci.yml   # pytest on every push
├── api/
│   ├── main.py                  # FastAPI app + all endpoints
│   ├── pipeline.py              # Audio → manifold pipeline
│   ├── store.py                 # SpeciesStore + kNN classify
│   ├── llm.py                   # Groq / Llama 3.1 integration
│   └── tests/
│       └── test_pipeline.py     # Unit tests (pytest)
├── src/
│   ├── main.js                  # Three.js scene + animation loop
│   ├── features.js              # Browser-side feature extraction
│   ├── knn.js                   # Browser-side kNN classifier
│   └── upload.js                # Drag-and-drop upload UI
├── birds/                       # Source audio files (.ogg)
├── birdsong_data.json           # Pre-processed manifold data
├── process_birds.py             # Batch audio → JSON pipeline
├── scripts/download_birds.py    # Download 20 UK bird recordings
├── render.yaml                  # Render deployment config
└── requirements.txt
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/hulashc/birdsong.git
cd birdsong

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download bird audio (20 UK species from Wikimedia Commons)
python scripts/download_birds.py

# 4. Process audio → birdsong_data.json
python process_birds.py

# 5. Run the API
PYTHONPATH=. uvicorn api.main:app --reload

# 6. Open the frontend
# Open index.html in your browser, or serve it:
npx serve .
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes (for LLM features) | Groq API key — get one free at [console.groq.com](https://console.groq.com) |
| `BIRDSONG_DATA_PATH` | No | Path to `birdsong_data.json` (default: `birdsong_data.json`) |

Create a `.env` file from the provided example:
```bash
cp .env.example .env
# Then add your GROQ_API_KEY
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe + species count + LLM status |
| `GET` | `/api/species` | List all species with metadata |
| `POST` | `/api/process` | Upload audio file → manifold JSON |
| `POST` | `/api/classify` | Manifold JSON → top-K nearest species |
| `POST` | `/api/describe` | Classification result → LLM species description |
| `POST` | `/api/search` | Natural language query → ranked species |
| `POST` | `/api/chat` | Multi-turn conversational birdsong assistant |

Full interactive docs available at `/docs` (Swagger UI) when the API is running.

---

## Running Tests

```bash
PYTHONPATH=. pytest api/tests/ -v
```

Tests cover:
- Feature extraction schema validation
- XYZ coordinate normalisation (`[-1, 1]`)
- Energy value clamping (`[0, 1]`)
- Short audio `ValueError` handling
- `SpeciesStore.classify` ranking correctness

---

## Tech Stack

| Layer | Technology |
|---|---|
| Audio features | librosa (MFCC, Chroma, Spectral) |
| Dimensionality reduction | scikit-learn PCA |
| Classification | kNN centroid distance (browser + server) |
| Backend | FastAPI + Uvicorn |
| Frontend 3D | Three.js + OrbitControls |
| LLM | Groq / Llama 3.1 8B Instant |
| Deployment | Render (API) + GitHub Pages (frontend) |
| CI | GitHub Actions |

---

## Research Paper

A full academic paper documenting the mathematical model, feature extraction pipeline, PCA reduction, and system architecture is available in the repository: [`Birdsong_paper.pdf`](Birdsong_paper.pdf)

---

## Data Sources

All bird audio sourced from [Wikimedia Commons](https://commons.wikimedia.org/) under public domain or CC licenses. Run `scripts/download_birds.py` to download them automatically. A `CREDITS.md` file is generated in the `birds/` directory listing every source URL.
