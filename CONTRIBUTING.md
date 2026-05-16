# Running locally

## Prerequisites
- Python 3.11+
- Node (for serving the frontend — any static server works)

## Backend

```bash
# 1. Create virtualenv
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate species data (put .mp3/.wav/.ogg files in birds/)
python process_birds.py

# 4. Run the API server
uvicorn api.main:app --reload
# API docs: http://localhost:8000/docs
```

## Frontend

The frontend is a plain ES-module site — just serve the repo root:

```bash
# Python one-liner
python -m http.server 5173
# Then open http://localhost:5173
```

To point the frontend at your local API instead of the static JSON,
set `VITE_API_BASE` (or edit `src/api.js` once added in Phase 3).

## Tests

```bash
pytest api/tests/ -v
```

## Deployment (Render)

1. Push to GitHub — Render auto-deploys on push to `main` via `render.yaml`.
2. Set env var `BIRDSONG_DATA_PATH` to the absolute path of `birdsong_data.json`
   on the Render disk, or upload the file as a [Render Disk](https://render.com/docs/disks).
3. Add a Render Cron Job: `GET https://<your-app>.onrender.com/health` every 14 min
   to prevent free-tier spin-down.
