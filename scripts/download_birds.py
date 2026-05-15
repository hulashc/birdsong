#!/usr/bin/env python3
"""
download_birds.py

Fetches ~20 high-quality bird recordings from the Xeno-canto API v2,
saves them as MP3s into ../birds/, then calls process_birds.py to
regenerate birdsong_data.json.

Usage (from repo root):
    pip install requests
    python scripts/download_birds.py

Xeno-canto recordings are licensed CC BY / CC BY-NC / CC BY-SA.
Credit is printed per file.
"""

import os
import sys
import json
import time
import subprocess
import urllib.request
import urllib.parse

# --------------------------------------------------------------------------
# Target species: chosen for sonic variety (calls, songs, trills, clicks)
# Common UK / European birds the portfolio audience will recognise
# --------------------------------------------------------------------------
SPECIES = [
    "Turdus merula",          # Blackbird
    "Erithacus rubecula",     # Robin
    "Luscinia megarhynchos",  # Nightingale
    "Troglodytes troglodytes", # Wren
    "Fringilla coelebs",      # Chaffinch
    "Carduelis carduelis",    # Goldfinch
    "Parus major",            # Great Tit
    "Cyanistes caeruleus",    # Blue Tit
    "Phylloscopus trochilus", # Willow Warbler
    "Sylvia atricapilla",     # Blackcap
    "Columba palumbus",       # Wood Pigeon
    "Cuculus canorus",        # Cuckoo
    "Apus apus",              # Common Swift
    "Alcedo atthis",          # Kingfisher
    "Corvus corone",          # Carrion Crow
    "Garrulus glandarius",    # Jay
    "Pica pica",              # Magpie
    "Sturnus vulgaris",       # Starling
    "Alauda arvensis",        # Skylark
    "Motacilla alba",         # Pied Wagtail
]

# Friendly display name (also becomes the filename / JSON key)
DISPLAY = [
    "blackbird", "robin", "nightingale", "wren", "chaffinch",
    "goldfinch", "great_tit", "blue_tit", "willow_warbler", "blackcap",
    "wood_pigeon", "cuckoo", "common_swift", "kingfisher", "carrion_crow",
    "jay", "magpie", "starling", "skylark", "pied_wagtail",
]

BIRDS_DIR = os.path.join(os.path.dirname(__file__), "..", "birds")
BIRDS_DIR = os.path.abspath(BIRDS_DIR)
os.makedirs(BIRDS_DIR, exist_ok=True)

XC_API = "https://xeno-canto.org/api/2/recordings"


def fetch_best_recording(latin_name):
    """Return (mp3_url, attribution) for the highest-quality A-graded recording."""
    query = urllib.parse.urlencode({"query": f"{latin_name} q:A type:song"})
    url = f"{XC_API}?{query}"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            data = json.loads(r.read())
    except Exception as e:
        print(f"  [API error] {e}")
        return None, None

    recordings = data.get("recordings", [])
    if not recordings:
        # Relax quality filter
        query = urllib.parse.urlencode({"query": f"{latin_name} type:song"})
        url = f"{XC_API}?{query}"
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                data = json.loads(r.read())
        except Exception:
            return None, None
        recordings = data.get("recordings", [])

    if not recordings:
        return None, None

    rec = recordings[0]
    file_url = rec.get("file", "")
    if not file_url.startswith("http"):
        file_url = "https:" + file_url
    attrib = f"{rec.get('rec','?')} / XC{rec.get('id','?')} / {rec.get('lic','?')}"
    return file_url, attrib


def download_file(url, dest_path):
    req = urllib.request.Request(url, headers={"User-Agent": "birdsong-portfolio/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r, open(dest_path, "wb") as f:
        f.write(r.read())


credits = {}
downloaded = []

for latin, name in zip(SPECIES, DISPLAY):
    dest = os.path.join(BIRDS_DIR, f"{name}.mp3")
    if os.path.exists(dest):
        print(f"  [skip] {name}.mp3 already exists")
        downloaded.append(name)
        continue

    print(f"Fetching: {name} ({latin})")
    mp3_url, attrib = fetch_best_recording(latin)
    if not mp3_url:
        print(f"  [skip] no recording found")
        continue

    try:
        download_file(mp3_url, dest)
        size_kb = os.path.getsize(dest) // 1024
        print(f"  Saved {name}.mp3 ({size_kb} KB) — {attrib}")
        credits[name] = attrib
        downloaded.append(name)
    except Exception as e:
        print(f"  [download error] {e}")

    time.sleep(0.6)  # polite rate-limit

# Save credits
credits_path = os.path.join(BIRDS_DIR, "CREDITS.md")
with open(credits_path, "w") as f:
    f.write("# Bird Audio Credits\n\nAll recordings from Xeno-canto (xeno-canto.org).\n\n")
    for name, attrib in credits.items():
        f.write(f"- **{name}**: {attrib}\n")

print(f"\nDownloaded {len(downloaded)} files. Running process_birds.py...\n")

# Run the existing processing pipeline
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, ".."))
result = subprocess.run(
    [sys.executable, os.path.join(repo_root, "process_birds.py")],
    cwd=repo_root
)

if result.returncode == 0:
    print("\nDone! birdsong_data.json updated.")
else:
    print("\nprocess_birds.py exited with errors — check output above.")
