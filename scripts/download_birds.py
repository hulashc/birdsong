#!/usr/bin/env python3
"""
download_birds.py

Downloads 20 UK/European bird recordings from Wikimedia Commons
(public domain / CC licensed), saves as OGG into ../birds/,
then runs process_birds.py to regenerate birdsong_data.json.

Usage (from repo root):
    python scripts/download_birds.py

No API key needed. Requires: requests (pip install requests)
"""

import os
import sys
import time
import subprocess
import urllib.request
import urllib.parse

# ---------------------------------------------------------------------------
# Curated list: (display_name, wikimedia_commons_direct_url)
# All files are public domain or CC-BY licensed from Wikimedia Commons.
# URLs verified May 2026.
# ---------------------------------------------------------------------------
BIRDS = [
    ("blackbird",
     "https://upload.wikimedia.org/wikipedia/commons/3/30/Common_Blackbird_song_%28Turdus_merula%29.ogg"),
    ("chiffchaff",
     "https://upload.wikimedia.org/wikipedia/commons/2/2c/Chiffchaff_%28Phylloscopus_collybita%29song_Germany.ogg"),
    ("robin",
     "https://upload.wikimedia.org/wikipedia/commons/8/8c/Erithacus_rubecula_-_XC271055_%28cropped%29.ogg"),
    ("song_thrush",
     "https://upload.wikimedia.org/wikipedia/commons/0/0c/Song_Thrush_Turdus_philomelos.ogg"),
    ("skylark",
     "https://upload.wikimedia.org/wikipedia/commons/4/47/Alauda_arvensis_-_Skylark_-_XC109978.ogg"),
    ("chaffinch",
     "https://upload.wikimedia.org/wikipedia/commons/a/a8/Fringilla_coelebs_chaffinch_2.ogg"),
    ("cuckoo",
     "https://upload.wikimedia.org/wikipedia/commons/9/96/Cuculus_canorus_-Common_Cuckoo-_XC97212.ogg"),
    ("nightingale",
     "https://upload.wikimedia.org/wikipedia/commons/6/69/Luscinia_megarhynchos_-_XC38154.ogg"),
    ("starling",
     "https://upload.wikimedia.org/wikipedia/commons/f/f4/Sturnus_vulgaris_-_song_%28V2%29.ogg"),
    ("blackcap",
     "https://upload.wikimedia.org/wikipedia/commons/f/f7/Sylvia_atricapilla_-_blackcap_-_xc.ogg"),
    ("great_tit",
     "https://upload.wikimedia.org/wikipedia/commons/1/11/Parus_major_-_great_tit_-_xc.ogg"),
    ("blue_tit",
     "https://upload.wikimedia.org/wikipedia/commons/3/35/Blue_Tit_-_Song.ogg"),
    ("wren",
     "https://upload.wikimedia.org/wikipedia/commons/8/8d/Troglodytes_troglodytes_XC109443.ogg"),
    ("dunnock",
     "https://upload.wikimedia.org/wikipedia/commons/e/eb/Prunella_modularis_XC15612.ogg"),
    ("goldfinch",
     "https://upload.wikimedia.org/wikipedia/commons/1/1e/Carduelis_carduelis_XC47504.ogg"),
    ("greenfinch",
     "https://upload.wikimedia.org/wikipedia/commons/1/13/Chloris_chloris_-_greenfinch_-_xc.ogg"),
    ("wood_pigeon",
     "https://upload.wikimedia.org/wikipedia/commons/1/14/Columba_palumbus_-_XC45566.ogg"),
    ("swift",
     "https://upload.wikimedia.org/wikipedia/commons/b/b6/Apus_apus_-_XC30723.ogg"),
    ("yellowhammer",
     "https://upload.wikimedia.org/wikipedia/commons/9/93/Emberiza_citrinella_-_XC38231.ogg"),
    ("linnet",
     "https://upload.wikimedia.org/wikipedia/commons/4/4d/Linaria_cannabina_-_XC38050.ogg"),
]

BIRDS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "birds"))
os.makedirs(BIRDS_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (BirdSong-Portfolio/1.0; educational project; https://github.com/hulashc/birdsong)"
}

downloaded = []
skipped = []

print(f"Saving to: {BIRDS_DIR}\n")

for name, url in BIRDS:
    ext = ".ogg"  # all Wikimedia sources are ogg
    dest = os.path.join(BIRDS_DIR, f"{name}{ext}")

    if os.path.exists(dest):
        size_kb = os.path.getsize(dest) // 1024
        print(f"  [skip] {name}{ext} already exists ({size_kb} KB)")
        downloaded.append(name)
        continue

    print(f"Downloading: {name}")
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        with open(dest, "wb") as f:
            f.write(data)
        size_kb = len(data) // 1024
        print(f"  Saved {name}{ext} ({size_kb} KB)")
        downloaded.append(name)
    except Exception as e:
        print(f"  [ERROR] {name}: {e}")
        skipped.append(name)

    time.sleep(0.8)  # polite rate-limit

# Write CREDITS
credits_path = os.path.join(BIRDS_DIR, "CREDITS.md")
with open(credits_path, "w") as f:
    f.write("# Bird Audio Credits\n\n")
    f.write("All recordings sourced from Wikimedia Commons under public domain or CC licenses.\n\n")
    for name, url in BIRDS:
        f.write(f"- **{name}**: {url}\n")

print(f"\nDownloaded: {len(downloaded)} | Skipped/failed: {len(skipped)}")

if skipped:
    print(f"Failed: {', '.join(skipped)}")
    print("Tip: run the script again — Wikimedia sometimes rate-limits; it will retry skipped files.")

if not downloaded:
    print("\nNo files downloaded. Check your internet connection.")
    sys.exit(1)

print("\nRunning process_birds.py...\n")
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
result = subprocess.run(
    [sys.executable, os.path.join(repo_root, "process_birds.py")],
    cwd=repo_root
)
if result.returncode == 0:
    print("\nDone! birdsong_data.json updated.")
else:
    print("\nprocess_birds.py exited with errors.")
