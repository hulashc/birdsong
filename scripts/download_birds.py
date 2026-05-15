#!/usr/bin/env python3
"""
download_birds.py

Downloads 20 UK/European bird recordings from Wikimedia Commons
(public domain / CC licensed), saves as OGG into ../birds/,
then runs process_birds.py to regenerate birdsong_data.json.

Usage (from repo root):
    python scripts/download_birds.py

No API key needed. Requires only the Python standard library.

Wikimedia rate-limits bulk downloads. This script retries with
exponential backoff automatically. If some species still fail,
just run it again — it skips already-downloaded files.
"""

import os
import sys
import time
import subprocess
import urllib.request
import urllib.error

# ---------------------------------------------------------------------------
# Curated list: (slug, wikimedia_commons_direct_url)
# All files are public domain or CC-BY licensed from Wikimedia Commons.
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
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
}

# Retry settings
MAX_RETRIES = 5
BASE_DELAY  = 3.0   # seconds between normal downloads
RETRY_WAIT  = [5, 15, 30, 60, 120]  # backoff schedule for 429/errors


def download_with_retry(name, url, dest):
    """Download url → dest. Returns True on success, False on permanent failure."""
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=40) as r:
                data = r.read()
            with open(dest, "wb") as f:
                f.write(data)
            return True, len(data) // 1024
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = RETRY_WAIT[min(attempt, len(RETRY_WAIT) - 1)]
                print(f"    [rate-limited] waiting {wait}s before retry {attempt + 1}/{MAX_RETRIES}...")
                time.sleep(wait)
            elif e.code == 404:
                return False, f"404 Not Found — URL may have moved"
            else:
                wait = RETRY_WAIT[min(attempt, len(RETRY_WAIT) - 1)]
                print(f"    [HTTP {e.code}] waiting {wait}s before retry {attempt + 1}/{MAX_RETRIES}...")
                time.sleep(wait)
        except Exception as e:
            wait = RETRY_WAIT[min(attempt, len(RETRY_WAIT) - 1)]
            print(f"    [error] {e} — waiting {wait}s before retry {attempt + 1}/{MAX_RETRIES}...")
            time.sleep(wait)
    return False, "max retries exceeded"


downloaded = []
skipped    = []

print(f"Saving to: {BIRDS_DIR}")
print(f"Note: Wikimedia rate-limits bulk downloads. Script will retry automatically.\n")

for name, url in BIRDS:
    dest = os.path.join(BIRDS_DIR, f"{name}.ogg")

    if os.path.exists(dest):
        size_kb = os.path.getsize(dest) // 1024
        print(f"  [skip] {name}.ogg already exists ({size_kb} KB)")
        downloaded.append(name)
        continue

    print(f"Downloading: {name}")
    ok, info = download_with_retry(name, url, dest)
    if ok:
        print(f"  Saved {name}.ogg ({info} KB)")
        downloaded.append(name)
    else:
        print(f"  [FAILED] {name}: {info}")
        skipped.append(name)
        # Clean up partial file if any
        if os.path.exists(dest):
            os.remove(dest)

    time.sleep(BASE_DELAY)

# Write CREDITS
credits_path = os.path.join(BIRDS_DIR, "CREDITS.md")
with open(credits_path, "w") as f:
    f.write("# Bird Audio Credits\n\n")
    f.write("All recordings sourced from Wikimedia Commons under public domain or CC licenses.\n\n")
    for name, url in BIRDS:
        f.write(f"- **{name}**: {url}\n")

print(f"\nDownloaded: {len(downloaded)} | Failed: {len(skipped)}")
if skipped:
    print(f"Failed: {', '.join(skipped)}")
    print("Run the script again — it will skip already-downloaded files and retry failed ones.")

if not downloaded:
    print("\nNo files downloaded at all. Check your internet connection.")
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
    print("\nprocess_birds.py exited with errors — check above.")
