import librosa
import numpy as np
from sklearn.decomposition import PCA
import json, os

birds_folder = "birds"
output = {}

for filename in os.listdir(birds_folder):
    if not filename.endswith((".mp3", ".wav")):
        continue

    species = filename.replace(".mp3", "").replace(".wav", "")
    filepath = os.path.join(birds_folder, filename)

    # Load audio and extract MFCCs (40 coefficients per frame)
    y, sr = librosa.load(filepath, duration=10)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs = mfccs.T  # shape: (frames, 40)

    # Reduce 40D → 3D using PCA
    pca = PCA(n_components=3)
    coords = pca.fit_transform(mfccs)

    # Normalise to roughly -1 to 1 range
    coords = coords / np.abs(coords).max()

    output[species] = coords.tolist()

with open("birdsong_data.json", "w") as f:
    json.dump(output, f)

print("Done! birdsong_data.json created.")