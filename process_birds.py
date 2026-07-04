import os
import json
import traceback

import numpy as np
import librosa
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BIRDS_FOLDER = "birds"
OUT_PATH = "birdsong_data.json"

DURATION = 10.0
SR = 22050
N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512


def _extract_features(y, sr):
    """Extract raw feature matrix (frames, 57) from audio array."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP_LENGTH)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)[np.newaxis, :]

    n_frames = min(
        mfcc.shape[1], chroma.shape[1], centroid.shape[1],
        bandwidth.shape[1], rolloff.shape[1], zcr.shape[1], onset_env.shape[1]
    )

    feature_matrix = np.vstack([
        mfcc[:, :n_frames],
        chroma[:, :n_frames],
        centroid[:, :n_frames],
        bandwidth[:, :n_frames],
        rolloff[:, :n_frames],
        zcr[:, :n_frames],
        onset_env[:, :n_frames],
    ]).T  # (frames, 57)

    return feature_matrix, centroid, n_frames


def main():
    output = {}

    for filename in os.listdir(BIRDS_FOLDER):
        if not filename.lower().endswith((".mp3", ".wav", ".ogg", ".flac")):
            continue

        species = os.path.splitext(filename)[0].lower().replace(" ", "_")
        filepath = os.path.join(BIRDS_FOLDER, filename)
        print(f"Processing: {filename} → '{species}'")

        try:
            y, sr = librosa.load(filepath, sr=SR, duration=DURATION, mono=True)

            if len(y) < N_FFT:
                print("  Skipped — too short")
                continue

            feature_matrix, centroid_raw, n_frames = _extract_features(y, sr)

            if len(feature_matrix) < 3:
                print("  Skipped — too few frames")
                continue

            # StandardScaler + PCA per species
            # NOTE: each species lives in its own PCA space — centroids are
            # comparable only within the same coordinate system.
            # For a shared embedding, run a global PCA across all species.
            X = StandardScaler().fit_transform(feature_matrix)
            coords = PCA(n_components=3, random_state=0).fit_transform(X)

            m = np.max(np.abs(coords))
            if m > 0:
                coords = coords / m

            rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
            rms = rms[:n_frames]
            rms = rms / (np.percentile(rms, 95) + 1e-9)
            rms = np.clip(rms, 0, 1)

            nyquist = sr / 2
            centroid_norm = np.clip(centroid_raw[0, :n_frames] / nyquist, 0, 1)

            t = librosa.frames_to_time(
                np.arange(n_frames), sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT
            )

            output[species] = {
                "sr": int(sr),
                "hop_length": int(HOP_LENGTH),
                "duration_s": float(len(y) / sr),
                "features_used": ["mfcc_40", "chroma_12", "centroid", "bandwidth", "rolloff", "zcr", "onset"],
                "t": t.tolist(),
                "xyz": coords.tolist(),
                "energy": rms.tolist(),
                "spectral_centroid": centroid_norm.tolist(),
            }

            print(f"  Done — {n_frames} frames, {output[species]['duration_s']:.2f}s, feature_dim=57")

        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()

    with open(OUT_PATH, "w") as f:
        json.dump(output, f)

    print(f"\nFinished. {len(output)} species written to {OUT_PATH}.")


if __name__ == "__main__":
    main()
