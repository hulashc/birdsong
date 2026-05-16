"""Audio → manifold pipeline.

Produces the same JSON schema as process_birds.py but as a callable
function rather than a script, so FastAPI can call it per-request.

Feature layout (must match features.js exactly):
  [mfcc_0..39, chroma_0..11, centroid, bandwidth, rolloff, zcr, onset]  → 57 dims
"""

from __future__ import annotations

from typing import Any

import numpy as np

SR = 22050
N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512
N_PCA = 3


def process_audio(
    filepath: str,
    species: str = "unknown",
    duration: float = 10.0,
) -> dict[str, Any]:
    """Load audio, extract features, run PCA, return manifold dict.

    Parameters
    ----------
    filepath : str
        Absolute path to the audio file.
    species : str
        Name tag embedded in the output JSON (used by the frontend).
    duration : float
        Seconds to analyse. Longer files are truncated.

    Returns
    -------
    dict matching birdsong_data.json per-species schema.

    Raises
    ------
    ValueError
        If the audio is too short to extract features from.
    """
    # Import here so the module loads without librosa installed (unit tests)
    import librosa  # noqa: PLC0415
    from sklearn.decomposition import PCA  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    y, sr = librosa.load(filepath, sr=SR, duration=duration, mono=True)

    if len(y) < N_FFT:
        raise ValueError(
            f"Audio too short ({len(y)} samples). Need at least {N_FFT}."
        )

    # ── Feature extraction ─────────────────────────────────────────────────
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    bandwidth = librosa.feature.spectral_bandwidth(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=N_FFT, hop_length=HOP_LENGTH
    )
    onset_env = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=HOP_LENGTH
    )[np.newaxis, :]  # (1, frames)

    n_frames = min(
        mfcc.shape[1],
        chroma.shape[1],
        centroid.shape[1],
        bandwidth.shape[1],
        rolloff.shape[1],
        zcr.shape[1],
        onset_env.shape[1],
    )

    if n_frames < N_PCA + 1:
        raise ValueError(
            f"Too few frames ({n_frames}). Increase audio duration."
        )

    # ── Stack: (frames, 57) ────────────────────────────────────────────────
    feature_matrix = np.vstack(
        [
            mfcc[:, :n_frames],       # 40
            chroma[:, :n_frames],     # 12
            centroid[:, :n_frames],   #  1
            bandwidth[:, :n_frames],  #  1
            rolloff[:, :n_frames],    #  1
            zcr[:, :n_frames],        #  1
            onset_env[:, :n_frames],  #  1
        ]
    ).T  # (frames, 57)

    # ── Scale + PCA ────────────────────────────────────────────────────────
    X = StandardScaler().fit_transform(feature_matrix)
    coords = PCA(n_components=N_PCA, random_state=0).fit_transform(X)

    m = np.max(np.abs(coords))
    if m > 0:
        coords = coords / m

    # ── Energy + spectral centroid ─────────────────────────────────────────
    rms = librosa.feature.rms(
        y=y, frame_length=N_FFT, hop_length=HOP_LENGTH
    )[0, :n_frames]
    rms = rms / (float(np.percentile(rms, 95)) + 1e-9)
    rms = np.clip(rms, 0.0, 1.0)

    nyquist = sr / 2.0
    cent_norm = np.clip(centroid[0, :n_frames] / nyquist, 0.0, 1.0)

    t = librosa.frames_to_time(
        np.arange(n_frames), sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT
    )

    return {
        "species": species,
        "sr": int(sr),
        "hop_length": int(HOP_LENGTH),
        "duration_s": float(len(y) / sr),
        "features_used": [
            "mfcc_40", "chroma_12", "centroid",
            "bandwidth", "rolloff", "zcr", "onset",
        ],
        "t": t.tolist(),
        "xyz": coords.tolist(),
        "energy": rms.tolist(),
        "spectral_centroid": cent_norm.tolist(),
    }
