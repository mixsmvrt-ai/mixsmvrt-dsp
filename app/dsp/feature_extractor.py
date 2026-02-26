"""Audio feature extraction for session-aware adaptive mixing.

Constraints:
- CPU only, lightweight signal processing
- Uses librosa/numpy/scipy only (no heavy ML models)
- Returns numeric-only feature dicts (float values)
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import librosa


_EPS = 1e-9


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        # Accept either [2, N] or [N, 2]
        if audio.shape[0] == 2:
            return np.asarray(audio.mean(axis=0), dtype=np.float32)
        if audio.shape[1] == 2:
            return np.asarray(audio.mean(axis=1), dtype=np.float32)
    raise ValueError("Unsupported audio shape; expected mono [N] or stereo [2, N]/[N, 2]")


def _stereo_channels(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if audio.ndim != 2:
        return None
    if audio.shape[0] == 2:
        left = np.asarray(audio[0], dtype=np.float32)
        right = np.asarray(audio[1], dtype=np.float32)
        return left, right
    if audio.shape[1] == 2:
        left = np.asarray(audio[:, 0], dtype=np.float32)
        right = np.asarray(audio[:, 1], dtype=np.float32)
        return left, right
    return None


def _db(x: float) -> float:
    return float(20.0 * np.log10(max(float(x), _EPS)))


def extract_track_features(audio: np.ndarray, sr: int) -> Dict[str, float]:
    """Extract lightweight per-track features.

    Args:
        audio: mono [N] or stereo [2, N]/[N, 2] float/pcm array.
        sr: sample rate.

    Returns:
        Numeric-only dict of features.
    """

    if sr <= 0:
        raise ValueError("Sample rate must be > 0")

    mono = _to_mono(audio)
    n = int(mono.shape[-1])
    duration_s = float(n / float(sr)) if n else 0.0

    # Loudness-related
    peak = float(np.max(np.abs(mono)) + _EPS) if n else _EPS
    rms = float(np.sqrt(np.mean(mono * mono) + _EPS)) if n else _EPS
    peak_db = _db(peak)
    rms_db = _db(rms)
    crest_factor = float(peak / max(rms, _EPS))

    # Dynamic range: frame-wise RMS percentiles (dB)
    if n:
        rms_frames = librosa.feature.rms(y=mono, frame_length=2048, hop_length=512, center=True)[0]
        rms_db_frames = 20.0 * np.log10(np.maximum(rms_frames, _EPS))
        dyn_range = float(np.percentile(rms_db_frames, 95) - np.percentile(rms_db_frames, 5))
    else:
        dyn_range = 0.0

    # Spectral features computed from STFT power
    if n:
        n_fft = 2048
        hop = 512
        S = librosa.stft(mono, n_fft=n_fft, hop_length=hop, center=True).astype(np.complex64)
        P = (np.abs(S) ** 2).astype(np.float32)

        centroid = librosa.feature.spectral_centroid(S=P, sr=sr, n_fft=n_fft, hop_length=hop)[0]
        rolloff = librosa.feature.spectral_rolloff(S=P, sr=sr, n_fft=n_fft, hop_length=hop, roll_percent=0.85)[0]
        spectral_centroid_mean = float(np.mean(centroid))
        spectral_rolloff_mean = float(np.mean(rolloff))

        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft).astype(np.float32)
        total_energy = float(np.sum(P) + _EPS)

        def band_ratio(lo: float, hi: float | None) -> float:
            if hi is None:
                mask = freqs >= lo
            else:
                mask = (freqs >= lo) & (freqs <= hi)
            band_energy = float(np.sum(P[mask, :]))
            return float(band_energy / total_energy)

        low_energy_ratio = band_ratio(20.0, 120.0)
        mid_energy_ratio = band_ratio(250.0, 4000.0)
        high_energy_ratio = band_ratio(6000.0, None)

        # Harmonic ratio via HPSS
        H, Pp = librosa.decompose.hpss(P)
        harmonic_ratio = float(np.sum(H) / (float(np.sum(H) + np.sum(Pp)) + _EPS))

        # Transients: onset count per second
        onset_frames = librosa.onset.onset_detect(y=mono, sr=sr, hop_length=hop, units="frames")
        onset_count = int(len(onset_frames))
        transient_density = float(onset_count / max(duration_s, 1e-3))

        # Free large arrays ASAP
        del S, P, centroid, rolloff, freqs, H, Pp
    else:
        spectral_centroid_mean = 0.0
        spectral_rolloff_mean = 0.0
        low_energy_ratio = 0.0
        mid_energy_ratio = 0.0
        high_energy_ratio = 0.0
        harmonic_ratio = 0.0
        transient_density = 0.0

    # Stereo width: mid/side energy ratio (side / (mid + eps))
    stereo = _stereo_channels(audio)
    if stereo is None:
        stereo_width = 0.0
    else:
        left, right = stereo
        mid = 0.5 * (left + right)
        side = 0.5 * (left - right)
        mid_e = float(np.mean(mid * mid) + _EPS)
        side_e = float(np.mean(side * side) + _EPS)
        stereo_width = float(side_e / mid_e)

    return {
        "peak_db": float(peak_db),
        "rms_db": float(rms_db),
        "crest_factor": float(crest_factor),
        "dynamic_range": float(dyn_range),
        "spectral_centroid_mean": float(spectral_centroid_mean),
        "spectral_rolloff_mean": float(spectral_rolloff_mean),
        "low_energy_ratio": float(low_energy_ratio),
        "mid_energy_ratio": float(mid_energy_ratio),
        "high_energy_ratio": float(high_energy_ratio),
        "stereo_width": float(stereo_width),
        "transient_density": float(transient_density),
        "harmonic_ratio": float(harmonic_ratio),
    }
