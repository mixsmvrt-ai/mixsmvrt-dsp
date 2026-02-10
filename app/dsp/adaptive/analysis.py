"""Adaptive audio analysis for mixing/mastering.

This module performs a detailed analysis pass BEFORE any processing.
It extracts:
- peak_db, rms_db, lufs (integrated)
- spectral_centroid
- band energies (low/mid/high, sibilance band)
- dynamic_range
- stereo_width
- transient_density
- noise_floor_db

It is intentionally stateless and returns a plain dict so that
classification and adaptive processing modules can consume it.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict

import logging

import numpy as np
import pyloudnorm as pyln
from scipy import signal

try:  # librosa is optional but strongly recommended
    import librosa  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    librosa = None  # type: ignore

logger = logging.getLogger("mixsmvrt_dsp.adaptive.analysis")


@dataclass
class BandEnergies:
    low_db: float
    mid_db: float
    high_db: float
    sibilance_db: float


@dataclass
class TrackAnalysis:
    peak_db: float
    rms_db: float
    lufs: float
    spectral_centroid_hz: float
    bands: BandEnergies
    dynamic_range_db: float
    stereo_width: float
    transient_density: float
    noise_floor_db: float


def _ensure_stereo(audio: np.ndarray) -> np.ndarray:
    """Return a 2xN float32 stereo array for consistent analysis."""

    if audio.ndim == 1:
        return np.vstack([audio, audio]).astype(np.float32)
    if audio.ndim == 2:
        if audio.shape[0] == 2:
            return audio.astype(np.float32)
        if audio.shape[1] == 2:
            return audio.T.astype(np.float32)
    # Fallback: flatten and duplicate as mono
    flat = audio.reshape(-1).astype(np.float32)
    return np.vstack([flat, flat])


def _ensure_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32)
    if audio.ndim == 2:
        if audio.shape[0] == 2:
            return audio.mean(axis=0).astype(np.float32)
        if audio.shape[1] == 2:
            return audio.mean(axis=1).astype(np.float32)
    return audio.reshape(-1).astype(np.float32)


def _safe_db(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(x), eps))


def compute_loudness_metrics(audio: np.ndarray, sr: int) -> tuple[float, float, float]:
    """Return (peak_db, rms_db, lufs_integrated)."""

    mono = _ensure_mono(audio)

    peak = float(np.max(np.abs(mono)))
    peak_db = float(_safe_db(np.array([peak]))[0])

    rms = float(np.sqrt(np.mean(mono ** 2)))
    rms_db = float(_safe_db(np.array([rms]))[0])

    try:
        meter = pyln.Meter(sr)  # EBU R128
        lufs = float(meter.integrated_loudness(mono.astype(np.float32)))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("[ANALYSIS] LUFS measurement failed: %s", exc)
        # Fallback: approximate LUFS from RMS
        lufs = rms_db

    return peak_db, rms_db, lufs


def compute_spectral_features(audio: np.ndarray, sr: int) -> tuple[float, BandEnergies]:
    """Compute spectral centroid and band energies in dB.

    Bands:
    - low:   20–200 Hz
    - mid:   200–2000 Hz
    - high:  2 kHz–16 kHz
    - sibilance: 5–10 kHz
    """

    mono = _ensure_mono(audio)

    if librosa is not None:  # pragma: no cover - optional path
        S = np.abs(librosa.stft(mono, n_fft=4096, hop_length=1024, window="hann"))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
        centroid = float(librosa.feature.spectral_centroid(S=S, sr=sr).mean())
    else:
        freqs, _, Zxx = signal.stft(
            mono,
            fs=sr,
            nperseg=4096,
            noverlap=4096 - 1024,
            window="hann",
        )
        S = np.abs(Zxx)
        # Power-weighted frequency mean
        power = S ** 2
        num = (freqs[:, None] * power).sum()
        den = power.sum() + 1e-12
        centroid = float(num / den)

    def _band_db(f_lo: float, f_hi: float) -> float:
        mask = (freqs >= f_lo) & (freqs < f_hi)
        if not np.any(mask):
            return -120.0
        band_mag = S[mask, :]
        band_power = float(np.mean(band_mag ** 2))
        return float(_safe_db(np.array([np.sqrt(band_power)]))[0])

    low_db = _band_db(20.0, 200.0)
    mid_db = _band_db(200.0, 2000.0)
    high_db = _band_db(2000.0, 16000.0)
    sibilance_db = _band_db(5000.0, 10000.0)

    bands = BandEnergies(low_db=low_db, mid_db=mid_db, high_db=high_db, sibilance_db=sibilance_db)
    return centroid, bands


def compute_dynamic_and_transients(audio: np.ndarray, sr: int) -> tuple[float, float, float]:
    """Compute dynamic_range_db, transient_density, noise_floor_db.

    - dynamic_range_db: 95th - 5th percentile of short-term loudness (dB).
    - transient_density: fraction of frames where short-term level rises
      rapidly compared to a smoothed envelope.
    - noise_floor_db: 10th percentile of frame RMS in dB.
    """

    mono = _ensure_mono(audio)

    frame_size = int(sr * 0.05)
    hop = frame_size // 2 or 1
    if frame_size <= 0 or frame_size > mono.shape[0]:
        return 0.0, 0.0, -120.0

    rms_vals: list[float] = []
    peaks: list[float] = []

    for start in range(0, mono.shape[0] - frame_size + 1, hop):
        frame = mono[start : start + frame_size]
        rms = float(np.sqrt(np.mean(frame ** 2)))
        rms_vals.append(rms)
        peaks.append(float(np.max(np.abs(frame))))

    if not rms_vals:
        return 0.0, 0.0, -120.0

    rms_arr = np.array(rms_vals)
    rms_db = _safe_db(rms_arr)

    p5 = float(np.percentile(rms_db, 5))
    p95 = float(np.percentile(rms_db, 95))
    dynamic_range_db = max(0.0, p95 - p5)

    # Transient density: proportion of frames where peak is significantly
    # above RMS (e.g. > 6 dB), which indicates sharp transients.
    peak_arr = np.array(peaks)
    peak_db = _safe_db(peak_arr)
    transient_mask = (peak_db - rms_db) > 6.0
    transient_density = float(np.mean(transient_mask.astype(float)))

    # Noise floor: a lower percentile of the RMS distribution.
    noise_floor_db = float(np.percentile(rms_db, 10))

    return dynamic_range_db, transient_density, noise_floor_db


def compute_stereo_width(audio: np.ndarray, sr: int) -> float:
    """Estimate stereo width via mid/side energy ratio and correlation.

    Returns a value roughly in [0, 1.5], where higher means wider.
    """

    stereo = _ensure_stereo(audio)
    L, R = stereo[0], stereo[1]

    mid = 0.5 * (L + R)
    side = 0.5 * (L - R)

    mid_energy = float(np.mean(mid ** 2))
    side_energy = float(np.mean(side ** 2))

    # Basic width metric: side / (mid + epsilon)
    width = float(side_energy / (mid_energy + 1e-12))

    # Phase correlation: high correlation -> narrower perceived width.
    frame_size = int(sr * 0.05)
    hop = frame_size // 2 or 1
    corrs: list[float] = []
    for start in range(0, len(mid) - frame_size + 1, hop):
        m = mid[start : start + frame_size]
        s = side[start : start + frame_size]
        if m.std() < 1e-6 or s.std() < 1e-6:
            continue
        c = float(np.corrcoef(m, s)[0, 1])
        corrs.append(c)
    phase_corr = float(np.median(corrs)) if corrs else 1.0

    # Map correlation into a soft width modifier (lower correlation -> wider)
    corr_factor = float(np.clip(1.0 - phase_corr, 0.0, 1.0))

    return float(width + 0.5 * corr_factor)


def analyze_track(audio: np.ndarray, sr: int) -> Dict[str, Any]:
    """Run the full adaptive analysis stack on a single track.

    Returns a JSON-serialisable dict suitable for logging, storage,
    and downstream adaptive decision logic.
    """

    peak_db, rms_db, lufs = compute_loudness_metrics(audio, sr)
    centroid, bands = compute_spectral_features(audio, sr)
    dynamic_range_db, transient_density, noise_floor_db = compute_dynamic_and_transients(audio, sr)
    stereo_width = compute_stereo_width(audio, sr)

    analysis = TrackAnalysis(
        peak_db=peak_db,
        rms_db=rms_db,
        lufs=lufs,
        spectral_centroid_hz=centroid,
        bands=bands,
        dynamic_range_db=dynamic_range_db,
        stereo_width=stereo_width,
        transient_density=transient_density,
        noise_floor_db=noise_floor_db,
    )

    result: Dict[str, Any] = asdict(analysis)
    # Flatten band energies for easier consumption in decision logic.
    band_dict = result.pop("bands", {}) or {}
    for key, value in band_dict.items():
        result[f"band_{key}_db"] = value

    logger.info(
        "[ANALYSIS] peak=%.2f dB rms=%.2f dB LUFS=%.2f centroid=%.0f Hz "
        "low=%.1f mid=%.1f high=%.1f sib=%.1f dyn=%.1f dB width=%.2f trans=%.2f noise=%.1f dB",
        peak_db,
        rms_db,
        lufs,
        centroid,
        bands.low_db,
        bands.mid_db,
        bands.high_db,
        bands.sibilance_db,
        dynamic_range_db,
        stereo_width,
        transient_density,
        noise_floor_db,
    )

    return result
