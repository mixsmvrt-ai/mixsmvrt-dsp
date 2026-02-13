"""Biquad EQ filters built on SciPy.

This module provides a small wrapper around scipy.signal to design
standard audio filters (HP, LP, peaking EQ, shelves) and apply them
safely with sensible constraints for a broadcast / production chain.

Constraints:
- Boosts are clamped to +3 dB
- Cuts are clamped to -5 dB
- Q values are clamped to a safe, musical range

All processing is real, no placeholders.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
from scipy.signal import lfilter, sosfilt, zpk2sos, bilinear_zpk

FilterType = Literal["highpass", "lowpass", "peaking", "lowshelf", "highshelf"]

_SAMPLE_RATE_FALLBACK = 44100


@dataclass
class BiquadFilter:
    """Container for a single biquad section.

    We store coefficients in sos form so we can easily chain
    multiple sections with `sosfilt`.
    """

    sos: np.ndarray  # shape (1, 6)

    def process(self, x: np.ndarray) -> np.ndarray:
        """Apply the filter to a mono or stereo signal.

        Args:
            x: np.ndarray [channels, samples] or [samples]
        """

        if x.ndim == 1:
            return np.asarray(sosfilt(self.sos, x.astype(np.float32)))

        # process channels independently but with same filter
        y = np.empty_like(x, dtype=np.float32)
        for ch in range(x.shape[0]):
            y[ch] = sosfilt(self.sos, x[ch].astype(np.float32))
        return y


def _clamp_freq(freq: float, sr: int) -> float:
    nyq = sr * 0.5
    return float(np.clip(freq, 20.0, nyq - 100.0))


def _clamp_gain_db(gain_db: float) -> float:
    # Global safety: boost <= +3 dB, cut >= -5 dB
    return float(np.clip(gain_db, -5.0, 3.0))


def _clamp_q(q: float) -> float:
    # Keep Q in a musical, numerically stable range
    return float(np.clip(q, 0.25, 10.0))


def design_biquad(
    ftype: FilterType,
    freq: float,
    sr: int,
    gain_db: float = 0.0,
    q: float = 0.707,
) -> BiquadFilter:
    """Design a single biquad filter section.

    Uses analog prototype -> bilinear transform for stability.
    """

    if sr <= 0:
        sr = _SAMPLE_RATE_FALLBACK

    freq = _clamp_freq(freq, sr)
    gain_db = _clamp_gain_db(gain_db)
    q = _clamp_q(q)

    w0 = 2.0 * np.pi * freq
    a = 10 ** (gain_db / 40.0)

    # analog prototype in ZPK form
    if ftype == "highpass":
        z = np.array([0.0, 0.0])
        p = np.array([-w0 / q, -w0 * q])
        k = 1.0
    elif ftype == "lowpass":
        z = np.array([])
        p = np.array([-w0 / q, -w0 * q])
        k = w0 * w0
    elif ftype == "peaking":
        # symmetric peak / dip around freq
        bw = w0 / q
        z = np.array([
            -bw / 2.0 + 1j * np.sqrt(w0**2 - (bw / 2.0) ** 2),
            -bw / 2.0 - 1j * np.sqrt(w0**2 - (bw / 2.0) ** 2),
        ])
        p = z / a
        k = 1.0
    elif ftype == "lowshelf":
        # simplified shelf using magnitude tilt
        z = np.array([-w0 / np.sqrt(a)])
        p = np.array([-w0 * np.sqrt(a)])
        k = np.sqrt(a)
    elif ftype == "highshelf":
        z = np.array([-w0 * np.sqrt(a)])
        p = np.array([-w0 / np.sqrt(a)])
        k = 1.0 / np.sqrt(a)
    else:
        raise ValueError(f"Unsupported filter type: {ftype}")

    # bilinear transform to digital z-domain
    z_d, p_d, k_d = bilinear_zpk(z, p, k, fs=sr)
    sos = zpk2sos(z_d, p_d, k_d)
    # We expect a single section; enforce shape (1, 6)
    if sos.ndim == 1:
        sos = sos[np.newaxis, :]
    return BiquadFilter(sos=sos.astype(np.float64))


def apply_eq_stack(
    x: np.ndarray,
    sr: int,
    bands: Tuple[Tuple[FilterType, float, float, float], ...],
) -> np.ndarray:
    """Apply a stack of biquad filters.

    Args:
        x: mono or stereo signal
        sr: sample rate
        bands: tuple of (type, freq, gain_db, q)
    """

    y = x.astype(np.float32)
    for ftype, freq, gain_db, q in bands:
        filt = design_biquad(ftype, freq, sr, gain_db=gain_db, q=q)
        y = filt.process(y)
    return y
