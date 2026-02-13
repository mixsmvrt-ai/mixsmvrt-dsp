"""Shared analysis helpers for the new DSP engine.

This keeps librosa / pyloudnorm isolated to analysis only so the
processing code can remain lightweight and dependency-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

import numpy as np
import pyloudnorm as pyln
import librosa


@dataclass
class LoudnessStats:
  integrated_lufs: float
  true_peak_dbfs: float


@dataclass
class SpectralStats:
  centroid_hz: float
  rolloff_hz: float
  bandwidth_hz: float


@lru_cache(maxsize=64)
def _meter_for_sr(sr: int) -> pyln.Meter:
  return pyln.Meter(sr)


def measure_loudness(x: np.ndarray, sr: int) -> LoudnessStats:
  mono = x.mean(axis=0) if x.ndim > 1 else x
  meter = _meter_for_sr(sr)
  try:
    integrated = float(meter.integrated_loudness(mono.astype(np.float32)))
  except Exception:
    # Fallback: RMS-based approximation
    rms = float(np.sqrt(np.mean(np.square(mono.astype(np.float32))) + 1e-12))
    integrated = -20.0 * np.log10(max(rms, 1e-6))

  peak = float(np.max(np.abs(mono)) + 1e-9)
  true_peak_dbfs = 20.0 * np.log10(peak)
  return LoudnessStats(integrated_lufs=integrated, true_peak_dbfs=true_peak_dbfs)


def basic_spectral_stats(x: np.ndarray, sr: int) -> SpectralStats:
  mono = x.mean(axis=0) if x.ndim > 1 else x
  centroid = float(librosa.feature.spectral_centroid(y=mono, sr=sr).mean())
  rolloff = float(librosa.feature.spectral_rolloff(y=mono, sr=sr, roll_percent=0.95).mean())
  bandwidth = float(librosa.feature.spectral_bandwidth(y=mono, sr=sr).mean())
  return SpectralStats(centroid_hz=centroid, rolloff_hz=rolloff, bandwidth_hz=bandwidth)
