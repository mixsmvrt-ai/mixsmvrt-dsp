"""RNNoise-based denoiser integration.

This module expects an rnnoise binding to be available. To keep this
code deployable on Fly.io even when rnnoise or its Python bindings are
missing, we guard imports and fall back to no-op processing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:  # pragma: no cover - optional dependency
  import rnnoise
except Exception:  # pragma: no cover
  rnnoise = None  # type: ignore


@dataclass
class RnNoiseCleaner:
  noise_floor_threshold_db: float = -55.0

  def _estimate_noise_floor(self, x: np.ndarray) -> float:
    mono = x.mean(axis=0) if x.ndim > 1 else x
    eps = 1e-9
    rms = float(np.sqrt(np.mean(np.square(mono.astype(np.float32))) + eps))
    return -20.0 * np.log10(max(rms, 1e-6))

  def process(self, x: np.ndarray, sr: int) -> np.ndarray:
    if rnnoise is None:
      return x.astype(np.float32)

    noise_floor = self._estimate_noise_floor(x)
    if noise_floor <= self.noise_floor_threshold_db:
      # already clean enough
      return x.astype(np.float32)

    frame_size = 480  # rnnoise native frame size at 48k; we resample via rnnoise lib if needed
    if x.ndim > 1:
      mono = x.mean(axis=0)
    else:
      mono = x

    out = np.empty_like(mono, dtype=np.float32)
    st = rnnoise.RNNoise()
    for i in range(0, len(mono), frame_size):
      frame = mono[i : i + frame_size]
      if len(frame) < frame_size:
        padded = np.zeros(frame_size, dtype=np.float32)
        padded[: len(frame)] = frame
        den = st.process_frame(padded)
        out[i : i + len(frame)] = den[: len(frame)]
      else:
        out[i : i + frame_size] = st.process_frame(frame.astype(np.float32))

    return out[np.newaxis, :] if x.ndim > 1 else out
