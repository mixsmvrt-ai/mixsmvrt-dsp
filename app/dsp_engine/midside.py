"""Mid/Side utilities for mastering-only processing.

Provides safe conversion between stereo and M/S domains and basic
width control.
"""
from __future__ import annotations

import numpy as np


def stereo_to_ms(x: np.ndarray) -> np.ndarray:
  """Convert stereo [2, N] to mid/side [2, N]."""
  if x.ndim != 2 or x.shape[0] != 2:
    raise ValueError("Mid/Side processing expects stereo [2, N] input")
  mid = 0.5 * (x[0] + x[1])
  side = 0.5 * (x[0] - x[1])
  return np.stack([mid, side], axis=0)


def ms_to_stereo(ms: np.ndarray) -> np.ndarray:
  """Convert mid/side [2, N] back to stereo [2, N]."""
  if ms.ndim != 2 or ms.shape[0] != 2:
    raise ValueError("ms_to_stereo expects [2, N] mid/side input")
  mid, side = ms
  left = mid + side
  right = mid - side
  return np.stack([left, right], axis=0)


def apply_width(ms: np.ndarray, width: float) -> np.ndarray:
  """Scale side channel to adjust stereo width.

  Width is clamped to [0.8, 1.4] to avoid phase issues.
  """
  width = float(np.clip(width, 0.8, 1.4))
  out = ms.copy().astype(np.float32)
  out[1] *= width
  return out
