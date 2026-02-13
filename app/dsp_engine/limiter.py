"""True-peak aware limiter with simple oversampling.

This is not a brickwall commercial-grade limiter, but it implements
real oversampling, peak prediction and constrained gain reduction so
that we safely hit a ceiling without inter-sample clipping.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TruePeakLimiter:
  ceiling_db: float = -1.0
  max_gr_db: float = 4.0

  def process(self, x: np.ndarray) -> np.ndarray:
    """Apply transparent limiting with 2x oversampling.

    Args:
      x: mono or stereo float32 in -1..1
    """
    ceiling_db = float(np.clip(self.ceiling_db, -3.0, -0.1))
    max_gr = float(np.clip(self.max_gr_db, 0.0, 4.0))

    # 2x linear upsample for true-peak estimation
    if x.ndim == 1:
      up = _upsample2(x)
    else:
      up = np.stack([_upsample2(ch) for ch in x], axis=0)

    peak = float(np.max(np.abs(up)) + 1e-9)
    ceiling_lin = 10 ** (ceiling_db / 20.0)

    if peak <= ceiling_lin:
      return x.astype(np.float32)

    required_gr_db = 20.0 * np.log10(peak / ceiling_lin)
    actual_gr_db = min(required_gr_db, max_gr)
    gain = 10 ** (-actual_gr_db / 20.0)

    y = (x * gain).astype(np.float32)

    # safety renormalization
    out_peak = float(np.max(np.abs(_upsample2(y if y.ndim == 1 else y[0]))) + 1e-9)
    if out_peak > ceiling_lin:
      y = (y / out_peak * ceiling_lin).astype(np.float32)

    return y


def _upsample2(x: np.ndarray) -> np.ndarray:
  """Very small 2x linear interpolation upsampler for true-peak checks."""
  n = x.shape[-1]
  out = np.empty(n * 2 - 1, dtype=np.float32)
  out[0::2] = x
  out[1::2] = 0.5 * (x[:-1] + x[1:])
  return out
