"""Subtle saturation helpers.

We keep saturation very conservative – default max around 3% THD – to
respect headroom and avoid obvious distortion artifacts.
"""
from __future__ import annotations

import numpy as np


def soft_saturation(x: np.ndarray, amount: float = 0.03) -> np.ndarray:
  """Apply gentle symmetrical soft clipping.

  `amount` is clipped to <= 0.03 (3%).
  """
  amt = float(np.clip(amount, 0.0, 0.03))
  if amt <= 0.0:
    return x.astype(np.float32)

  # simple waveshaper: tanh blend
  dry = x.astype(np.float32)
  wet = np.tanh(dry * (1.0 + 10.0 * amt))
  mix = amt / 0.03  # normalise 0..1
  return ((1.0 - mix) * dry + mix * wet).astype(np.float32)
