"""Dynamic EQ utilities.

Implements band-limited level monitoring with envelope detection and
conditional gain changes (downward only) for:
- Vocal harshness control
- Beat vocal pocketing (with optional sidechain)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .biquad_eq import BiquadFilter, design_biquad


@dataclass
class DynamicEQBand:
  """Single dynamic EQ band.

  Only cuts (negative gain) are applied, with max depth of -4 dB.
  """

  detector_filter: BiquadFilter
  gain_filter: BiquadFilter
  threshold_db: float
  max_reduction_db: float
  attack_ms: float
  release_ms: float

  def process(self, x: np.ndarray, sr: int, sidechain: Optional[np.ndarray] = None) -> np.ndarray:
    if sidechain is None:
      sc = x
    else:
      sc = sidechain

    # mono level for detection
    if sc.ndim > 1:
      sc_mono = sc.mean(axis=0)
    else:
      sc_mono = sc

    # band-pass / peaking filter for detector
    band = self.detector_filter.process(sc_mono)

    eps = 1e-9
    level_db = 20.0 * np.log10(np.maximum(np.abs(band), eps))

    # envelope follower
    attack = np.exp(-1.0 / (0.001 * self.attack_ms * sr))
    release = np.exp(-1.0 / (0.001 * self.release_ms * sr))

    env = np.zeros_like(level_db)
    prev = 0.0
    for i, s in enumerate(level_db):
      if s > prev:
        coeff = attack
      else:
        coeff = release
      prev = coeff * prev + (1.0 - coeff) * s
      env[i] = prev

    # compute gain reduction
    over = np.maximum(env - self.threshold_db, 0.0)
    # map overshoot to reduction up to max_reduction_db (negative)
    reduction = -np.minimum(over, abs(self.max_reduction_db))

    # convert reduction envelope to linear
    gain = 10 ** (reduction / 20.0)

    # apply static filter then modulate output by dynamic gain
    y = self.gain_filter.process(x)
    if y.ndim == 1:
      return (y * gain).astype(np.float32)

    return (y * gain[np.newaxis, :]).astype(np.float32)


def create_dynamic_eq_band(
  center_freq: float,
  sr: int,
  q: float = 2.0,
  threshold_db: float = -24.0,
  max_reduction_db: float = -4.0,
  attack_ms: float = 10.0,
  release_ms: float = 120.0,
) -> DynamicEQBand:
  """Factory with safety clamping for dynamic EQ band."""

  max_reduction_db = float(np.clip(max_reduction_db, -4.0, -0.5))
  attack_ms = float(np.clip(attack_ms, 5.0, 20.0))
  release_ms = float(np.clip(release_ms, 80.0, 150.0))

  detector = design_biquad("peaking", center_freq, sr, gain_db=0.0, q=q)
  gain_filter = design_biquad("peaking", center_freq, sr, gain_db=0.0, q=q)

  return DynamicEQBand(
    detector_filter=detector,
    gain_filter=gain_filter,
    threshold_db=threshold_db,
    max_reduction_db=max_reduction_db,
    attack_ms=attack_ms,
    release_ms=release_ms,
  )
