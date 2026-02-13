"""Simple but production-safe multiband compressor.

Splits the signal into four bands and applies gentle compression
per band with soft knees and constrained gain reduction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .biquad_eq import design_biquad, BiquadFilter


@dataclass
class BandCompressor:
  threshold_db: float
  ratio: float
  attack_ms: float
  release_ms: float
  max_reduction_db: float

  def process(self, level_db: np.ndarray) -> np.ndarray:
    # soft knee: small region around threshold
    knee_width = 3.0
    over = level_db - self.threshold_db
    # soft knee curve
    gr = np.where(
      over <= -knee_width,
      0.0,
      np.where(
        over >= knee_width,
        (1.0 - 1.0 / self.ratio) * over,
        (1.0 - 1.0 / self.ratio) * ((over + knee_width) ** 2) / (4.0 * knee_width),
      ),
    )

    # envelope smoothing
    attack = np.exp(-1.0 / (0.001 * self.attack_ms))
    release = np.exp(-1.0 / (0.001 * self.release_ms))

    env = np.zeros_like(gr)
    prev = 0.0
    for i, g in enumerate(gr):
      if g > prev:
        coeff = attack
      else:
        coeff = release
      prev = coeff * prev + (1.0 - coeff) * g
      env[i] = prev

    env = np.minimum(env, abs(self.max_reduction_db))
    return env


@dataclass
class MultibandCompressor:
  low: BandCompressor
  lowmid: BandCompressor
  mid: BandCompressor
  high: BandCompressor
  split_filters: Tuple[BiquadFilter, BiquadFilter, BiquadFilter]

  def process(self, x: np.ndarray, sr: int) -> np.ndarray:
    # split using serial HP/LP filters around 120, 600, 4k
    hp120 = self.split_filters[0]
    hp600 = self.split_filters[1]
    hp4k = self.split_filters[2]

    # duplicate for safety
    sig = x.astype(np.float32)

    low = sig - hp120.process(sig)
    lowmid_src = hp120.process(sig)
    lowmid = lowmid_src - hp600.process(lowmid_src)
    mid_src = hp600.process(lowmid_src)
    mid = mid_src - hp4k.process(mid_src)
    high = hp4k.process(mid_src)

    def compress_band(band: np.ndarray, comp: BandCompressor) -> np.ndarray:
      if band.ndim > 1:
        mono = band.mean(axis=0)
      else:
        mono = band
      eps = 1e-9
      level = 20.0 * np.log10(np.maximum(np.abs(mono), eps))
      gr = comp.process(level)
      gain = 10 ** (-gr / 20.0)
      if band.ndim == 1:
        return (band * gain).astype(np.float32)
      return (band * gain[np.newaxis, :]).astype(np.float32)

    low_c = compress_band(low, self.low)
    lowmid_c = compress_band(lowmid, self.lowmid)
    mid_c = compress_band(mid, self.mid)
    high_c = compress_band(high, self.high)

    # recombine; soft limiting of sum to avoid overs
    out = low_c + lowmid_c + mid_c + high_c
    peak = np.max(np.abs(out)) + 1e-9
    if peak > 1.2:
      out = out / peak * 1.2
    return out


def create_default_multiband(sr: int) -> MultibandCompressor:
  # ratios and thresholds chosen for gentle bus control
  def band(threshold: float) -> BandCompressor:
    return BandCompressor(
      threshold_db=threshold,
      ratio=2.0,
      attack_ms=20.0,
      release_ms=120.0,
      max_reduction_db=3.0,
    )

  split_filters = (
    design_biquad("highpass", 120.0, sr, gain_db=0.0, q=0.707),
    design_biquad("highpass", 600.0, sr, gain_db=0.0, q=0.707),
    design_biquad("highpass", 4000.0, sr, gain_db=0.0, q=0.707),
  )

  return MultibandCompressor(
    low=band(-30.0),
    lowmid=band(-32.0),
    mid=band(-32.0),
    high=band(-34.0),
    split_filters=split_filters,
  )
