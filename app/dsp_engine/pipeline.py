"""High-level processing pipelines for MixSmvrt.

This module exposes four main flows:
- audio_cleanup
- mixing_only
- mix_master
- mastering_only

Each flow uses the shared building blocks from dsp_engine.*
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Literal, Optional

import numpy as np

from .analysis import LoudnessStats, measure_loudness
from .biquad_eq import apply_eq_stack, FilterType
from .dynamic_eq import create_dynamic_eq_band
from .multiband_compressor import create_default_multiband
from .midside import stereo_to_ms, ms_to_stereo, apply_width
from .saturation import soft_saturation
from .limiter import TruePeakLimiter
from .rnnoise_cleaner import RnNoiseCleaner


FlowType = Literal["audio_cleanup", "mixing_only", "mix_master", "mastering_only"]


@dataclass
class ProcessingReport:
  track_name: str
  processing_chain: List[str]
  parameter_values: Dict[str, float]
  loudness_before: float
  loudness_after: float
  true_peak_after: float


def _ensure_stereo(x: np.ndarray) -> np.ndarray:
  if x.ndim == 1:
    return np.stack([x, x], axis=0)
  if x.ndim == 2 and x.shape[0] == 2:
    return x
  raise ValueError("Expected mono [N] or stereo [2, N] audio")


def process_audio_cleanup(track_name: str, x: np.ndarray, sr: int) -> tuple[np.ndarray, ProcessingReport]:
  x = _ensure_stereo(x).astype(np.float32)
  loud_before = measure_loudness(x, sr)

  chain: List[str] = []
  params: Dict[str, float] = {}

  # RNNoise cleanup
  cleaner = RnNoiseCleaner()
  mono_clean = cleaner.process(x, sr)
  if mono_clean.ndim == 1:
    x_clean = np.stack([mono_clean, mono_clean], axis=0)
  else:
    x_clean = mono_clean
  chain.append("rnnoise_clean")

  # gentle high-pass and low-mid cut
  bands: tuple[tuple[FilterType, float, float, float], ...] = (
    ("highpass", 70.0, 0.0, 0.707),
    ("peaking", 250.0, -3.0, 1.2),
  )
  x_eq = apply_eq_stack(x_clean, sr, bands)
  chain.append("cleanup_eq")

  # simple loudness normalisation to -18 LUFS (dialogue-ish)
  loud_mid = measure_loudness(x_eq, sr)
  target_lufs = -18.0
  gain_db = float(np.clip(target_lufs - loud_mid.integrated_lufs, -6.0, 6.0))
  gain_lin = 10 ** (gain_db / 20.0)
  x_out = (x_eq * gain_lin).astype(np.float32)
  chain.append("level_normalise")
  params["cleanup_gain_db"] = gain_db

  loud_after = measure_loudness(x_out, sr)

  report = ProcessingReport(
    track_name=track_name,
    processing_chain=chain,
    parameter_values=params,
    loudness_before=loud_before.integrated_lufs,
    loudness_after=loud_after.integrated_lufs,
    true_peak_after=loud_after.true_peak_dbfs,
  )

  return x_out, report


def process_mixing_only(
  track_name: str,
  x: np.ndarray,
  sr: int,
  is_vocal: bool = False,
  beat_sidechain: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, ProcessingReport]:
  x = _ensure_stereo(x).astype(np.float32)
  loud_before = measure_loudness(x, sr)

  chain: List[str] = []
  params: Dict[str, float] = {}

  # adaptive EQ: gentle tilt based on role
  if is_vocal:
    bands: tuple[tuple[FilterType, float, float, float], ...] = (
      ("highpass", 80.0, 0.0, 0.707),
      ("peaking", 250.0, -3.0, 1.0),
      ("peaking", 3500.0, 2.0, 1.5),
    )
  else:
    bands = (
      ("highpass", 30.0, 0.0, 0.707),
      ("peaking", 300.0, -2.5, 1.2),
    )
  x_eq = apply_eq_stack(x, sr, bands)
  chain.append("adaptive_eq")

  # dynamic EQ: vocal harshness or beat vocal pocketing
  if is_vocal:
    harsh_band = create_dynamic_eq_band(4500.0, sr)
    x_dyn = harsh_band.process(x_eq, sr)
    chain.append("dynamic_eq_vocal_harshness")
  else:
    if beat_sidechain is not None:
      pocket = create_dynamic_eq_band(3000.0, sr)
      x_dyn = pocket.process(x_eq, sr, sidechain=beat_sidechain)
      chain.append("dynamic_eq_vocal_pocket")
    else:
      x_dyn = x_eq

  # multiband bus compressor
  mb = create_default_multiband(sr)
  x_bus = mb.process(x_dyn, sr)
  chain.append("multiband_bus_comp")

  loud_after = measure_loudness(x_bus, sr)

  report = ProcessingReport(
    track_name=track_name,
    processing_chain=chain,
    parameter_values=params,
    loudness_before=loud_before.integrated_lufs,
    loudness_after=loud_after.integrated_lufs,
    true_peak_after=loud_after.true_peak_dbfs,
  )

  return x_bus, report


def process_mix_master(
  track_name: str,
  x: np.ndarray,
  sr: int,
  is_vocal: bool = False,
  beat_sidechain: Optional[np.ndarray] = None,
  target_lufs: float = -9.0,
) -> tuple[np.ndarray, ProcessingReport]:
  # start from mixing chain
  mix, mix_report = process_mixing_only(track_name, x, sr, is_vocal=is_vocal, beat_sidechain=beat_sidechain)

  # subtle saturation
  sat = soft_saturation(mix, amount=0.02)

  # mid/side mastering EQ on stereo bus
  ms = stereo_to_ms(sat)
  ms_width = apply_width(ms, width=1.1)
  ms_back = ms_to_stereo(ms_width)

  # gentle full-band multiband again for final glue
  mb = create_default_multiband(sr)
  glued = mb.process(ms_back, sr)

  # true peak limiter
  limiter = TruePeakLimiter(ceiling_db=-1.0, max_gr_db=4.0)
  limited = limiter.process(glued)

  # final loudness adjust towards target (small gain only)
  loud_after = measure_loudness(limited, sr)
  gain_db = float(np.clip(target_lufs - loud_after.integrated_lufs, -3.0, 3.0))
  gain_lin = 10 ** (gain_db / 20.0)
  out = (limited * gain_lin).astype(np.float32)

  final_stats = measure_loudness(out, sr)

  chain = mix_report.processing_chain + [
    "saturation",
    "mid_side_master_eq",
    "multiband_master_glue",
    "true_peak_limiter",
    "final_loudness_trim",
  ]

  params = dict(mix_report.parameter_values)
  params["final_gain_db"] = gain_db

  report = ProcessingReport(
    track_name=track_name,
    processing_chain=chain,
    parameter_values=params,
    loudness_before=mix_report.loudness_before,
    loudness_after=final_stats.integrated_lufs,
    true_peak_after=final_stats.true_peak_dbfs,
  )

  return out, report


def process_mastering_only(
  track_name: str,
  x: np.ndarray,
  sr: int,
  target_lufs: float = -9.0,
) -> tuple[np.ndarray, ProcessingReport]:
  x = _ensure_stereo(x).astype(np.float32)
  loud_before = measure_loudness(x, sr)

  chain: List[str] = []
  params: Dict[str, float] = {}

  # mid/side EQ – gentle tonal balancing
  ms = stereo_to_ms(x)
  ms = apply_width(ms, width=1.05)

  # low shelf on mid, gentle high shelf on side using biquads via apply_eq_stack
  mid = ms[0:1, :]
  side = ms[1:2, :]
  mid_bands: tuple[tuple[FilterType, float, float, float], ...] = (("lowshelf", 120.0, -2.5, 0.7),)
  side_bands: tuple[tuple[FilterType, float, float, float], ...] = (("highshelf", 8000.0, 2.0, 0.7),)
  mid_eq = apply_eq_stack(mid, sr, mid_bands)
  side_eq = apply_eq_stack(side, sr, side_bands)
  ms_eq = np.vstack([mid_eq, side_eq])
  chain.append("ms_eq")

  stereo = ms_to_stereo(ms_eq)

  # multiband compression
  mb = create_default_multiband(sr)
  glued = mb.process(stereo, sr)
  chain.append("multiband_master")

  # subtle harmonic enhancement
  sat = soft_saturation(glued, amount=0.02)
  chain.append("saturation")

  # limiter + target loudness
  limiter = TruePeakLimiter(ceiling_db=-1.0, max_gr_db=4.0)
  limited = limiter.process(sat)

  loud_limited = measure_loudness(limited, sr)
  gain_db = float(np.clip(target_lufs - loud_limited.integrated_lufs, -3.0, 3.0))
  gain_lin = 10 ** (gain_db / 20.0)
  out = (limited * gain_lin).astype(np.float32)
  chain.append("final_loudness_trim")
  params["final_gain_db"] = gain_db

  final_stats = measure_loudness(out, sr)

  report = ProcessingReport(
    track_name=track_name,
    processing_chain=chain,
    parameter_values=params,
    loudness_before=loud_before.integrated_lufs,
    loudness_after=final_stats.integrated_lufs,
    true_peak_after=final_stats.true_peak_dbfs,
  )

  return out, report
