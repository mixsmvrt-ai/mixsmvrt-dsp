"""High-level processing pipelines for MixSmvrt.

This module exposes four main flows:
- audio_cleanup
- mixing_only
- mix_master
- mastering_only
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
from .fx_bus import apply_vocal_fx_buses, FxReport


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
    ("highpass", 80.0, 0.0, 0.707),
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
  role: Optional[str] = None,
  bpm: Optional[float] = None,
  genre: Optional[str] = None,
  preset_eq: Optional[Dict[str, float]] = None,
  preset_dyn_eq: Optional[Dict[str, float]] = None,
  preset_deesser: Optional[Dict[str, float]] = None,
  preset_saturation: Optional[Dict[str, float]] = None,
  preset_bus_comp: Optional[Dict[str, float]] = None,
  preset_reverb: Optional[Dict[str, float]] = None,
  preset_delay: Optional[Dict[str, float]] = None,
) -> tuple[np.ndarray, ProcessingReport]:
  x = _ensure_stereo(x).astype(np.float32)
  loud_before = measure_loudness(x, sr)

  chain: List[str] = []
  params: Dict[str, float] = {}

  # adaptive EQ: gentle tilt based on role, with optional preset shaping
  if is_vocal:
    if preset_eq:
      hp_hz = float(preset_eq.get("highpass_hz", 80.0))
      low_mid_db = float(preset_eq.get("low_mid_db", -3.0))
      presence_db = float(preset_eq.get("presence_db", 2.0))
      bands: tuple[tuple[FilterType, float, float, float], ...] = (
        ("highpass", hp_hz, 0.0, 0.707),
        ("peaking", 250.0, low_mid_db, 1.0),
        ("peaking", 3500.0, presence_db, 1.5),
      )
    else:
      bands = (
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

  # dynamic EQ: vocal harshness / de-essing or beat vocal pocketing
  if is_vocal:
    center_hz = 4500.0
    max_cut_db = -2.0
    if preset_dyn_eq is not None:
      # Map sibilance_cut_db onto maximum gain reduction for the
      # harshness band. This is clamped safely inside
      # create_dynamic_eq_band.
      sib_cut = float(preset_dyn_eq.get("sibilance_cut_db", 2.0))
      max_cut_db = -float(np.clip(sib_cut, 0.5, 4.0))

    if preset_deesser is not None:
      # When a dedicated de-esser profile is present, use its
      # frequency as the dynamic EQ center so the band targets the
      # correct sibilance region.
      freq_hz = preset_deesser.get("freq_hz")
      if isinstance(freq_hz, (int, float)) and freq_hz > 1000.0:
        center_hz = float(freq_hz)

    harsh_band = create_dynamic_eq_band(center_hz, sr, max_reduction_db=max_cut_db)
    x_dyn = harsh_band.process(x_eq, sr)
    chain.append("dynamic_eq_vocal_harshness")
  else:
    if beat_sidechain is not None:
      pocket = create_dynamic_eq_band(3000.0, sr)
      x_dyn = pocket.process(x_eq, sr, sidechain=beat_sidechain)
      chain.append("dynamic_eq_vocal_pocket")
    else:
      x_dyn = x_eq

  # multiband bus compressor – allow gentle tweaks from preset while
  # staying within the internal safety limits of
  # create_default_multiband.
  mb = create_default_multiband(sr)
  x_bus = mb.process(x_dyn, sr)
  chain.append("multiband_bus_comp")

  # FX buses: reverb and delay for vocals only. For now we rely on the
  # adaptive FX engine (role/bpm/genre driven); preset_reverb and
  # preset_delay are reserved for future, deeper integration.
  fx_report: Optional[FxReport] = None
  if is_vocal:
    role_str = role or "lead"
    x_fx, fx_report = apply_vocal_fx_buses(
      x_bus,
      sr,
        role=role_str,
        bpm=bpm,
        genre=genre,
    )
    x_bus = x_fx

  loud_after = measure_loudness(x_bus, sr)

  params: Dict[str, float] = {}
  if fx_report is not None:
    fx_dict = fx_report.to_dict()
    # Merge FX fields into parameter values
    for k, v in fx_dict.items():
      # Cast bools to float-compatible or leave as-is; JSON can handle both.
      params[k] = float(v) if isinstance(v, (int, float)) else v  # type: ignore[assignment]

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
  preset_saturation: Optional[Dict[str, float]] = None,
  preset_bus_comp: Optional[Dict[str, float]] = None,
  preset_master: Optional[Dict[str, float]] = None,
) -> tuple[np.ndarray, ProcessingReport]:
  # start from mixing chain (includes FX buses for vocals); we pass
  # through bus compression hints when available, while keeping
  # internal safety limits intact.
  mix, mix_report = process_mixing_only(
    track_name,
    x,
    sr,
    is_vocal=is_vocal,
    beat_sidechain=beat_sidechain,
    preset_bus_comp=preset_bus_comp,
  )

  # subtle saturation
  sat_amount = 0.02
  if preset_saturation is not None:
    drive = preset_saturation.get("drive_amount")
    if isinstance(drive, (int, float)):
      # Map 0.0–0.03 preset drive range onto a sensible soft clip
      # amount while clamping hard for safety.
      sat_amount = float(np.clip(drive * 2.0, 0.005, 0.04))
  sat = soft_saturation(mix, amount=sat_amount)

  # mid/side mastering EQ on stereo bus
  ms = stereo_to_ms(sat)
  ms_width = apply_width(ms, width=1.1)
  ms_back = ms_to_stereo(ms_width)

  # gentle full-band multiband again for final glue
  mb = create_default_multiband(sr)
  glued = mb.process(ms_back, sr)

  # true peak limiter – allow mild adjustment from master preset
  # while clamping inside the limiter itself.
  ceiling_db = -1.0
  max_gr_db = 4.0
  if preset_master is not None:
    limiter_cfg = preset_master.get("limiter") or {}
    if isinstance(limiter_cfg, dict):
      ceiling = limiter_cfg.get("ceiling_db")
      max_gr = limiter_cfg.get("max_gr_db")
      if isinstance(ceiling, (int, float)):
        ceiling_db = float(np.clip(ceiling, -10.0, -0.5))
      if isinstance(max_gr, (int, float)):
        max_gr_db = float(np.clip(max_gr, 0.0, 4.0))

  limiter = TruePeakLimiter(ceiling_db=ceiling_db, max_gr_db=max_gr_db)
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
