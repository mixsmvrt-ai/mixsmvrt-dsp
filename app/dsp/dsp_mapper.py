"""Map predicted parameters onto the existing DSP chain.

This module uses the existing `app.dsp_engine` primitives (EQ, dynamic EQ,
multiband comp, limiter, mid/side width) and clamps all parameters to
production-safe ranges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from app.dsp_engine.biquad_eq import apply_eq_stack, FilterType
from app.dsp_engine.dynamic_eq import create_dynamic_eq_band
from app.dsp_engine.multiband_compressor import create_default_multiband
from app.dsp_engine.midside import stereo_to_ms, ms_to_stereo, apply_width
from app.dsp_engine.limiter import TruePeakLimiter
from app.dsp_engine.analysis import measure_loudness


def _ensure_stereo(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return np.stack([x, x], axis=0).astype(np.float32)
    if x.ndim == 2 and x.shape[0] == 2:
        return x.astype(np.float32)
    if x.ndim == 2 and x.shape[1] == 2:
        return x.T.astype(np.float32)
    raise ValueError("Expected mono [N] or stereo [2,N]/[N,2]")


def _compressor(
    x: np.ndarray,
    sr: int,
    *,
    ratio: float,
    attack_ms: float,
    release_ms: float,
    threshold_db: float = -18.0,
) -> np.ndarray:
    """Lightweight feed-forward compressor.

    This is designed to be stable and fast for CPU-only inference.
    """

    x = _ensure_stereo(x)
    ratio = float(np.clip(ratio, 1.5, 8.0))
    attack_ms = float(np.clip(attack_ms, 1.0, 100.0))
    release_ms = float(np.clip(release_ms, 20.0, 500.0))

    mono = x.mean(axis=0)
    level = 20.0 * np.log10(np.maximum(np.abs(mono), 1e-9))
    over = level - threshold_db
    gr_db = np.where(over > 0.0, (1.0 - 1.0 / ratio) * over, 0.0).astype(np.float32)

    # Envelope smoothing
    attack = float(np.exp(-1.0 / (0.001 * attack_ms * sr)))
    release = float(np.exp(-1.0 / (0.001 * release_ms * sr)))
    env = np.zeros_like(gr_db)
    prev = 0.0
    for i, g in enumerate(gr_db):
        coeff = attack if g > prev else release
        prev = coeff * prev + (1.0 - coeff) * float(g)
        env[i] = prev

    gain = (10 ** (-env / 20.0)).astype(np.float32)
    return (x * gain[np.newaxis, :]).astype(np.float32)


@dataclass
class MappedTrackResult:
    audio: np.ndarray
    role: str


def apply_predicted_parameters(
    track_audio: np.ndarray,
    sr: int,
    role: str,
    predicted_params: Dict[str, float],
    *,
    vocal_sidechain: Optional[np.ndarray] = None,
) -> MappedTrackResult:
    """Apply predicted parameters to a single track based on its role."""

    x = _ensure_stereo(track_audio).astype(np.float32)

    # Extract + clamp
    vocal_eq_high_gain = float(np.clip(predicted_params.get("vocal_eq_high_gain", 1.0), -3.0, 6.0))
    vocal_ratio = float(np.clip(predicted_params.get("vocal_compression_ratio", 3.0), 1.5, 8.0))
    vocal_attack = float(np.clip(predicted_params.get("vocal_attack", 10.0), 1.0, 100.0))
    vocal_release = float(np.clip(predicted_params.get("vocal_release", 120.0), 20.0, 500.0))

    beat_mid_dip_gain = float(np.clip(predicted_params.get("beat_mid_dip_gain", -3.0), -9.0, 0.0))
    beat_dip_freq = float(np.clip(predicted_params.get("beat_dip_freq", 3200.0), 250.0, 8000.0))

    role_key = (role or "").lower().strip()

    if role_key in {"lead_vocal", "bg_vocal"}:
        # High shelf for air
        bands: tuple[tuple[FilterType, float, float, float], ...] = (
            ("highpass", 80.0, 0.0, 0.707),
            ("highshelf", 12000.0, vocal_eq_high_gain, 0.7),
        )
        x = apply_eq_stack(x, sr, bands)
        x = _compressor(
            x,
            sr,
            ratio=vocal_ratio,
            attack_ms=vocal_attack,
            release_ms=vocal_release,
            threshold_db=-18.0 if role_key == "lead_vocal" else -20.0,
        )
    else:
        # Beat/instruments: dynamic mid dip (with vocal sidechain if provided)
        if beat_mid_dip_gain < -0.25:
            band = create_dynamic_eq_band(
                beat_dip_freq,
                sr,
                max_reduction_db=float(beat_mid_dip_gain),
                attack_ms=10.0,
                release_ms=140.0,
            )
            x = band.process(x, sr, sidechain=vocal_sidechain)

    # Safety sanitization
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    peak = float(np.max(np.abs(x)) + 1e-9)
    if peak > 1.2:
        x = (x / peak * 1.2).astype(np.float32)

    return MappedTrackResult(audio=x, role=role)


def apply_master_processing(
    mix: np.ndarray,
    sr: int,
    predicted_params: Dict[str, float],
) -> tuple[np.ndarray, Dict[str, float]]:
    """Apply master bus processing based on predicted parameters."""

    x = _ensure_stereo(mix).astype(np.float32)

    target_lufs = float(np.clip(predicted_params.get("master_target_lufs", -10.0), -16.0, -6.0))
    mb_ratio = float(np.clip(predicted_params.get("master_multiband_ratio", 1.6), 1.2, 3.5))
    widen = float(np.clip(predicted_params.get("stereo_widen_amount", 1.05), 0.8, 1.35))

    # Width
    ms = stereo_to_ms(x)
    ms_w = apply_width(ms, width=widen)
    x = ms_to_stereo(ms_w).astype(np.float32)

    # Multiband glue (adjust ratios)
    mb = create_default_multiband(sr)
    mb.low.ratio = mb_ratio
    mb.lowmid.ratio = mb_ratio
    mb.mid.ratio = mb_ratio
    mb.high.ratio = mb_ratio
    x = mb.process(x, sr).astype(np.float32)

    # Loudness targeting (gain) before limiting
    loud = measure_loudness(x, sr)
    gain_db = float(np.clip(target_lufs - float(loud.integrated_lufs), -12.0, 12.0))
    gain = float(10 ** (gain_db / 20.0))
    x = (x * gain).astype(np.float32)

    # Limiter ceiling fixed; limiter will clamp
    limiter = TruePeakLimiter(ceiling_db=-1.0, max_gr_db=4.0)
    x = limiter.process(x).astype(np.float32)

    loud_after = measure_loudness(x, sr)
    report = {
        "target_lufs": float(target_lufs),
        "gain_db": float(gain_db),
        "integrated_lufs": float(loud_after.integrated_lufs),
        "true_peak_dbfs": float(loud_after.true_peak_dbfs),
        "stereo_widen": float(widen),
        "multiband_ratio": float(mb_ratio),
    }
    return x, report
