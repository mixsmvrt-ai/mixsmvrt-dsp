"""Shared FX bus system for reverb and delay.

Implements a simple send-based FX architecture:
- One global reverb bus
- One global delay bus

Tracks send into these buses via a send level (0–1). The buses are
then mixed back into the main stereo signal.

This module is designed to stay CPU/memory-light for Fly.io:
- Preallocated buffers
- Sequential processing
- Float32 only
- Optional auto-bypass based on system load
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, Optional

import numpy as np

from .reverb import AlgorithmicReverb, ReverbSettings, create_vocal_reverb_settings
from .delay import (
    StereoDelay,
    DelaySettings,
    create_slap_delay_settings,
    create_ping_pong_delay_settings,
)


try:  # Optional; if missing we simply don't auto-bypass on system load.
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore


@dataclass
class FxReport:
    reverb_enabled: bool
    reverb_decay: float
    reverb_wet: float
    delay_enabled: bool
    delay_time_ms: float
    delay_feedback: float
    reverb_time_ms: float
    delay_time_ms_stage: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reverb_enabled": self.reverb_enabled,
            "reverb_decay": self.reverb_decay,
            "reverb_wet": self.reverb_wet,
            "delay_enabled": self.delay_enabled,
            "delay_time_ms": self.delay_time_ms,
            "delay_feedback": self.delay_feedback,
            "reverb_time_ms": self.reverb_time_ms,
            "delay_process_time_ms": self.delay_time_ms_stage,
        }


def _system_allows_fx(min_avail_mb: float = 256.0, max_cpu_percent: float = 90.0) -> bool:
    """Best-effort system load check for auto-bypass.

    If psutil is not available, always returns True.
    """

    if psutil is None:
        return True

    try:
        vm = psutil.virtual_memory()
        if vm.available < min_avail_mb * 1024 * 1024:
            return False
        cpu = psutil.cpu_percent(interval=0.0)
        if cpu > max_cpu_percent:
            return False
    except Exception:
        return True

    return True


def _estimate_stereo_width(x: np.ndarray) -> float:
    """Rough stereo width estimate based on L/R correlation.

    Returns a value in [0, 1], where 0 ~= mono and 1 ~= very wide.
    """

    if x.ndim != 2 or x.shape[0] != 2 or x.shape[1] == 0:
        return 0.0

    left = x[0].astype(np.float32, copy=False)
    right = x[1].astype(np.float32, copy=False)
    left -= float(left.mean())
    right -= float(right.mean())
    num = float(np.dot(left, right))
    den = float(np.linalg.norm(left) * np.linalg.norm(right) + 1e-9)
    corr = num / den if den > 0.0 else 0.0
    corr = max(min(corr, 1.0), -1.0)
    width = 1.0 - abs(corr)
    return float(max(min(width, 1.0), 0.0))


def _estimate_dynamic_range_db(x: np.ndarray) -> float:
    """Simple crest-factor style dynamic range estimate in dB."""

    mono = x.mean(axis=0) if x.ndim > 1 else x
    mono = mono.astype(np.float32, copy=False)
    rms = float(np.sqrt(np.mean(mono * mono) + 1e-12))
    peak = float(np.max(np.abs(mono)) + 1e-9)
    if rms <= 0.0 or peak <= 0.0:
        return 0.0
    return float(20.0 * np.log10(peak / rms))


def apply_vocal_fx_buses(
    vocal_bus: np.ndarray,
    sr: int,
    *,
    role: str = "lead",
    bpm: Optional[float] = None,
    genre: Optional[str] = None,
) -> tuple[np.ndarray, FxReport]:
    """Apply reverb + delay FX buses to a vocal bus.

    - Takes post-compression stereo vocal bus [2, N]
    - Derives reverb/delay settings adaptively
    - Applies FX only if system resources allow

    Returns (processed_vocal, fx_report).
    """

    if vocal_bus.ndim != 2 or vocal_bus.shape[0] != 2:
        raise ValueError("vocal_bus must be stereo [2, N]")

    x = vocal_bus.astype(np.float32, copy=False)
    n = x.shape[1]
    if n == 0:
        empty_report = FxReport(
            reverb_enabled=False,
            reverb_decay=0.0,
            reverb_wet=0.0,
            delay_enabled=False,
            delay_time_ms=0.0,
            delay_feedback=0.0,
            reverb_time_ms=0.0,
            delay_time_ms_stage=0.0,
        )
        return x, empty_report

    # Estimate context features for adaptivity
    stereo_width = _estimate_stereo_width(x)
    dynamic_range_db = _estimate_dynamic_range_db(x)

    # Derive settings
    reverb_settings: ReverbSettings = create_vocal_reverb_settings(
        role=role,
        dynamic_range_db=dynamic_range_db,
        bpm=bpm,
        genre=genre,
        stereo_width=stereo_width,
    )

    # Default: lead uses slap delay, background uses ping-pong.
    role_lower = (role or "lead").lower()
    if "bg" in role_lower or "background" in role_lower:
        delay_settings: DelaySettings = create_ping_pong_delay_settings(bpm=bpm, note_fraction=0.5)
    else:
        delay_settings = create_slap_delay_settings()

    # Safety caps (reinforced here)
    reverb_settings = reverb_settings.clamped()
    delay_settings = delay_settings.clamped()

    reverb_enabled = False
    delay_enabled = False

    y = x.copy()

    reverb_start = perf_counter()
    delay_start = 0.0

    if _system_allows_fx():
        # Reverb send is mono derived from vocal
        reverb = AlgorithmicReverb(sr, reverb_settings)
        send = x.mean(axis=0)  # mono send
        wet_rev = reverb.process_mono(send)
        # Mix reverb back in as a bus
        y = y + wet_rev * reverb_settings.wet
        reverb_enabled = reverb_settings.wet > 0.0
    reverb_time_ms = (perf_counter() - reverb_start) * 1000.0

    if _system_allows_fx():
        delay_start = perf_counter()
        delay = StereoDelay(sr, delay_settings)
        # Send level = 1; wet mix controlled in settings
        wet_delay = delay.process(y)
        y = y + wet_delay
        delay_time_ms_stage = (perf_counter() - delay_start) * 1000.0
        delay_enabled = delay_settings.wet > 0.0 and delay_settings.feedback > 0.0
    else:
        delay_time_ms_stage = 0.0

    fx_report = FxReport(
        reverb_enabled=reverb_enabled,
        reverb_decay=reverb_settings.decay_s,
        reverb_wet=reverb_settings.wet,
        delay_enabled=delay_enabled,
        delay_time_ms=delay_settings.time_ms,
        delay_feedback=delay_settings.feedback,
        reverb_time_ms=reverb_time_ms,
        delay_time_ms_stage=delay_time_ms_stage,
    )

    return y.astype(np.float32, copy=False), fx_report
