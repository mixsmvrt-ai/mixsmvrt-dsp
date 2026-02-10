"""High-level adaptive mixing & mastering pipeline configuration.

This module does *not* render audio. Instead, it:
- runs adaptive analysis
- classifies the track role
- designs gain staging
- builds adaptive EQ, dynamics, sidechain, and mastering configs

The resulting config object can be consumed by the main DSP engine to
apply processing in a single pass, while the studio UI reads the same
config to mirror what the engine actually did.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

import logging

import numpy as np

from .analysis import analyze_track
from .classification import classify_track
from .adaptive_eq import build_adaptive_eq
from .adaptive_compression import build_adaptive_dynamics
from .adaptive_mastering import build_adaptive_master_config

logger = logging.getLogger("mixsmvrt_dsp.adaptive.pipeline")


def _compute_gain_staging(analysis: Dict[str, Any], role: str) -> Dict[str, float]:
    """Design adaptive gain staging based on peak levels and target roles.

    Targets (peak):
    - beat: -6 dBFS
    - lead_vocal: -9 dBFS
    - background_vocal: -13 dBFS

    Clamp gain adjustments to ±6 dB.
    """

    peak_db = float(analysis.get("peak_db", -18.0))

    if role == "beat":
        target_peak = -6.0
    elif role == "lead_vocal":
        target_peak = -9.0
    elif role in {"background_vocal", "adlibs"}:
        target_peak = -13.0
    else:
        target_peak = -9.0

    needed_change = target_peak - peak_db
    gain_db = max(-6.0, min(6.0, needed_change))

    logger.info(
        "[GAIN] role=%s, peak_db=%.2f -> target_peak=%.2f, raw_change=%.2f, clamped_gain=%.2f",
        role,
        peak_db,
        target_peak,
        needed_change,
        gain_db,
    )

    return {"gain_db": gain_db}


def build_adaptive_pipeline(
    audio: np.ndarray,
    sr: int,
    *,
    user_tag: str | None = None,
    genre: str | None = None,
    gender: str | None = None,
) -> Dict[str, Any]:
    """Top-level function to build an adaptive DSP configuration.

    This function is intentionally pure: it only inspects `audio` and
    returns a nested dict describing the intended processing chain.
    Rendering is delegated to the main engine.
    """

    analysis = analyze_track(audio, sr)

    role = classify_track(analysis, user_tag=user_tag)

    gain = _compute_gain_staging(analysis, role)

    eq_config = build_adaptive_eq(analysis, role)

    dynamics_config = build_adaptive_dynamics(
        analysis=analysis,
        role=role,
        gender_hint=gender,
    )

    master_config = build_adaptive_master_config(analysis, genre)

    config: Dict[str, Any] = {
        "analysis": analysis,
        "role": role,
        "gain_staging": gain,
        "eq": eq_config,
        "dynamics": dynamics_config,
        "mastering": master_config,
    }

    logger.info(
        "[PIPELINE] Built adaptive config for role=%s (gain=%.2f dB, genre=%s)",
        role,
        gain.get("gain_db", 0.0),
        genre,
    )

    return config
