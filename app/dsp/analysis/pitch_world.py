from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
from scipy.signal import resample_poly

try:  # WORLD vocoder for high-quality f0 tracking
    import pyworld as pw  # type: ignore
    _WORLD_AVAILABLE = True
except Exception:  # pragma: no cover - runtime fallback if WORLD is missing
    pw = None  # type: ignore[assignment]
    _WORLD_AVAILABLE = False

logger = logging.getLogger(__name__)


def _resample_to_16k(audio: np.ndarray, sr: int, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    if sr == target_sr:
        return audio.astype(np.float64, copy=False), sr
    # Use polyphase resampling for efficiency and numerical stability.
    audio64 = audio.astype(np.float64, copy=False)
    gcd = np.gcd(sr, target_sr)
    up = target_sr // gcd
    down = sr // gcd
    resampled = resample_poly(audio64, up, down)
    return resampled.astype(np.float64, copy=False), target_sr


def analyze_pitch_world(audio: np.ndarray, sr: int) -> Dict[str, Any]:
    """Analyse pitch using WORLD and classify tuning stability.

    Returns a dict with:
        - mean_f0_hz
        - median_f0_hz
        - pitch_std_semitones
        - voiced_ratio
        - tuning_quality: "stable" | "slightly_off" | "unstable"

    The function is intentionally conservative and does *not* apply any
    auto-tuning; it simply reports statistics that can influence gentle
    de-essing, saturation, or optional pitch-assist thresholds.
    """

    if audio.size == 0:
        return {
            "mean_f0_hz": 0.0,
            "median_f0_hz": 0.0,
            "pitch_std_semitones": 0.0,
            "voiced_ratio": 0.0,
            "tuning_quality": "stable",
        }

    # Downmix to mono for pitch tracking.
    if audio.ndim > 1:
        mono = audio.mean(axis=-1)
    else:
        mono = audio

    mono = mono.astype(np.float64, copy=False)

    if not _WORLD_AVAILABLE:
        # Fallback: derive very rough pitch stability from zero-crossing rate.
        zc = np.mean(np.abs(np.diff(np.sign(mono))))
        # Map approximate ZCR to a pseudo stability score.
        if zc < 0.1:
            tuning = "stable"
        elif zc < 0.3:
            tuning = "slightly_off"
        else:
            tuning = "unstable"
        return {
            "mean_f0_hz": 0.0,
            "median_f0_hz": 0.0,
            "pitch_std_semitones": 0.0,
            "voiced_ratio": 0.0,
            "tuning_quality": tuning,
        }

    try:
        mono_rs, sr_rs = _resample_to_16k(mono, sr)
        # WORLD expects double precision
        _f0, t = pw.dio(mono_rs, sr_rs)  # type: ignore[operator]
        f0 = pw.stonemask(mono_rs, _f0, t, sr_rs)  # type: ignore[operator]

        f0 = np.asarray(f0, dtype=np.float64)
        voiced = f0 > 0.0
        if not np.any(voiced):
            return {
                "mean_f0_hz": 0.0,
                "median_f0_hz": 0.0,
                "pitch_std_semitones": 0.0,
                "voiced_ratio": 0.0,
                "tuning_quality": "unstable",
            }

        f0_voiced = f0[voiced]
        mean_f0 = float(np.mean(f0_voiced))
        median_f0 = float(np.median(f0_voiced))
        voiced_ratio = float(np.mean(voiced.astype(np.float64)))

        # Pitch variability in semitones relative to the median f0.
        ref = max(median_f0, 1e-6)
        semitone_offsets = 12.0 * np.log2(f0_voiced / ref)
        pitch_std = float(np.std(semitone_offsets))

        if pitch_std <= 0.5:
            quality = "stable"
        elif pitch_std <= 1.5:
            quality = "slightly_off"
        else:
            quality = "unstable"

        logger.debug(
            "[DSP] WORLD pitch analysis completed: mean_f0=%.1f, std_semitones=%.2f, voiced_ratio=%.2f, quality=%s",
            mean_f0,
            pitch_std,
            voiced_ratio,
            quality,
        )

        return {
            "mean_f0_hz": mean_f0,
            "median_f0_hz": median_f0,
            "pitch_std_semitones": pitch_std,
            "voiced_ratio": voiced_ratio,
            "tuning_quality": quality,
        }

    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("WORLD pitch analysis failed, falling back to rough ZCR: %s", exc)
        # Fallback to the simple ZCR-based heuristic above.
        return analyze_pitch_world(mono.astype(np.float32, copy=False), sr)
