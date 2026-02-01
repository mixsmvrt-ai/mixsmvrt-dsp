from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
from pedalboard import load_plugin, Plugin

try:  # WORLD vocoder (pyworld) for analysis + resynthesis
    import pyworld as pw  # type: ignore
    _WORLD_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    pw = None  # type: ignore[assignment]
    _WORLD_AVAILABLE = False


_TUNER_PLUGIN: Optional[Plugin] = None
_TUNER_ENV_VAR = "VOCAL_TUNER_PLUGIN"


def _get_tuner_plugin() -> Optional[Plugin]:
    """Lazily load an external pitch‑correction plugin if configured.

    If the environment variable ``VOCAL_TUNER_PLUGIN`` points to a valid
    VST3/Audio Unit effect, it will be loaded once and cached. If loading
    fails for any reason, pitch correction is silently disabled.
    """
    global _TUNER_PLUGIN

    if _TUNER_PLUGIN is not None:
        return _TUNER_PLUGIN

    plugin_path = os.getenv(_TUNER_ENV_VAR)
    if not plugin_path:
        return None

    try:
        _TUNER_PLUGIN = load_plugin(plugin_path)
    except Exception:
        # If the plugin can't be loaded, just disable pitch correction.
        _TUNER_PLUGIN = None
    return _TUNER_PLUGIN


_NOTE_TO_SEMITONE = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}


def _scale_degrees(scale: str | None) -> Sequence[int]:
    """Return scale degrees (in semitones) for Major/Minor.

    Uses natural minor; this is intentionally simple and safe.
    """

    if not scale:
        return (0, 2, 4, 5, 7, 9, 11)
    s = scale.lower()
    if s == "minor":
        return (0, 2, 3, 5, 7, 8, 10)
    return (0, 2, 4, 5, 7, 9, 11)


def _quantize_f0_to_key(f0: np.ndarray, key: str, scale: str | None) -> np.ndarray:
    """Snap WORLD f0 curve to the nearest note in the given key/scale.

    Unvoiced frames (f0 <= 0) are passed through unchanged.
    """

    if f0.size == 0:
        return f0

    root = _NOTE_TO_SEMITONE.get(key.upper())
    if root is None:
        return f0

    degrees = _scale_degrees(scale)
    allowed_pcs = {int((root + d) % 12) for d in degrees}

    f0_quantized = f0.copy()
    # Avoid log of zero
    voiced_mask = f0 > 0.0
    if not np.any(voiced_mask):
        return f0

    f0_voiced = f0[voiced_mask]
    # MIDI note numbers for voiced frames
    midi = 69.0 + 12.0 * np.log2(f0_voiced / 440.0)

    # For each frame, find nearest note in key within +/- 6 semitones.
    midi_q = np.empty_like(midi)
    for i, m in enumerate(midi):
        best_m = m
        best_dist = 999.0
        for delta in range(-6, 7):
            cand = m + float(delta)
            pc = int(round(cand)) % 12
            if pc not in allowed_pcs:
                continue
            dist = abs(cand - m)
            if dist < best_dist:
                best_dist = dist
                best_m = cand
        midi_q[i] = best_m

    f0_q = 440.0 * (2.0 ** ((midi_q - 69.0) / 12.0))
    f0_quantized[voiced_mask] = f0_q
    return f0_quantized


def _world_key_aware_tune(audio: np.ndarray, sr: int, key: str, scale: str | None) -> np.ndarray:
    """WORLD-based gentle key-aware pitch correction.

    This operates on mono internally and copies back to the original
    channel layout. If WORLD is unavailable or anything fails, the
    input audio is returned unchanged.
    """

    if not _WORLD_AVAILABLE or audio.size == 0:
        return audio

    # Normalise to mono for analysis/resynthesis
    if audio.ndim == 1:
        mono = audio.astype(np.float64)
    else:
        # Treat last dimension as samples for safety
        if audio.shape[0] <= audio.shape[1]:
            mono = np.mean(audio, axis=0, dtype=np.float64)
        else:
            mono = np.mean(audio, axis=1, dtype=np.float64)

    try:
        # WORLD analysis
        f0, t = pw.dio(mono, sr)  # type: ignore[call-arg]
        f0 = pw.stonemask(mono, f0, t, sr)  # type: ignore[call-arg]
        sp = pw.cheaptrick(mono, f0, t, sr)  # type: ignore[call-arg]
        ap = pw.d4c(mono, f0, t, sr)  # type: ignore[call-arg]

        f0_q = _quantize_f0_to_key(f0, key, scale)

        tuned = pw.synthesize(f0_q, sp, ap, sr)  # type: ignore[call-arg]
    except Exception:
        return audio

    tuned = tuned.astype(np.float32)

    # Restore original layout: copy tuned mono into all channels.
    if audio.ndim == 1:
        return tuned[: audio.shape[0]]

    if audio.shape[0] <= audio.shape[1]:
        # (channels, samples)
        out = np.stack([tuned[: audio.shape[1]]] * audio.shape[0], axis=0)
    else:
        # (samples, channels)
        out = np.stack([tuned[: audio.shape[0]]] * audio.shape[1], axis=1)
    return out.astype(np.float32)


def apply_pitch_correction(
    audio: np.ndarray,
    sr: int,
    session_key: str | None = None,
    session_scale: str | None = None,
) -> np.ndarray:
    """Apply optional pitch correction.

    Preference order:
    1. If a session key is provided and WORLD is available, run a
       gentle key-aware tuning pass in that key/scale.
    2. Otherwise, fall back to an external tuner plugin if configured
       via ``VOCAL_TUNER_PLUGIN``.
    3. On any error, return the original audio.
    """

    if audio.size == 0:
        return audio

    # First preference: internal WORLD-based key-aware tuning.
    if session_key and _WORLD_AVAILABLE:
        tuned = _world_key_aware_tune(audio, sr, session_key, session_scale)
        return tuned

    # Fallback: external tuner plugin (key configured in the plugin).
    tuner = _get_tuner_plugin()
    if tuner is None:
        return audio

    try:
        processed = tuner(audio.astype("float32"), sr)
    except Exception:
        return audio

    if not isinstance(processed, np.ndarray) or processed.size == 0:
        return audio
    return processed.astype("float32")
