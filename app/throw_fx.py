from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np


ThrowFxMode = Literal["reverb", "delay", "both"]


def _to_2d(audio: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Ensure (n_samples, n_channels) float32, return (audio_2d, channels_first)."""

    if audio.ndim == 1:
        return audio.astype(np.float32)[:, None], True
    channels_first = audio.shape[0] <= audio.shape[1]
    arr = audio if not channels_first else audio.T
    return arr.astype(np.float32), channels_first


def _from_2d(arr: np.ndarray, original: np.ndarray, channels_first: bool) -> np.ndarray:
    if original.ndim == 1:
        return arr[:, 0].astype(original.dtype)
    out = arr.T if channels_first else arr
    return out.astype(original.dtype)


def apply_throw_fx_to_vocal(
    audio: np.ndarray,
    sr: int,
    mode: Optional[str],
) -> np.ndarray:
    """Fill vocal gaps with reverb/delay throws in a musical way.

    This is designed as a gentle, radio-ready enhancement:
    - Uses the existing Pedalboard Reverb/Delay where available.
    - Builds an envelope from the dry vocal to detect gaps.
    - Lets throws bloom into the empty spaces with a smooth gate.
    """

    if audio.size == 0:
        return audio
    if mode not in {"reverb", "delay", "both"}:
        return audio

    try:
        from pedalboard import Pedalboard, Reverb, Delay, Gain  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        # If pedalboard is unavailable on this deployment, skip throws.
        return audio

    dry_2d, channels_first = _to_2d(audio)

    # 1) Build a wet throw chain tuned for modern hip-hop / dancehall / R&B.
    throw_chain = []
    if mode in {"reverb", "both"}:
        throw_chain.append(
            Reverb(
                room_size=0.28,
                damping=0.45,
                wet_level=0.9,
                dry_level=0.1,
                width=1.0,
            ),
        )
    if mode in {"delay", "both"}:
        throw_chain.append(
            Delay(
                delay_seconds=0.26,
                feedback=0.32,
                mix=1.0,
            ),
        )
    # Gentle trim on the throw path to keep headroom.
    throw_chain.append(Gain(gain_db=-3.0))

    board = Pedalboard(throw_chain)
    wet = board(dry_2d, sr)

    # 2) Build an amplitude envelope from the dry vocal.
    mono = dry_2d.mean(axis=1)
    n = mono.shape[0]
    frame = max(int(0.02 * sr), 1)  # 20 ms
    hop = frame

    env = np.zeros_like(mono, dtype=np.float32)
    for i in range(0, n, hop):
        win = mono[i : i + frame]
        if win.size == 0:
            break
        rms = float(np.sqrt(np.mean(win**2)))
        env[i : i + frame] = rms

    nonzero = env[env > 0]
    if nonzero.size == 0:
        return audio

    # Adaptive threshold so behaviour matches different input levels.
    thresh = float(np.percentile(nonzero, 35))
    if thresh <= 0:
        return audio

    # High env = vocal present; we want throws in the gaps, so invert.
    gate = np.clip((thresh - env) / thresh, 0.0, 1.0)

    # 3) Smooth gate for musical blooms.
    smooth_len = max(int(0.08 * sr), 1)  # 80 ms
    kernel = np.ones(smooth_len, dtype=np.float32)
    kernel /= kernel.sum()
    gate = np.convolve(gate, kernel, mode="same")

    gate = gate[:, None]  # broadcast across channels

    # 4) Apply gate to wet path and blend under the dry vocal.
    throw_level = 0.35  # tasteful, radio-ready level
    throw_sig = wet * gate * throw_level
    mixed = dry_2d + throw_sig
    mixed = np.clip(mixed, -1.0, 1.0)

    return _from_2d(mixed, audio, channels_first)
