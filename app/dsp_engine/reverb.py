"""CPU-optimised algorithmic reverb for MixSmvrt.

Lightweight Schroeder/Freeverb-style structure:
- Parallel feedback comb filters
- Series all-pass filters
- Mono in, stereo out

Constraints:
- No convolution / IR loading
- No oversampling
- Float32 only
- Sequential processing with preallocated buffers
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ReverbSettings:
    decay_s: float = 1.2
    pre_delay_ms: float = 25.0
    wet: float = 0.12
    damping: float = 0.3  # 0..1, higher = darker

    def clamped(self) -> "ReverbSettings":
        d = float(min(max(self.decay_s, 0.5), 2.5))
        pre = float(min(max(self.pre_delay_ms, 10.0), 50.0))
        wet = float(min(max(self.wet, 0.0), 0.25))
        damp = float(min(max(self.damping, 0.0), 0.9))
        return ReverbSettings(decay_s=d, pre_delay_ms=pre, wet=wet, damping=damp)


class AlgorithmicReverb:
    """Simple Schroeder-style algorithmic reverb.

    Designed for vocals, with modest CPU use. Works on mono input and
    returns stereo output.
    """

    # Prime-ish comb delays in milliseconds (scaled by sr)
    _comb_delays_ms_left: Tuple[float, ...] = (29.7, 37.1, 41.1, 43.7)
    _comb_delays_ms_right: Tuple[float, ...] = (31.1, 33.3, 39.3, 45.1)

    # Small all-pass delays in milliseconds
    _allpass_delays_ms: Tuple[float, ...] = (5.0, 1.7)

    def __init__(self, sr: int, settings: ReverbSettings) -> None:
        self.sr = int(max(sr, 8000))
        self.settings = settings.clamped()

    @staticmethod
    def _pre_delay(x: np.ndarray, samples: int) -> np.ndarray:
        if samples <= 0:
            return x
        n = x.shape[-1]
        out = np.zeros_like(x, dtype=np.float32)
        if samples < n:
            out[..., samples:] = x[..., : n - samples]
        return out

    @staticmethod
    def _comb_filter(
        x: np.ndarray,
        delay_samples: int,
        feedback: float,
        damping: float,
    ) -> np.ndarray:
        """Feedback comb with simple HF damping in feedback path.

        y[n] = buffer[idx]
        buffer[idx] = x[n] + feedback * damped
        damped is a one-pole low-pass of y.
        """

        n = x.shape[-1]
        delay = max(int(delay_samples), 1)
        buf = np.zeros(delay, dtype=np.float32)
        out = np.zeros(n, dtype=np.float32)
        lp_state = 0.0
        idx = 0

        fb = float(feedback)
        damp = float(damping)
        inv_damp = 1.0 - damp

        for i in range(n):
            buf_out = buf[idx]
            # low-pass in feedback path
            lp_state = inv_damp * buf_out + damp * lp_state
            buf[idx] = x[i] + fb * lp_state
            out[i] = buf_out
            idx += 1
            if idx >= delay:
                idx = 0
        return out

    @staticmethod
    def _allpass_filter(x: np.ndarray, delay_samples: int, feedback: float) -> np.ndarray:
        """Simple all-pass based on a circular buffer.

        y[n] = -x[n] + buffer[idx]
        buffer[idx] = x[n] + feedback * y[n]
        """

        n = x.shape[-1]
        delay = max(int(delay_samples), 1)
        buf = np.zeros(delay, dtype=np.float32)
        out = np.zeros(n, dtype=np.float32)
        idx = 0
        fb = float(feedback)

        for i in range(n):
            buf_out = buf[idx]
            yn = -x[i] + buf_out
            buf[idx] = x[i] + fb * yn
            out[i] = yn
            idx += 1
            if idx >= delay:
                idx = 0
        return out

    def _build_comb_bank(self, x: np.ndarray, delays_ms: Tuple[float, ...]) -> np.ndarray:
        """Run a bank of combs in parallel and average their outputs."""

        x = x.astype(np.float32, copy=False)
        n = x.shape[-1]
        acc = np.zeros(n, dtype=np.float32)

        for d_ms in delays_ms:
            delay_samples = int(self.sr * (d_ms / 1000.0))
            # Feedback chosen so that energy decays ~60 dB over decay_s.
            t = delay_samples / float(self.sr)
            if t <= 0.0:
                continue
            g = 10.0 ** (-3.0 * t / self.settings.decay_s)
            g = float(min(max(g, 0.1), 0.9))
            comb_out = self._comb_filter(x, delay_samples, g, self.settings.damping)
            acc += comb_out

        if len(delays_ms) > 0:
            acc *= 1.0 / float(len(delays_ms))
        return acc

    def _apply_allpasses(self, x: np.ndarray) -> np.ndarray:
        out = x
        for d_ms in self._allpass_delays_ms:
            delay_samples = int(self.sr * (d_ms / 1000.0))
            out = self._allpass_filter(out, delay_samples, feedback=0.5)
        return out

    def process_mono(self, x_mono: np.ndarray) -> np.ndarray:
        """Process mono input and return stereo wet signal.

        The caller is responsible for mixing this wet signal back into
        a stereo dry signal using the configured wet level.
        """

        x = x_mono.astype(np.float32, copy=False)
        n = x.shape[-1]
        if n == 0:
            return np.zeros((2, 0), dtype=np.float32)

        s = self.settings

        # Pre-delay
        pre_delay_samples = int(self.sr * (s.pre_delay_ms / 1000.0))
        x_pre = self._pre_delay(x, pre_delay_samples)

        # Left and right comb banks with slightly different delays.
        left = self._build_comb_bank(x_pre, self._comb_delays_ms_left)
        right = self._build_comb_bank(x_pre, self._comb_delays_ms_right)

        # Shared all-pass chain per channel for extra diffusion.
        left = self._apply_allpasses(left)
        right = self._apply_allpasses(right)

        wet = np.stack([left, right], axis=0)
        return wet.astype(np.float32, copy=False)


def create_vocal_reverb_settings(
    role: str = "lead",
    dynamic_range_db: float | None = None,
    bpm: float | None = None,
    genre: str | None = None,
    stereo_width: float | None = None,
) -> ReverbSettings:
    """Derive reverb settings for a vocal role with simple adaptivity.

    - Lead vocal: 0.8–1.6 s decay, 20–40 ms pre-delay, 8–15% wet
    - Background: 1.2–2.2 s decay, 15–22% wet

    Adjustments:
    - High dynamic range -> shorter decay
    - BPM > 120 -> shorter decay
    - Trap/dancehall -> shorter/brighter
    - R&B -> longer/smoother
    - High stereo width -> reduce wet by 20%%
    """

    role_lower = (role or "lead").lower()

    if "bg" in role_lower or "background" in role_lower:
        base_decay = 1.7
        min_decay, max_decay = 1.2, 2.2
        base_wet = 0.18
    else:
        base_decay = 1.2
        min_decay, max_decay = 0.8, 1.6
        base_wet = 0.12

    decay = base_decay
    pre_delay_ms = 30.0
    wet = base_wet
    damping = 0.3

    # Dynamic range (crest factor) based adjustment
    if dynamic_range_db is not None:
        if dynamic_range_db > 14.0:
            decay *= 0.75
        elif dynamic_range_db < 8.0:
            decay *= 1.1

    # BPM-based adjustment
    if bpm is not None and bpm > 120.0:
        decay *= 0.8

    # Genre-based tweaks
    g = (genre or "").lower()
    if any(key in g for key in ["trap", "dancehall"]):
        decay *= 0.85
        damping = 0.2  # brighter plate-like tail
    elif any(key in g for key in ["rnb", "r&b"]):
        decay *= 1.15
        damping = 0.4

    # Clamp decay within role-specific band and global cap.
    decay = float(min(max(decay, min_decay), max_decay, 2.5))

    # Stereo width adjustment
    if stereo_width is not None and stereo_width > 0.5:
        wet *= 0.8

    # Global wet cap
    wet = float(min(max(wet, 0.0), 0.25))

    return ReverbSettings(decay_s=decay, pre_delay_ms=pre_delay_ms, wet=wet, damping=damping).clamped()
