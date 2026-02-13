"""CPU-safe delay / echo effect for MixSmvrt.

Implements a simple stereo delay with optional ping-pong mode using
circular buffers.

Constraints:
- Float32 only
- Feedback clamped to <= 30%
- Wet mix clamped to <= 25%
- No FFT / convolution
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DelaySettings:
    time_ms: float = 100.0
    feedback: float = 0.15
    wet: float = 0.12
    ping_pong: bool = False

    def clamped(self) -> "DelaySettings":
        t = float(min(max(self.time_ms, 40.0), 800.0))
        fb = float(min(max(self.feedback, 0.0), 0.3))
        wet = float(min(max(self.wet, 0.0), 0.25))
        return DelaySettings(time_ms=t, feedback=fb, wet=wet, ping_pong=self.ping_pong)


class StereoDelay:
    """Stereo delay with optional ping-pong.

    Processes stereo [2, N] float32 arrays in-place style.
    """

    def __init__(self, sr: int, settings: DelaySettings) -> None:
        self.sr = int(max(sr, 8000))
        self.settings = settings.clamped()

    def process(self, x_stereo: np.ndarray) -> np.ndarray:
        """Apply delay, returning the delayed signal only (no dry).

        The caller is expected to mix this with the dry signal using
        the configured wet mix.
        """

        if x_stereo.ndim != 2 or x_stereo.shape[0] != 2:
            raise ValueError("StereoDelay expects shape [2, N]")

        x = x_stereo.astype(np.float32, copy=False)
        n = x.shape[1]
        if n == 0:
            return np.zeros_like(x, dtype=np.float32)

        s = self.settings
        delay_samples = int(self.sr * (s.time_ms / 1000.0))
        delay_samples = max(delay_samples, 1)

        # Circular buffers per channel
        buf = np.zeros((2, delay_samples), dtype=np.float32)
        out = np.zeros_like(x, dtype=np.float32)
        idx = 0

        # Simple first-order high-pass in the feedback loop to avoid mud.
        hp_state = np.zeros(2, dtype=np.float32)
        hp_alpha = 0.995  # close to 1 => removes just a bit of low end

        fb = float(s.feedback)
        wet = float(s.wet)
        ping_pong = bool(s.ping_pong)

        for i in range(n):
            # Read from buffer (delayed signal)
            delayed = buf[:, idx]

            # High-pass filter in feedback path
            hp = delayed - hp_state + hp_alpha * hp_state
            hp_state = hp

            # Write new value into buffer: input + feedback * filtered delayed
            if ping_pong:
                # Cross-feed L<->R for ping-pong character
                buf[0, idx] = x[0, i] + fb * hp[1]
                buf[1, idx] = x[1, i] + fb * hp[0]
            else:
                buf[:, idx] = x[:, i] + fb * hp

            # Output is the delayed signal scaled by wet
            out[:, i] = delayed * wet

            idx += 1
            if idx >= delay_samples:
                idx = 0

        return out.astype(np.float32, copy=False)


def create_slap_delay_settings() -> DelaySettings:
    """Default slap delay for lead vocals.

    80–120 ms, 10–15% feedback, 8–12% wet. We choose a middle-ground
    setting and clamp in DelaySettings.
    """

    return DelaySettings(time_ms=100.0, feedback=0.12, wet=0.1, ping_pong=False).clamped()


def create_ping_pong_delay_settings(
    bpm: Optional[float] = None,
    note_fraction: float = 0.5,
) -> DelaySettings:
    """Ping-pong delay using tempo sync when BPM is available.

    note_fraction is the note length in beats (e.g. 0.5 = 1/8 note,
    1.0 = 1/4 note). If bpm is missing, falls back to 350 ms.
    """

    if bpm is not None and bpm > 0.0:
        beat_period_s = 60.0 / bpm
        delay_s = beat_period_s * note_fraction
        time_ms = float(delay_s * 1000.0)
    else:
        time_ms = 350.0

    return DelaySettings(time_ms=time_ms, feedback=0.2, wet=0.15, ping_pong=True).clamped()
