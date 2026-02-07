from __future__ import annotations

import numpy as np
import pyloudnorm as pyln
from pedalboard import Pedalboard

try:
    from pedalboard import Gain  # type: ignore
except Exception:  # pragma: no cover - fallback only
    class Gain:
        def __init__(self, gain_db: float = 0.0, **kwargs) -> None:
            self.gain = gain_db

        def __call__(self, audio, sample_rate):
            factor = 10.0 ** (self.gain / 20.0)
            return audio * factor

try:
    from pedalboard import HighpassFilter, PeakFilter, HighShelfFilter, LowShelfFilter  # type: ignore
except Exception:  # pragma: no cover - fallback only
    class HighpassFilter:
        def __init__(self, cutoff_frequency_hz: float = 30.0, **kwargs) -> None:
            self.cutoff = cutoff_frequency_hz

        def __call__(self, audio, sample_rate):
            import numpy as _np

            rc = 1.0 / (2 * _np.pi * self.cutoff)
            dt = 1.0 / float(sample_rate)
            alpha = rc / (rc + dt)
            out = _np.zeros_like(audio)
            out[0] = audio[0]
            for i in range(1, len(audio)):
                out[i] = alpha * (out[i - 1] + audio[i] - audio[i - 1])
            return out

    class PeakFilter:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __call__(self, audio, sample_rate):
            return audio

    class HighShelfFilter(PeakFilter):
        pass

    class LowShelfFilter(PeakFilter):
        pass

try:
    from pedalboard import Compressor  # type: ignore
except Exception:  # pragma: no cover - fallback only
    class Compressor:
        def __init__(self, threshold_db: float = -12.0, ratio: float = 2.0, **kwargs) -> None:
            self.threshold = threshold_db
            self.ratio = ratio

        def __call__(self, audio, sample_rate):
            import numpy as _np

            threshold_lin = 10.0 ** (self.threshold / 20.0)
            mag = _np.abs(audio)
            over = _np.maximum(mag - threshold_lin, 0.0)
            gain = 1.0 / (1.0 + (self.ratio - 1.0) * over)
            return audio * gain

try:
    from pedalboard import Saturation  # type: ignore
except Exception:  # pragma: no cover - fallback only
    class Saturation:
        def __init__(self, drive_db: float = 0.0, **kwargs) -> None:
            self.drive = drive_db

        def __call__(self, audio, sample_rate):
            import numpy as _np

            gain = 10.0 ** (self.drive / 20.0)
            return _np.tanh(audio * gain)

try:
    from pedalboard import Limiter  # type: ignore
except Exception:  # pragma: no cover - fallback only
    class Limiter:
        def __init__(self, threshold_db: float = -1.0, **kwargs) -> None:
            self.threshold = 10.0 ** (threshold_db / 20.0)

        def __call__(self, audio, sample_rate):
            import numpy as _np

            return _np.clip(audio, -self.threshold, self.threshold)


def _pre_loudness_normalize(audio: np.ndarray, sr: int, target_lufs: float = -14.0) -> np.ndarray:
    """Normalize mix to a consistent loudness before dynamics.

    Slightly louder target than vocals so beats/masters hit glue
    compression and saturation in a predictable way.
    """

    if audio.ndim == 1:
        mono = audio.astype(np.float32)
    else:
        mono = (
            audio.mean(axis=0).astype(np.float32)
            if audio.shape[0] < audio.shape[1]
            else audio.mean(axis=1).astype(np.float32)
        )

    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(mono)
    loudness_diff = target_lufs - loudness
    gain_linear = 10.0 ** (loudness_diff / 20.0)
    return audio * gain_linear


def _prepare_for_pedalboard(audio: np.ndarray) -> np.ndarray:
    """Ensure (n_samples, n_channels) float32 for Pedalboard."""

    if audio.ndim == 1:
        audio_2d = audio[np.newaxis, :]
    else:
        audio_2d = audio if audio.shape[0] <= audio.shape[1] else audio.T
    return audio_2d.T.astype(np.float32)


def _restore_shape(processed: np.ndarray, original: np.ndarray) -> np.ndarray:
    """Return audio to original shape and dtype (float32)."""

    if original.ndim == 1:
        return processed[:, 0].astype(np.float32)
    channels_first = original.shape[0] <= original.shape[1]
    out = processed.T.astype(np.float32)
    return out if channels_first else out.T


def process_beat_or_master(audio: np.ndarray, sr: int, overrides: dict | None = None) -> np.ndarray:
    """Beat/master bus preset for more polished tone.

    - Gentle low-end clean-up
    - Subtle mid cut and presence boost
    - Glue-style compression
    - Harmonic saturation
    - Final limiter with streaming headroom
    """

    if audio.size == 0:
        return audio

    norm = _pre_loudness_normalize(audio, sr, target_lufs=-14.0)
    pb_input = _prepare_for_pedalboard(norm)

    # Allow reference analysis (via the "streaming_master" preset overrides)
    # to gently steer bus compression toward the reference loudness.
    base_comp_threshold = -14.0
    if overrides and isinstance(overrides, dict):
        comp_cfg = overrides.get("compressor") or {}
        try:
            th_value = comp_cfg.get("threshold")
            if th_value is not None:
                th = float(th_value)
                base_comp_threshold = th
        except (TypeError, ValueError):
            pass

    plugins = [
        HighpassFilter(cutoff_frequency_hz=30.0),
        LowShelfFilter(cutoff_frequency_hz=120.0, gain_db=-2.5),
        PeakFilter(cutoff_frequency_hz=350.0, gain_db=-1.5, q=0.9),
        PeakFilter(cutoff_frequency_hz=2500.0, gain_db=2.0, q=0.9),
        HighShelfFilter(cutoff_frequency_hz=10000.0, gain_db=2.0),
        Compressor(threshold_db=base_comp_threshold, ratio=2.0, attack_ms=12.0, release_ms=120.0),
        Saturation(drive_db=4.0),
        Limiter(threshold_db=-1.0, release_ms=120.0),
        Gain(gain_db=0.0),
    ]

    # Keep only real Pedalboard plugins when available
    plugins = [p for p in plugins if p.__class__.__module__.startswith("pedalboard")]
    board = Pedalboard(plugins)
    processed = board(pb_input, sr)
    return _restore_shape(processed, audio)
