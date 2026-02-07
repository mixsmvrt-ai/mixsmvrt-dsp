from __future__ import annotations

from typing import Any

import numpy as np
import pyloudnorm as pyln
from pedalboard import Pedalboard

try:
    from pedalboard import Gain  # type: ignore
except Exception:
    class Gain:
        def __init__(self, gain_db: float = 0.0, **kwargs) -> None:
            self.gain = gain_db

        def __call__(self, audio, sample_rate):
            factor = 10.0 ** (self.gain / 20.0)
            return audio * factor

try:
    from pedalboard import HighpassFilter  # type: ignore
except Exception:
    class HighpassFilter:
        def __init__(self, cutoff_frequency_hz: float = 80.0, **kwargs) -> None:
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

try:
    from pedalboard import Compressor  # type: ignore
except Exception:
    class Compressor:
        def __init__(self, threshold_db: float = -18.0, ratio: float = 2.0, **kwargs) -> None:
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
    from pedalboard import Limiter  # type: ignore
except Exception:
    class Limiter:
        def __init__(self, threshold_db: float = -1.0, **kwargs) -> None:
            self.threshold = 10.0 ** (threshold_db / 20.0)

        def __call__(self, audio, sample_rate):
            import numpy as _np
            return _np.clip(audio, -self.threshold, self.threshold)

try:
    from pedalboard import NoiseGate  # type: ignore
except Exception:
    class NoiseGate:
        def __init__(self, threshold_db: float = -60.0, **kwargs) -> None:
            self.threshold = 10.0 ** (threshold_db / 20.0)

        def __call__(self, audio, sample_rate):
            import numpy as _np
            mag = _np.abs(audio)
            mask = mag >= self.threshold
            return audio * mask

try:
    from pedalboard import PeakFilter, HighShelfFilter, LowShelfFilter  # type: ignore
except Exception:
    class PeakFilter:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __call__(self, audio, sample_rate):
            return audio

    class HighShelfFilter(PeakFilter):
        pass

    class LowShelfFilter(PeakFilter):
        pass

try:
    from pedalboard import Saturation  # type: ignore
except Exception:
    class Saturation:
        """Fallback saturation using soft clipping."""

        def __init__(self, drive_db: float = 0.0, **kwargs) -> None:
            self.drive = drive_db

        def __call__(self, audio, sample_rate):
            import numpy as _np
            gain = 10.0 ** (self.drive / 20.0)
            return _np.tanh(audio * gain)

try:
    from pedalboard import Reverb, Delay  # type: ignore
except Exception:
    class Reverb:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __call__(self, audio, sample_rate):
            return audio

    class Delay:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __call__(self, audio, sample_rate):
            return audio

try:
    from pedalboard import Deesser  # type: ignore
except Exception:
    class Deesser:
        """Fallback De-Esser used when pedalboard's Deesser is unavailable.

        This implementation is intentionally conservative and leaves the
        signal unchanged rather than risking artifacts.
        """

        def __init__(
            self,
            frequency: float = 7000.0,
            threshold_db: float = -30.0,
            ratio: float = 3.0,
            **kwargs,
        ) -> None:
            self.frequency = frequency
            self.threshold_db = threshold_db
            self.ratio = ratio

        def __call__(self, audio, sample_rate):
            return audio

from .tuning import apply_pitch_correction


def _pre_loudness_normalize(audio: np.ndarray, sr: int, target_lufs: float = -19.0) -> np.ndarray:
    """Normalize input to a consistent loudness before dynamics.

    Slightly softer target for smoother R&B feel.
    """
    if audio.ndim == 1:
        mono = audio.astype(np.float32)
    else:
        mono = audio.mean(axis=0).astype(np.float32) if audio.shape[0] < audio.shape[1] else audio.mean(axis=1).astype(np.float32)

    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(mono)
    loudness_diff = target_lufs - loudness
    gain_linear = 10.0 ** (loudness_diff / 20.0)
    return audio * gain_linear


def _prepare_for_pedalboard(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        audio_2d = audio[np.newaxis, :]
    else:
        audio_2d = audio if audio.shape[0] <= audio.shape[1] else audio.T
    return audio_2d.T.astype(np.float32)


def _restore_shape(processed: np.ndarray, original: np.ndarray) -> np.ndarray:
    if original.ndim == 1:
        return processed[:, 0].astype(np.float32)
    channels_first = original.shape[0] <= original.shape[1]
    out = processed.T.astype(np.float32)
    return out if channels_first else out.T


def _process_vocal_gender(audio: np.ndarray, sr: int, gender: str | None) -> np.ndarray:
    """R&B vocal preset with male/female variants."""
    if audio.size == 0:
        return audio

    norm = _pre_loudness_normalize(audio, sr, target_lufs=-19.0)
    pb_input = _prepare_for_pedalboard(norm)

    # Optional pitch correction using an external tuner plugin if available.
    pb_input = apply_pitch_correction(pb_input, sr)

    is_female = (gender or "male").lower() == "female"

    highpass_cutoff = 80.0 if not is_female else 95.0
    deesser_freq = 6600.0 if not is_female else 7100.0
    highshelf_gain = 3.0 if not is_female else 3.5

    plugins = [
        # Front-of-chain cleanup
        HighpassFilter(cutoff_frequency_hz=highpass_cutoff),
        NoiseGate(threshold_db=-52.0, ratio=1.6, release_ms=180.0),
        Deesser(
            frequency=deesser_freq,
            threshold_db=-32.0,
            ratio=2.7,
        ),
        # Subtractive EQ – gentle low/low-mid shaping for warm R&B body
        LowShelfFilter(
            cutoff_frequency_hz=150.0,
            gain_db=-0.5,
        ),
        PeakFilter(
            cutoff_frequency_hz=260.0,
            gain_db=-1.5,
            q=1.0,
        ),
        # Level compressor – keep dynamics under control without killing feel
        Compressor(
            threshold_db=-18.5,
            ratio=3.4,
            attack_ms=9.0,
            release_ms=160.0,
        ),
        # Additive EQ – body + top sheen
        PeakFilter(
            cutoff_frequency_hz=2200.0,
            gain_db=2.5,
            q=0.9,
        ),
        HighShelfFilter(
            cutoff_frequency_hz=10500.0,
            gain_db=highshelf_gain,
        ),
        # Glue compressor – slower, softer to keep things smooth
        Compressor(
            threshold_db=-17.0,
            ratio=2.4,
            attack_ms=18.0,
            release_ms=190.0,
        ),
        # Colour + space + safety
        Saturation(drive_db=5.5),
        Reverb(
            room_size=0.32,
            damping=0.5,
            wet_level=0.24,
            dry_level=0.76,
            width=1.0,
        ),
        Delay(
            delay_seconds=0.33,
            feedback=0.27,
            mix=0.22,
        ),
        Limiter(
            threshold_db=-1.5,
            release_ms=140.0,
        ),
        Gain(gain_db=-0.7),
    ]

    plugins = [p for p in plugins if p.__class__.__module__.startswith("pedalboard")]
    board = Pedalboard(plugins)  # type: ignore[arg-type]
    processed = board(pb_input, sr)
    return _restore_shape(processed, audio)


def process_vocal_male(audio: np.ndarray, sr: int) -> np.ndarray:
    """R&B vocal preset tuned for male voices."""
    return _process_vocal_gender(audio, sr, "male")


def process_vocal_female(audio: np.ndarray, sr: int) -> np.ndarray:
    """R&B vocal preset tuned for female voices."""
    return _process_vocal_gender(audio, sr, "female")


def process_vocal(audio: np.ndarray, sr: int) -> np.ndarray:
    """Backward‑compatible entry point (defaults to male variant)."""
    return _process_vocal_gender(audio, sr, "male")


def _process_background_gender(audio: np.ndarray, sr: int, gender: str | None) -> np.ndarray:
    """R&B background vocals – smooth, darker and more spacious."""
    if audio.size == 0:
        return audio

    lead = _process_vocal_gender(audio, sr, gender)
    pb_input = _prepare_for_pedalboard(lead)

    bg_plugins: list[Any] = [
        HighpassFilter(cutoff_frequency_hz=150.0),
        LowShelfFilter(cutoff_frequency_hz=210.0, gain_db=-1.8),
        HighShelfFilter(cutoff_frequency_hz=10000.0, gain_db=-0.5),
        Reverb(
            room_size=0.42,
            damping=0.5,
            wet_level=0.34,
            dry_level=0.66,
            width=1.0,
        ),
        Delay(
            delay_seconds=0.34,
            feedback=0.32,
            mix=0.3,
        ),
        Gain(gain_db=-4.0),
    ]
    bg_board = Pedalboard(bg_plugins)

    processed = bg_board(pb_input, sr)
    return _restore_shape(processed, lead)


def _process_adlib_gender(audio: np.ndarray, sr: int, gender: str | None) -> np.ndarray:
    """R&B adlibs – airy, expressive, wetter but still smooth."""
    if audio.size == 0:
        return audio

    lead = _process_vocal_gender(audio, sr, gender)
    pb_input = _prepare_for_pedalboard(lead)

    adlib_plugins: list[Any] = [
        HighpassFilter(cutoff_frequency_hz=180.0),
        PeakFilter(cutoff_frequency_hz=3000.0, gain_db=2.5, q=1.0),
        HighShelfFilter(cutoff_frequency_hz=11500.0, gain_db=2.0),
        Saturation(drive_db=5.0),
        Reverb(
            room_size=0.5,
            damping=0.52,
            wet_level=0.38,
            dry_level=0.62,
            width=1.0,
        ),
        Delay(
            delay_seconds=0.36,
            feedback=0.34,
            mix=0.32,
        ),
        Gain(gain_db=-5.0),
    ]
    adlib_board = Pedalboard(adlib_plugins)

    processed = adlib_board(pb_input, sr)
    return _restore_shape(processed, lead)


def process_vocal_background(audio: np.ndarray, sr: int) -> np.ndarray:
    return _process_background_gender(audio, sr, "male")


def process_vocal_adlib(audio: np.ndarray, sr: int) -> np.ndarray:
    return _process_adlib_gender(audio, sr, "male")
