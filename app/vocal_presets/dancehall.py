from __future__ import annotations

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
        """Fallback De-Esser using a gentle high-shelf cut as approximation."""

        def __init__(self, frequency: float = 7000.0, threshold_db: float = -30.0, ratio: float = 3.0, **kwargs) -> None:
            self.frequency = frequency
            self.threshold_db = threshold_db
            self.ratio = ratio
            self._board = Pedalboard([
                HighShelfFilter(cutoff_frequency_hz=self.frequency, gain_db=-3.0),
            ])

        def __call__(self, audio, sample_rate):
            return self._board(audio, sample_rate)

from .tuning import apply_pitch_correction


def _pre_loudness_normalize(audio: np.ndarray, sr: int, target_lufs: float = -18.0) -> np.ndarray:
    """Normalize input to a consistent loudness before dynamics.

    This keeps the vocal hitting the compressors in a predictable way
    regardless of how it was recorded.
    """
    if audio.ndim == 1:
        mono = audio.astype(np.float32)
    else:
        # Downmix to mono for loudness measurement.
        mono = audio.mean(axis=0).astype(np.float32) if audio.shape[0] < audio.shape[1] else audio.mean(axis=1).astype(np.float32)

    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(mono)
    loudness_diff = target_lufs - loudness
    gain_linear = 10.0 ** (loudness_diff / 20.0)
    return audio * gain_linear


def _prepare_for_pedalboard(audio: np.ndarray) -> np.ndarray:
    """Ensure shape (n_samples, n_channels) float32 for Pedalboard."""
    if audio.ndim == 1:
        audio_2d = audio[np.newaxis, :]
    else:
        # Expect (channels, samples); if not, transpose.
        audio_2d = audio if audio.shape[0] <= audio.shape[1] else audio.T
    return audio_2d.T.astype(np.float32)


def _restore_shape(processed: np.ndarray, original: np.ndarray) -> np.ndarray:
    """Return audio to original shape and dtype (float32)."""
    # processed is (n_samples, n_channels)
    if original.ndim == 1:
        return processed[:, 0].astype(np.float32)
    # original was (channels, samples) or (samples, channels)
    channels_first = original.shape[0] <= original.shape[1]
    out = processed.T.astype(np.float32)
    return out if channels_first else out.T


def _process_vocal_gender(audio: np.ndarray, sr: int, gender: str | None) -> np.ndarray:
    """Dancehall vocal preset with simple gender variants.

    The core character stays the same, but female vocals get a slightly
    higher high‑pass, a touch more de‑essing, and a bit more air.
    """
    if audio.size == 0:
        return audio

    # 1) Loudness normalization (pre‑gain)
    norm = _pre_loudness_normalize(audio, sr, target_lufs=-18.0)

    # 2) Prepare for Pedalboard (samples, channels)
    pb_input = _prepare_for_pedalboard(norm)

    # Optional pitch correction using an external tuner plugin, if
    # configured via the VOCAL_TUNER_PLUGIN environment variable.
    pb_input = apply_pitch_correction(pb_input, sr)

    is_female = (gender or "male").lower() == "female"

    highpass_cutoff = 95.0 if not is_female else 115.0
    deesser_freq = 7100.0 if not is_female else 7600.0
    deesser_threshold = -30.0 if not is_female else -32.0
    compressor_threshold = -20.0 if not is_female else -22.0
    highshelf_gain = 2.0 if not is_female else 2.5

    plugins = [
        # 2) High‑pass filter – slightly higher for female voices.
        HighpassFilter(cutoff_frequency_hz=highpass_cutoff),
        # 3) Gentle noise gate to clean breaths/room between phrases.
        NoiseGate(threshold_db=-45.0, ratio=2.0, release_ms=120.0),
        # 4) De‑esser to tame harsh sibilance from bright presence boosts.
        Deesser(
            frequency=deesser_freq,
            threshold_db=deesser_threshold,
            ratio=4.0,
        ),
        # 5) Subtractive EQ – clean low end before hard compression.
        LowShelfFilter(
            cutoff_frequency_hz=180.0,
            gain_db=-2.5,
        ),
        # 6) Primary compressor – punchy, modern dancehall feel.
        Compressor(
            threshold_db=compressor_threshold,
            ratio=4.8,
            attack_ms=4.0,
            release_ms=90.0,
        ),
        # 7) Additive EQ – presence and top end.
        PeakFilter(
            cutoff_frequency_hz=2500.0,
            gain_db=4.0,
            q=1.0,
        ),
        HighShelfFilter(
            cutoff_frequency_hz=9500.0,
            gain_db=highshelf_gain,
        ),
        # 8) Glue compressor – softer, to sit the bright tone.
        Compressor(
            threshold_db=compressor_threshold + 2.0,
            ratio=2.6,
            attack_ms=15.0,
            release_ms=140.0,
        ),
        # 9) Harmonic saturation for energy and edge.
        Saturation(drive_db=7.0),
        # 10) Space and vibe – short plate reverb and slap delay.
        Reverb(
            room_size=0.26,
            damping=0.42,
            wet_level=0.2,
            dry_level=0.8,
            width=1.0,
        ),
        Delay(
            delay_seconds=0.27,
            feedback=0.26,
            mix=0.21,
        ),
        # 11) Final limiter with safe streaming headroom.
        Limiter(
            threshold_db=-1.3,
            release_ms=110.0,
        ),
        Gain(gain_db=-0.7),
    ]

    # Filter out any local fallback processors that are not real pedalboard plugins
    plugins = [p for p in plugins if p.__class__.__module__.startswith("pedalboard")]
    board = Pedalboard(plugins)

    processed = board(pb_input, sr)
    return _restore_shape(processed, audio)


def process_vocal_male(audio: np.ndarray, sr: int) -> np.ndarray:
    """Dancehall vocal preset tuned for male voices."""
    return _process_vocal_gender(audio, sr, "male")


def process_vocal_female(audio: np.ndarray, sr: int) -> np.ndarray:
    """Dancehall vocal preset tuned for female voices."""
    return _process_vocal_gender(audio, sr, "female")


def process_vocal(audio: np.ndarray, sr: int) -> np.ndarray:
    """Backward‑compatible entry point (defaults to male variant)."""
    return _process_vocal_gender(audio, sr, "male")
