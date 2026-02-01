from __future__ import annotations

# pyright: strict=false, reportGeneralTypeIssues=false, reportUnknownMemberType=false, reportUnknownArgumentType=false

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
    """Normalize input to a consistent loudness before dynamics."""
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
    """Trap Dancehall vocal preset with male/female variants."""
    if audio.size == 0:
        return audio

    # Trap‑dancehall sits dense in modern mixes; keep it upfront
    # but leave a little more headroom for throw FX and mastering.
    norm = _pre_loudness_normalize(audio, sr, target_lufs=-18.5)
    pb_input = _prepare_for_pedalboard(norm)

    # Optional pitch correction using an external tuner plugin if available.
    pb_input = apply_pitch_correction(pb_input, sr)

    is_female = (gender or "male").lower() == "female"

    highpass_cutoff = 120.0 if not is_female else 135.0
    deesser_freq = 7400.0 if not is_female else 7900.0
    deesser_threshold = -32.0 if not is_female else -34.0
    highshelf_gain = 2.8 if not is_female else 3.2

    plugins = [
        # Front-of-chain cleanup
        HighpassFilter(cutoff_frequency_hz=highpass_cutoff),
        NoiseGate(threshold_db=-46.0, ratio=2.8, release_ms=150.0),
        Deesser(
            frequency=deesser_freq,
            threshold_db=deesser_threshold,
            ratio=4.8,
        ),
        # Subtractive EQ block – similar in spirit to a Neutron-style
        # cleanup: cut rumble, mud, and a couple of harsh bands.
        LowShelfFilter(
            cutoff_frequency_hz=140.0,
            gain_db=-3.0,
        ),
        PeakFilter(
            cutoff_frequency_hz=280.0,
            gain_db=-3.0,
            q=1.0,
        ),
        PeakFilter(
            cutoff_frequency_hz=550.0,
            gain_db=-2.5,
            q=1.2,
        ),
        PeakFilter(
            cutoff_frequency_hz=4475.0,
            gain_db=-7.0,
            q=8.5,
        ),
        # Level/peak compressor – fast and assertive, after cleanup
        Compressor(
            threshold_db=-22.0,
            ratio=5.4,
            attack_ms=2.5,
            release_ms=80.0,
        ),
        # Additive/shape EQ block – bring back controlled presence and air
        PeakFilter(
            cutoff_frequency_hz=2400.0,
            gain_db=3.0,
            q=0.9,
        ),
        HighShelfFilter(
            cutoff_frequency_hz=10500.0,
            gain_db=highshelf_gain,
        ),
        # Glue compressor – slightly gentler bus-style squeeze
        Compressor(
            threshold_db=-18.0,
            ratio=2.6,
            attack_ms=15.0,
            release_ms=140.0,
        ),
        # Colour + space + safety
        Saturation(drive_db=8.5),
        Reverb(
            room_size=0.24,
            damping=0.42,
            wet_level=0.2,
            dry_level=0.8,
            width=1.0,
        ),
        Delay(
            delay_seconds=0.25,
            feedback=0.24,
            mix=0.2,
        ),
        Limiter(
            threshold_db=-1.3,
            release_ms=90.0,
        ),
        Gain(gain_db=-0.7),
    ]

    plugins = [p for p in plugins if p.__class__.__module__.startswith("pedalboard")]
    board = Pedalboard(plugins)

    processed = board(pb_input, sr)
    return _restore_shape(processed, audio)


def process_vocal_male(audio: np.ndarray, sr: int) -> np.ndarray:
    """Trap Dancehall vocal preset tuned for male voices."""
    return _process_vocal_gender(audio, sr, "male")


def process_vocal_female(audio: np.ndarray, sr: int) -> np.ndarray:
    """Trap Dancehall vocal preset tuned for female voices."""
    return _process_vocal_gender(audio, sr, "female")


def process_vocal(audio: np.ndarray, sr: int) -> np.ndarray:
    """Backward‑compatible entry point (defaults to male variant)."""
    return _process_vocal_gender(audio, sr, "male")
