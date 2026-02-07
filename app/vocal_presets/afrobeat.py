from __future__ import annotations

# pyright: reportGeneralTypeIssues=false, reportUnknownMemberType=false, reportUnknownArgumentType=false

import numpy as np
import pyloudnorm as pyln
from pedalboard import Pedalboard

# Try to import all processors from pedalboard; if any are missing on the
# deployed version, fall back to simple implementations so the service
# still runs without import errors.
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
            # Simple first-order high-pass approximation
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

    Slightly lower target LUFS than hard rap presets for more openness.
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
    """Afrobeat vocal preset with male/female variants."""
    if audio.size == 0:
        return audio

    # Afrobeat leads are bright but a bit more open; keep them
    # slightly less dense than hard rap presets.
    norm = _pre_loudness_normalize(audio, sr, target_lufs=-19.5)
    pb_input = _prepare_for_pedalboard(norm)

    # Optional pitch correction using an external tuner plugin if available.
    pb_input = apply_pitch_correction(pb_input, sr)

    is_female = (gender or "male").lower() == "female"

    highpass_cutoff = 88.0 if not is_female else 98.0
    deesser_freq = 6700.0 if not is_female else 7200.0
    highshelf_gain = 2.0 if not is_female else 2.4

    plugins = [
        # Front-of-chain cleanup
        HighpassFilter(cutoff_frequency_hz=highpass_cutoff),
        NoiseGate(threshold_db=-51.0, ratio=1.9, release_ms=155.0),
        Deesser(
            frequency=deesser_freq,
            threshold_db=-29.0,
            ratio=3.0,
        ),
        # Subtractive EQ – tidy low end and low-mids before heavy compression
        LowShelfFilter(
            cutoff_frequency_hz=170.0,
            gain_db=-1.5,
        ),
        PeakFilter(
            cutoff_frequency_hz=260.0,
            gain_db=-2.0,
            q=1.0,
        ),
        # Level compressor – main peak/density control
        Compressor(
            threshold_db=-19.5,
            ratio=3.5,
            attack_ms=6.0,
            release_ms=135.0,
        ),
        # Additive / corrective EQ – presence + air
        PeakFilter(
            cutoff_frequency_hz=2300.0,
            gain_db=2.8,
            q=0.9,
        ),
        HighShelfFilter(
            cutoff_frequency_hz=9500.0,
            gain_db=highshelf_gain,
        ),
        # Glue compressor – softer ratio to gel the shaped tone
        Compressor(
            threshold_db=-17.5,
            ratio=2.6,
            attack_ms=16.0,
            release_ms=160.0,
        ),
        # Colour + space + safety
        Saturation(drive_db=5.5),
        Reverb(
            room_size=0.3,
            damping=0.46,
            wet_level=0.22,
            dry_level=0.78,
            width=1.0,
        ),
        Delay(
            delay_seconds=0.3,
            feedback=0.26,
            mix=0.2,
        ),
        Limiter(
            threshold_db=-1.4,
            release_ms=130.0,
        ),
        Gain(gain_db=-0.7),
    ]

    plugins = [p for p in plugins if p.__class__.__module__.startswith("pedalboard")]
    board = Pedalboard(plugins)  # type: ignore
    processed = board(pb_input, sr)
    return _restore_shape(processed, audio)


def process_vocal_male(audio: np.ndarray, sr: int) -> np.ndarray:
    """Afrobeat vocal preset tuned for male voices."""
    return _process_vocal_gender(audio, sr, "male")


def process_vocal_female(audio: np.ndarray, sr: int) -> np.ndarray:
    """Afrobeat vocal preset tuned for female voices."""
    return _process_vocal_gender(audio, sr, "female")


def process_vocal(audio: np.ndarray, sr: int) -> np.ndarray:
    """Backward‑compatible entry point (defaults to male variant)."""
    return _process_vocal_gender(audio, sr, "male")


def _process_background_gender(audio: np.ndarray, sr: int, gender: str | None) -> np.ndarray:
    """Afrobeat background vocals – softer, a bit darker and wetter."""
    if audio.size == 0:
        return audio

    lead = _process_vocal_gender(audio, sr, gender)
    pb_input = _prepare_for_pedalboard(lead)

    bg_plugins = [
        HighpassFilter(cutoff_frequency_hz=165.0),
        LowShelfFilter(cutoff_frequency_hz=210.0, gain_db=-2.0),
        HighShelfFilter(cutoff_frequency_hz=10500.0, gain_db=-0.5),
        Reverb(
            room_size=0.38,
            damping=0.48,
            wet_level=0.3,
            dry_level=0.7,
            width=1.0,
        ),
        Delay(
            delay_seconds=0.3,
            feedback=0.3,
            mix=0.26,
        ),
        Gain(gain_db=-4.0),
    ]

    # Filter out any local fallback processors that are not real
    # pedalboard-native plugins to avoid Chain() type errors.
    bg_plugins = [
        p for p in bg_plugins if p.__class__.__module__.startswith("pedalboard")
    ]
    bg_board = Pedalboard(bg_plugins)  # type: ignore

    processed = bg_board(pb_input, sr)
    return _restore_shape(processed, lead)


def _process_adlib_gender(audio: np.ndarray, sr: int, gender: str | None) -> np.ndarray:
    """Afrobeat adlibs – brighter, hyped and wetter than the lead."""
    if audio.size == 0:
        return audio

    lead = _process_vocal_gender(audio, sr, gender)
    pb_input = _prepare_for_pedalboard(lead)

    adlib_plugins = [
        HighpassFilter(cutoff_frequency_hz=185.0),
        PeakFilter(cutoff_frequency_hz=3100.0, gain_db=2.5, q=1.0),
        HighShelfFilter(cutoff_frequency_hz=12000.0, gain_db=2.0),
        Saturation(drive_db=5.0),
        Reverb(
            room_size=0.48,
            damping=0.5,
            wet_level=0.36,
            dry_level=0.64,
            width=1.0,
        ),
        Delay(
            delay_seconds=0.34,
            feedback=0.32,
            mix=0.32,
        ),
        Gain(gain_db=-5.0),
    ]

    # Filter out any local fallback processors that are not real
    # pedalboard-native plugins to avoid Chain() type errors when the
    # deployed pedalboard build lacks certain modules (e.g. Saturation).
    adlib_plugins = [
        p for p in adlib_plugins if p.__class__.__module__.startswith("pedalboard")
    ]
    adlib_board = Pedalboard(adlib_plugins)  # type: ignore

    processed = adlib_board(pb_input, sr)
    return _restore_shape(processed, lead)


def process_vocal_background(audio: np.ndarray, sr: int) -> np.ndarray:
    return _process_background_gender(audio, sr, "male")


def process_vocal_adlib(audio: np.ndarray, sr: int) -> np.ndarray:
    return _process_adlib_gender(audio, sr, "male")
