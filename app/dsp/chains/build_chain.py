from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional

import numpy as np
from pedalboard import Pedalboard

try:
    from pedalboard import (  # type: ignore
        HighpassFilter,  # type: ignore
        HighShelfFilter,  # type: ignore
        Compressor,  # type: ignore
        Saturation,  # type: ignore
        Deesser,  # type: ignore
        Limiter,  # type: ignore
    )  # type: ignore
except Exception:  # pragma: no cover - lightweight fallbacks for environments
    class HighpassFilter:  # type: ignore[override]
        def __init__(self, cutoff_frequency_hz: float = 80.0, **_: Any) -> None:
            self.cutoff = cutoff_frequency_hz

        def __call__(self, audio, sample_rate):  # type: ignore[override]
            import numpy as _np
            rc = 1.0 / (2 * _np.pi * self.cutoff)
            dt = 1.0 / float(sample_rate)
            alpha = rc / (rc + dt)
            out = _np.zeros_like(audio)
            out[0] = audio[0]
            for i in range(1, len(audio)):
                out[i] = alpha * (out[i - 1] + audio[i] - audio[i - 1])
            return out

    class HighShelfFilter:  # type: ignore[override]
        def __init__(self, cutoff_frequency_hz: float = 8000.0, gain_db: float = 0.0, **_: Any) -> None:
            self.cutoff = cutoff_frequency_hz
            self.gain = gain_db

        def __call__(self, audio, sample_rate):  # type: ignore[override]
            return audio

    class Compressor:  # type: ignore[override]
        def __init__(self, threshold_db: float = -18.0, ratio: float = 2.0, attack_ms: float = 5.0, release_ms: float = 80.0, **_: Any) -> None:
            self.threshold = threshold_db
            self.ratio = ratio

        def __call__(self, audio, sample_rate):  # type: ignore[override]
            import numpy as _np
            thr = 10.0 ** (self.threshold / 20.0)
            mag = _np.abs(audio)
            over = _np.maximum(mag - thr, 0.0)
            gain = 1.0 / (1.0 + (self.ratio - 1.0) * over)
            return audio * gain

    class Saturation:  # type: ignore[override]
        def __init__(self, drive_db: float = 0.0, **_: Any) -> None:
            self.drive = drive_db

        def __call__(self, audio, sample_rate):  # type: ignore[override]
            import numpy as _np
            gain = 10.0 ** (self.drive / 20.0)
            return _np.tanh(audio * gain)

    class Deesser:  # type: ignore[override]
        def __init__(self, frequency: float = 7000.0, threshold_db: float = -30.0, ratio: float = 3.0, **_: Any) -> None:
            self.frequency = frequency
            self.threshold = threshold_db
            self.ratio = ratio

        def __call__(self, audio, sample_rate):  # type: ignore[override]
            return audio

    class Limiter:  # type: ignore[override]
        def __init__(self, threshold_db: float = -1.0, **_: Any) -> None:
            self.threshold = 10.0 ** (threshold_db / 20.0)

        def __call__(self, audio, sample_rate):  # type: ignore[override]
            import numpy as _np
            return _np.clip(audio, -self.threshold, self.threshold)


logger = logging.getLogger(__name__)


def _prepare_for_pedalboard(audio: np.ndarray) -> np.ndarray:
    """Return (samples, channels) float32 array for pedalboard.

    This keeps the transformation lightweight and reversible so the
    rest of the engine can continue working in its existing layout.
    """

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


def _map_brightness_to_shelf(brightness_hz: float, genre: Optional[str]) -> float:
    """Map spectral brightness to a gentle high-shelf gain.

    Genre subtly shapes the base tilt so trap/rap/dancehall feel
    a bit more hyped on top, while reggae/R&B stay smoother.
    """

    base = 0.0
    if genre:
        g = genre.lower()
        if any(k in g for k in ("trap", "dancehall", "rap", "hiphop")):
            base = 1.5
        elif any(k in g for k in ("rnb", "r&b")):
            base = 0.8
        elif "reggae" in g:
            base = 0.2

    if brightness_hz <= 1800.0:
        delta = 2.2
    elif brightness_hz >= 4500.0:
        delta = -2.2
    else:
        # Smooth interpolation between dark and bright
        t = (brightness_hz - 1800.0) / (4500.0 - 1800.0)
        delta = 2.2 * (1.0 - 2.0 * t)
    return float(np.clip(base + delta, -3.5, 3.5))


def _map_transients_to_attack(transients: float) -> float:
    """Weaker transients → slower attack, strong transients → faster attack."""

    transients = float(np.clip(transients, 0.0, 1.0))
    # Map [0,1] to [20ms, 5ms]
    return float(20.0 - 15.0 * transients)


def _map_dynamic_range_to_ratio(dynamic_range: float, genre: Optional[str], track_type: str) -> float:
    """Map measured dynamic range to compressor ratio with genre flavour.

    Trap/rap/dancehall lean a bit more controlled, R&B/Reggae a
    touch looser, and masters are slightly firmer than vocals.
    """

    dynamic_range = float(max(dynamic_range, 0.0))

    # Base curve for vocal-style sources
    if dynamic_range <= 6.0:
        base = 2.0
    elif dynamic_range >= 14.0:
        base = 3.8
    else:
        t = (dynamic_range - 6.0) / (14.0 - 6.0)
        base = 2.0 + t * (3.8 - 2.0)

    # Genre flavour
    flavour = 0.0
    if genre:
        g = genre.lower()
        if any(k in g for k in ("trap", "dancehall", "rap", "hiphop")):
            flavour = 0.3
        elif any(k in g for k in ("rnb", "r&b")):
            flavour = -0.1
        elif "reggae" in g:
            flavour = -0.2

    # Masters get a slightly firmer ratio than vocals at the same DR.
    if track_type == "master":
        flavour += 0.2

    return float(np.clip(base + flavour, 1.6, 4.2))


def _map_pitch_to_saturation(pitch_info: Mapping[str, Any] | None, default_drive: float = 4.0) -> float:
    if not pitch_info:
        return default_drive
    quality = str(pitch_info.get("tuning_quality", "stable"))
    if quality == "stable":
        return default_drive + 1.0
    if quality == "slightly_off":
        return default_drive
    return default_drive - 1.5


def _map_pitch_to_deesser(pitch_info: Mapping[str, Any] | None, base_freq: float = 7000.0) -> tuple[float, float]:
    if not pitch_info:
        return base_freq, 0.0
    median_f0 = float(pitch_info.get("median_f0_hz", 0.0) or 0.0)
    quality = str(pitch_info.get("tuning_quality", "stable"))

    if median_f0 <= 0.0:
        freq = base_freq
    else:
        # Rough mapping: brighter, higher voices → slightly higher de-ess band.
        # Clamp between 6.5 kHz and 9 kHz.
        freq = float(np.clip(base_freq * (median_f0 / 200.0) ** 0.1, 6500.0, 9000.0))

    if quality == "unstable":
        amount = 1.5
    elif quality == "slightly_off":
        amount = 1.0
    else:
        amount = 0.7
    return freq, amount


def build_pedalboard_chain(
    preset_key: str,
    track_type: str,
    analysis: Mapping[str, float] | None,
    pitch_info: Mapping[str, Any] | None,
    genre: Optional[str] = None,
) -> Pedalboard:
    """Build an EQ→Compressor→Saturation→De-esser→Limiter chain.

    The chain stays intentionally conservative and parameter ranges are
    kept narrow so that analysis steers flavour, not radical changes.
    """

    analysis = analysis or {}
    brightness_hz = float(analysis.get("brightness", 3000.0) or 3000.0)
    transients = float(analysis.get("transients", 0.5) or 0.5)
    dynamic_range = float(analysis.get("dynamic_range", 10.0) or 10.0)

    shelf_gain = _map_brightness_to_shelf(brightness_hz, genre)
    attack_ms = _map_transients_to_attack(transients)
    ratio = _map_dynamic_range_to_ratio(dynamic_range, genre, track_type)
    drive_db = _map_pitch_to_saturation(pitch_info, default_drive=4.0)
    deesser_freq, deesser_amount = _map_pitch_to_deesser(pitch_info, base_freq=7200.0)

    # Thresholds tuned for vocal/mix/master context, may be refined per preset.
    if track_type == "master":
        comp_threshold = -16.0
        limiter_ceiling = -1.0
    elif track_type == "beat":
        comp_threshold = -18.0
        limiter_ceiling = -1.2
    else:  # vocal / other
        comp_threshold = -20.0
        limiter_ceiling = -1.0

    plugins: list[Any] = [
        HighpassFilter(cutoff_frequency_hz=80.0),
        HighShelfFilter(cutoff_frequency_hz=9000.0, gain_db=shelf_gain),
        Compressor(
            threshold_db=comp_threshold,
            ratio=ratio,
            attack_ms=attack_ms,
            release_ms=120.0,
        ),
        Saturation(drive_db=drive_db),
        Deesser(
            frequency=deesser_freq,
            threshold_db=-30.0,
            ratio=3.5,
        ),
        Limiter(threshold_db=limiter_ceiling),
    ]

    # Only keep real pedalboard plugins when available.
    plugins = [p for p in plugins if p.__class__.__module__.startswith("pedalboard")]
    board = Pedalboard(plugins)

    logger.debug(
        "[DSP] Pedalboard chain built: preset=%s track_type=%s shelf=%.2f drive=%.2f ratio=%.2f",
        preset_key,
        track_type,
        shelf_gain,
        drive_db,
        ratio,
    )

    return board


def process_with_dynamic_chain(
    audio: np.ndarray,
    sr: int,
    preset_key: str,
    track_type: str,
    analysis: Mapping[str, float] | None,
    pitch_info: Mapping[str, Any] | None,
    genre: Optional[str] = None,
) -> np.ndarray:
    """High-level helper to build and apply the dynamic Pedalboard chain."""

    if audio.size == 0:
        return audio

    pb_input = _prepare_for_pedalboard(audio)
    board = build_pedalboard_chain(
        preset_key=preset_key,
        track_type=track_type,
        analysis=analysis,
        pitch_info=pitch_info,
        genre=genre,
    )
    processed = board(pb_input, sr)
    return _restore_shape(processed, audio)
