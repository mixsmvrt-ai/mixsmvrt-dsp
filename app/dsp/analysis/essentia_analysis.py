from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

try:  # Essentia is optional at runtime but recommended
    import essentia
    import essentia.standard as es  # type: ignore
    _ESSENTIA_AVAILABLE = True
except Exception:  # pragma: no cover - runtime fallback if Essentia is missing
    essentia = None  # type: ignore[assignment]
    es = None  # type: ignore[assignment]
    _ESSENTIA_AVAILABLE = False

logger = logging.getLogger(__name__)


def _to_mono_16k(audio: np.ndarray, sr: int, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Downmix to mono and resample to a lightweight analysis rate.

    The function avoids modifying the input array in-place and keeps
    the returned signal small enough for Render free-tier memory
    limits while still providing meaningful spectral information.
    """

    if audio.ndim > 1:
        # Downmix channels without allocating an excessively large buffer
        audio_mono = audio.mean(axis=-1)
    else:
        audio_mono = audio

    audio_mono = audio_mono.astype(np.float32, copy=False)

    if sr == target_sr or not _ESSENTIA_AVAILABLE:
        return audio_mono, sr

    try:
        resampler = es.Resample(inputSampleRate=float(sr), outputSampleRate=float(target_sr))  # type: ignore[attr-defined]
        audio_resampled = resampler(audio_mono)
        return np.asarray(audio_resampled, dtype=np.float32), target_sr
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Essentia resample failed, keeping original SR: %s", exc)
        return audio_mono, sr


def analyze_mono_signal(audio: np.ndarray, sr: int) -> Dict[str, float]:
    """Compute compact spectral and dynamics features using Essentia.

    The analysis is strictly read-only: it never mutates ``audio`` and
    only returns scalar descriptors to drive higher-level DSP choices.

    Returned keys:
        - rms: overall RMS level
        - brightness: average spectral centroid in Hz
        - transients: normalised transient strength (0..1)
        - dynamic_range: estimated crest factor (dB)
    """

    if audio.size == 0:
        return {"rms": 0.0, "brightness": 0.0, "transients": 0.0, "dynamic_range": 0.0}

    # Downmix + lightweight resample
    mono, sr_mono = _to_mono_16k(audio, sr)

    # Pure numpy fallbacks if Essentia is not available at runtime.
    if not _ESSENTIA_AVAILABLE:
        rms = float(np.sqrt(np.mean(mono ** 2)))
        spectrum = np.fft.rfft(mono)
        freqs = np.fft.rfftfreq(len(mono), d=1.0 / float(sr_mono))
        mag = np.abs(spectrum) + 1e-12
        brightness = float(np.sum(freqs * mag) / np.sum(mag))
        def _numpy_analysis(mono: np.ndarray, sr: int) -> Dict[str, float]:
            """Lightweight numpy-based feature extraction used as a fallback.

            This keeps behaviour well-defined even when Essentia is not
            available or fails at runtime.
            """

            rms = float(np.sqrt(np.mean(mono ** 2)))
            spectrum = np.fft.rfft(mono)
            freqs = np.fft.rfftfreq(len(mono), d=1.0 / float(sr))
            mag = np.abs(spectrum) + 1e-12
            brightness = float(np.sum(freqs * mag) / np.sum(mag))

            # Simple dynamic range estimate from percentiles of absolute amplitude.
            mag_abs = np.abs(mono)
            p95 = float(np.percentile(mag_abs, 95)) + 1e-9
            p5 = float(np.percentile(mag_abs, 5)) + 1e-9
            dynamic_range = float(20.0 * np.log10(p95 / p5))

            # Approximate transient strength via frame-to-frame energy changes.
            frame = 1024
            hop = 512
            if len(mono) < frame + hop:
                transients = 0.0
            else:
                frames = np.lib.stride_tricks.sliding_window_view(mono, frame)[::hop]
                energies = np.mean(frames ** 2, axis=1)
                diffs = np.abs(np.diff(energies))
                transients = float(np.clip(np.mean(diffs) / (np.max(energies) + 1e-9), 0.0, 1.0))

            return {
                "rms": rms,
                "brightness": brightness,
                "transients": transients,
                "dynamic_range": dynamic_range,
            }

        # Simple dynamic range estimate from percentiles of absolute amplitude.
        mag_abs = np.abs(mono)
        p95 = float(np.percentile(mag_abs, 95)) + 1e-9
        p5 = float(np.percentile(mag_abs, 5)) + 1e-9
        dynamic_range = float(20.0 * np.log10(p95 / p5))

        # Approximate transient strength via frame-to-frame energy changes.
        frame = 1024
                return _numpy_analysis(mono, sr_mono)
        for frame in es.FrameGenerator(mono, frame_size=frame_size, hop_size=hop_size, startFromZero=True):  # type: ignore[attr-defined]
            spec = spectrum(window(frame))
            centroids.append(float(centroid_alg(spec)))
            rms_frame = float(rms_alg(frame))
            rms_vals.append(rms_frame)
            energies.append(rms_frame ** 2)

        if not rms_vals:
            return {"rms": 0.0, "brightness": 0.0, "transients": 0.0, "dynamic_range": 0.0}

        rms = float(np.mean(rms_vals))
        brightness = float(np.mean(centroids))

        energies_arr = np.asarray(energies, dtype=np.float32)
        if energies_arr.size <= 1:
            transients = 0.0
            dynamic_range = 0.0
        else:
            diffs = np.abs(np.diff(energies_arr))
            transients = float(np.clip(np.mean(diffs) / (np.max(energies_arr) + 1e-9), 0.0, 1.0))

            p95 = float(np.percentile(energies_arr, 95)) + 1e-9
            p5 = float(np.percentile(energies_arr, 5)) + 1e-9
            dynamic_range = float(10.0 * np.log10(p95 / p5))

        logger.debug("[DSP] Essentia analysis completed: rms=%.3f, bright=%.1f, trans=%.3f, dr=%.2f", rms, brightness, transients, dynamic_range)

        return {
            "rms": rms,
            "brightness": brightness,
            "transients": transients,
            "dynamic_range": dynamic_range,
        }

    except Exception as exc:  # pragma: no cover - defensive path
        logger.warning("Essentia analysis failed, falling back to numpy: %s", exc)
        return analyze_mono_signal(audio, sr)
