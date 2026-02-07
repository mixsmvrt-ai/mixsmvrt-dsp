import librosa
import numpy as np

from app.dsp.analysis.essentia_analysis import analyze_mono_signal
from app.dsp.analysis.pitch_world import analyze_pitch_world
from app.processors.loudness import measure_loudness


def _build_preset_overrides(lufs: float, brightness: float) -> dict:
    """Generate simple preset overrides from reference metrics.

    This is intentionally conservative: it nudges existing presets rather than
    replacing them entirely, so the reference track acts like a taste profile
    for loudness and brightness.
    """

    # Target streaming loudness around -14 LUFS; use the reference offset to
    # gently push compressor thresholds.
    loudness_offset = float(lufs - (-14.0))
    comp_delta = float(np.clip(loudness_offset / 4.0, -3.0, 3.0))

    # Normalise brightness (roughly) between 0..1 around ~2 kHz.
    norm_brightness = float(np.clip((brightness - 2000.0) / 4000.0, 0.0, 1.0))

    # Map brightness to presence gain range.
    presence = 4.0 + (norm_brightness - 0.5) * 4.0
    presence = float(np.clip(presence, 1.0, 8.0))

    # Slightly tighten or relax compression based on loudness offset.
    vocal_threshold = float(-18.0 - comp_delta * 2.0)
    master_threshold = float(-12.0 - comp_delta * 1.5)

    return {
        # Lead vocal chain tweaks – presence and compression intensity.
        "clean_vocal": {
            "eq": {"presence": presence},
            "compressor": {"threshold": vocal_threshold},
        },
        # Background stack glue – follow the same curve but a bit softer.
        "bg_vocal_glue": {
            "eq": {"presence": presence - 1.0},
            "compressor": {"threshold": vocal_threshold + 1.5},
        },
        # Adlibs – slightly brighter and more compressed.
        "adlib_hype": {
            "eq": {"presence": presence + 0.5},
            "compressor": {"threshold": vocal_threshold - 1.0},
        },
        # Master bus – gently adjust compressor threshold.
        "streaming_master": {
            "compressor": {"threshold": master_threshold},
        },
    }


def analyze_audio(file):
    """Return loudness/peak/duration plus preset overrides from a reference.

    The frontend can feed the returned ``preset_overrides`` object back into
    the /process endpoint to adapt chain params for this session.
    """

    # librosa can read directly from a file-like object
    y, sr = librosa.load(file.file, sr=None, mono=True)

    rms = float(np.sqrt(np.mean(y ** 2)))
    peak = float(np.max(np.abs(y)))
    duration = float(librosa.get_duration(y=y, sr=int(sr)))

    # Integrated loudness in LUFS.
    lufs = measure_loudness(y, int(sr))

    # Crude tonal brightness via spectral centroid.
    centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())

    # Tier-1 analysis: Essentia + WORLD (with safe fallbacks when
    # libraries are not available at runtime).
    essentia_features = analyze_mono_signal(y, int(sr))
    pitch_profile = analyze_pitch_world(y, int(sr))

    preset_overrides = _build_preset_overrides(lufs, centroid)

    return {
        "sample_rate": int(sr),
        "rms": rms,
        "peak": peak,
        "duration": duration,
        "lufs": lufs,
        "brightness_hz": centroid,
        "preset_overrides": preset_overrides,
        "essentia": essentia_features,
        "pitch": pitch_profile,
    }
