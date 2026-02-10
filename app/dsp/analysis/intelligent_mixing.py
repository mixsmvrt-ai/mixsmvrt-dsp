"""Intelligent analysis + decision logic for AI-assisted mixing.

This module does NOT render audio.
It analyzes a single track (beat, lead, bg, adlibs) and returns:
- rich analysis metrics (loudness, spectrum, dynamics, masking hints)
- a suggested, flow-aware plugin chain for the Studio UI.

Designed to be called from FastAPI endpoints in app.main/engine.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
import pyloudnorm as pyln
from scipy import signal

try:  # librosa is optional
    import librosa  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    librosa = None  # type: ignore


TrackRole = Literal["beat", "vocal_lead", "vocal_bg", "vocal_adlib"]
FeatureFlow = Literal[
    "audio_cleanup",
    "mixing_only",
    "mix_master",
    "mastering_only",
    "beat_only",
]


@dataclass
class LoudnessAnalysis:
    integrated_lufs: float
    short_term_lufs: float
    rms_dbfs: float
    peak_dbfs: float
    true_peak_dbfs: float
    crest_factor_db: float
    silence_ratio: float
    duration_sec: float


@dataclass
class SpectralBandEnergy:
    band_rms_db: Dict[str, float]
    band_delta_db: Dict[str, float]


@dataclass
class MaskingAnalysis:
    has_masking: bool
    masking_bands: List[str]


@dataclass
class DynamicsAnalysis:
    rms_variance: float
    peak_density: float
    transient_sharpness: float


@dataclass
class StereoAnalysis:
    mid_energy: float
    side_energy: float
    mid_side_ratio: float
    phase_correlation: float


@dataclass
class TrackAnalysis:
    loudness: LoudnessAnalysis
    spectral: SpectralBandEnergy
    dynamics: DynamicsAnalysis
    stereo: StereoAnalysis
    masking: Optional[MaskingAnalysis]
    muddy: bool
    harsh: bool


@dataclass
class PluginConfig:
    plugin: str
    params: Dict[str, Any]


GenreKey = Literal[
    "afrobeat",
    "trap_dancehall",
    "dancehall",
    "hiphop",
    "rap",
    "rnb",
    "reggae",
    "generic",
]


# Simple genre-dependent target LUFS for individual tracks
GENRE_TARGET_LUFS: Dict[GenreKey, Dict[TrackRole, float]] = {
    "generic": {"beat": -12.0, "vocal_lead": -18.0, "vocal_bg": -20.0, "vocal_adlib": -19.0},
    "afrobeat": {"beat": -11.5, "vocal_lead": -18.0, "vocal_bg": -20.0, "vocal_adlib": -19.0},
    "trap_dancehall": {"beat": -10.5, "vocal_lead": -17.0, "vocal_bg": -19.0, "vocal_adlib": -18.0},
    "dancehall": {"beat": -11.0, "vocal_lead": -17.5, "vocal_bg": -19.0, "vocal_adlib": -18.0},
    "hiphop": {"beat": -11.5, "vocal_lead": -17.5, "vocal_bg": -19.0, "vocal_adlib": -18.0},
    "rap": {"beat": -11.5, "vocal_lead": -17.0, "vocal_bg": -19.0, "vocal_adlib": -18.0},
    "rnb": {"beat": -12.0, "vocal_lead": -19.0, "vocal_bg": -20.0, "vocal_adlib": -19.0},
    "reggae": {"beat": -13.0, "vocal_lead": -19.0, "vocal_bg": -20.0, "vocal_adlib": -19.0},
}


# Genre target curves: approximate dB offsets per band from a flat reference
# Positive = typical genre leans hotter in that band.
GENRE_TARGET_SPECTRAL: Dict[GenreKey, Dict[str, float]] = {
    "generic": {
        "sub": 0.0,
        "low": 0.0,
        "low_mid": 0.0,
        "mid": 0.0,
        "presence": 0.0,
        "air": 0.0,
    },
    "afrobeat": {
        "sub": +1.0,
        "low": +0.5,
        "low_mid": -0.5,
        "mid": 0.0,
        "presence": +0.5,
        "air": +1.0,
    },
    "trap_dancehall": {
        "sub": +2.0,
        "low": +1.5,
        "low_mid": -1.0,
        "mid": 0.0,
        "presence": +0.5,
        "air": +1.0,
    },
    "dancehall": {
        "sub": +1.5,
        "low": +1.0,
        "low_mid": -0.5,
        "mid": 0.0,
        "presence": +0.5,
        "air": +0.5,
    },
    "hiphop": {
        "sub": +1.5,
        "low": +1.0,
        "low_mid": -0.5,
        "mid": 0.0,
        "presence": +0.5,
        "air": +0.5,
    },
    "rap": {
        "sub": +1.5,
        "low": +1.0,
        "low_mid": -0.5,
        "mid": 0.0,
        "presence": +0.5,
        "air": +0.5,
    },
    "rnb": {
        "sub": +0.5,
        "low": 0.0,
        "low_mid": -0.5,
        "mid": 0.0,
        "presence": +0.5,
        "air": +1.0,
    },
    "reggae": {
        "sub": +1.0,
        "low": +0.5,
        "low_mid": -0.5,
        "mid": 0.0,
        "presence": 0.0,
        "air": +0.5,
    },
}


SPECTRAL_BANDS: List[Tuple[str, float, float]] = [
    ("sub", 20.0, 60.0),
    ("low", 60.0, 200.0),
    ("low_mid", 200.0, 500.0),
    ("mid", 500.0, 2000.0),
    ("presence", 2000.0, 6000.0),
    ("air", 6000.0, 16000.0),
]


def _ensure_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    if audio.ndim == 2:
        return audio.mean(axis=0).astype(np.float32, copy=False)
    return audio.reshape(-1).astype(np.float32, copy=False)


def _safe_db(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(x), eps))


def analyze_loudness(audio: np.ndarray, sr: int) -> LoudnessAnalysis:
    """Compute LUFS, RMS, peaks, crest factor, silence ratio."""

    mono = _ensure_mono(audio)
    duration_sec = mono.shape[0] / float(sr)

    # EBU R128 loudness. Very short clips can raise
    # "Audio must have length greater than the block size" in pyloudnorm,
    # so fall back to an RMS-derived estimate instead of failing.
    try:
        meter = pyln.Meter(sr)  # EBU R128
        integrated_lufs = float(meter.integrated_loudness(mono))
        # pyloudnorm Meter does not have short_term_loudness; use integrated as fallback
        short_term_lufs = integrated_lufs
    except Exception:
        rms_fallback = float(np.sqrt(np.mean(mono ** 2))) if mono.size > 0 else 0.0
        rms_dbfs_fallback = float(_safe_db(np.array([rms_fallback]))[0])
        integrated_lufs = rms_dbfs_fallback
        short_term_lufs = rms_dbfs_fallback

    rms = float(np.sqrt(np.mean(mono ** 2)))
    rms_dbfs = float(_safe_db(np.array([rms]))[0])

    peak = float(np.max(np.abs(mono)))
    peak_dbfs = float(_safe_db(np.array([peak]))[0])

    # Quick true-peak approximation via 4x oversampling
    up = signal.resample_poly(mono, 4, 1)
    true_peak = float(np.max(np.abs(up)))
    true_peak_dbfs = float(_safe_db(np.array([true_peak]))[0])

    crest_factor_db = float(true_peak_dbfs - rms_dbfs)

    # Silence ratio: fraction of frames below -60 dBFS
    frame_size = int(sr * 0.05)
    hop = frame_size // 2 or 1
    if frame_size <= 0 or frame_size > mono.shape[0]:
        silence_ratio = 0.0
    else:
        frames: List[np.ndarray] = []
        for start in range(0, mono.shape[0] - frame_size + 1, hop):
            frames.append(mono[start : start + frame_size])
        if not frames:
            silence_ratio = 0.0
        else:
            frame_rms = np.array([np.sqrt(np.mean(f ** 2)) for f in frames])
            frame_db = _safe_db(frame_rms)
            silence_ratio = float(np.mean(frame_db < -60.0))

    return LoudnessAnalysis(
        integrated_lufs=integrated_lufs,
        short_term_lufs=short_term_lufs,
        rms_dbfs=rms_dbfs,
        peak_dbfs=peak_dbfs,
        true_peak_dbfs=true_peak_dbfs,
        crest_factor_db=crest_factor_db,
        silence_ratio=silence_ratio,
        duration_sec=duration_sec,
    )


def analyze_spectral(audio: np.ndarray, sr: int, genre: GenreKey) -> SpectralBandEnergy:
    """Compute RMS per band and deltas vs genre target curve."""

    mono = _ensure_mono(audio)

    # Short FFT window to capture spectral balance; use Hann window
    n_fft = 4096
    hop = n_fft // 4
    if librosa is not None:  # pragma: no cover - optional
        spec = np.abs(librosa.stft(mono, n_fft=n_fft, hop_length=hop, window="hann"))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    else:
        freqs, _, Zxx = signal.stft(
            mono,
            fs=sr,
            nperseg=n_fft,
            noverlap=n_fft - hop,
            window="hann",
        )
        spec = np.abs(Zxx)

    # Power per band
    band_rms_db: Dict[str, float] = {}
    band_delta_db: Dict[str, float] = {}

    genre_targets = GENRE_TARGET_SPECTRAL.get(genre, GENRE_TARGET_SPECTRAL["generic"]).copy()

    # Use overall median band level as 0 dB reference
    overall_ref_levels: List[float] = []

    for name, f_lo, f_hi in SPECTRAL_BANDS:
        mask = (freqs >= f_lo) & (freqs < f_hi)
        if not np.any(mask):
            band_rms_db[name] = -120.0
            continue
        band_mag = spec[mask, :]
        band_power = np.mean(band_mag ** 2)
        band_db = float(_safe_db(np.array([np.sqrt(band_power)]))[0])
        band_rms_db[name] = band_db
        overall_ref_levels.append(band_db)

    if overall_ref_levels:
        ref = float(np.median(overall_ref_levels))
    else:
        ref = 0.0

    for name in band_rms_db.keys():
        raw = band_rms_db[name]
        target_offset = genre_targets.get(name, 0.0)
        # How far from a genre-weighted reference we are
        band_delta_db[name] = raw - (ref + target_offset)

    return SpectralBandEnergy(band_rms_db=band_rms_db, band_delta_db=band_delta_db)


def analyze_masking(
    beat_audio: np.ndarray,
    vocal_audio: np.ndarray,
    sr: int,
    freq_range: Tuple[float, float] = (200.0, 6000.0),
    threshold_db: float = 4.0,
) -> MaskingAnalysis:
    """Detect broad masking regions between beat and vocal.

    We look for bands where beat >> vocal within 200 Hz–6 kHz.
    """

    beat = _ensure_mono(beat_audio)
    vocal = _ensure_mono(vocal_audio)

    n_fft = 4096
    hop = n_fft // 4

    if librosa is not None:  # pragma: no cover - optional
        beat_spec = np.abs(librosa.stft(beat, n_fft=n_fft, hop_length=hop, window="hann"))
        vocal_spec = np.abs(librosa.stft(vocal, n_fft=n_fft, hop_length=hop, window="hann"))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    else:
        freqs, _, Bxx = signal.stft(
            beat,
            fs=sr,
            nperseg=n_fft,
            noverlap=n_fft - hop,
            window="hann",
        )
        _, _, Vxx = signal.stft(
            vocal,
            fs=sr,
            nperseg=n_fft,
            noverlap=n_fft - hop,
            window="hann",
        )
        beat_spec = np.abs(Bxx)
        vocal_spec = np.abs(Vxx)

    masking_bands: List[str] = []
    has_masking = False

    for name, f_lo, f_hi in SPECTRAL_BANDS:
        if f_hi < freq_range[0] or f_lo > freq_range[1]:
            continue
        mask = (freqs >= f_lo) & (freqs < f_hi)
        if not np.any(mask):
            continue
        b = beat_spec[mask, :]
        v = vocal_spec[mask, :]
        b_power = np.mean(b ** 2)
        v_power = np.mean(v ** 2) + 1e-12
        diff_db = float(_safe_db(np.array([np.sqrt(b_power / v_power)]))[0])
        if diff_db > threshold_db:
            masking_bands.append(name)
            has_masking = True

    return MaskingAnalysis(has_masking=has_masking, masking_bands=masking_bands)


def analyze_dynamics(audio: np.ndarray, sr: int) -> DynamicsAnalysis:
    """Characterize dynamics via RMS variance, peak density, transient sharpness."""

    mono = _ensure_mono(audio)

    # Short windows for RMS variance
    frame_size = int(sr * 0.05)
    hop = frame_size // 2 or 1
    if frame_size <= 0 or frame_size > mono.shape[0]:
        return DynamicsAnalysis(rms_variance=0.0, peak_density=0.0, transient_sharpness=0.0)

    rms_vals: List[float] = []
    peaks = 0
    total_frames = 0

    for start in range(0, mono.shape[0] - frame_size + 1, hop):
        frame = mono[start : start + frame_size]
        total_frames += 1
        rms = float(np.sqrt(np.mean(frame ** 2)))
        rms_vals.append(rms)
        # Simple peak detection: frame max above -6 dBFS
        peak = float(np.max(np.abs(frame)))
        if peak > 10 ** (-6.0 / 20.0):
            peaks += 1

    if not rms_vals or total_frames == 0:
        return DynamicsAnalysis(rms_variance=0.0, peak_density=0.0, transient_sharpness=0.0)

    rms_arr = np.array(rms_vals)
    rms_db = _safe_db(rms_arr)
    rms_variance = float(np.var(rms_db))

    peak_density = float(peaks / total_frames)

    # Transient sharpness: compare full-band energy vs a smoothed envelope
    nyquist_freq = 0.5 * sr
    normalized_cutoff = 50.0 / nyquist_freq
    # Clamp to valid range (0, 1) to avoid filter design failure
    normalized_cutoff = np.clip(normalized_cutoff, 0.001, 0.999)
    try:
        result = signal.butter(2, normalized_cutoff, btype="low", output="ba")
        if result is None:
            transient_sharpness = 0.0
        else:
            b, a = result
            env = signal.filtfilt(b, a, np.abs(mono))
            sharp = np.mean(np.maximum(0.0, mono - env))
            transient_sharpness = float(sharp)
    except Exception:
        transient_sharpness = 0.0

    return DynamicsAnalysis(
        rms_variance=rms_variance,
        peak_density=peak_density,
        transient_sharpness=transient_sharpness,
    )


def analyze_stereo(audio: np.ndarray, sr: int) -> StereoAnalysis:
    """Compute simple mid/side energy and phase correlation."""

    if audio.ndim == 1 or audio.shape[0] == 1:
        mono = _ensure_mono(audio)
        mid = mono
        side = np.zeros_like(mono)
    else:
        # assume (channels, samples) or (samples, channels)
        if audio.shape[0] == 2:
            L, R = audio[0], audio[1]
        else:
            L, R = audio[:, 0], audio[:, 1]
        mid = 0.5 * (L + R)
        side = 0.5 * (L - R)

    mid_energy = float(np.mean(mid ** 2))
    side_energy = float(np.mean(side ** 2))
    mid_side_ratio = float(side_energy / (mid_energy + 1e-12))

    # Phase correlation via Pearson across blocks
    frame_size = int(sr * 0.05)
    hop = frame_size // 2 or 1
    corrs: List[float] = []
    for start in range(0, min(len(mid), len(side)) - frame_size + 1, hop):
        m = mid[start : start + frame_size]
        s = side[start : start + frame_size]
        if m.std() < 1e-6 or s.std() < 1e-6:
            continue
        c = float(np.corrcoef(m, s)[0, 1])
        corrs.append(c)
    phase_correlation = float(np.median(corrs)) if corrs else 1.0

    return StereoAnalysis(
        mid_energy=mid_energy,
        side_energy=side_energy,
        mid_side_ratio=mid_side_ratio,
        phase_correlation=phase_correlation,
    )


def summarize_flags(loud: LoudnessAnalysis, spec: SpectralBandEnergy) -> Tuple[bool, bool]:
    """Heuristic muddy/harsh flags for quick UI summary."""

    muddy = spec.band_delta_db.get("low_mid", 0.0) > 2.0
    harsh = spec.band_delta_db.get("presence", 0.0) > 2.0 or spec.band_delta_db.get("air", 0.0) > 2.0

    # Very low crest factor and high presence can also signal harsh masters
    if loud.crest_factor_db < 6.0 and spec.band_delta_db.get("presence", 0.0) > 1.0:
        harsh = True

    return muddy, harsh


def decide_plugins_for_track(
    analysis: TrackAnalysis,
    role: TrackRole,
    genre: GenreKey,
    flow: FeatureFlow,
    target_lufs_override: Optional[float] = None,
    masking: Optional[MaskingAnalysis] = None,
) -> List[PluginConfig]:
    """Turn analysis into an ordered plugin suggestion chain.

    This is intentionally conservative and meant as a *starting point*.
    The Studio UI can expose these as editable AI-generated plugins.
    """

    plugins: List[PluginConfig] = []

    genre_targets = GENRE_TARGET_LUFS.get(genre, GENRE_TARGET_LUFS["generic"])
    target_lufs = target_lufs_override or genre_targets[role]

    loud = analysis.loudness
    spec = analysis.spectral
    dyn = analysis.dynamics

    # ------------------------------------------------------------------
    # 1. EQ based on spectral deltas and masking
    # ------------------------------------------------------------------
    eq_bands: List[Dict[str, Any]] = []
    for band_name, delta in spec.band_delta_db.items():
        if delta > 1.5:  # too hot
            gain = -min(delta, 6.0)
            eq_bands.append({"type": "bell", "band": band_name, "gain_db": gain, "q": 1.3})
        elif delta < -1.5:  # too weak
            gain = min(-delta, 3.0)
            eq_bands.append({"type": "bell", "band": band_name, "gain_db": gain, "q": 1.0})

    # Extra midrange shaping for vocals, low-mid cleanup for beat
    if role.startswith("vocal"):
        if analysis.muddy:
            eq_bands.append({"type": "bell", "band": "low_mid", "gain_db": -3.0, "q": 1.4})
        if analysis.harsh:
            eq_bands.append({"type": "bell", "band": "presence", "gain_db": -2.5, "q": 3.0})
    elif role == "beat" and analysis.muddy:
        eq_bands.append({"type": "bell", "band": "low_mid", "gain_db": -2.5, "q": 1.2})

    # Masking-driven dynamic EQ suggestions on beat
    if masking and masking.has_masking and role == "beat":
        for band_name in masking.masking_bands:
            eq_bands.append(
                {
                    "type": "dynamic_dip",
                    "band": band_name,
                    "gain_db": -3.0,
                    "q": 2.0,
                    "sidechain": "vocal_bus",
                }
            )

    if eq_bands and flow != "mastering_only":  # cleanup/mix-focused
        plugins.append(
            PluginConfig(
                plugin="EQ",
                params={
                    "bands": eq_bands,
                    "mode": "surgical" if flow == "audio_cleanup" else "musical",
                },
            )
        )

    # ------------------------------------------------------------------
    # 2. De-Esser for vocals if presence/air are hot
    # ------------------------------------------------------------------
    if role.startswith("vocal") and (spec.band_delta_db.get("presence", 0.0) > 1.0 or spec.band_delta_db.get("air", 0.0) > 1.0):
        plugins.append(
            PluginConfig(
                plugin="De-esser",
                params={
                    "freq_hz": 6500.0,
                    "ratio": 3.0,
                    "threshold_db": -20.0,
                    "mix": 0.7,
                },
            )
        )

    # ------------------------------------------------------------------
    # 3. Compression based on dynamics
    # ------------------------------------------------------------------
    comp_needed = dyn.rms_variance > 4.0 or dyn.peak_density > 0.3
    if comp_needed and flow in {"audio_cleanup", "mixing_only", "mix_master"}:
        ratio = 2.5 if dyn.rms_variance < 8.0 else 3.5
        attack_ms = 25.0 if dyn.transient_sharpness > 0.02 else 10.0
        release_ms = 120.0
        mix = 0.8

        # Flat/boring sources: use more parallel style
        if dyn.rms_variance < 2.0:
            mix = 0.4

        plugins.append(
            PluginConfig(
                plugin="Compressor",
                params={
                    "ratio": ratio,
                    "attack_ms": attack_ms,
                    "release_ms": release_ms,
                    "threshold_db": -18.0,
                    "mix": mix,
                },
            )
        )

    # ------------------------------------------------------------------
    # 4. Saturation / tone density
    # ------------------------------------------------------------------
    # Compare perceived loudness vs peak headroom
    headroom_db = loud.true_peak_dbfs

    add_saturation = False
    sat_style = "tape"

    if role == "beat":
        if loud.integrated_lufs < target_lufs - 3.0:
            add_saturation = True
            sat_style = "bus"
    else:  # vocals
        if loud.integrated_lufs < target_lufs - 2.0:
            add_saturation = True
            sat_style = "vocal"

    if add_saturation and flow in {"mixing_only", "mix_master"}:
        drive = 5.0
        if analysis.harsh:
            sat_mode = "even_harmonics"
            drive = 3.0
        else:
            sat_mode = "tape" if sat_style == "bus" else "tube"
        plugins.append(
            PluginConfig(
                plugin="Saturation",
                params={
                    "mode": sat_mode,
                    "drive_percent": drive,
                    "mix": 0.3 if role == "beat" else 0.4,
                },
            )
        )

    # ------------------------------------------------------------------
    # 5. Stereo / imaging decisions
    # ------------------------------------------------------------------
    if role == "beat" and flow in {"mixing_only", "mix_master", "mastering_only"}:
        plugins.append(
            PluginConfig(
                plugin="Stereo Imager",
                params={
                    "low_mono_hz": 120.0,
                    "low_width": 0.95,
                    "mid_width": 1.05,
                    "high_width": 1.15,
                },
            )
        )

    # ------------------------------------------------------------------
    # 6. Limiting / output stage
    # ------------------------------------------------------------------
    # Only in mastering-style flows OR for beat-only with mild limiting.
    if flow in {"mix_master", "mastering_only", "beat_only"} and role in {"beat", "vocal_lead"}:
        target_out = target_lufs
        ceiling = -1.0
        if flow == "mastering_only" or flow == "mix_master":
            ceiling = -0.8
        plugins.append(
            PluginConfig(
                plugin="Limiter",
                params={
                    "ceiling_db": ceiling,
                    "target_lufs": target_out,
                    "release_ms": 120.0,
                },
            )
        )

    return plugins


def analyze_track_and_suggest_chain(
    audio: np.ndarray,
    sr: int,
    role: TrackRole,
    genre: GenreKey = "generic",
    flow: FeatureFlow = "mixing_only",
    beat_audio_for_masking: Optional[np.ndarray] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Full pipeline for a single track.

    Returns a FastAPI-serializable dict with analysis + plugin_chain.
    """

    def _step(msg: str) -> None:
        if progress_cb is not None:
            try:
                progress_cb(msg)
            except Exception:  # pragma: no cover - defensive
                pass

    _step("loudness")
    loud = analyze_loudness(audio, sr)

    _step("spectral")
    spec = analyze_spectral(audio, sr, genre)

    _step("dynamics")
    dyn = analyze_dynamics(audio, sr)

    _step("stereo")
    stereo = analyze_stereo(audio, sr)

    masking: Optional[MaskingAnalysis] = None
    if beat_audio_for_masking is not None and role.startswith("vocal"):
        _step("masking")
        masking = analyze_masking(beat_audio_for_masking, audio, sr)

    muddy, harsh = summarize_flags(loud, spec)

    track_analysis = TrackAnalysis(
        loudness=loud,
        spectral=spec,
        dynamics=dyn,
        stereo=stereo,
        masking=masking,
        muddy=muddy,
        harsh=harsh,
    )

    _step("decision")
    plugins = decide_plugins_for_track(track_analysis, role=role, genre=genre, flow=flow, masking=masking)

    # Flatten dataclasses into plain dicts for FastAPI JSON responses
    analysis_dict: Dict[str, Any] = {
        "loudness": asdict(loud),
        "spectral": {
            "band_rms_db": spec.band_rms_db,
            "band_delta_db": spec.band_delta_db,
        },
        "dynamics": asdict(dyn),
        "stereo": asdict(stereo),
        "masking": asdict(masking) if masking is not None else None,
        "muddy": muddy,
        "harsh": harsh,
    }

    return {
        "analysis": analysis_dict,
        "plugin_chain": [asdict(p) for p in plugins],
    }
