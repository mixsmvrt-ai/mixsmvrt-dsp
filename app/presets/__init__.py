from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

FlowType = Literal["mixing_only", "mixing_mastering", "mix_master"]

GENRE_KEY = Literal[
    "dancehall",
    "trap_dancehall",
    "afrobeats",
    "hip_hop",
    "rap",
    "rnb",
    "reggae",
]


@dataclass(frozen=True)
class VocalPresetProfile:
    preset_name: str
    ui_subtitle: str
    eq_profile: Dict[str, Any]
    compression_profile: Dict[str, Any]
    dynamic_eq_profile: Dict[str, Any]
    deesser_profile: Dict[str, Any]
    saturation_profile: Dict[str, Any]
    reverb_profile: Dict[str, Any]
    delay_profile: Dict[str, Any]
    bus_processing: Dict[str, Any]
    master_processing: Optional[Dict[str, Any]] = None


# Import all genre/flow modules so we can index presets centrally.
from .mixing_only import (  # noqa: E402
    dancehall as mo_dancehall,
    trap_dancehall as mo_trap_dancehall,
    afrobeats as mo_afrobeats,
    hip_hop as mo_hip_hop,
    rap as mo_rap,
    rnb as mo_rnb,
    reggae as mo_reggae,
)
from .mixing_mastering import (  # noqa: E402
    dancehall as mm_dancehall,
    trap_dancehall as mm_trap_dancehall,
    afrobeats as mm_afrobeats,
    hip_hop as mm_hip_hop,
    rap as mm_rap,
    rnb as mm_rnb,
    reggae as mm_reggae,
)


_PRESET_SOURCES: List[Tuple[str, str, Any]] = [
    ("mixing_only", "dancehall", mo_dancehall),
    ("mixing_only", "trap_dancehall", mo_trap_dancehall),
    ("mixing_only", "afrobeats", mo_afrobeats),
    ("mixing_only", "hip_hop", mo_hip_hop),
    ("mixing_only", "rap", mo_rap),
    ("mixing_only", "rnb", mo_rnb),
    ("mixing_only", "reggae", mo_reggae),
    ("mixing_mastering", "dancehall", mm_dancehall),
    ("mixing_mastering", "trap_dancehall", mm_trap_dancehall),
    ("mixing_mastering", "afrobeats", mm_afrobeats),
    ("mixing_mastering", "hip_hop", mm_hip_hop),
    ("mixing_mastering", "rap", mm_rap),
    ("mixing_mastering", "rnb", mm_rnb),
    ("mixing_mastering", "reggae", mm_reggae),
]


# (flow_type, genre) -> preset_name -> VocalPresetProfile
_PRESET_INDEX: Dict[Tuple[str, str], Dict[str, VocalPresetProfile]] = {}


def _normalise_flow_type(flow_type: str) -> str:
    ft = (flow_type or "").strip().lower()
    if ft in {"mix_master", "mixing_mastering"}:
        return "mixing_mastering"
    if ft in {"mix_only", "mixing_only"}:
        return "mixing_only"
    return ft or "mixing_mastering"


def _normalise_genre(genre: str) -> str:
    g = (genre or "").strip().lower()
    if g in {"afrobeat", "afrobeats"}:
        return "afrobeats"
    if g in {"hiphop", "hip-hop"}:
        return "hip_hop"
    return g


def _build_index() -> None:
    for flow_type, genre, module in _PRESET_SOURCES:
        key = (_normalise_flow_type(flow_type), _normalise_genre(genre))
        presets_for_key = _PRESET_INDEX.setdefault(key, {})
        for raw in getattr(module, "PRESETS", []):
            name = str(raw.get("preset_name"))
            if not name:
                continue
            profile = VocalPresetProfile(
                preset_name=name,
                ui_subtitle=str(raw.get("ui_subtitle", "")),
                eq_profile=dict(raw.get("eq_profile", {})),
                compression_profile=dict(raw.get("compression_profile", {})),
                dynamic_eq_profile=dict(raw.get("dynamic_eq_profile", {})),
                deesser_profile=dict(raw.get("deesser_profile", {})),
                saturation_profile=dict(raw.get("saturation_profile", {})),
                reverb_profile=dict(raw.get("reverb_profile", {})),
                delay_profile=dict(raw.get("delay_profile", {})),
                bus_processing=dict(raw.get("bus_processing", {})),
                master_processing=(
                    dict(raw["master_processing"]) if "master_processing" in raw else None
                ),
            )
            presets_for_key[name] = _clamp_preset(profile)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _clamp_preset(profile: VocalPresetProfile) -> VocalPresetProfile:
    eq = dict(profile.eq_profile)
    for key in ["low_shelf_db", "low_mid_db", "presence_db", "high_shelf_db"]:
        if key in eq:
            eq[key] = float(_clamp(float(eq[key]), -5.0, 3.0))

    comp = dict(profile.compression_profile)
    if "max_gr_db" in comp:
        comp["max_gr_db"] = float(_clamp(float(comp["max_gr_db"]), 0.0, 4.0))

    dyn = dict(profile.dynamic_eq_profile)
    for key in ["harsh_cut_db", "mud_cut_db", "sibilance_cut_db"]:
        if key in dyn:
            dyn[key] = float(_clamp(float(dyn[key]), 0.0, 4.0))

    deess = dict(profile.deesser_profile)
    if "max_reduction_db" in deess:
        deess["max_reduction_db"] = float(_clamp(float(deess["max_reduction_db"]), 0.0, 4.0))

    sat = dict(profile.saturation_profile)
    if "drive_amount" in sat:
        sat["drive_amount"] = float(_clamp(float(sat["drive_amount"]), 0.0, 0.03))

    rev = dict(profile.reverb_profile)
    if "wet_pct" in rev:
        rev["wet_pct"] = float(_clamp(float(rev["wet_pct"]), 0.0, 25.0))

    delay = dict(profile.delay_profile)
    if "feedback_pct" in delay:
        delay["feedback_pct"] = float(_clamp(float(delay["feedback_pct"]), 0.0, 30.0))
    if "wet_pct" in delay:
        delay["wet_pct"] = float(_clamp(float(delay["wet_pct"]), 0.0, 25.0))

    bus = dict(profile.bus_processing)
    bus_comp = dict(bus.get("bus_comp", {}))
    if "max_gr_db" in bus_comp:
        bus_comp["max_gr_db"] = float(_clamp(float(bus_comp["max_gr_db"]), 0.0, 4.0))
    bus["bus_comp"] = bus_comp

    master = dict(profile.master_processing or {})
    limiter = dict(master.get("limiter", {}))
    if "ceiling_db" in limiter:
        limiter["ceiling_db"] = float(_clamp(float(limiter["ceiling_db"]), -10.0, -0.5))
    if "max_gr_db" in limiter:
        limiter["max_gr_db"] = float(_clamp(float(limiter["max_gr_db"]), 0.0, 4.0))
    master["limiter"] = limiter

    return VocalPresetProfile(
        preset_name=profile.preset_name,
        ui_subtitle=profile.ui_subtitle,
        eq_profile=eq,
        compression_profile=comp,
        dynamic_eq_profile=dyn,
        deesser_profile=deess,
        saturation_profile=sat,
        reverb_profile=rev,
        delay_profile=delay,
        bus_processing=bus,
        master_processing=master if master else None,
    )


_build_index()


def list_presets_for_ui(flow_type: FlowType, genre: str) -> List[Dict[str, str]]:
    ft = _normalise_flow_type(flow_type)
    g = _normalise_genre(genre)
    key = (ft, g)
    results: List[Dict[str, str]] = []
    for name, profile in _PRESET_INDEX.get(key, {}).items():
        results.append(
            {
                "genre": g,
                "flow_type": ft,
                "preset_name": name,
                "ui_subtitle": profile.ui_subtitle,
            }
        )
    return results


def list_all_presets_for_ui() -> List[Dict[str, str]]:
    """Return a flat list of all vocal presets for UI consumption.

    Each entry includes ``flow_type``, ``genre``, ``preset_name`` and
    ``ui_subtitle`` so callers can build user-facing labels and stable
    identifiers without depending on internal module layout.
    """

    results: List[Dict[str, str]] = []

    for (flow_type, genre), presets_for_key in _PRESET_INDEX.items():
        for name, profile in presets_for_key.items():
            results.append(
                {
                    "genre": genre,
                    "flow_type": flow_type,
                    "preset_name": name,
                    "ui_subtitle": profile.ui_subtitle,
                }
            )

    return results


def get_preset(flow_type: FlowType, genre: str, preset_name: str) -> Optional[VocalPresetProfile]:
    ft = _normalise_flow_type(flow_type)
    g = _normalise_genre(genre)
    key = (ft, g)
    presets_for_key = _PRESET_INDEX.get(key)
    if not presets_for_key:
        return None
    return presets_for_key.get(preset_name)


def apply_adaptive_tweaks(
    profile: VocalPresetProfile,
    *,
    analysis: Dict[str, Any],
    tempo_bpm: Optional[float] = None,
    already_distorted: Optional[bool] = None,
    median_pitch_hz: Optional[float] = None,
) -> VocalPresetProfile:
    """Return a copy of ``profile`` with small adaptive tweaks.

    - Slightly adjust HPF / presence and de-esser frequency based on pitch.
    - Increase harshness control if sibilance band is hot.
    - Reduce reverb/delay wet and feedback at high tempos.
    - Reduce saturation when the vocal already looks heavily compressed.
    All adjustments are clamped by _clamp_preset so they cannot exceed
    the global DSP safety limits.
    """

    eq = dict(profile.eq_profile)
    deess = dict(profile.deesser_profile)
    dyn = dict(profile.dynamic_eq_profile)
    sat = dict(profile.saturation_profile)
    rev = dict(profile.reverb_profile)
    delay = dict(profile.delay_profile)

    bands = analysis.get("bands") or {}
    high_db = float(bands.get("high_db", 0.0))
    sibilance_db = float(bands.get("sibilance_db", high_db))
    harshness = sibilance_db - high_db

    if harshness > 3.0:
        dyn["sibilance_cut_db"] = float(min(4.0, dyn.get("sibilance_cut_db", 2.0) + 0.5))
        deess["max_reduction_db"] = float(min(4.0, deess.get("max_reduction_db", 2.5) + 0.5))
        eq["presence_db"] = float(max(-5.0, eq.get("presence_db", 0.0) - 0.5))

    if median_pitch_hz is None:
        median_pitch_hz = float(analysis.get("median_f0_hz", 0.0) or 0.0)

    if median_pitch_hz > 220.0:
        # Higher voices: slightly higher HPF and de-esser frequency.
        eq["highpass_hz"] = float(min(140.0, eq.get("highpass_hz", 100.0) + 10.0))
        deess["freq_hz"] = float(max(6000.0, deess.get("freq_hz", 7500.0) + 200.0))
    elif 0.0 < median_pitch_hz < 140.0:
        # Lower voices: keep more low end and de-ess a bit lower.
        eq["highpass_hz"] = float(max(60.0, eq.get("highpass_hz", 100.0) - 10.0))
        deess["freq_hz"] = float(min(9000.0, deess.get("freq_hz", 7500.0) - 200.0))

    if tempo_bpm is not None and tempo_bpm > 130.0:
        rev["wet_pct"] = float(rev.get("wet_pct", 18.0) * 0.7)
        delay["feedback_pct"] = float(delay.get("feedback_pct", 20.0) * 0.7)

    if already_distorted is None:
        dyn_range = float(analysis.get("dynamic_range_db", 10.0))
        rms_db = float(analysis.get("rms_db", -18.0))
        already_distorted = dyn_range < 6.0 and rms_db > -10.0

    if already_distorted:
        sat["drive_amount"] = float(sat.get("drive_amount", 0.015) * 0.5)

    tweaked = VocalPresetProfile(
        preset_name=profile.preset_name,
        ui_subtitle=profile.ui_subtitle,
        eq_profile=eq,
        compression_profile=dict(profile.compression_profile),
        dynamic_eq_profile=dyn,
        deesser_profile=deess,
        saturation_profile=sat,
        reverb_profile=rev,
        delay_profile=delay,
        bus_processing=dict(profile.bus_processing),
        master_processing=dict(profile.master_processing or {}),
    )

    return _clamp_preset(tweaked)
