"""High-level preset registry for MIXSMVRT DSP.

This module provides structured metadata for all vocal, mix-bus and
mastering presets that the DSP engine knows about. It is intentionally
thin and sits on top of the lower-level PRESETS dict and
VOCAL_GENRE_PROCESSORS mapping in engine.py.

The goals are:
- Give the studio/frontend a single source of truth for available presets
  (ids, names, categories, descriptions, targets).
- Make it easy for the backend to reason about presets when recording
  analytics (preset_usage, processing_jobs.preset_key).
- Keep backwards compatibility with existing preset keys used in
  app.presets.PRESETS and the genre-specific vocal presets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, TypedDict


PresetKind = Literal["vocal", "mix", "master"]
ChainType = Literal["generic_chain", "genre_vocal"]


class EqRanges(TypedDict, total=False):
    hpf_hz: tuple[float, float]
    presence_hz: tuple[float, float]
    presence_boost_db: tuple[float, float]
    air_hz: tuple[float, float]
    air_boost_db: tuple[float, float]
    sub_cut_hz: tuple[float, float]
    high_shelf_hz: tuple[float, float]
    high_shelf_boost_db: tuple[float, float]


class CompressionRanges(TypedDict, total=False):
    ratio: tuple[float, float]
    attack_ms: tuple[float, float]
    release_ms: tuple[float, float]
    gain_reduction_db: tuple[float, float]


class SaturationRanges(TypedDict, total=False):
    drive_percent: tuple[float, float]


class DeesserRanges(TypedDict, total=False):
    freq_hz: tuple[float, float]
    reduction_db: tuple[float, float]


class BusRanges(TypedDict, total=False):
    ratio: tuple[float, float]
    attack_ms: tuple[float, float]
    release_ms: tuple[float, float]
    gain_reduction_db: tuple[float, float]


class LimiterRanges(TypedDict, total=False):
    true_peak_db: tuple[float, float]
    target_lufs: tuple[float, float]


class DspRanges(TypedDict, total=False):
    eq: EqRanges
    compression: CompressionRanges
    saturation: SaturationRanges
    deesser: DeesserRanges
    bus: BusRanges
    limiter: LimiterRanges


@dataclass(frozen=True)
class PresetMeta:
    """Describes a user-facing preset and safe DSP ranges.

    key:         Internal preset key (used in PRESETS / engine).
    name:        Human-readable label for UI.
    kind:        High-level role: vocal | mix | master.
    chain_type:  Whether it uses the generic TRACK_CHAINS + PRESETS
                 system or the dedicated genre-specific vocal chains.
    description: Short marketing/UX description.
    genre:       Optional genre tag (for discovery/filters).
    target_lufs: Optional loudness target; mostly for masters.
    dsp_ranges:  Optional safe parameter ranges for AI/macro control.
    """

    key: str
    name: str
    kind: PresetKind
    chain_type: ChainType
    description: str = ""
    genre: Optional[str] = None
    target_lufs: Optional[float] = None
    dsp_ranges: Optional[DspRanges] = None


# Core vocal presets that map directly onto app.presets.PRESETS
VOCAL_PRESETS: List[PresetMeta] = [
    PresetMeta(
        key="clean_vocal",
        name="Clean Lead Vocal",
        kind="vocal",
        chain_type="generic_chain",
        description="Modern, clean lead vocal with gentle de-essing and compression.",
        genre="any",
        dsp_ranges={
            "eq": {
                "hpf_hz": (70.0, 90.0),
                "presence_hz": (2000.0, 4000.0),
                "presence_boost_db": (1.0, 3.0),
                "air_hz": (10000.0, 14000.0),
                "air_boost_db": (1.0, 2.0),
            },
            "compression": {
                "ratio": (2.0, 3.0),
                "attack_ms": (15.0, 30.0),
                "release_ms": (60.0, 120.0),
                "gain_reduction_db": (-5.0, -3.0),
            },
            "saturation": {
                "drive_percent": (3.0, 8.0),
            },
            "deesser": {
                "freq_hz": (6000.0, 8000.0),
                "reduction_db": (-4.0, -2.0),
            },
            "bus": {},
            "limiter": {},
        },
    ),
    PresetMeta(
        key="bg_vocal_glue",
        name="BG Vocal Glue",
        kind="vocal",
        chain_type="generic_chain",
        description="Stacked backing vocals with soft glue and extra width.",
        genre="any",
    ),
    PresetMeta(
        key="adlib_hype",
        name="Adlib Hype",
        kind="vocal",
        chain_type="generic_chain",
        description="Bright, wide, aggressive chain for hype adlibs.",
        genre="any",
    ),
    PresetMeta(
        key="aggressive_rap",
        name="Aggressive Rap Vocal",
        kind="vocal",
        chain_type="generic_chain",
        description="Forward, punchy rap vocal with more compression and bite.",
        genre="rap/hiphop",
        dsp_ranges={
            "eq": {
                "hpf_hz": (80.0, 100.0),
                "presence_hz": (3000.0, 5000.0),
                "presence_boost_db": (2.0, 4.0),
            },
            "compression": {
                "ratio": (3.0, 4.0),
                "attack_ms": (5.0, 15.0),
                "release_ms": (40.0, 80.0),
                "gain_reduction_db": (-7.0, -5.0),
            },
            "saturation": {
                "drive_percent": (6.0, 12.0),
            },
            "deesser": {
                "freq_hz": (6000.0, 8000.0),
                "reduction_db": (-5.0, -3.0),
            },
            "bus": {},
            "limiter": {},
        },
    ),
]


# Genre-specific vocal chains exposed via VOCAL_GENRE_PROCESSORS in engine.py.
# These rely on dedicated pedalboard-based processing in app.vocal_presets.*
GENRE_VOCAL_PRESETS: List[PresetMeta] = [
    PresetMeta(
        key="dancehall",
        name="Dancehall Vocal",
        kind="vocal",
        chain_type="genre_vocal",
        description="Genre-tuned vocal chain for modern dancehall leads.",
        genre="dancehall",
    ),
    PresetMeta(
        key="trap_dancehall",
        name="Trap Dancehall Vocal",
        kind="vocal",
        chain_type="genre_vocal",
        description="Hybrid trap/dancehall vocal chain with extra bite.",
        genre="trap_dancehall",
    ),
    PresetMeta(
        key="afrobeat",
        name="Afrobeat Vocal",
        kind="vocal",
        chain_type="genre_vocal",
        description="Smooth but present vocal chain tuned for afrobeat.",
        genre="afrobeat",
    ),
    PresetMeta(
        key="reggae",
        name="Reggae Vocal",
        kind="vocal",
        chain_type="genre_vocal",
        description="Warm, rounded vocal chain for roots and modern reggae.",
        genre="reggae",
    ),
    PresetMeta(
        key="hiphop",
        name="Hip-Hop Vocal",
        kind="vocal",
        chain_type="genre_vocal",
        description="Punchy hip-hop vocal chain with controlled highs.",
        genre="hiphop",
    ),
    PresetMeta(
        key="rap",
        name="Rap Vocal",
        kind="vocal",
        chain_type="genre_vocal",
        description="Focused rap chain with extra articulation.",
        genre="rap",
    ),
    PresetMeta(
        key="rnb",
        name="R&B Vocal",
        kind="vocal",
        chain_type="genre_vocal",
        description="Silky, smoothed-out R&B vocal chain.",
        genre="rnb",
    ),
    # Simple aliases that can be mapped in the frontend to these base genres
    PresetMeta(
        key="reggaeton",
        name="Reggaeton Vocal (alias)",
        kind="vocal",
        chain_type="genre_vocal",
        description="Reggaeton vocal chain (routes to afrobeat/reggaeton tuning).",
        genre="reggaeton",
    ),
]


# Mix-bus and mastering presets. These are intended to map to PRESETS keys
# as we expand the dictionary; for now we surface streaming_master and leave
# room for future variants.
MASTERING_PRESETS: List[PresetMeta] = [
    PresetMeta(
        key="streaming_master",
        name="Streaming Master -14 LUFS",
        kind="master",
        chain_type="generic_chain",
        description="Balanced streaming master around -14 LUFS.",
        genre="any",
        target_lufs=-14.0,
        dsp_ranges={
            "eq": {
                "sub_cut_hz": (25.0, 30.0),
                "high_shelf_hz": (10000.0, 12000.0),
                "high_shelf_boost_db": (0.5, 1.5),
            },
            "compression": {},
            "saturation": {},
            "deesser": {},
            "bus": {
                "ratio": (1.5, 2.0),
                "attack_ms": (20.0, 40.0),
                "release_ms": (80.0, 160.0),
                "gain_reduction_db": (-2.0, -1.0),
            },
            "limiter": {
                "true_peak_db": (-1.2, -1.0),
                "target_lufs": (-14.0, -13.0),
            },
        },
    ),
]


# Simple helper maps for quick lookup.
ALL_PRESETS: List[PresetMeta] = VOCAL_PRESETS + GENRE_VOCAL_PRESETS + MASTERING_PRESETS
PRESET_BY_KEY: Dict[str, PresetMeta] = {p.key: p for p in ALL_PRESETS}


def list_presets(kind: Optional[PresetKind] = None) -> List[PresetMeta]:
    """Return all presets, optionally filtered by kind."""

    if kind is None:
        return list(ALL_PRESETS)
    return [p for p in ALL_PRESETS if p.kind == kind]


def get_preset(key: str) -> Optional[PresetMeta]:
    """Look up preset metadata by key.

    Returns None if the preset is not registered. This keeps the registry
    non-invasive for callers that still rely on raw PRESETS keys.
    """

    return PRESET_BY_KEY.get(key)
