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
from typing import Dict, List, Literal, Optional


PresetKind = Literal["vocal", "mix", "master"]
ChainType = Literal["generic_chain", "genre_vocal"]


@dataclass(frozen=True)
class PresetMeta:
    """Describes a user-facing preset.

    key:        Internal preset key (used in PRESETS / engine).
    name:       Human-readable label for UI.
    kind:       High-level role: vocal | mix | master.
    chain_type: Whether it uses the generic TRACK_CHAINS + PRESETS
                system or the dedicated genre-specific vocal chains.
    description:Short marketing/UX description.
    genre:      Optional genre tag (for discovery/filters).
    target_lufs:Optional loudness target; mostly for masters.
   """

    key: str
    name: str
    kind: PresetKind
    chain_type: ChainType
    description: str = ""
    genre: Optional[str] = None
    target_lufs: Optional[float] = None


# Core vocal presets that map directly onto app.presets.PRESETS
VOCAL_PRESETS: List[PresetMeta] = [
    PresetMeta(
        key="clean_vocal",
        name="Clean Lead Vocal",
        kind="vocal",
        chain_type="generic_chain",
        description="Modern, clean lead vocal with gentle de-essing and compression.",
        genre="any",
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
    # Background and adlib variants for key genres. These map directly
    # to the *_bg / *_adlib entries in VOCAL_GENRE_PROCESSORS in
    # app.engine so that each role (lead, BG, adlib) has its own
    # explicitly named preset instead of being hidden inside a single
    # generic preset.
    PresetMeta(
        key="trap_dancehall_bg",
        name="Trap Dancehall BG Vox",
        kind="vocal",
        chain_type="genre_vocal",
        description="Background vocal stack tuned for trap dancehall mixes.",
        genre="trap_dancehall",
    ),
    PresetMeta(
        key="trap_dancehall_adlib",
        name="Trap Dancehall Adlibs",
        kind="vocal",
        chain_type="genre_vocal",
        description="Hyped adlib chain for trap dancehall vocals.",
        genre="trap_dancehall",
    ),
    PresetMeta(
        key="hiphop_bg",
        name="Hip-Hop BG Vox",
        kind="vocal",
        chain_type="genre_vocal",
        description="Glue-style backing vocals for hip-hop.",
        genre="hiphop",
    ),
    PresetMeta(
        key="hiphop_adlib",
        name="Hip-Hop Adlibs",
        kind="vocal",
        chain_type="genre_vocal",
        description="Brighter, more aggressive hip-hop adlib chain.",
        genre="hiphop",
    ),
    PresetMeta(
        key="afrobeat_bg",
        name="Afrobeat BG Vox",
        kind="vocal",
        chain_type="genre_vocal",
        description="Smooth background stacks for afrobeat.",
        genre="afrobeat",
    ),
    PresetMeta(
        key="afrobeat_adlib",
        name="Afrobeat Adlibs",
        kind="vocal",
        chain_type="genre_vocal",
        description="Wide, characterful afrobeat adlibs.",
        genre="afrobeat",
    ),
    PresetMeta(
        key="rnb_bg",
        name="R&B BG Vox",
        kind="vocal",
        chain_type="genre_vocal",
        description="Soft, wide background vocals for R&B.",
        genre="rnb",
    ),
    PresetMeta(
        key="rnb_adlib",
        name="R&B Adlibs",
        kind="vocal",
        chain_type="genre_vocal",
        description="Expressive R&B adlib chain with extra space.",
        genre="rnb",
    ),
    PresetMeta(
        key="reggae_bg",
        name="Reggae BG Vox",
        kind="vocal",
        chain_type="genre_vocal",
        description="Warm, supporting background vocals for reggae.",
        genre="reggae",
    ),
    PresetMeta(
        key="reggae_adlib",
        name="Reggae Adlibs",
        kind="vocal",
        chain_type="genre_vocal",
        description="Echo-friendly reggae adlib chain.",
        genre="reggae",
    ),
    PresetMeta(
        key="rap_bg",
        name="Rap BG Vox",
        kind="vocal",
        chain_type="genre_vocal",
        description="Supporting rap backgrounds that sit under the lead.",
        genre="rap",
    ),
    PresetMeta(
        key="rap_adlib",
        name="Rap Adlibs",
        kind="vocal",
        chain_type="genre_vocal",
        description="Punchy, hyped rap adlib chain.",
        genre="rap",
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
