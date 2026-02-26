"""Contextual (relative) role classification for multi-track sessions.

This is intentionally rule-based and lightweight.
"""

from __future__ import annotations

from typing import Dict


RoleKey = str


def _argmax(track_feature_map: Dict[str, dict], score_fn) -> str | None:
    best_id = None
    best_score = None
    for track_id, feats in track_feature_map.items():
        try:
            score = float(score_fn(feats))
        except Exception:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_id = track_id
    return best_id


def classify_roles(track_feature_map: Dict[str, dict]) -> Dict[str, RoleKey]:
    """Assign contextual roles by comparing tracks relative to each other.

    Input:
        {"track_id": feature_dict}

    Returns:
        {"track_id": "lead_vocal"}
    """

    if not track_feature_map:
        return {}

    remaining = dict(track_feature_map)
    role_map: Dict[str, RoleKey] = {}

    # Lead vocal: highest mid energy + harmonic content
    lead = _argmax(
        remaining,
        lambda f: float(f.get("mid_energy_ratio", 0.0)) + float(f.get("harmonic_ratio", 0.0)),
    )
    if lead is not None:
        role_map[lead] = "lead_vocal"
        remaining.pop(lead, None)

    # 808 / bass: highest low-frequency energy
    bass = _argmax(remaining, lambda f: float(f.get("low_energy_ratio", 0.0)))
    if bass is not None:
        role_map[bass] = "808"
        remaining.pop(bass, None)

    # Drums: highest transient density
    drums = _argmax(remaining, lambda f: float(f.get("transient_density", 0.0)))
    if drums is not None:
        role_map[drums] = "drums"
        remaining.pop(drums, None)

    # Background vocal: next most vocal-like (if any tracks left)
    bg = _argmax(
        remaining,
        lambda f: 0.8 * float(f.get("mid_energy_ratio", 0.0)) + 0.2 * float(f.get("harmonic_ratio", 0.0)),
    )
    if bg is not None:
        role_map[bg] = "bg_vocal"
        remaining.pop(bg, None)

    # Melody: wide stereo + sustained (lower transient density + lower crest)
    melody = _argmax(
        remaining,
        lambda f: float(f.get("stereo_width", 0.0))
        - 0.15 * float(f.get("transient_density", 0.0))
        - 0.05 * float(f.get("crest_factor", 0.0)),
    )
    if melody is not None:
        role_map[melody] = "melody"
        remaining.pop(melody, None)

    # FX: whatever remains
    for track_id in remaining.keys():
        role_map[track_id] = "fx"

    return role_map
