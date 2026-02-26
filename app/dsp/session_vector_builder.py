"""Build a fixed-length session feature vector."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


ROLE_ORDER: List[str] = ["lead_vocal", "bg_vocal", "808", "drums", "melody", "fx"]

GENRE_ORDER: List[str] = [
    "dancehall",
    "trap",
    "afrobeats",
    "reggaeton",
    "hiphop",
    "pop",
    "other",
]


FEATURE_KEYS: List[str] = [
    "peak_db",
    "rms_db",
    "crest_factor",
    "dynamic_range",
    "spectral_centroid_mean",
    "spectral_rolloff_mean",
    "low_energy_ratio",
    "mid_energy_ratio",
    "high_energy_ratio",
    "stereo_width",
    "transient_density",
    "harmonic_ratio",
]


def _role_to_track_id(role_map: Dict[str, str]) -> Dict[str, str]:
    """Invert {track_id: role} to {role: track_id}, keeping first match."""
    out: Dict[str, str] = {}
    for track_id, role in role_map.items():
        if role not in out:
            out[role] = track_id
    return out


def _genre_embedding(genre: str) -> np.ndarray:
    g = (genre or "other").lower().strip()
    # lightweight normalization
    if g in {"afrobeat", "afrobeats"}:
        g = "afrobeats"
    elif g in {"hiphop", "hip-hop"}:
        g = "hiphop"
    elif g in {"reggaeton", "latin"}:
        g = "reggaeton"
    elif g in {"dancehall"}:
        g = "dancehall"
    elif g in {"trap"}:
        g = "trap"
    elif g in {"pop"}:
        g = "pop"
    else:
        g = "other"

    vec = np.zeros((len(GENRE_ORDER),), dtype=np.float32)
    try:
        idx = GENRE_ORDER.index(g)
    except ValueError:
        idx = GENRE_ORDER.index("other")
    vec[idx] = 1.0
    return vec


def build_session_vector(track_features: Dict[str, dict], role_map: Dict[str, str], genre: str) -> np.ndarray:
    """Create a fixed-length session vector.

    - Roles are in fixed order: lead_vocal, bg_vocal, 808, drums, melody, fx
    - Missing roles are zero-filled
    - Features are flattened in FEATURE_KEYS order
    - Genre one-hot embedding appended
    """

    role_to_track = _role_to_track_id(role_map)

    role_vectors: List[np.ndarray] = []
    for role in ROLE_ORDER:
        track_id = role_to_track.get(role)
        if track_id is None:
            role_vectors.append(np.zeros((len(FEATURE_KEYS),), dtype=np.float32))
            continue

        feats = track_features.get(track_id, {})
        values = [float(feats.get(k, 0.0)) for k in FEATURE_KEYS]
        role_vectors.append(np.asarray(values, dtype=np.float32))

    session_vec = np.concatenate([*role_vectors, _genre_embedding(genre)], axis=0).astype(np.float32)
    return session_vec
