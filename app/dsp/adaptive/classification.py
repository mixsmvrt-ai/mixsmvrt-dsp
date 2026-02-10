"""Automatic track classification for adaptive DSP.

Uses analysis metrics to classify tracks into:
- beat / instrumental
- lead_vocal
- background_vocal
- adlibs

User-provided tags can override the automatic classification.
"""

from __future__ import annotations

from typing import Dict, Any

import logging

logger = logging.getLogger("mixsmvrt_dsp.adaptive.classification")


TrackRole = str  # alias for clarity: "beat" | "lead_vocal" | "background_vocal" | "adlibs"


def classify_track(analysis: Dict[str, Any], user_tag: str | None = None) -> TrackRole:
    """Classify a track based on analysis metrics and optional user tag.

    Heuristics (can be refined over time):
    - High low-band energy, wide stereo, lower transient density -> beat
    - Strong mid/high, higher transient density, moderate width -> lead_vocal
    - Softer level, similar spectral shape to lead, less transient -> background_vocal
    - Very spiky transients, intermittent level -> adlibs

    User tag has priority and can be one of:
    - "beat" / "instrumental"
    - "lead_vocal" / "lead"
    - "background_vocal" / "bg"
    - "adlibs" / "adlib"
    """

    if user_tag:
        normalized = user_tag.strip().lower()
        if normalized in {"beat", "instrumental"}:
            logger.info("[CLASSIFY] Using user tag override: beat")
            return "beat"
        if normalized in {"lead", "lead_vocal", "vocal_lead"}:
            logger.info("[CLASSIFY] Using user tag override: lead_vocal")
            return "lead_vocal"
        if normalized in {"bg", "background", "background_vocal", "vocal_bg"}:
            logger.info("[CLASSIFY] Using user tag override: background_vocal")
            return "background_vocal"
        if normalized in {"adlib", "adlibs", "vocal_adlib"}:
            logger.info("[CLASSIFY] Using user tag override: adlibs")
            return "adlibs"

    peak_db = float(analysis.get("peak_db", -18.0))
    band_low = float(analysis.get("band_low_db", -40.0))
    band_mid = float(analysis.get("band_mid_db", -40.0))
    band_high = float(analysis.get("band_high_db", -40.0))
    transient_density = float(analysis.get("transient_density", 0.0))
    stereo_width = float(analysis.get("stereo_width", 0.0))

    # Rough loudness threshold between beat and more sparse sources
    is_loud = peak_db > -8.0
    low_heavy = band_low > (band_mid + 1.0) and band_low > (band_high + 1.5)

    # Beats / instrumentals: strong low band, relatively wide, not too spiky.
    if (is_loud and low_heavy and stereo_width > 0.25 and transient_density < 0.25):
        logger.info("[CLASSIFY] Heuristic classified track as beat (low_heavy, loud, wide, smooth transients)")
        return "beat"

    # Lead vocals: pronounced mid/high, clear transient presence.
    mid_high_strong = band_mid > -24.0 and band_high > -24.0
    if mid_high_strong and transient_density >= 0.25:
        logger.info("[CLASSIFY] Heuristic classified track as lead_vocal (mid/high strong, transienty)")
        return "lead_vocal"

    # Adlibs: very spiky, often louder peaks relative to body.
    if transient_density >= 0.45:
        logger.info("[CLASSIFY] Heuristic classified track as adlibs (very high transient density)")
        return "adlibs"

    # Background vocals: fallback for vocal-like material that is not as transienty.
    if mid_high_strong:
        logger.info("[CLASSIFY] Heuristic classified track as background_vocal (mid/high strong, softer transients)")
        return "background_vocal"

    # Default: treat unknown material as beat with conservative processing.
    logger.info("[CLASSIFY] Defaulting classification to beat (no clear vocal pattern)")
    return "beat"
