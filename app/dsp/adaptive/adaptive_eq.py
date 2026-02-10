"""Adaptive EQ decision logic.

Uses analysis features to propose EQ moves for low-mid mud, harshness,
air, and overall tonal balance. This module returns a *config* that can
be consumed by a downstream processor (e.g. Pedalboard, JUCE, ffmpeg).

Hard constraints:
- Never boost more than +3 dB
- Never cut more than -5 dB

Rule set:
- Detect mud (200–350 Hz) and cut -1 to -4 dB dynamically
- Detect harshness (2.5–4.5 kHz) and cut -1 to -3 dB
- Detect lack of air (10–16 kHz) and boost +1 to +3 dB
- Never exceed ±5 dB per band
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List

import logging

logger = logging.getLogger("mixsmvrt_dsp.adaptive.eq")


@dataclass
class EqBand:
    """Single EQ band configuration.

    gain_db is clamped according to global safety rules.
    """

    type: str  # e.g. "bell", "high_shelf", "low_shelf"
    frequency_hz: float
    q: float
    gain_db: float


@dataclass
class AdaptiveEqConfig:
    bands: List[EqBand]


def _clamp_gain(gain_db: float) -> float:
    """Clamp gain to global safety bounds.

    - Boost <= +3 dB
    - Cut >= -5 dB
    """

    if gain_db > 3.0:
        return 3.0
    if gain_db < -5.0:
        return -5.0
    return gain_db


def build_adaptive_eq(analysis: Dict[str, Any], role: str) -> Dict[str, Any]:
    """Build an adaptive EQ configuration from analysis metrics.

    `role` is one of: "beat", "lead_vocal", "background_vocal", "adlibs".
    """

    bands: List[EqBand] = []

    low_db = float(analysis.get("band_low_db", -40.0))
    mid_db = float(analysis.get("band_mid_db", -40.0))
    high_db = float(analysis.get("band_high_db", -40.0))
    noise_floor_db = float(analysis.get("noise_floor_db", -80.0))

    # ---------------------------------------------
    # Mud detection: 200–350 Hz region
    # ---------------------------------------------
    # Use low vs mid comparison and overall noise floor to infer mud.
    mud_excess = (low_db - mid_db)
    mud_severity = max(0.0, mud_excess - 1.0)  # ignore very small differences

    if mud_severity > 0.0:
        # Map severity into -1 to -4 dB range.
        cut_db = -min(4.0, 1.0 + mud_severity)
        cut_db = _clamp_gain(cut_db)
        bands.append(
            EqBand(
                type="bell",
                frequency_hz=260.0,
                q=1.2,
                gain_db=cut_db,
            )
        )
        logger.info(
            "[EQ] Mud detected (low-mid excess %.2f dB). Applying cut %.2f dB at 260 Hz",
            mud_excess,
            cut_db,
        )
    else:
        logger.info("[EQ] No significant mud detected (low-mid delta %.2f dB)", mud_excess)

    # ---------------------------------------------
    # Harshness detection: 2.5–4.5 kHz
    # ---------------------------------------------
    # Approximate harshness using high-band vs mid-band balance.
    harsh_excess = (high_db - mid_db)
    harsh_severity = max(0.0, harsh_excess - 1.0)

    if harsh_severity > 0.0:
        cut_db = -min(3.0, 1.0 + 0.5 * harsh_severity)
        cut_db = _clamp_gain(cut_db)
        bands.append(
            EqBand(
                type="bell",
                frequency_hz=3200.0,
                q=2.0,
                gain_db=cut_db,
            )
        )
        logger.info(
            "[EQ] Harshness detected (high-mid excess %.2f dB). Applying cut %.2f dB at 3.2 kHz",
            harsh_excess,
            cut_db,
        )
    else:
        logger.info("[EQ] No significant harshness detected (high-mid delta %.2f dB)", harsh_excess)

    # ---------------------------------------------
    # Air detection: 10–16 kHz
    # ---------------------------------------------
    # Re-use high band as a proxy for upper-air balance relative to noise floor.
    air_gap = (high_db - noise_floor_db)
    # If high band is not far above noise floor, consider it lacking air.
    if air_gap < 25.0:
        # Boost between +1 and +3 dB depending on how far below 25 dB we are.
        deficit = 25.0 - air_gap
        boost_db = min(3.0, 1.0 + (deficit / 10.0))
        boost_db = _clamp_gain(boost_db)
        if boost_db > 0.0:
            bands.append(
                EqBand(
                    type="high_shelf",
                    frequency_hz=12000.0,
                    q=0.7,
                    gain_db=boost_db,
                )
            )
            logger.info(
                "[EQ] Air deficiency detected (air_gap %.1f dB). Applying boost %.2f dB at 12 kHz",
                air_gap,
                boost_db,
            )
    else:
        logger.info("[EQ] Air balance acceptable (air_gap %.1f dB)", air_gap)

    # Additional gentle tilt for role-specific voicing
    if role in {"lead_vocal", "background_vocal"}:
        # Slight presence lift for vocals when not already harsh.
        if harsh_severity == 0.0:
            bands.append(
                EqBand(
                    type="bell",
                    frequency_hz=2800.0,
                    q=1.4,
                    gain_db=_clamp_gain(0.5),
                )
            )
            logger.info("[EQ] Adding gentle vocal presence lift (+0.5 dB at 2.8 kHz)")

    config = AdaptiveEqConfig(bands=bands)
    return asdict(config)
