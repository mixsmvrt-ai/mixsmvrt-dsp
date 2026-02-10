"""Adaptive mastering decision logic.

Provides loudness targets, multiband compression config, and final
limiter settings based on genre/style while enforcing safety limits.

Goals:
- Select loudness target by genre (streaming vs club/trap/dancehall)
- Use light multiband compression
- Final limiter ceiling -1.0 dBTP
- Avoid over-limiting (> 4 dB GR)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List

import logging

logger = logging.getLogger("mixsmvrt_dsp.adaptive.mastering")


@dataclass
class MultibandCompressorBand:
    low_hz: float
    high_hz: float
    threshold_db: float
    ratio: float
    attack_ms: float
    release_ms: float
    max_gr_db: float


@dataclass
class LimiterConfig:
    ceiling_db: float
    target_lufs: float
    max_gr_db: float


@dataclass
class AdaptiveMasterConfig:
    target_lufs: float
    multiband_bands: List[MultibandCompressorBand]
    limiter: LimiterConfig


STREAMING_TARGET_LUFS = -14.0
CLUB_TARGET_LUFS = -9.5  # between -9 and -8


def _infer_target_lufs(genre: str | None) -> float:
    if not genre:
        return STREAMING_TARGET_LUFS
    g = genre.lower()
    if any(k in g for k in ["trap", "dancehall", "club"]):
        logger.info("[MASTER] Genre=%s -> using club target LUFS %.1f", genre, CLUB_TARGET_LUFS)
        return CLUB_TARGET_LUFS
    logger.info("[MASTER] Genre=%s -> using streaming target LUFS %.1f", genre, STREAMING_TARGET_LUFS)
    return STREAMING_TARGET_LUFS


def build_adaptive_master_config(analysis: Dict[str, Any], genre: str | None) -> Dict[str, Any]:
    """Build mastering configuration from analysis features and genre.

    Analysis is expected to include `lufs` and band levels.
    """

    current_lufs = float(analysis.get("lufs", -16.0))
    target_lufs = _infer_target_lufs(genre)

    # Light multiband compression bands covering low, mid, and high.
    bands: list[MultibandCompressorBand] = [
        MultibandCompressorBand(
            low_hz=20.0,
            high_hz=160.0,
            threshold_db=-24.0,
            ratio=1.5,
            attack_ms=30.0,
            release_ms=200.0,
            max_gr_db=2.0,
        ),
        MultibandCompressorBand(
            low_hz=160.0,
            high_hz=4000.0,
            threshold_db=-20.0,
            ratio=1.7,
            attack_ms=20.0,
            release_ms=160.0,
            max_gr_db=3.0,
        ),
        MultibandCompressorBand(
            low_hz=4000.0,
            high_hz=18000.0,
            threshold_db=-18.0,
            ratio=1.5,
            attack_ms=10.0,
            release_ms=140.0,
            max_gr_db=2.5,
        ),
    ]

    # Limiter: aim to bridge the gap between current and target with
    # at most 4 dB of GR.
    loudness_gap = target_lufs - current_lufs  # negative if we need to get louder
    needed_gr = max(0.0, -loudness_gap)
    limiter_gr = min(4.0, needed_gr)

    limiter = LimiterConfig(
        ceiling_db=-1.0,
        target_lufs=target_lufs,
        max_gr_db=limiter_gr,
    )

    logger.info(
        "[MASTER] current LUFS=%.2f, target=%.2f, loudness_gap=%.2f -> limiter max GR=%.2f dB",
        current_lufs,
        target_lufs,
        loudness_gap,
        limiter_gr,
    )

    config = AdaptiveMasterConfig(
        target_lufs=target_lufs,
        multiband_bands=bands,
        limiter=limiter,
    )

    return asdict(config)
