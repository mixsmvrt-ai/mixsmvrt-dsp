"""Adaptive compression, de-essing, saturation, and interaction logic.

This module builds *config objects* for compressors, de-essers, saturation,
multiband sidechain EQ, and vocal-bus processing based on analysis metrics.

Hard constraints:
- Never compress more than ~4 dB GR per stage
- Vocal saturation drive <= 3%, mix <= 10%
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List

import logging

logger = logging.getLogger("mixsmvrt_dsp.adaptive.compression")


@dataclass
class CompressorConfig:
    threshold_db: float
    ratio: float
    attack_ms: float
    release_ms: float
    makeup_db: float
    max_gr_db: float


@dataclass
class DeEsserConfig:
    center_freq_hz: float
    bandwidth_hz: float
    max_reduction_db: float
    threshold_db: float


@dataclass
class SaturationConfig:
    drive_percent: float
    mix_percent: float


@dataclass
class SidechainDuckingBand:
    low_hz: float
    high_hz: float
    gain_reduction_db: float
    attack_ms: float
    release_ms: float


@dataclass
class VocalBusConfig:
    glue_compressor: CompressorConfig
    eq_tilt_db: float
    saturation: SaturationConfig


@dataclass
class AdaptiveDynamicsConfig:
    """Container for all adaptive dynamics-related configs."""

    track_compressors: List[CompressorConfig]
    deesser: DeEsserConfig | None
    saturation: SaturationConfig | None
    beat_sidechain_bands: List[SidechainDuckingBand]
    vocal_bus: VocalBusConfig | None


def _choose_compression(dynamic_range_db: float, role: str) -> List[CompressorConfig]:
    """Design one or two compressor stages given dynamic range and role.

    - For lead vocals: two stages with gentle ratios and different envelopes.
    - For others: single stage.
    """

    configs: List[CompressorConfig] = []

    # Map dynamic range into approximate needed GR between 1–4 dB.
    # Very low dynamic range -> very light compression.
    # Very high dynamic range -> closer to 4 dB.
    target_gr = max(1.0, min(4.0, dynamic_range_db / 4.0))

    if role == "lead_vocal":
        # Two-stage vocal compression: first stage slower, second stage faster.
        first_ratio = 2.0 if dynamic_range_db < 10.0 else 2.5
        second_ratio = 1.8

        first = CompressorConfig(
            threshold_db=-20.0,
            ratio=first_ratio,
            attack_ms=25.0,
            release_ms=120.0,
            makeup_db=2.0,
            max_gr_db=min(4.0, target_gr * 0.6),
        )
        second = CompressorConfig(
            threshold_db=-18.0,
            ratio=second_ratio,
            attack_ms=8.0,
            release_ms=80.0,
            makeup_db=1.0,
            max_gr_db=min(4.0, target_gr * 0.6),
        )
        configs.extend([first, second])
        logger.info(
            "[COMP] Lead vocal dynamic range %.1f dB -> two-stage compression (GR≈%.1f dB per stage)",
            dynamic_range_db,
            target_gr * 0.6,
        )
    else:
        # Single-stage compression tuned by dynamic range.
        if dynamic_range_db < 6.0:
            ratio = 1.5
            attack = 30.0
            release = 150.0
        elif dynamic_range_db < 12.0:
            ratio = 2.0
            attack = 20.0
            release = 120.0
        else:
            ratio = 2.5
            attack = 15.0
            release = 100.0

        config = CompressorConfig(
            threshold_db=-18.0,
            ratio=ratio,
            attack_ms=attack,
            release_ms=release,
            makeup_db=2.0,
            max_gr_db=min(4.0, target_gr),
        )
        configs.append(config)
        logger.info(
            "[COMP] %s dynamic range %.1f dB -> single-stage compression (ratio=%.1f, target GR≈%.1f dB)",
            role,
            dynamic_range_db,
            ratio,
            target_gr,
        )

    return configs


def _choose_deesser(analysis: Dict[str, Any], role: str, gender_hint: str | None) -> DeEsserConfig | None:
    """Design de-esser or return None if sibilance is low.

    - Uses sibilance band level and role.
    - Gender hint steers center frequency range.
    """

    sibilance_db = float(analysis.get("band_sibilance_db", -60.0))
    mid_db = float(analysis.get("band_mid_db", -40.0))

    # Relative sibilance prominence.
    sib_excess = sibilance_db - mid_db
    if sib_excess < 2.0 or role not in {"lead_vocal", "background_vocal", "adlibs"}:
        logger.info(
            "[DEESSER] Bypassed (sib_excess %.2f dB, role=%s)",
            sib_excess,
            role,
        )
        return None

    # Map sib_excess into 2–6 dB reduction.
    max_reduction = max(2.0, min(6.0, 2.0 + (sib_excess / 2.0)))

    # Center frequency based on gender hint.
    if gender_hint and gender_hint.lower().startswith("f"):
        center_freq = 7000.0
    else:
        center_freq = 6000.0

    config = DeEsserConfig(
        center_freq_hz=center_freq,
        bandwidth_hz=3000.0,
        max_reduction_db=max_reduction,
        threshold_db=-24.0,
    )
    logger.info(
        "[DEESSER] Enabled for role=%s (sib_excess=%.2f dB, center=%.0f Hz, max_reduction=%.1f dB)",
        role,
        sib_excess,
        center_freq,
        max_reduction,
    )
    return config


def _choose_saturation(analysis: Dict[str, Any], role: str) -> SaturationConfig | None:
    """Design gentle saturation based on tonal balance.

    - Increase saturation for thin vocals (weak low band, strong high).
    - Reduce saturation for already dense signals.
    - Clamp drive 1–3 %, mix 5–10 %.
    """

    low_db = float(analysis.get("band_low_db", -40.0))
    mid_db = float(analysis.get("band_mid_db", -40.0))
    high_db = float(analysis.get("band_high_db", -40.0))

    # Perceived thinness: high minus low content.
    thinness = (high_db - low_db)

    if role in {"lead_vocal", "background_vocal", "adlibs"}:
        if thinness > 3.0:
            drive = min(3.0, 1.0 + (thinness / 4.0))
            mix = 8.0
            logger.info(
                "[SAT] Thin vocal detected (thinness=%.1f dB). Applying drive=%.1f%%, mix=%.1f%%",
                thinness,
                drive,
                mix,
            )
        elif thinness < 0.0:
            # Already dense -> lighter saturation
            drive = 1.0
            mix = 5.0
            logger.info(
                "[SAT] Dense vocal detected (thinness=%.1f dB). Applying gentle drive=%.1f%%, mix=%.1f%%",
                thinness,
                drive,
                mix,
            )
        else:
            drive = 1.5
            mix = 6.0
            logger.info(
                "[SAT] Neutral vocal body (thinness=%.1f dB). Applying moderate drive=%.1f%%, mix=%.1f%%",
                thinness,
                drive,
                mix,
            )
    else:
        # Beats/master: very subtle saturation only.
        drive = 1.0
        mix = 5.0
        logger.info("[SAT] Beat/master saturation: drive=%.1f%%, mix=%.1f%%", drive, mix)

    drive = max(1.0, min(3.0, drive))
    mix = max(5.0, min(10.0, mix))

    return SaturationConfig(drive_percent=drive, mix_percent=mix)


def _build_beat_sidechain_bands(role: str) -> List[SidechainDuckingBand]:
    """Create dynamic EQ ducking bands for beat sidechain.

    Only enabled when role is a vocal (lead/background/adlibs).
    """

    if role not in {"lead_vocal", "background_vocal", "adlibs"}:
        return []

    # Single broad band around 2–5 kHz for vocal presence.
    band = SidechainDuckingBand(
        low_hz=2000.0,
        high_hz=5000.0,
        gain_reduction_db=-2.0,
        attack_ms=10.0,
        release_ms=120.0,
    )
    logger.info(
        "[INT] Enabling vocal-beat sidechain ducking: 2–5 kHz, -2 dB, attack=10 ms, release=120 ms",
    )
    return [band]


def _build_vocal_bus(role: str) -> VocalBusConfig | None:
    """Create vocal-bus processing where applicable.

    - Glue compression 1–2 dB GR
    - Gentle EQ tilt
    - Subtle saturation (< 1.5 %)
    """

    if role not in {"lead_vocal", "background_vocal", "adlibs"}:
        return None

    glue = CompressorConfig(
        threshold_db=-10.0,
        ratio=1.5,
        attack_ms=30.0,
        release_ms=150.0,
        makeup_db=1.0,
        max_gr_db=2.0,
    )
    sat = SaturationConfig(drive_percent=1.2, mix_percent=6.0)

    config = VocalBusConfig(
        glue_compressor=glue,
        eq_tilt_db=0.5,
        saturation=sat,
    )
    logger.info("[BUS] Enabling vocal bus processing (glue GR<=2 dB, tilt=+0.5 dB, sat~1.2%%)")
    return config


def build_adaptive_dynamics(
    analysis: Dict[str, Any],
    role: str,
    gender_hint: str | None = None,
) -> Dict[str, Any]:
    """Top-level helper to assemble all adaptive dynamics configs.

    Returns a JSON-serialisable dict.
    """

    dynamic_range_db = float(analysis.get("dynamic_range_db", 8.0))

    compressors = _choose_compression(dynamic_range_db, role)
    deesser = _choose_deesser(analysis, role, gender_hint)
    saturation = _choose_saturation(analysis, role)
    beat_sidechain = _build_beat_sidechain_bands(role)
    vocal_bus = _build_vocal_bus(role)

    config = AdaptiveDynamicsConfig(
        track_compressors=compressors,
        deesser=deesser,
        saturation=saturation,
        beat_sidechain_bands=beat_sidechain,
        vocal_bus=vocal_bus,
    )

    return asdict(config)
