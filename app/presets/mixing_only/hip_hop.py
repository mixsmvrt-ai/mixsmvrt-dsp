from __future__ import annotations

from typing import Any, Dict, List


def _base(eq_presence: float, air: float, warmth: float, reverb_wet: float, delay_fb: float) -> Dict[str, Any]:
    return {
        "eq_profile": {
            "highpass_hz": 85.0,
            "low_shelf_db": warmth,
            "low_mid_db": -1.8,
            "presence_db": eq_presence,
            "high_shelf_db": air,
        },
        "compression_profile": {
            "threshold_db": -21.0,
            "ratio": 3.3,
            "attack_ms": 5.0,
            "release_ms": 90.0,
            "max_gr_db": 3.5,
        },
        "dynamic_eq_profile": {
            "mud_cut_db": 2.2,
            "harsh_cut_db": 2.2,
            "sibilance_cut_db": 2.0,
        },
        "deesser_profile": {
            "freq_hz": 7600.0,
            "max_reduction_db": 2.5,
        },
        "saturation_profile": {
            "drive_amount": 0.02,
            "mode": "soft_clip",
        },
        "reverb_profile": {
            "pre_delay_ms": 14.0,
            "decay_s": 1.1,
            "size": 0.6,
            "wet_pct": reverb_wet,
        },
        "delay_profile": {
            "time_ms": 310.0,
            "feedback_pct": delay_fb,
            "wet_pct": 17.0,
            "sync_note": "1/8",
        },
        "bus_processing": {
            "bus_comp": {
                "threshold_db": -10.5,
                "ratio": 2.4,
                "attack_ms": 18.0,
                "release_ms": 110.0,
                "max_gr_db": 3.5,
            },
            "bus_saturation": {
                "drive_amount": 0.017,
            },
            "stereo_image": {
                "width": 0.14,
            },
        },
    }


PRESETS: List[Dict[str, Any]] = [
    {
        "preset_name": "Classic Boom Bap",
        "ui_subtitle": "Inspired by modern hip_hop vocal tone",
        **_base(eq_presence=1.8, air=1.4, warmth=1.2, reverb_wet=13.0, delay_fb=18.0),
    },
    {
        "preset_name": "Modern East Coast",
        "ui_subtitle": "Inspired by modern hip_hop vocal tone",
        **_base(eq_presence=2.4, air=2.0, warmth=0.3, reverb_wet=15.0, delay_fb=20.0),
    },
    {
        "preset_name": "West Coast Clean",
        "ui_subtitle": "Inspired by modern hip_hop vocal tone",
        **_base(eq_presence=2.2, air=2.2, warmth=0.0, reverb_wet=16.0, delay_fb=20.0),
    },
    {
        "preset_name": "Tight Punch Rap",
        "ui_subtitle": "Inspired by modern hip_hop vocal tone",
        **_base(eq_presence=2.5, air=1.8, warmth=-0.5, reverb_wet=14.0, delay_fb=22.0),
    },
    {
        "preset_name": "Deep Baritone",
        "ui_subtitle": "Inspired by modern hip_hop vocal tone",
        **_base(eq_presence=1.5, air=1.2, warmth=1.8, reverb_wet=13.0, delay_fb=18.0),
    },
    {
        "preset_name": "Airy Modern",
        "ui_subtitle": "Inspired by modern hip_hop vocal tone",
        **_base(eq_presence=2.3, air=2.6, warmth=-0.3, reverb_wet=16.0, delay_fb=20.0),
    },
    {
        "preset_name": "Radio Clean",
        "ui_subtitle": "Inspired by modern hip_hop vocal tone",
        **_base(eq_presence=2.1, air=2.0, warmth=0.4, reverb_wet=15.0, delay_fb=18.0),
    },
    {
        "preset_name": "Dark Underground",
        "ui_subtitle": "Inspired by modern hip_hop vocal tone",
        **_base(eq_presence=1.4, air=1.0, warmth=1.5, reverb_wet=13.0, delay_fb=18.0),
    },
    {
        "preset_name": "Smooth Hook Vocal",
        "ui_subtitle": "Inspired by modern hip_hop vocal tone",
        **_base(eq_presence=2.0, air=2.2, warmth=0.8, reverb_wet=16.0, delay_fb=20.0),
    },
    {
        "preset_name": "Hard Rap Attack",
        "ui_subtitle": "Inspired by modern hip_hop vocal tone",
        **_base(eq_presence=2.7, air=1.9, warmth=-0.2, reverb_wet=14.0, delay_fb=22.0),
    },
]
