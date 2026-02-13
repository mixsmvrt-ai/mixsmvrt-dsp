from __future__ import annotations

from typing import Any, Dict, List


def _base(eq_presence: float, air: float, warmth: float, reverb_wet: float, delay_fb: float) -> Dict[str, Any]:
    return {
        "eq_profile": {
            "highpass_hz": 70.0,
            "low_shelf_db": warmth,
            "low_mid_db": -0.5,
            "presence_db": eq_presence,
            "high_shelf_db": air,
        },
        "compression_profile": {
            "threshold_db": -18.5,
            "ratio": 2.4,
            "attack_ms": 7.0,
            "release_ms": 150.0,
            "max_gr_db": 3.0,
        },
        "dynamic_eq_profile": {
            "mud_cut_db": 1.3,
            "harsh_cut_db": 1.3,
            "sibilance_cut_db": 1.0,
        },
        "deesser_profile": {
            "freq_hz": 6900.0,
            "max_reduction_db": 2.0,
        },
        "saturation_profile": {
            "drive_amount": 0.017,
            "mode": "soft_clip",
        },
        "reverb_profile": {
            "pre_delay_ms": 18.0,
            "decay_s": 1.9,
            "size": 0.82,
            "wet_pct": reverb_wet,
        },
        "delay_profile": {
            "time_ms": 430.0,
            "feedback_pct": delay_fb,
            "wet_pct": 22.0,
            "sync_note": "1/4",
        },
        "bus_processing": {
            "bus_comp": {
                "threshold_db": -9.5,
                "ratio": 2.1,
                "attack_ms": 24.0,
                "release_ms": 180.0,
                "max_gr_db": 3.0,
            },
            "bus_saturation": {
                "drive_amount": 0.015,
            },
            "stereo_image": {
                "width": 0.2,
            },
        },
    }


PRESETS: List[Dict[str, Any]] = [
    {
        "preset_name": "Classic Roots",
        "ui_subtitle": "Inspired by modern reggae vocal tone",
        **_base(eq_presence=1.7, air=1.6, warmth=1.8, reverb_wet=23.0, delay_fb=22.0),
    },
    {
        "preset_name": "Modern Reggae Clean",
        "ui_subtitle": "Inspired by modern reggae vocal tone",
        **_base(eq_presence=2.0, air=2.0, warmth=1.2, reverb_wet=22.0, delay_fb=22.0),
    },
    {
        "preset_name": "Warm Dub Style",
        "ui_subtitle": "Inspired by modern reggae vocal tone",
        **_base(eq_presence=1.6, air=1.8, warmth=2.0, reverb_wet=24.0, delay_fb=24.0),
    },
    {
        "preset_name": "Bright Radio Reggae",
        "ui_subtitle": "Inspired by modern reggae vocal tone",
        **_base(eq_presence=2.2, air=2.4, warmth=1.0, reverb_wet=22.0, delay_fb=22.0),
    },
    {
        "preset_name": "Deep Roots Voice",
        "ui_subtitle": "Inspired by modern reggae vocal tone",
        **_base(eq_presence=1.5, air=1.4, warmth=2.2, reverb_wet=23.0, delay_fb=20.0),
    },
    {
        "preset_name": "Smooth Lovers Rock",
        "ui_subtitle": "Inspired by modern reggae vocal tone",
        **_base(eq_presence=1.8, air=2.0, warmth=1.6, reverb_wet=23.0, delay_fb=22.0),
    },
    {
        "preset_name": "Wide Stereo Reggae",
        "ui_subtitle": "Inspired by modern reggae vocal tone",
        **_base(eq_presence=1.9, air=2.1, warmth=1.4, reverb_wet=23.0, delay_fb=22.0),
    },
    {
        "preset_name": "Raw Yard Style",
        "ui_subtitle": "Inspired by modern reggae vocal tone",
        **_base(eq_presence=1.6, air=1.5, warmth=1.9, reverb_wet=22.0, delay_fb=20.0),
    },
    {
        "preset_name": "Clean Festival Tone",
        "ui_subtitle": "Inspired by modern reggae vocal tone",
        **_base(eq_presence=2.1, air=2.1, warmth=1.3, reverb_wet=22.0, delay_fb=22.0),
    },
    {
        "preset_name": "Warm Mid Classic",
        "ui_subtitle": "Inspired by modern reggae vocal tone",
        **_base(eq_presence=1.8, air=1.7, warmth=2.0, reverb_wet=23.0, delay_fb=22.0),
    },
]
