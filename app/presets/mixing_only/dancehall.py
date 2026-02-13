from __future__ import annotations

from typing import Any, Dict, List


def _base(eq_presence: float, air: float, warmth: float, reverb_wet: float, delay_fb: float) -> Dict[str, Any]:
    return {
        "eq_profile": {
            "highpass_hz": 90.0,
            "low_shelf_db": warmth,
            "low_mid_db": -1.5,
            "presence_db": eq_presence,
            "high_shelf_db": air,
        },
        "compression_profile": {
            "threshold_db": -20.0,
            "ratio": 3.2,
            "attack_ms": 4.0,
            "release_ms": 80.0,
            "max_gr_db": 3.5,
        },
        "dynamic_eq_profile": {
            "mud_cut_db": 2.0,
            "harsh_cut_db": 2.0,
            "sibilance_cut_db": 2.0,
        },
        "deesser_profile": {
            "freq_hz": 7500.0,
            "max_reduction_db": 2.5,
        },
        "saturation_profile": {
            "drive_amount": 0.02,
            "mode": "soft_clip",
        },
        "reverb_profile": {
            "pre_delay_ms": 18.0,
            "decay_s": 1.4,
            "size": 0.7,
            "wet_pct": reverb_wet,
        },
        "delay_profile": {
            "time_ms": 340.0,
            "feedback_pct": delay_fb,
            "wet_pct": 18.0,
            "sync_note": "1/8",
        },
        "bus_processing": {
            "bus_comp": {
                "threshold_db": -10.0,
                "ratio": 2.2,
                "attack_ms": 20.0,
                "release_ms": 120.0,
                "max_gr_db": 3.0,
            },
            "bus_saturation": {
                "drive_amount": 0.015,
            },
            "stereo_image": {
                "width": 0.15,
            },
        },
    }


PRESETS: List[Dict[str, Any]] = [
    {
        "preset_name": "Bright Island Lead",
        "ui_subtitle": "Inspired by modern dancehall vocal tone",
        **_base(eq_presence=2.5, air=2.5, warmth=-0.5, reverb_wet=20.0, delay_fb=20.0),
    },
    {
        "preset_name": "Gritty Street Tone",
        "ui_subtitle": "Inspired by modern dancehall vocal tone",
        **_base(eq_presence=1.5, air=1.5, warmth=0.5, reverb_wet=16.0, delay_fb=18.0),
    },
    {
        "preset_name": "Female Island Pop",
        "ui_subtitle": "Inspired by modern dancehall vocal tone",
        **_base(eq_presence=2.8, air=2.8, warmth=-0.8, reverb_wet=22.0, delay_fb=18.0),
    },
    {
        "preset_name": "Dark Minimal Dancehall",
        "ui_subtitle": "Inspired by modern dancehall vocal tone",
        **_base(eq_presence=0.5, air=0.5, warmth=1.5, reverb_wet=14.0, delay_fb=16.0),
    },
    {
        "preset_name": "Aggressive Yard Voice",
        "ui_subtitle": "Inspired by modern dancehall vocal tone",
        **_base(eq_presence=2.8, air=2.0, warmth=-1.0, reverb_wet=18.0, delay_fb=22.0),
    },
    {
        "preset_name": "Smooth Radio Dancehall",
        "ui_subtitle": "Inspired by modern dancehall vocal tone",
        **_base(eq_presence=1.8, air=2.0, warmth=0.8, reverb_wet=17.0, delay_fb=18.0),
    },
    {
        "preset_name": "Modern High-Air Vocal",
        "ui_subtitle": "Inspired by modern dancehall vocal tone",
        **_base(eq_presence=2.2, air=3.0, warmth=-0.5, reverb_wet=21.0, delay_fb=20.0),
    },
    {
        "preset_name": "Raw Underground Style",
        "ui_subtitle": "Inspired by modern dancehall vocal tone",
        **_base(eq_presence=1.2, air=1.2, warmth=0.3, reverb_wet=15.0, delay_fb=16.0),
    },
    {
        "preset_name": "Clean Festival Vocal",
        "ui_subtitle": "Inspired by modern dancehall vocal tone",
        **_base(eq_presence=2.0, air=2.2, warmth=0.0, reverb_wet=19.0, delay_fb=18.0),
    },
    {
        "preset_name": "Heavy Mid Punch",
        "ui_subtitle": "Inspired by modern dancehall vocal tone",
        **_base(eq_presence=2.5, air=1.5, warmth=0.5, reverb_wet=17.0, delay_fb=20.0),
    },
]
