from __future__ import annotations

from typing import Any, Dict, List


def _base(eq_presence: float, air: float, warmth: float, reverb_wet: float, delay_fb: float) -> Dict[str, Any]:
    return {
        "eq_profile": {
            "highpass_hz": 90.0,
            "low_shelf_db": warmth,
            "low_mid_db": -2.0,
            "presence_db": eq_presence,
            "high_shelf_db": air,
        },
        "compression_profile": {
            "threshold_db": -23.0,
            "ratio": 3.8,
            "attack_ms": 4.0,
            "release_ms": 80.0,
            "max_gr_db": 4.0,
        },
        "dynamic_eq_profile": {
            "mud_cut_db": 2.5,
            "harsh_cut_db": 2.5,
            "sibilance_cut_db": 2.0,
        },
        "deesser_profile": {
            "freq_hz": 7700.0,
            "max_reduction_db": 3.0,
        },
        "saturation_profile": {
            "drive_amount": 0.022,
            "mode": "soft_clip",
        },
        "reverb_profile": {
            "pre_delay_ms": 10.0,
            "decay_s": 0.9,
            "size": 0.55,
            "wet_pct": reverb_wet,
        },
        "delay_profile": {
            "time_ms": 290.0,
            "feedback_pct": delay_fb,
            "wet_pct": 16.0,
            "sync_note": "1/8",
        },
        "bus_processing": {
            "bus_comp": {
                "threshold_db": -11.0,
                "ratio": 2.6,
                "attack_ms": 16.0,
                "release_ms": 100.0,
                "max_gr_db": 3.8,
            },
            "bus_saturation": {
                "drive_amount": 0.02,
            },
            "stereo_image": {
                "width": 0.12,
            },
        },
    }


PRESETS: List[Dict[str, Any]] = [
    {
        "preset_name": "Aggressive Trap Rap",
        "ui_subtitle": "Inspired by modern rap vocal tone",
        **_base(eq_presence=2.8, air=2.0, warmth=-0.8, reverb_wet=13.0, delay_fb=24.0),
    },
    {
        "preset_name": "Punchy Drill",
        "ui_subtitle": "Inspired by modern rap vocal tone",
        **_base(eq_presence=2.6, air=1.8, warmth=-0.5, reverb_wet=13.0, delay_fb=22.0),
    },
    {
        "preset_name": "Bright Modern Rap",
        "ui_subtitle": "Inspired by modern rap vocal tone",
        **_base(eq_presence=2.7, air=2.5, warmth=-0.5, reverb_wet=14.0, delay_fb=22.0),
    },
    {
        "preset_name": "Heavy Compression Rap",
        "ui_subtitle": "Inspired by modern rap vocal tone",
        **_base(eq_presence=2.5, air=1.7, warmth=-0.3, reverb_wet=13.0, delay_fb=24.0),
    },
    {
        "preset_name": "Radio Polished",
        "ui_subtitle": "Inspired by modern rap vocal tone",
        **_base(eq_presence=2.3, air=2.1, warmth=0.3, reverb_wet=14.0, delay_fb=20.0),
    },
    {
        "preset_name": "Street Raw",
        "ui_subtitle": "Inspired by modern rap vocal tone",
        **_base(eq_presence=1.8, air=1.4, warmth=0.8, reverb_wet=12.0, delay_fb=18.0),
    },
    {
        "preset_name": "Air Boosted Rap",
        "ui_subtitle": "Inspired by modern rap vocal tone",
        **_base(eq_presence=2.4, air=2.8, warmth=-0.5, reverb_wet=14.0, delay_fb=20.0),
    },
    {
        "preset_name": "Minimal Dry Rap",
        "ui_subtitle": "Inspired by modern rap vocal tone",
        **_base(eq_presence=2.1, air=1.5, warmth=0.0, reverb_wet=10.0, delay_fb=14.0),
    },
    {
        "preset_name": "Wide Hook Rap",
        "ui_subtitle": "Inspired by modern rap vocal tone",
        **_base(eq_presence=2.2, air=2.2, warmth=0.4, reverb_wet=15.0, delay_fb=20.0),
    },
    {
        "preset_name": "Loud Energy Rap",
        "ui_subtitle": "Inspired by modern rap vocal tone",
        **_base(eq_presence=2.8, air=2.0, warmth=-0.2, reverb_wet=13.0, delay_fb=24.0),
    },
]
