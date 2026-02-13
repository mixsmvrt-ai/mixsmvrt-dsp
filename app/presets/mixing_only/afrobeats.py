from __future__ import annotations

from typing import Any, Dict, List


def _base(eq_presence: float, air: float, warmth: float, reverb_wet: float, delay_fb: float) -> Dict[str, Any]:
    return {
        "eq_profile": {
            "highpass_hz": 80.0,
            "low_shelf_db": warmth,
            "low_mid_db": -1.0,
            "presence_db": eq_presence,
            "high_shelf_db": air,
        },
        "compression_profile": {
            "threshold_db": -18.0,
            "ratio": 2.8,
            "attack_ms": 6.0,
            "release_ms": 120.0,
            "max_gr_db": 3.0,
        },
        "dynamic_eq_profile": {
            "mud_cut_db": 1.8,
            "harsh_cut_db": 1.8,
            "sibilance_cut_db": 1.5,
        },
        "deesser_profile": {
            "freq_hz": 7200.0,
            "max_reduction_db": 2.0,
        },
        "saturation_profile": {
            "drive_amount": 0.018,
            "mode": "soft_clip",
        },
        "reverb_profile": {
            "pre_delay_ms": 20.0,
            "decay_s": 1.8,
            "size": 0.8,
            "wet_pct": reverb_wet,
        },
        "delay_profile": {
            "time_ms": 380.0,
            "feedback_pct": delay_fb,
            "wet_pct": 20.0,
            "sync_note": "1/4",
        },
        "bus_processing": {
            "bus_comp": {
                "threshold_db": -9.0,
                "ratio": 2.0,
                "attack_ms": 22.0,
                "release_ms": 160.0,
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
        "preset_name": "Smooth Afro Pop",
        "ui_subtitle": "Inspired by modern afrobeats vocal tone",
        **_base(eq_presence=2.0, air=2.0, warmth=0.5, reverb_wet=22.0, delay_fb=20.0),
    },
    {
        "preset_name": "Airy Festival Vocal",
        "ui_subtitle": "Inspired by modern afrobeats vocal tone",
        **_base(eq_presence=2.3, air=2.8, warmth=0.0, reverb_wet=24.0, delay_fb=22.0),
    },
    {
        "preset_name": "Deep Warm Afro",
        "ui_subtitle": "Inspired by modern afrobeats vocal tone",
        **_base(eq_presence=1.5, air=1.5, warmth=1.5, reverb_wet=21.0, delay_fb=18.0),
    },
    {
        "preset_name": "Bright Female Afro",
        "ui_subtitle": "Inspired by modern afrobeats vocal tone",
        **_base(eq_presence=2.5, air=2.8, warmth=-0.5, reverb_wet=23.0, delay_fb=20.0),
    },
    {
        "preset_name": "Radio Afro Clean",
        "ui_subtitle": "Inspired by modern afrobeats vocal tone",
        **_base(eq_presence=2.0, air=2.2, warmth=0.3, reverb_wet=21.0, delay_fb=18.0),
    },
    {
        "preset_name": "Wide Stereo Afro",
        "ui_subtitle": "Inspired by modern afrobeats vocal tone",
        **_base(eq_presence=2.1, air=2.4, warmth=0.5, reverb_wet=23.0, delay_fb=20.0),
    },
    {
        "preset_name": "Intimate Afro Soul",
        "ui_subtitle": "Inspired by modern afrobeats vocal tone",
        **_base(eq_presence=1.8, air=2.0, warmth=1.2, reverb_wet=20.0, delay_fb=16.0),
    },
    {
        "preset_name": "Percussive Tight Vocal",
        "ui_subtitle": "Inspired by modern afrobeats vocal tone",
        **_base(eq_presence=2.2, air=2.0, warmth=0.0, reverb_wet=19.0, delay_fb=18.0),
    },
    {
        "preset_name": "Warm Mid Boost",
        "ui_subtitle": "Inspired by modern afrobeats vocal tone",
        **_base(eq_presence=2.0, air=1.8, warmth=1.0, reverb_wet=21.0, delay_fb=18.0),
    },
    {
        "preset_name": "Club Afro Energy",
        "ui_subtitle": "Inspired by modern afrobeats vocal tone",
        **_base(eq_presence=2.4, air=2.4, warmth=0.3, reverb_wet=22.0, delay_fb=22.0),
    },
]
