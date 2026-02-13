from __future__ import annotations

from typing import Any, Dict, List


def _base(eq_presence: float, air: float, warmth: float, reverb_wet: float, delay_fb: float) -> Dict[str, Any]:
    return {
        "eq_profile": {
            "highpass_hz": 100.0,
            "low_shelf_db": warmth,
            "low_mid_db": -2.0,
            "presence_db": eq_presence,
            "high_shelf_db": air,
        },
        "compression_profile": {
            "threshold_db": -22.0,
            "ratio": 3.5,
            "attack_ms": 3.0,
            "release_ms": 70.0,
            "max_gr_db": 3.8,
        },
        "dynamic_eq_profile": {
            "mud_cut_db": 2.5,
            "harsh_cut_db": 2.5,
            "sibilance_cut_db": 2.0,
        },
        "deesser_profile": {
            "freq_hz": 7800.0,
            "max_reduction_db": 3.0,
        },
        "saturation_profile": {
            "drive_amount": 0.022,
            "mode": "soft_clip",
        },
        "reverb_profile": {
            "pre_delay_ms": 16.0,
            "decay_s": 1.2,
            "size": 0.65,
            "wet_pct": reverb_wet,
        },
        "delay_profile": {
            "time_ms": 320.0,
            "feedback_pct": delay_fb,
            "wet_pct": 19.0,
            "sync_note": "1/8",
        },
        "bus_processing": {
            "bus_comp": {
                "threshold_db": -11.0,
                "ratio": 2.5,
                "attack_ms": 18.0,
                "release_ms": 110.0,
                "max_gr_db": 3.5,
            },
            "bus_saturation": {
                "drive_amount": 0.018,
            },
            "stereo_image": {
                "width": 0.18,
            },
        },
    }


PRESETS: List[Dict[str, Any]] = [
    {
        "preset_name": "Dark Auto Style",
        "ui_subtitle": "Inspired by modern trap_dancehall vocal tone",
        **_base(eq_presence=1.8, air=1.2, warmth=0.8, reverb_wet=17.0, delay_fb=22.0),
    },
    {
        "preset_name": "Aggressive Punch",
        "ui_subtitle": "Inspired by modern trap_dancehall vocal tone",
        **_base(eq_presence=2.6, air=1.8, warmth=-0.5, reverb_wet=18.0, delay_fb=24.0),
    },
    {
        "preset_name": "Bright Street Trap",
        "ui_subtitle": "Inspired by modern trap_dancehall vocal tone",
        **_base(eq_presence=2.8, air=2.5, warmth=-1.0, reverb_wet=19.0, delay_fb=22.0),
    },
    {
        "preset_name": "Airy Hybrid Trap",
        "ui_subtitle": "Inspired by modern trap_dancehall vocal tone",
        **_base(eq_presence=2.2, air=2.8, warmth=-0.5, reverb_wet=20.0, delay_fb=20.0),
    },
    {
        "preset_name": "Deep Resonant Voice",
        "ui_subtitle": "Inspired by modern trap_dancehall vocal tone",
        **_base(eq_presence=1.5, air=1.0, warmth=1.5, reverb_wet=16.0, delay_fb=18.0),
    },
    {
        "preset_name": "High Energy Club",
        "ui_subtitle": "Inspired by modern trap_dancehall vocal tone",
        **_base(eq_presence=2.7, air=2.0, warmth=-0.8, reverb_wet=18.0, delay_fb=24.0),
    },
    {
        "preset_name": "Tight Dry Vocal",
        "ui_subtitle": "Inspired by modern trap_dancehall vocal tone",
        **_base(eq_presence=2.0, air=1.5, warmth=0.0, reverb_wet=14.0, delay_fb=16.0),
    },
    {
        "preset_name": "Wide Stereo Trap",
        "ui_subtitle": "Inspired by modern trap_dancehall vocal tone",
        **_base(eq_presence=2.3, air=2.3, warmth=0.3, reverb_wet=19.0, delay_fb=22.0),
    },
    {
        "preset_name": "Radio Trap Clean",
        "ui_subtitle": "Inspired by modern trap_dancehall vocal tone",
        **_base(eq_presence=2.1, air=2.1, warmth=0.2, reverb_wet=17.0, delay_fb=20.0),
    },
    {
        "preset_name": "Heavy Grit Trap",
        "ui_subtitle": "Inspired by modern trap_dancehall vocal tone",
        **_base(eq_presence=2.5, air=1.7, warmth=0.5, reverb_wet=16.0, delay_fb=24.0),
    },
]
