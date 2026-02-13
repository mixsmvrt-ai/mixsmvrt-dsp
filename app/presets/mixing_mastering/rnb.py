from __future__ import annotations

from typing import Any, Dict, List


def _base(eq_presence: float, air: float, warmth: float, reverb_wet: float, delay_fb: float) -> Dict[str, Any]:
    return {
        "eq_profile": {
            "highpass_hz": 75.0,
            "low_shelf_db": warmth,
            "low_mid_db": -0.8,
            "presence_db": eq_presence,
            "high_shelf_db": air,
        },
        "compression_profile": {
            "threshold_db": -17.0,
            "ratio": 2.5,
            "attack_ms": 8.0,
            "release_ms": 140.0,
            "max_gr_db": 3.0,
        },
        "dynamic_eq_profile": {
            "mud_cut_db": 1.5,
            "harsh_cut_db": 1.5,
            "sibilance_cut_db": 1.5,
        },
        "deesser_profile": {
            "freq_hz": 7100.0,
            "max_reduction_db": 2.0,
        },
        "saturation_profile": {
            "drive_amount": 0.016,
            "mode": "soft_clip",
        },
        "reverb_profile": {
            "pre_delay_ms": 22.0,
            "decay_s": 2.0,
            "size": 0.85,
            "wet_pct": reverb_wet,
        },
        "delay_profile": {
            "time_ms": 420.0,
            "feedback_pct": delay_fb,
            "wet_pct": 21.0,
            "sync_note": "1/4",
        },
        "bus_processing": {
            "bus_comp": {
                "threshold_db": -8.5,
                "ratio": 2.0,
                "attack_ms": 24.0,
                "release_ms": 170.0,
                "max_gr_db": 3.0,
            },
            "bus_saturation": {
                "drive_amount": 0.014,
            },
            "stereo_image": {
                "width": 0.22,
            },
        },
        "master_processing": {
            "target_lufs": -10.5,
            "multiband": {
                "low_band_max_gr_db": 3.0,
                "mid_band_max_gr_db": 3.0,
                "high_band_max_gr_db": 3.0,
            },
            "midside": {
                "mid_eq_db": 0.0,
                "side_eq_db": 1.0,
                "width": 0.18,
            },
            "saturation": {
                "drive_amount": 0.015,
            },
            "limiter": {
                "ceiling_db": -1.0,
                "max_gr_db": 4.0,
            },
        },
    }


PRESETS: List[Dict[str, Any]] = [
    {
        "preset_name": "Smooth Silk Vocal",
        "ui_subtitle": "Inspired by modern rnb vocal tone",
        **_base(eq_presence=1.8, air=2.0, warmth=1.5, reverb_wet=24.0, delay_fb=20.0),
    },
    {
        "preset_name": "Airy Female R&B",
        "ui_subtitle": "Inspired by modern rnb vocal tone",
        **_base(eq_presence=2.3, air=2.8, warmth=0.5, reverb_wet=25.0, delay_fb=22.0),
    },
    {
        "preset_name": "Warm Intimate",
        "ui_subtitle": "Inspired by modern rnb vocal tone",
        **_base(eq_presence=1.6, air=1.8, warmth=1.8, reverb_wet=23.0, delay_fb=18.0),
    },
    {
        "preset_name": "Modern Pop R&B",
        "ui_subtitle": "Inspired by modern rnb vocal tone",
        **_base(eq_presence=2.2, air=2.4, warmth=0.8, reverb_wet=24.0, delay_fb=20.0),
    },
    {
        "preset_name": "Wide Stereo Ballad",
        "ui_subtitle": "Inspired by modern rnb vocal tone",
        **_base(eq_presence=1.9, air=2.1, warmth=1.2, reverb_wet=24.0, delay_fb=20.0),
    },
    {
        "preset_name": "Dark Emotional",
        "ui_subtitle": "Inspired by modern rnb vocal tone",
        **_base(eq_presence=1.4, air=1.6, warmth=1.8, reverb_wet=23.0, delay_fb=18.0),
    },
    {
        "preset_name": "Soft Compression Vocal",
        "ui_subtitle": "Inspired by modern rnb vocal tone",
        **_base(eq_presence=1.7, air=1.9, warmth=1.3, reverb_wet=23.0, delay_fb=18.0),
    },
    {
        "preset_name": "Radio R&B Clean",
        "ui_subtitle": "Inspired by modern rnb vocal tone",
        **_base(eq_presence=2.0, air=2.2, warmth=1.0, reverb_wet=23.0, delay_fb=18.0),
    },
    {
        "preset_name": "High Air Falsetto",
        "ui_subtitle": "Inspired by modern rnb vocal tone",
        **_base(eq_presence=2.4, air=2.9, warmth=0.2, reverb_wet=24.0, delay_fb=20.0),
    },
    {
        "preset_name": "Deep Warm Soul",
        "ui_subtitle": "Inspired by modern rnb vocal tone",
        **_base(eq_presence=1.6, air=1.8, warmth=2.0, reverb_wet=23.0, delay_fb=18.0),
    },
]
