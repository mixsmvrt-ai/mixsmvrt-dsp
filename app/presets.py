PRESETS = {
    "clean_vocal": {
        "eq": {"highpass": 100, "presence": 4},
        "deesser": {"freq": 7000, "amount": 0.4},
        "compressor": {"threshold": -18, "ratio": 3},
        "saturation": {"drive": 0.1},
        "stereo": {"width": 0.0},
        "limiter": {"ceiling": -1},
    },
    "aggressive_rap": {
        "eq": {"highpass": 90, "presence": 6},
        "compressor": {"threshold": -22, "ratio": 5},
        "saturation": {"drive": 0.3},
        "limiter": {"ceiling": -0.8},
    },
    "streaming_master": {
        "compressor": {"threshold": -12, "ratio": 2},
        "limiter": {"ceiling": -1},
    },
    # Background / stack vocals – a bit softer, more glue, less high‑end bite
    "bg_vocal_glue": {
        "eq": {"highpass": 150, "presence": 2},
        "deesser": {"freq": 6500, "amount": 0.3},
        "compressor": {"threshold": -16, "ratio": 2.5},
        "saturation": {"drive": 0.15},
        "stereo": {"width": 0.25},
        "limiter": {"ceiling": -1.5},
    },
    # Adlibs / hype vocals – brighter, more aggressive, a bit wider
    "adlib_hype": {
        "eq": {"highpass": 140, "presence": 5},
        "deesser": {"freq": 7500, "amount": 0.5},
        "compressor": {"threshold": -20, "ratio": 4},
        "saturation": {"drive": 0.35},
        "stereo": {"width": 0.35},
        "limiter": {"ceiling": -1},
    },
}
