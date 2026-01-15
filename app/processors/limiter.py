import numpy as np


def limiter(audio, sr, params):
    """Simple hard limiter with linear ceiling in dBFS."""

    ceiling = float(params.get("ceiling", -1.0))
    limit = 10 ** (ceiling / 20.0)
    return np.clip(audio, -limit, limit)
