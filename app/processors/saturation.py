import numpy as np


def saturation(audio, sr, params):
    """Simple tanh saturation with adjustable drive.

    This is a very rough non-linear curve, not band-limited.
    """

    drive = float(params.get("drive", 0.1))
    return np.tanh(audio * (1.0 + drive))
