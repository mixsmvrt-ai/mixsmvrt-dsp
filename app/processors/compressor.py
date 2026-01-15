import numpy as np


def compressor(audio, sr, params):
    """Very simple RMS-based compressor stub.

    This is *not* a production compressor, just a gain adjustment
    based on overall RMS vs. a threshold.
    """

    threshold = params.get("threshold", -18)
    ratio = params.get("ratio", 3)

    # Convert threshold from dBFS to linear
    thresh_linear = 10 ** (threshold / 20.0)

    # Overall RMS of the signal
    rms = float(np.sqrt(np.mean(audio.astype(float) ** 2)))

    # Amount above threshold
    over = max(0.0, rms - thresh_linear)

    if over <= 0:
        return audio

    gain_reduction = over * (1.0 - 1.0 / ratio)
    gain = max(0.0, 1.0 - gain_reduction)

    return (audio * gain).astype(audio.dtype)
