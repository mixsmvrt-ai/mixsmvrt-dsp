import pyloudnorm as pyln


def measure_loudness(audio, sr):
    """Integrated loudness helper using ITU-R BS.1770.

    Not currently wired into the processing chain, but available
    for future adaptive presets.
    """

    meter = pyln.Meter(sr)
    return float(meter.integrated_loudness(audio))
