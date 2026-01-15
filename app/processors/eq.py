import numpy as np
import librosa


def eq(audio, sr, params):
    """Very lightweight EQ stub.

    - Optional high-pass style tilt via pre-emphasis
    """

    highpass = params.get("highpass")

    # Work on mono or stereo transparently
    if highpass is not None:
        if audio.ndim == 1:
            audio = librosa.effects.preemphasis(audio)
        else:
            # Apply per channel
            processed_channels = []
            for ch in range(audio.shape[1]):
                processed_channels.append(librosa.effects.preemphasis(audio[:, ch]))
            audio = np.stack(processed_channels, axis=1)

    return audio
