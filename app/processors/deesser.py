

def deesser(audio, sr, params):
    """Very rough de-esser stub.

    Currently just applies a broadband attenuation factor.
    """

    amount = params.get("amount", 0.3)
    amount = max(0.0, min(float(amount), 1.0))
    return audio * (1.0 - amount)
