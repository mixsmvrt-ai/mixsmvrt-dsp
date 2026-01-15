from app.processors.eq import eq
from app.processors.deesser import deesser
from app.processors.compressor import compressor
from app.processors.saturation import saturation
from app.processors.stereo import stereo
from app.processors.limiter import limiter


TRACK_CHAINS = {
    "vocal": [
        eq,
        deesser,
        compressor,
        eq,
        saturation,
        stereo,
        limiter,
    ],
    "beat": [
        eq,
        compressor,
        saturation,
        stereo,
        limiter,
    ],
    "master": [
        eq,
        compressor,
        saturation,
        stereo,
        limiter,
    ],
}
