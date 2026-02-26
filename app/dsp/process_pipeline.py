"""Sequential session processing pipeline.

This is the integration layer for the session-aware adaptive engine.

It enforces:
- CPU-only sequential processing (one session at a time)
- temp file cleanup
- optional S3 upload of outputs
"""

from __future__ import annotations

import os
import json
import logging
import tempfile
import threading
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from app.storage import upload_file_to_s3

from .adaptive_engine import AdaptiveSessionEngine
from .parameter_model import ParameterPredictor


logger = logging.getLogger("mixsmvrt_dsp.process_pipeline")


_SESSION_LOCK = threading.Lock()


def process_session(
    track_audio_map: Dict[str, tuple[np.ndarray, int]],
    *,
    genre: str,
    engine: AdaptiveSessionEngine,
) -> Dict[str, Any]:
    """Process a multi-track session sequentially and upload result to S3."""

    with _SESSION_LOCK:
        result = engine.process_session(track_audio_map, genre)

        # Always write WAV (canonical output)
        out_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        out_wav.close()
        try:
            audio = result.audio
            # soundfile expects [N, C]
            if audio.ndim == 2 and audio.shape[0] == 2:
                sf_audio = audio.T
            else:
                sf_audio = audio
            sf.write(out_wav.name, sf_audio, int(result.sr))

            wav_url_or_path = upload_file_to_s3(out_wav.name)
        finally:
            try:
                os.remove(out_wav.name)
            except Exception:
                pass

        payload: Dict[str, Any] = {
            "status": "processed",
            "output_files": {
                "wav": wav_url_or_path,
            },
            "roles": result.role_map,
            "master": result.master_report,
            "processing_time_s": result.processing_time_s,
        }

        if os.getenv("AI_DEBUG", "false").lower() in {"1", "true", "yes", "on"}:
            payload["session_vector"] = result.session_vector.tolist()
            payload["predicted_params"] = result.predicted_params

        return payload
