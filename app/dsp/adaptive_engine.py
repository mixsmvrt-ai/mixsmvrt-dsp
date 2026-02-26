"""Session-aware Adaptive AI Mix Engine.

This is the main orchestrator that:
- extracts per-track features
- classifies roles contextually
- builds a fixed-length session vector
- runs a pretrained LightGBM model
- maps predictions to DSP processing using existing DSP engine primitives

All processing is sequential and CPU-only.
"""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from .feature_extractor import extract_track_features
from .role_classifier import classify_roles
from .session_vector_builder import build_session_vector
from .parameter_model import ParameterPredictor
from .dsp_mapper import apply_predicted_parameters, apply_master_processing


logger = logging.getLogger("mixsmvrt_dsp.ai_mix_engine")


def _parse_track_audio_map(track_audio_map: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], int]:
    """Accept {id: (audio, sr)} or {id: {audio, sr}} and return consistent sr."""

    parsed: dict[str, np.ndarray] = {}
    sr_value: int | None = None

    for track_id, value in track_audio_map.items():
        audio: np.ndarray
        sr: int
        if isinstance(value, (tuple, list)) and len(value) == 2:
            audio = value[0]
            sr = int(value[1])
        elif isinstance(value, dict) and "audio" in value and "sr" in value:
            audio = value["audio"]
            sr = int(value["sr"])
        else:
            raise ValueError(
                "track_audio_map values must be (audio, sr) or {'audio':..., 'sr':...}"
            )

        if sr_value is None:
            sr_value = sr
        elif sr != sr_value:
            raise ValueError("All tracks must have the same sample rate")

        parsed[str(track_id)] = np.asarray(audio, dtype=np.float32)

    if sr_value is None:
        raise ValueError("Empty track_audio_map")

    return parsed, sr_value


def _ensure_stereo(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return np.stack([x, x], axis=0).astype(np.float32)
    if x.ndim == 2 and x.shape[0] == 2:
        return x.astype(np.float32)
    if x.ndim == 2 and x.shape[1] == 2:
        return x.T.astype(np.float32)
    raise ValueError("Expected mono [N] or stereo [2,N]/[N,2]")


@dataclass
class SessionProcessResult:
    audio: np.ndarray
    sr: int
    role_map: Dict[str, str]
    session_vector: np.ndarray
    predicted_params: Dict[str, float]
    master_report: Dict[str, float]
    processing_time_s: float


class AdaptiveSessionEngine:
    def __init__(self, predictor: ParameterPredictor) -> None:
        self._predictor = predictor
        self._debug = os.getenv("AI_DEBUG", "false").lower() in {"1", "true", "yes", "on"}

    def process_session(self, track_audio_map: Mapping[str, Any], genre: str) -> SessionProcessResult:
        t0 = time.perf_counter()

        parsed_audio, sr = _parse_track_audio_map(track_audio_map)

        # 1) Extract features sequentially
        track_features: Dict[str, dict] = {}
        for track_id, audio in parsed_audio.items():
            track_features[track_id] = extract_track_features(audio, sr)

        # 2) Contextual role classification
        role_map = classify_roles(track_features)

        # 3) Build fixed-length session vector
        session_vector = build_session_vector(track_features, role_map, genre)

        # 4) Predict DSP parameters
        predicted_params = self._predictor.predict(session_vector)

        if self._debug:
            logger.info("[AI_DEBUG] roles=%s", role_map)
            logger.info("[AI_DEBUG] session_vector=%s", session_vector.tolist())
            logger.info("[AI_DEBUG] predicted_params=%s", predicted_params)

        # 5) Apply parameters per track (use lead vocal as sidechain)
        lead_vocal_id = next((tid for tid, r in role_map.items() if r == "lead_vocal"), None)
        vocal_sc = None
        if lead_vocal_id is not None:
            vocal_sc = _ensure_stereo(parsed_audio[lead_vocal_id])

        processed_tracks: Dict[str, np.ndarray] = {}
        for track_id, audio in parsed_audio.items():
            role = role_map.get(track_id, "fx")
            res = apply_predicted_parameters(
                audio,
                sr,
                role,
                predicted_params,
                vocal_sidechain=vocal_sc,
            )
            processed_tracks[track_id] = res.audio

        # 6) Mix tracks (simple summing with headroom)
        mix = None
        for audio in processed_tracks.values():
            x = _ensure_stereo(audio)
            mix = x if mix is None else (mix + x)

        if mix is None:
            raise ValueError("No tracks to mix")

        mix = mix.astype(np.float32)
        peak = float(np.max(np.abs(mix)) + 1e-9)
        if peak > 0.95:
            mix = (mix / peak * 0.95).astype(np.float32)

        # 7) Master processing
        mastered, master_report = apply_master_processing(mix, sr, predicted_params)

        t1 = time.perf_counter()
        total_s = float(t1 - t0)

        logger.info(
            "[AI_MIX] session processed tracks=%d genre=%s lufs=%.2f time=%.3fs",
            len(processed_tracks),
            genre,
            float(master_report.get("integrated_lufs", 0.0)),
            total_s,
        )

        return SessionProcessResult(
            audio=mastered,
            sr=sr,
            role_map=role_map,
            session_vector=session_vector,
            predicted_params=predicted_params,
            master_report=master_report,
            processing_time_s=total_s,
        )
