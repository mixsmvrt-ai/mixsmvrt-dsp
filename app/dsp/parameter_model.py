"""LightGBM parameter predictor for adaptive mixing.

Loads a pretrained model once at server startup and performs fast CPU inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Any

import numpy as np


PARAMETER_KEYS: List[str] = [
    "vocal_eq_high_gain",
    "vocal_compression_ratio",
    "vocal_attack",
    "vocal_release",
    "beat_mid_dip_gain",
    "beat_dip_freq",
    "master_target_lufs",
    "master_multiband_ratio",
    "stereo_widen_amount",
]


@dataclass
class ParameterPredictor:
    model_path: str
    _model: Any = None

    def __post_init__(self) -> None:
        self._load_model()

    def _load_model(self) -> None:
        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"LightGBM model not found at '{self.model_path}'. "
                "Expected a .txt or .pkl file."
            )

        suffix = path.suffix.lower()
        if suffix == ".txt":
            import lightgbm as lgb

            self._model = lgb.Booster(model_file=str(path))
            return

        if suffix == ".pkl":
            import pickle

            with path.open("rb") as f:
                self._model = pickle.load(f)
            return

        raise ValueError(f"Unsupported model format '{suffix}'. Use .txt or .pkl")

    def predict(self, session_vector: np.ndarray) -> Dict[str, float]:
        """Predict DSP parameters from a fixed-length session vector."""

        if self._model is None:
            raise RuntimeError("Model not loaded")

        x = np.asarray(session_vector, dtype=np.float32).reshape(1, -1)

        # Support:
        # - LightGBM Booster (.txt)
        # - A sklearn-like estimator with .predict
        # - A list/tuple of per-parameter models
        raw: Any
        if isinstance(self._model, (list, tuple)):
            preds: List[float] = []
            for m in self._model:
                y = m.predict(x)
                y = np.asarray(y)
                preds.append(float(y.reshape(-1)[0]))
            raw = np.asarray(preds, dtype=np.float32)
        else:
            raw = self._model.predict(x)

        arr = np.asarray(raw)
        if arr.ndim == 2:
            out = arr[0]
        else:
            out = arr.reshape(-1)

        if out.shape[0] != len(PARAMETER_KEYS):
            raise ValueError(
                f"Model output has {out.shape[0]} values; expected {len(PARAMETER_KEYS)}. "
                "Ensure the LightGBM model was trained for these 9 parameters."
            )

        return {k: float(out[i]) for i, k in enumerate(PARAMETER_KEYS)}
