import json
from dataclasses import asdict
from typing import Optional, Any, Dict

import logging
import os
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.engine import process_audio
from app.analysis import analyze_audio
from app.preset_registry import list_presets
from app.presets import list_all_presets_for_ui
from app.dsp.analysis.intelligent_mixing import (
    analyze_track_and_suggest_chain,
    GenreKey,
    TrackRole,
    FeatureFlow,
)

from app.dsp.parameter_model import ParameterPredictor
from app.dsp.adaptive_engine import AdaptiveSessionEngine
from app.dsp.process_pipeline import process_session as process_session_pipeline

logger = logging.getLogger("mixsmvrt_dsp")

app = FastAPI(title="MixSmvrt DSP Engine")


@app.on_event("startup")
def _startup_load_ai_model() -> None:
    """Preload the LightGBM parameter model once per server process.

    This is best-effort so existing single-track endpoints keep working
    even if the session model file isn't present in dev.
    """

    model_path = os.getenv("AI_MODEL_PATH", "models/mix_model.txt")
    require = os.getenv("AI_REQUIRE_MODEL", "false").lower() in {"1", "true", "yes", "on"}
    try:
        predictor = ParameterPredictor(model_path=model_path)
        app.state.parameter_predictor = predictor
        app.state.session_engine = AdaptiveSessionEngine(predictor)
        logger.info("[AI_MIX] Loaded LightGBM model from %s", model_path)
    except Exception as exc:
        app.state.parameter_predictor = None
        app.state.session_engine = None
        if require:
            logger.exception("[AI_MIX] Model preload failed and AI_REQUIRE_MODEL=true: %s", exc)
            raise
        logger.warning("[AI_MIX] Model preload skipped/failed (%s): %s", model_path, exc)

# Allow the Vercel studio and local development to call this DSP service
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://mixsmvrt.vercel.app",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Lightweight health endpoint for uptime checks.

    Returns a static JSON payload so Render and external monitors can verify
    that the DSP service is up without exercising the full processing stack.
    """

    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Return analysis stats and reference-based preset overrides."""
    return analyze_audio(file)


@app.post("/analyze-track")
async def analyze_track(
    file: UploadFile = File(...),
    track_role: str = Form("vocal_lead"),
    genre: Optional[str] = Form(None),
    flow: str = Form("mixing_only"),
    beat_file: Optional[UploadFile] = File(default=None),
    job_id: Optional[str] = Form(None),
    track_id: Optional[str] = Form(None),
):
    """Intelligent per-track analysis for AI-assisted mixing.

    This endpoint does *not* render audio. It inspects the uploaded
    track (and optionally a beat reference) and returns:
    - rich analysis metrics (loudness, spectrum, dynamics, stereo, masking)
    - a suggested plugin_chain suitable for the Studio UI.
    """

    # Decode main track
    try:
        audio, sr = sf.read(file.file)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail=f"Failed to read audio: {exc}") from exc

    # Optional beat for masking detection
    beat_audio = None
    if beat_file is not None:
        try:
            beat_audio, _ = sf.read(beat_file.file)
        except Exception:
            beat_audio = None

    normalized_role: TrackRole
    tr = (track_role or "").lower().strip()
    if tr in {"beat", "drums"}:
        normalized_role = "beat"
    elif tr in {"bg", "background", "vocal_bg", "background_vocal"}:
        normalized_role = "vocal_bg"
    elif tr in {"adlib", "adlibs", "vocal_adlib"}:
        normalized_role = "vocal_adlib"
    else:
        normalized_role = "vocal_lead"

    normalized_genre: GenreKey
    g = (genre or "generic").lower().strip()
    if g in {"afrobeat", "afrobeats"}:
        normalized_genre = "afrobeat"
    elif g in {"trap_dancehall", "trap-dancehall"}:
        normalized_genre = "trap_dancehall"
    elif g == "dancehall":
        normalized_genre = "dancehall"
    elif g in {"hiphop", "hip-hop"}:
        normalized_genre = "hiphop"
    elif g == "rap":
        normalized_genre = "rap"
    elif g in {"rnb", "r&b"}:
        normalized_genre = "rnb"
    elif g == "reggae":
        normalized_genre = "reggae"
    else:
        normalized_genre = "generic"

    normalized_flow: FeatureFlow
    f = (flow or "mixing_only").lower().strip()
    if f in {"audio_cleanup", "cleanup"}:
        normalized_flow = "audio_cleanup"
    elif f in {"mix_only", "mixing_only", "mix"}:
        normalized_flow = "mixing_only"
    elif f in {"mix_master", "mix+master", "mix_and_master"}:
        normalized_flow = "mix_master"
    elif f in {"master_only", "mastering_only", "master"}:
        normalized_flow = "mastering_only"
    elif f in {"beat_only", "beat"}:
        normalized_flow = "beat_only"
    else:
        normalized_flow = "mixing_only"

    result = analyze_track_and_suggest_chain(
        audio=audio,
        sr=int(sr),
        role=normalized_role,
        genre=normalized_genre,
        flow=normalized_flow,
        beat_audio_for_masking=beat_audio,
    )

    # Attach job/track identifiers for easier storage in processing_jobs
    if job_id is not None:
        result["job_id"] = job_id
    if track_id is not None:
        result["track_id"] = track_id

    return result


@app.post("/process")
async def process(
    file: UploadFile = File(...),
    track_type: str = Form("vocal"),
    preset: str = Form("clean_vocal"),
    genre: str | None = Form(None),
    reference_profile: str | None = Form(None),
    target: str | None = Form(None),
    gender: str | None = Form(None),
    throw_fx_mode: str | None = Form(None),
    session_key: str | None = Form(None),
    session_scale: str | None = Form(None),
    plugin_chain: str | None = Form(None),
    job_id: str | None = Form(None),
    track_id: str | None = Form(None),
    track_role: str | None = Form(None),
    vocal_preset: str | None = Form(None),
):
    """Process an uploaded audio file with the given track type + preset.

    Optionally accepts a ``reference_profile`` JSON string containing
    ``preset_overrides`` generated by the /analyze endpoint.
    """

    overrides_dict = None
    if reference_profile:
        try:
            parsed = json.loads(reference_profile)
            # Frontend typically passes the ``preset_overrides`` object.
            overrides_dict = parsed.get("preset_overrides", parsed)
        except Exception:  # pragma: no cover - defensive parsing
            overrides_dict = None

    plugin_chain_overrides = None
    if plugin_chain:
        try:
            plugin_chain_overrides = json.loads(plugin_chain)
        except Exception:  # pragma: no cover - defensive parsing
            plugin_chain_overrides = None

    output_paths: Dict[str, Any] | str
    try:
        output_paths = process_audio(
            file,
            track_type,
            preset,
            genre,
            gender,
            overrides_dict,
            target,
            throw_fx_mode=throw_fx_mode,
            session_key=session_key,
            session_scale=session_scale,
            plugin_chain=plugin_chain_overrides,
            job_id=job_id,
            track_id=track_id,
            track_role=track_role,
            vocal_preset=vocal_preset,
        )
    except MemoryError as exc:  # pragma: no cover - defensive
        # Explicitly surface out-of-memory conditions so orchestrators can
        # distinguish them from generic DSP failures.
        logger.exception("[DSP] MemoryError while processing job_id=%s track_id=%s", job_id, track_id)
        detail: Dict[str, Any] = {
            "error": "DSP_MEMORY_ERROR",
            "message": str(exc),
        }
        if job_id is not None:
            detail["job_id"] = job_id
        if track_id is not None:
            detail["track_id"] = track_id
        raise HTTPException(status_code=500, detail=detail) from exc
    except Exception as exc:  # pragma: no cover - defensive, logs via HTTP detail
        # Surface a more descriptive error than the default "Internal Server Error"
        # so upstream services and the studio UI can see what went wrong.
        logger.exception("[DSP] Processing failed job_id=%s track_id=%s: %s", job_id, track_id, exc)
        detail = {
            "error": "DSP_PROCESSING_FAILED",
            "message": str(exc),
        }
        if job_id is not None:
            detail["job_id"] = job_id
        if track_id is not None:
            detail["track_id"] = track_id
        raise HTTPException(status_code=500, detail=detail) from exc
    finally:
        # Ensure file handles are released promptly so each request only
        # consumes memory for the lifetime of a single track.
        try:
            file.file.close()
        except Exception:  # pragma: no cover - best effort
            pass
    # Normalise outputs so callers always have a primary output_file (WAV)
    # while also exposing a richer output_files map for multi-format exports.
    if isinstance(output_paths, dict):
        wav_path = output_paths.get("wav")
        mp3_path = output_paths.get("mp3")
    else:  # backwards-compat: older engine versions may return a single string
        wav_path = output_paths
        mp3_path = None

    response = {
        "status": "processed",
        # Primary output remains the WAV path for backwards compatibility.
        "output_file": wav_path,
        # New multi-format map used by the studio for WAV/MP3 downloads.
        "output_files": {
            "wav": wav_path,
            "mp3": mp3_path,
        },
        "track_type": track_type,
        "preset": preset,
        "genre": genre,
        "gender": gender,
    }

    # Surface optional metrics and context if the engine provided them.
    if isinstance(output_paths, dict):
        if "lufs" in output_paths:
            response["lufs"] = output_paths.get("lufs")
        if "true_peak" in output_paths:
            response["true_peak"] = output_paths.get("true_peak")
        if "plugin_chain" in output_paths:
            response["plugin_chain"] = output_paths.get("plugin_chain")
        # Intelligent analysis + suggested plugin chain for the Studio UI.
        if "intelligent_analysis" in output_paths:
            response["intelligent_analysis"] = output_paths.get("intelligent_analysis")
        if "intelligent_plugin_chain" in output_paths:
            response["intelligent_plugin_chain"] = output_paths.get("intelligent_plugin_chain")
        if "job_id" in output_paths:
            response["job_id"] = output_paths.get("job_id")
        if "track_id" in output_paths:
            response["track_id"] = output_paths.get("track_id")
        if "track_role" in output_paths:
            response["track_role"] = output_paths.get("track_role")

    return response


@app.post("/process-session")
async def process_session(
    files: list[UploadFile] = File(...),
    genre: str = Form("other"),
    track_ids: str | None = Form(None),
):
    """Process a multi-track session with the adaptive AI mix engine.

    Inputs:
    - `files`: multiple audio files (stems)
    - `track_ids`: optional JSON array or comma-separated ids matching `files`
    - `genre`: genre string for the session vector

    Output:
    - S3 URL (or local path fallback) to the rendered WAV.
    """

    engine = getattr(app.state, "session_engine", None)
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "AI_MODEL_NOT_AVAILABLE",
                "message": "Session AI mix model is not loaded. Set AI_MODEL_PATH and deploy the model file.",
            },
        )

    ids: list[str] | None = None
    if track_ids:
        # Accept JSON array or comma-separated
        raw = track_ids.strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                ids = [str(x) for x in parsed]
        except Exception:
            ids = [s.strip() for s in raw.split(",") if s.strip()]

    if ids is not None and len(ids) != len(files):
        raise HTTPException(status_code=400, detail="track_ids must match number of files")

    track_audio_map: dict[str, tuple[Any, int]] = {}
    try:
        for i, f in enumerate(files):
            try:
                audio, sr = sf.read(f.file)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Failed to read audio '{f.filename}': {exc}") from exc

            track_id = (ids[i] if ids is not None else None) or (f.filename or f"track_{i}")
            track_audio_map[str(track_id)] = (audio, int(sr))
    finally:
        for f in files:
            try:
                f.file.close()
            except Exception:
                pass

    try:
        return process_session_pipeline(track_audio_map, genre=genre, engine=engine)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("[AI_MIX] Session processing failed: %s", exc)
        raise HTTPException(status_code=500, detail={"error": "AI_SESSION_PROCESSING_FAILED", "message": str(exc)}) from exc


@app.get("/presets")
async def list_available_presets(kind: str | None = Query(default=None)):
    """Return the catalog of presets known to the DSP engine.

    Optional query parameter ``kind`` can be one of ``vocal``, ``mix`` or
    ``master`` to filter the list. The response is structured for direct
    consumption by the studio or admin UI.
    """

    normalized: str | None
    if kind in {"vocal", "mix", "master"}:
        normalized = kind
    else:
        normalized = None

    presets = list_presets(kind=normalized)  # type: ignore[arg-type]
    return {
        "presets": [asdict(p) for p in presets],
    }


@app.get("/vocal-presets")
async def list_vocal_presets(flow: str | None = Query(default=None), genre: str | None = Query(default=None)):
    """Expose the full catalog of vocal tone presets.

    Returns entries that can be used directly by the studio UI to
    populate a "vocal tone" selector. Each preset is given a stable
    ``id`` of the form ``"{flow_type}:{genre}:{preset_name}"`` that
    can be sent back as the ``vocal_preset`` form field to /process.
    """

    if flow and genre:
        rows = list_all_presets_for_ui()
        filtered = [
            r
            for r in rows
            if r.get("flow_type") == flow and r.get("genre") == genre
        ]
    else:
        filtered = list_all_presets_for_ui()

    presets = []
    for row in filtered:
        ft = row.get("flow_type") or "mixing_only"
        g = row.get("genre") or "generic"
        name = row.get("preset_name") or ""
        if not name:
            continue
        presets.append(
            {
                "id": f"{ft}:{g}:{name}",
                "flow_type": ft,
                "genre": g,
                "preset_name": name,
                "ui_subtitle": row.get("ui_subtitle", ""),
            }
        )

    return {"presets": presets}
