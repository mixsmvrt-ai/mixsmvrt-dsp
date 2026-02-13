import logging
import tempfile
import subprocess

import numpy as np
import soundfile as sf

from app.beat_mastering import process_beat_or_master
from app.storage import upload_file_to_s3
from app.throw_fx import apply_throw_fx_to_vocal
from app.vocal_presets.tuning import apply_pitch_correction
from app.vocal_presets import (
    dancehall,
    trap_dancehall,
    afrobeat,
    reggae,
    hiphop,
    rnb,
    rap,
)
from app.dsp.analysis.essentia_analysis import analyze_mono_signal
from app.dsp.analysis.pitch_world import analyze_pitch_world
from app.dsp.chains.build_chain import process_with_dynamic_chain
from app.dsp.analysis.intelligent_mixing import (
    analyze_track_and_suggest_chain,
    GenreKey,
    TrackRole,
    FeatureFlow,
)
from app.dsp.adaptive.pipeline import build_adaptive_pipeline
from app.dsp_engine import (
    process_audio_cleanup as engine_audio_cleanup,
    process_mixing_only as engine_mixing_only,
    process_mix_master as engine_mix_master,
    process_mastering_only as engine_mastering_only,
)


logger = logging.getLogger(__name__)


VOCAL_GENRE_PROCESSORS = {
    # Gender‑agnostic keys (for backwards compatibility)
    "dancehall": dancehall.process_vocal,
    "trap_dancehall": trap_dancehall.process_vocal,
    "trap-dancehall": trap_dancehall.process_vocal,
    "afrobeat": afrobeat.process_vocal,
    "reggae": reggae.process_vocal,
    "hiphop": getattr(hiphop, "process_vocal", lambda audio, sr: audio),
    "hip-hop": getattr(hiphop, "process_vocal", lambda audio, sr: audio),
    "rnb": rnb.process_vocal,
    "r&b": rnb.process_vocal,
    "rap": rap.process_vocal,
    # Explicit male/female variants
    "dancehall_male": getattr(dancehall, "process_vocal_male", dancehall.process_vocal),
    "dancehall_female": getattr(dancehall, "process_vocal_female", dancehall.process_vocal),
    "trap_dancehall_male": getattr(trap_dancehall, "process_vocal_male", trap_dancehall.process_vocal),
    "trap_dancehall_female": getattr(trap_dancehall, "process_vocal_female", trap_dancehall.process_vocal),
    "afrobeat_male": getattr(afrobeat, "process_vocal_male", afrobeat.process_vocal),
    "afrobeat_female": getattr(afrobeat, "process_vocal_female", afrobeat.process_vocal),
    "reggae_male": getattr(reggae, "process_vocal_male", reggae.process_vocal),
    "reggae_female": getattr(reggae, "process_vocal_female", reggae.process_vocal),
    "hiphop_male": getattr(hiphop, "process_vocal_male", getattr(hiphop, "process_vocal", lambda audio, sr: audio)),
    "hiphop_female": getattr(hiphop, "process_vocal_female", getattr(hiphop, "process_vocal", lambda audio, sr: audio)),
    "rnb_male": getattr(rnb, "process_vocal_male", rnb.process_vocal),
    "rnb_female": getattr(rnb, "process_vocal_female", rnb.process_vocal),
    "rap_male": getattr(rap, "process_vocal_male", rap.process_vocal),
    "rap_female": getattr(rap, "process_vocal_female", rap.process_vocal),
    # Background / adlib variants – these rely on preset names like
    # ``trap_dancehall_bg``, ``hiphop_adlib``, etc., which the studio
    # passes for background and adlib vocal tracks.
    "trap_dancehall_bg": getattr(trap_dancehall, "process_vocal_background", trap_dancehall.process_vocal),
    "trap_dancehall_bg_male": getattr(trap_dancehall, "process_vocal_background", trap_dancehall.process_vocal),
    "trap_dancehall_bg_female": getattr(trap_dancehall, "process_vocal_background", trap_dancehall.process_vocal),
    "trap_dancehall_adlib": getattr(trap_dancehall, "process_vocal_adlib", trap_dancehall.process_vocal),
    "trap_dancehall_adlib_male": getattr(trap_dancehall, "process_vocal_adlib", trap_dancehall.process_vocal),
    "trap_dancehall_adlib_female": getattr(trap_dancehall, "process_vocal_adlib", trap_dancehall.process_vocal),
    "hiphop_bg": getattr(hiphop, "process_vocal_background", getattr(hiphop, "process_vocal", lambda audio, sr: audio)),
    "hiphop_bg_male": getattr(hiphop, "process_vocal_background", getattr(hiphop, "process_vocal", lambda audio, sr: audio)),
    "hiphop_bg_female": getattr(hiphop, "process_vocal_background", getattr(hiphop, "process_vocal", lambda audio, sr: audio)),
    "hiphop_adlib": getattr(hiphop, "process_vocal_adlib", getattr(hiphop, "process_vocal", lambda audio, sr: audio)),
    "hiphop_adlib_male": getattr(hiphop, "process_vocal_adlib", getattr(hiphop, "process_vocal", lambda audio, sr: audio)),
    "hiphop_adlib_female": getattr(hiphop, "process_vocal_adlib", getattr(hiphop, "process_vocal", lambda audio, sr: audio)),
    "afrobeat_bg": getattr(afrobeat, "process_vocal_background", afrobeat.process_vocal),
    "afrobeat_bg_male": getattr(afrobeat, "process_vocal_background", afrobeat.process_vocal),
    "afrobeat_bg_female": getattr(afrobeat, "process_vocal_background", afrobeat.process_vocal),
    "afrobeat_adlib": getattr(afrobeat, "process_vocal_adlib", afrobeat.process_vocal),
    "afrobeat_adlib_male": getattr(afrobeat, "process_vocal_adlib", afrobeat.process_vocal),
    "afrobeat_adlib_female": getattr(afrobeat, "process_vocal_adlib", afrobeat.process_vocal),
    "rnb_bg": getattr(rnb, "process_vocal_background", rnb.process_vocal),
    "rnb_bg_male": getattr(rnb, "process_vocal_background", rnb.process_vocal),
    "rnb_bg_female": getattr(rnb, "process_vocal_background", rnb.process_vocal),
    "rnb_adlib": getattr(rnb, "process_vocal_adlib", rnb.process_vocal),
    "rnb_adlib_male": getattr(rnb, "process_vocal_adlib", rnb.process_vocal),
    "rnb_adlib_female": getattr(rnb, "process_vocal_adlib", rnb.process_vocal),
    "reggae_bg": getattr(reggae, "process_vocal_background", reggae.process_vocal),
    "reggae_bg_male": getattr(reggae, "process_vocal_background", reggae.process_vocal),
    "reggae_bg_female": getattr(reggae, "process_vocal_background", reggae.process_vocal),
    "reggae_adlib": getattr(reggae, "process_vocal_adlib", reggae.process_vocal),
    "reggae_adlib_male": getattr(reggae, "process_vocal_adlib", reggae.process_vocal),
    "reggae_adlib_female": getattr(reggae, "process_vocal_adlib", reggae.process_vocal),
    "rap_bg": getattr(rap, "process_vocal_background", rap.process_vocal),
    "rap_bg_male": getattr(rap, "process_vocal_background", rap.process_vocal),
    "rap_bg_female": getattr(rap, "process_vocal_background", rap.process_vocal),
    "rap_adlib": getattr(rap, "process_vocal_adlib", rap.process_vocal),
    "rap_adlib_male": getattr(rap, "process_vocal_adlib", rap.process_vocal),
    "rap_adlib_female": getattr(rap, "process_vocal_adlib", rap.process_vocal),
}


def _merge_preset_with_overrides(base: dict, overrides: dict | None) -> dict:
    """Deep-merge per-processor overrides into a base preset dict."""

    if not overrides:
        return base

    merged = {**base}
    for processor_id, params_override in overrides.items():
        base_params = dict(merged.get(processor_id, {}))
        base_params.update(params_override or {})
        merged[processor_id] = base_params
    return merged


def _apply_beat_safe_overrides(preset: dict) -> dict:
    """Return a beat-safe version of a preset.

    This lightly reduces compression and saturation to better preserve
    dynamics on stereo beats and instrumentals while keeping the
    original character as much as possible.
    """

    safe = {k: dict(v) for k, v in preset.items()}
    for processor_id, params in safe.items():
        ratio = params.get("ratio")
        if isinstance(ratio, (int, float)):
            params["ratio"] = max(1.0, min(ratio, 3.0))

        drive = params.get("drive")
        if isinstance(drive, (int, float)):
            params["drive"] = drive * 0.5

        saturation = params.get("saturation")
        if isinstance(saturation, (int, float)):
            params["saturation"] = saturation * 0.6

        makeup = params.get("makeup_gain")
        if isinstance(makeup, (int, float)):
            params["makeup_gain"] = makeup * 0.5

    return safe


def process_audio(
    file,
    track_type: str,
    preset_name: str,
    genre: str | None = None,
    gender: str | None = None,
    reference_overrides: dict | None = None,
    target: str | None = None,
    throw_fx_mode: str | None = None,
    session_key: str | None = None,
    session_scale: str | None = None,
    plugin_chain: dict | None = None,
    job_id: str | None = None,
    track_id: str | None = None,
    track_role: str | None = None,
) -> dict:
    """Run a simple offline DSP chain over the uploaded audio.

    - Reads the uploaded file via soundfile
    - Looks up the processor chain for the given track type
    - Looks up per-processor params from the chosen preset
    - Writes a processed WAV to a temp file and returns the path
    """

    # Read raw samples once; downstream layers must not mutate ``audio``
    # in-place to keep analysis and processing well separated.
    audio, sr = sf.read(file.file)

    # ---------------------------------
    # 1) ANALYSIS LAYER
    # ---------------------------------
    analysis_features = None
    pitch_profile = None
    intelligent_analysis: dict | None = None
    adaptive_config: dict | None = None

    try:
        analysis_features = analyze_mono_signal(audio, sr)
        logger.debug("[DSP] Analysis completed: %s", analysis_features)
    except Exception as exc:  # pragma: no cover - defensive path
        logger.warning("[DSP] Analysis failed (Essentia/numpy): %s", exc)

    # Build the high-level adaptive pipeline configuration. This is
    # pure analysis/decision logic and does not render audio.
    try:
        adaptive_config = build_adaptive_pipeline(
            audio=audio,
            sr=int(sr),
            user_tag=track_role or track_type,
            genre=genre,
            gender=gender,
        )
    except Exception as exc:  # pragma: no cover - defensive path
        logger.warning("[DSP] Adaptive pipeline build failed: %s", exc)

    # High-level intelligent analysis + plugin suggestions for the
    # Studio UI and processing_jobs table. This is non-destructive and
    # does not alter the DSP chain; it only returns analysis metadata
    # and an AI-suggested plugin_chain.
    try:
        # Infer a TrackRole from the incoming track_type / track_role.
        normalized_role: TrackRole
        role_key = (track_role or track_type or "").lower().strip()
        if track_type == "beat" or "beat" in role_key:
            normalized_role = "beat"
        elif any(key in role_key for key in ["bg", "background"]):
            normalized_role = "vocal_bg"
        elif "adlib" in role_key:
            normalized_role = "vocal_adlib"
        else:
            normalized_role = "vocal_lead"

        # Map free-text genre to a GenreKey understood by the analysis.
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

        # Infer the high-level flow for this call.
        normalized_flow: FeatureFlow
        if target == "full_mix" and track_type in {"master", "beat"}:
            normalized_flow = "mix_master"
        elif target == "full_mix":
            normalized_flow = "mastering_only"
        elif target == "beat" and track_type in {"beat", "master"}:
            normalized_flow = "beat_only"
        else:
            normalized_flow = "mixing_only"

        intelligent_analysis = analyze_track_and_suggest_chain(
            audio=audio,
            sr=int(sr),
            role=normalized_role,
            genre=normalized_genre,
            flow=normalized_flow,
            beat_audio_for_masking=None,
        )
    except Exception as exc:  # pragma: no cover - defensive path
        logger.warning("[DSP] Intelligent mixing analysis failed: %s", exc)

    # WORLD-based pitch analysis is only relevant for vocals.
    if track_type == "vocal":
        try:
            pitch_profile = analyze_pitch_world(audio, sr)
            logger.debug("[DSP] Pitch analysis completed: %s", pitch_profile)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("[DSP] WORLD pitch analysis failed: %s", exc)

    # If this is a vocal track and the studio provided a key/scale,
    # run a gentle key-aware pitch correction pass before the main
    # vocal processing chain. This uses WORLD-based analysis when
    # available and falls back gracefully when not.
    if track_type == "vocal" and session_key:
        try:
            audio = apply_pitch_correction(
                audio,
                sr,
                session_key=session_key,
                session_scale=session_scale,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("[DSP] Key-aware pitch correction failed: %s", exc)

    # Core processing is now delegated to the new dsp_engine pipelines.
    audio_for_processing = audio.astype(np.float32)

    # Extract simple BPM estimate if available from adaptive_config.analysis
    bpm_value: float | None = None
    if isinstance(adaptive_config, dict):
        try:
            analysis_section = adaptive_config.get("analysis") or {}
            bpm_candidate = analysis_section.get("bpm") or analysis_section.get("tempo")
            if isinstance(bpm_candidate, (int, float)):
                bpm_value = float(bpm_candidate)
        except Exception:  # pragma: no cover - defensive
            bpm_value = None

    # Normalised role/genre strings for FX adaptivity
    normalized_role_str: str | None = None
    if isinstance(adaptive_config, dict):
        role_val = adaptive_config.get("role")
        if isinstance(role_val, str):
            normalized_role_str = role_val
    if not normalized_role_str:
        normalized_role_str = track_role or track_type

    normalized_genre_str: str | None = None
    if genre is not None:
        normalized_genre_str = genre

    try:
        if target == "cleanup" or track_type == "cleanup":
            processed, core_report = engine_audio_cleanup(preset_name or "track", audio_for_processing, sr)
        elif target == "full_mix" and track_type in {"master", "beat"}:
            # mix + master flow on a stereo bus
            processed, core_report = engine_mix_master(
                preset_name or "mix_master_bus",
                audio_for_processing,
                sr,
                is_vocal=False,
            )
        elif target == "master_only" or track_type == "master":
            processed, core_report = engine_mastering_only(preset_name or "master_bus", audio_for_processing, sr)
        else:
            # default: mixing_only per-track processing
            is_vocal_track = track_type == "vocal"
            processed, core_report = engine_mixing_only(
                preset_name or "track",
                audio_for_processing,
                sr,
                is_vocal=is_vocal_track,
                role=normalized_role_str,
                bpm=bpm_value,
                genre=normalized_genre_str,
            )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("[DSP] dsp_engine pipeline failed, falling back to legacy chain: %s", exc)
        processed = audio_for_processing
        core_report = None

    # ---------------------------------
    # 1c) SAFETY LAYER (sanity & peak guard)
    # ---------------------------------
    # Ensure the processed signal is finite and not wildly clipped before
    # any further processing/export. This helps avoid harsh static or
    # intermittent distortion caused by NaNs/Infs or extreme gains in
    # individual presets.
    if isinstance(processed, np.ndarray) and processed.size:
        processed = np.nan_to_num(processed.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        peak = float(np.max(np.abs(processed)))
        if np.isfinite(peak) and peak > 1.0:
            # Soft-clip by normalising down to just under full scale.
            processed = processed / (peak * 1.01)

    # Optional throw FX for vocals  applied after the core vocal chain so
    # throws sit around the already-shaped vocal.
    if track_type == "vocal" and throw_fx_mode:
        processed = apply_throw_fx_to_vocal(processed, sr, throw_fx_mode)

    # ---------------------------------
    # 2b) MEASUREMENT LAYER (loudness + peaks)
    # ---------------------------------
    lufs_value: float | None = None
    true_peak_value: float | None = None

    try:
        if processed.size and core_report is not None:
            lufs_value = core_report.loudness_after
            true_peak_value = core_report.true_peak_after
    except Exception as exc:  # pragma: no cover - defensive path
        logger.warning("[DSP] Loudness/peak measurement failed: %s", exc)

    # ---------------------------------
    # 3) FINALISATION LAYER (render + export)
    # ---------------------------------
    # Always render a high-quality WAV first.
    out_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(out_wav.name, processed, sr)
    wav_url_or_path = upload_file_to_s3(out_wav.name)

    # Best-effort MP3 encode using ffmpeg when available. If encoding fails
    # for any reason, we still return the WAV so the caller has a valid
    # output to work with.
    mp3_url_or_path: str | None = None
    try:
        out_mp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        cmd = [
            "ffmpeg",
            "-y",  # overwrite without prompting
            "-i",
            out_wav.name,
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "320k",
            out_mp3.name,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        mp3_url_or_path = upload_file_to_s3(out_mp3.name)
    except Exception:
        mp3_url_or_path = None

    result: dict = {"wav": wav_url_or_path, "mp3": mp3_url_or_path}

    # Attach metrics and context for callers that care about them while
    # keeping the original keys for backwards compatibility.
    if lufs_value is not None:
        result["lufs"] = lufs_value
    if true_peak_value is not None:
        result["true_peak"] = true_peak_value
    if core_report is not None:
        result["processing_report"] = {
            "track_name": core_report.track_name,
            "processing_chain": core_report.processing_chain,
            "parameter_values": core_report.parameter_values,
            "loudness_before": core_report.loudness_before,
            "loudness_after": core_report.loudness_after,
            "true_peak_after": core_report.true_peak_after,
        }
    if intelligent_analysis is not None:
        # Attach both analysis metrics and the AI-suggested chain
        # without affecting the actual rendered audio.
        result["intelligent_analysis"] = intelligent_analysis.get("analysis")
        result["intelligent_plugin_chain"] = intelligent_analysis.get("plugin_chain")
    if adaptive_config is not None:
        result["adaptive_config"] = adaptive_config
    if plugin_chain is not None:
        result["plugin_chain"] = plugin_chain
    if target is not None:
        result["target"] = target
    if track_role is not None:
        result["track_role"] = track_role
    if job_id is not None:
        result["job_id"] = job_id
    if track_id is not None:
        result["track_id"] = track_id

    logger.info(
        "[DSP] Processing finished job_id=%s track_id=%s track_type=%s role=%s preset=%s genre=%s lufs=%s true_peak=%s wav=%s mp3=%s",
        job_id,
        track_id,
        track_type,
        track_role,
        preset_name,
        genre,
        lufs_value,
        true_peak_value,
        wav_url_or_path,
        mp3_url_or_path,
    )

    return result


def process_track(
    audio_path: str,
    track_role: str,
    preset: str,
    *,
    track_type: str | None = None,
    genre: str | None = None,
    gender: str | None = None,
    reference_overrides: dict | None = None,
    target: str | None = None,
    throw_fx_mode: str | None = None,
    session_key: str | None = None,
    session_scale: str | None = None,
    plugin_chain: dict | None = None,
    job_id: str | None = None,
    track_id: str | None = None,
) -> dict:
    """Convenience entry point for single-track processing by file path.

    This wraps ``process_audio`` so other Python callers can work with the
    same engine using a simple (audio_path, track_role, preset) API.
    """

    normalized_role = (track_role or "").lower()

    inferred_type = track_type
    if not inferred_type:
        if "beat" in normalized_role:
            inferred_type = "beat"
        elif "master" in normalized_role:
            inferred_type = "master"
        else:
            # Default to vocal-style processing for all other roles
            inferred_type = "vocal"

    class _FileLike:
        def __init__(self, path: str) -> None:
            self._fh = open(path, "rb")
            self.file = self._fh

        def close(self) -> None:
            try:
                self._fh.close()
            except Exception:  # pragma: no cover - defensive
                pass

    wrapper = _FileLike(audio_path)
    try:
        raw_result = process_audio(
            file=wrapper,
            track_type=inferred_type,
            preset_name=preset,
            genre=genre,
            gender=gender,
            reference_overrides=reference_overrides,
            target=target,
            throw_fx_mode=throw_fx_mode,
            session_key=session_key,
            session_scale=session_scale,
            plugin_chain=plugin_chain,
            job_id=job_id,
            track_id=track_id,
            track_role=track_role,
        )
    finally:
        wrapper.close()

    processed_audio_path = raw_result.get("wav") or raw_result.get("output_file")

    return {
        "track_id": track_id,
        "processed_audio_path": processed_audio_path,
        "plugin_chain": raw_result.get("plugin_chain"),
        "lufs": raw_result.get("lufs"),
        "true_peak": raw_result.get("true_peak"),
    }
