import logging
import tempfile
import subprocess

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


logger = logging.getLogger(__name__)


VOCAL_GENRE_PROCESSORS = {
    # Gender‑agnostic keys (for backwards compatibility)
    "dancehall": dancehall.process_vocal,
    "trap_dancehall": trap_dancehall.process_vocal,
    "trap-dancehall": trap_dancehall.process_vocal,
    "afrobeat": afrobeat.process_vocal,
    "reggae": reggae.process_vocal,
    "hiphop": hiphop.process_vocal,
    "hip-hop": hiphop.process_vocal,
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
    "hiphop_male": getattr(hiphop, "process_vocal_male", hiphop.process_vocal),
    "hiphop_female": getattr(hiphop, "process_vocal_female", hiphop.process_vocal),
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
    "hiphop_bg": getattr(hiphop, "process_vocal_background", hiphop.process_vocal),
    "hiphop_bg_male": getattr(hiphop, "process_vocal_background", hiphop.process_vocal),
    "hiphop_bg_female": getattr(hiphop, "process_vocal_background", hiphop.process_vocal),
    "hiphop_adlib": getattr(hiphop, "process_vocal_adlib", hiphop.process_vocal),
    "hiphop_adlib_male": getattr(hiphop, "process_vocal_adlib", hiphop.process_vocal),
    "hiphop_adlib_female": getattr(hiphop, "process_vocal_adlib", hiphop.process_vocal),
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
) -> dict[str, str | None]:
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

    try:
        analysis_features = analyze_mono_signal(audio, sr)
        logger.debug("[DSP] Analysis completed: %s", analysis_features)
    except Exception as exc:  # pragma: no cover - defensive path
        logger.warning("[DSP] Analysis failed (Essentia/numpy): %s", exc)

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

    # If this is a vocal track and a genre or matching preset is provided,
    # route through the dedicated genre-specific vocal chains.
    if track_type == "vocal":
        base_key = (genre or preset_name).lower()
        gender_key = (gender or "").lower().strip()

        vocal_processor = None
        if gender_key:
            # Prefer explicit gender-specific variant if available, e.g. "dancehall_male".
            combined = f"{base_key}_{gender_key}"
            vocal_processor = VOCAL_GENRE_PROCESSORS.get(combined)

        if vocal_processor is None:
            # Fallback to gender-agnostic preset, including bg/adlib variants.
            vocal_processor = VOCAL_GENRE_PROCESSORS.get(base_key)
        if vocal_processor is not None:
            # Genre-aware, pedalboard-centric vocal chains stay in charge
            # for named presets like trap_dancehall, hiphop, etc.
            processed = vocal_processor(audio, sr)
        else:
            # Generic vocal path: upgrade to the dynamic Pedalboard chain
            # driven by analysis + WORLD pitch metrics.
            processed = process_with_dynamic_chain(
                audio=audio,
                sr=sr,
                preset_key=preset_name,
                track_type=track_type,
                analysis=analysis_features or {},
                pitch_info=pitch_profile or {},
                genre=genre,
            )
    else:
        # Non‑vocal tracks: use a dedicated pedalboard-based bus chain for
        # beats/masters, and fall back to the original stub chain for others.
        if track_type in {"beat", "master"}:
            master_overrides = None
            if isinstance(reference_overrides, dict):
                master_overrides = reference_overrides.get("streaming_master")

            processed = process_beat_or_master(audio, sr, master_overrides)
        else:
            # For other non-vocal tracks, use the upgraded dynamic
            # Pedalboard chain, but still allow the backend to hint
            # whether this should be treated as a dedicated beat bus.
            processed = process_with_dynamic_chain(
                audio=audio,
                sr=sr,
                preset_key=preset_name,
                track_type=track_type,
                analysis=analysis_features or {},
                pitch_info=None,
                genre=genre,
            )

    # Optional throw FX for vocals – applied after the core vocal chain so
    # throws sit around the already-shaped vocal.
    if track_type == "vocal" and throw_fx_mode:
        processed = apply_throw_fx_to_vocal(processed, sr, throw_fx_mode)

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

    logger.debug("[DSP] Processing finished: wav=%s mp3=%s", wav_url_or_path, mp3_url_or_path)

    return {"wav": wav_url_or_path, "mp3": mp3_url_or_path}
