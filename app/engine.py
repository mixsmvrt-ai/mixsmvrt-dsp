import tempfile

import soundfile as sf

from app.chains import TRACK_CHAINS
from app.presets import PRESETS
from app.storage import upload_file_to_s3
from app.vocal_presets import (
    dancehall,
    trap_dancehall,
    afrobeat,
    reggae,
    hiphop,
    rnb,
    rap,
)


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


def process_audio(
    file,
    track_type: str,
    preset_name: str,
    genre: str | None = None,
    gender: str | None = None,
    reference_overrides: dict | None = None,
) -> str:
    """Run a simple offline DSP chain over the uploaded audio.

    - Reads the uploaded file via soundfile
    - Looks up the processor chain for the given track type
    - Looks up per-processor params from the chosen preset
    - Writes a processed WAV to a temp file and returns the path
    """

    # Read raw samples
    audio, sr = sf.read(file.file)

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
            # Fallback to gender-agnostic preset.
            vocal_processor = VOCAL_GENRE_PROCESSORS.get(base_key)
        if vocal_processor is not None:
            processed = vocal_processor(audio, sr)
        else:
            # Fallback to the generic vocal chain + preset config.
            chain = TRACK_CHAINS.get(track_type)
            if chain is None:
                raise ValueError(f"Unknown track_type: {track_type}")

            base_preset = PRESETS.get(preset_name, {})
            preset = _merge_preset_with_overrides(
                base_preset,
                (reference_overrides or {}).get(preset_name) if isinstance(reference_overrides, dict) else None,
            )
            processed = audio.copy()
            for processor in chain:
                params = preset.get(processor.__name__, {})
                processed = processor(processed, sr, params)
    else:
        # Non‑vocal tracks use the original chain/preset system.
        chain = TRACK_CHAINS.get(track_type)
        if chain is None:
            raise ValueError(f"Unknown track_type: {track_type}")

        base_preset = PRESETS.get(preset_name, {})
        preset = _merge_preset_with_overrides(
            base_preset,
            (reference_overrides or {}).get(preset_name) if isinstance(reference_overrides, dict) else None,
        )
        processed = audio.copy()
        for processor in chain:
            params = preset.get(processor.__name__, {})
            processed = processor(processed, sr, params)

    out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(out_file.name, processed, sr)

    # Store processed output in S3 when configured; otherwise return local path.
    return upload_file_to_s3(out_file.name)
