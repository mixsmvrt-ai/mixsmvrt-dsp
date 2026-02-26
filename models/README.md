# MixSmvrt AI Mix Model

Place the pretrained LightGBM model file here:

- `mix_model.txt` (recommended LightGBM Booster text format), or
- `mix_model.pkl`

The DSP service loads this at startup via `AI_MODEL_PATH` (defaults to `models/mix_model.txt`).

Production tip:
- Set `AI_REQUIRE_MODEL=true` to fail startup if the model is missing.
- Set `AI_DEBUG=true` to include vectors/params in `/process-session` responses.
