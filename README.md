# MIXSMVRT DSP Engine

Python FastAPI microservice that exposes lightweight DSP processing chains for MIXSMVRT.

## Folder structure

```bash
mixsmvrt-dsp/
  app/
    main.py
    config.py
    models.py
    engine.py
    analysis.py
    presets.py
    chains.py
    processors/
      eq.py
      compressor.py
      deesser.py
      saturation.py
      stereo.py
      limiter.py
      loudness.py
  requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Run locally

From the `mixsmvrt-dsp` folder:

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000` by default.
