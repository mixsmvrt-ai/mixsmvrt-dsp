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

First install the system-level audio tools, then the Python packages.

### System dependencies

The DSP engine relies on:

- **libsndfile** (backend used by `soundfile` for reading/writing audio)
- **ffmpeg** (recommended so you can handle a wide range of formats like mp3, m4a, flac)

Install them on your platform before running the service:

- **Ubuntu / Debian**

  ```bash
  sudo apt-get update
  sudo apt-get install -y ffmpeg libsndfile1
  ```

- **macOS (Homebrew)**

  ```bash
  brew install ffmpeg libsndfile
  ```

- **Windows (chocolatey)**

  ```powershell
  choco install ffmpeg
  choco install libsndfile
  ```

  Or install prebuilt binaries from the official ffmpeg/libsndfile sites and ensure they are on your `PATH`.

### Python packages

From the `mixsmvrt-dsp` folder:

```bash
pip install -r requirements.txt
```

## Run locally

From the `mixsmvrt-dsp` folder:

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000` by default.
