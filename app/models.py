"""Pydantic-style response models (optional helpers).

FastAPI will happily return plain dicts, but these are useful
for documentation and editor hints if you want to import them
into `main.py` later.
"""

from typing import Optional

from pydantic import BaseModel


class AnalysisResponse(BaseModel):
    sample_rate: int
    rms: float
    peak: float
    duration: float


class ProcessResponse(BaseModel):
    status: str
    output_file: str
    track_type: str
    preset: str
    message: Optional[str] = None
