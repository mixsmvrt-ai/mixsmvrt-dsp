import os
import uuid
from pathlib import Path
from typing import Optional

try:  # Optional dependency handling
    import boto3  # type: ignore
except Exception:  # pragma: no cover - boto3 may not be installed in some envs
    boto3 = None  # type: ignore


def _get_s3_client():
    """Return a configured S3 client or None if not available.

    This keeps the DSP engine flexible: if AWS env vars or boto3 are missing,
    it will fall back to returning local file paths instead of URLs.
    """

    if boto3 is None:
        return None

    bucket = os.getenv("DSP_S3_BUCKET") or os.getenv("S3_BUCKET")
    region = os.getenv("DSP_S3_REGION") or os.getenv("AWS_REGION")
    if not bucket or not region:
        return None

    session = boto3.session.Session()
    return session.client("s3", region_name=region)


def upload_file_to_s3(local_path: str, *, key_prefix: Optional[str] = None) -> str:
    """Upload a local file to S3 and return a URL.

    If S3 is not configured, this simply returns the original local path so
    existing behaviour remains unchanged.
    """

    client = _get_s3_client()
    bucket = os.getenv("DSP_S3_BUCKET") or os.getenv("S3_BUCKET")
    region = os.getenv("DSP_S3_REGION") or os.getenv("AWS_REGION")

    if client is None or not bucket or not region:
        # Fallback: keep using local filesystem path
        return local_path

    path = Path(local_path)
    prefix = (key_prefix or os.getenv("DSP_S3_PREFIX") or "processed/").rstrip("/")
    unique_id = uuid.uuid4().hex
    object_key = f"{prefix}/{unique_id}_{path.name}"

    extra_args = {"ContentType": "audio/wav"}

    client.upload_file(str(path), bucket, object_key, ExtraArgs=extra_args)

    # Optionally remove the local temp file after upload
    try:
        path.unlink()
    except OSError:
        pass

    # Construct a standard virtual-hosted–style S3 URL
    return f"https://{bucket}.s3.{region}.amazonaws.com/{object_key}"
