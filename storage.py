#!/usr/bin/env python3
"""
storage.py
==========
A thin storage adapter so the rest of the code doesn't care whether files live
on your laptop or in an S3 bucket. Flip one environment variable and the same
scripts run locally or in the cloud.

Configure with environment variables:
    STORAGE_BACKEND   "local" (default) or "s3"
    STORAGE_ROOT      local mode: base folder (default: current dir)
    S3_BUCKET         s3 mode: bucket name (required)
    S3_PREFIX         s3 mode: optional key prefix, e.g. "wildfire"

Local mode needs nothing extra. S3 mode needs `pip install boto3` and AWS
credentials available the usual way (env vars, ~/.aws/credentials, or the
IAM role attached to your Lambda/Fargate task — you never hard-code keys).

Why an adapter instead of calling boto3 everywhere: it keeps AWS out of the
business logic, lets you develop and test offline, and means the cloud
migration is a config change, not a rewrite. That separation is itself a good
engineering talking point.
"""
import io
import os
import tempfile

import pandas as pd

BACKEND = os.environ.get("STORAGE_BACKEND", "local").lower()
BUCKET = os.environ.get("S3_BUCKET")
PREFIX = os.environ.get("S3_PREFIX", "").strip("/")
LOCAL_ROOT = os.environ.get("STORAGE_ROOT", ".")

_S3_CLIENT = None


def _s3():
    """Lazily create one boto3 client (only imported when actually in s3 mode)."""
    global _S3_CLIENT
    if _S3_CLIENT is None:
        import boto3
        if not BUCKET:
            raise RuntimeError("STORAGE_BACKEND=s3 but S3_BUCKET is not set")
        _S3_CLIENT = boto3.client("s3")
    return _S3_CLIENT


def _key(path):
    return f"{PREFIX}/{path}" if PREFIX else path


def using_s3():
    return BACKEND == "s3"


def exists(path):
    if using_s3():
        try:
            _s3().head_object(Bucket=BUCKET, Key=_key(path))
            return True
        except Exception:
            return False
    return os.path.exists(os.path.join(LOCAL_ROOT, path))


def local_copy(path):
    """Return a real filesystem path for a stored object.

    Use this for libraries that demand a path on disk -- joblib.load(),
    geopandas.read_file(). In s3 mode the object is downloaded to a temp file.
    """
    if using_s3():
        dst = os.path.join(tempfile.gettempdir(), os.path.basename(path))
        _s3().download_file(BUCKET, _key(path), dst)
        return dst
    return os.path.join(LOCAL_ROOT, path)


def read_csv(path, **kwargs):
    if using_s3():
        obj = _s3().get_object(Bucket=BUCKET, Key=_key(path))
        return pd.read_csv(io.BytesIO(obj["Body"].read()), **kwargs)
    return pd.read_csv(os.path.join(LOCAL_ROOT, path), **kwargs)


def write_bytes(path, data, content_type=None, cache_control=None):
    if using_s3():
        extra = {}
        if content_type:
            extra["ContentType"] = content_type
        if cache_control:
            extra["CacheControl"] = cache_control  # CloudFront honors this for TTL
        _s3().put_object(Bucket=BUCKET, Key=_key(path), Body=data, **extra)
    else:
        full = os.path.join(LOCAL_ROOT, path)
        os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
        with open(full, "wb") as fh:
            fh.write(data)


def write_text(path, text, content_type=None, cache_control=None):
    write_bytes(path, text.encode("utf-8"), content_type=content_type,
                cache_control=cache_control)


def write_dataframe(path, df):
    write_text(path, df.to_csv(index=False), content_type="text/csv")


def upload_file(local_path, dest_path, content_type=None):
    """Copy a file that was written locally (e.g. a matplotlib PNG) to storage."""
    with open(local_path, "rb") as fh:
        write_bytes(dest_path, fh.read(), content_type=content_type)


# Quick self-test:  python storage.py
if __name__ == "__main__":
    import pandas as pd
    print(f"backend = {BACKEND!r}")
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    write_dataframe("storage_selftest.csv", df)
    back = read_csv("storage_selftest.csv")
    assert back.equals(df), "round-trip mismatch"
    write_text("storage_selftest.txt", "hello", content_type="text/plain")
    assert exists("storage_selftest.csv")
    print("round-trip OK:", "s3" if using_s3() else os.path.join(LOCAL_ROOT, "storage_selftest.csv"))
