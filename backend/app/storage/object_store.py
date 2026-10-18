"""Durable object storage for source video, memory databases and evidence.

The pipeline always works on local files (OpenCV needs a path). The object
store is the durable copy: after an ingest, the source video, the SQLite
memory and every evidence frame are mirrored under a per-session prefix.
``LocalObjectStore`` is a no-op so a laptop run needs no AWS at all;
``S3ObjectStore`` writes to Amazon S3.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Protocol


class ObjectStore(Protocol):
    name: str

    def put_file(self, path: Path, key: str) -> None: ...

    def put_dir(self, directory: Path, prefix: str, pattern: str = "*") -> int: ...

    def get_file(self, key: str, path: Path) -> bool: ...

    def delete_prefix(self, prefix: str) -> int: ...


class LocalObjectStore:
    """No-op store: local disk is already the source of truth."""

    name = "local"

    def put_file(self, path: Path, key: str) -> None:
        return None

    def put_dir(self, directory: Path, prefix: str, pattern: str = "*") -> int:
        return 0

    def get_file(self, key: str, path: Path) -> bool:
        return False

    def delete_prefix(self, prefix: str) -> int:
        return 0


class S3ObjectStore:
    """Amazon S3 backed store. Credentials come from the standard AWS chain
    (instance role on EC2, env vars or ``~/.aws`` locally)."""

    def __init__(
        self, bucket: str, prefix: str = "vista", region: str | None = None
    ) -> None:
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "boto3 is required for storage_backend='s3'. "
                "Install it with: uv sync --group aws"
            ) from exc
        if not bucket:
            raise ValueError("VISTA_S3_BUCKET must be set when VISTA_STORAGE=s3")
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._client = boto3.client("s3", region_name=region or None)
        self.name = f"s3://{bucket}/{self.prefix}"

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key.lstrip('/')}" if self.prefix else key.lstrip("/")

    def put_file(self, path: Path, key: str) -> None:
        content_type = mimetypes.guess_type(path.name)[0]
        extra = {"ContentType": content_type} if content_type else None
        self._client.upload_file(
            str(path), self.bucket, self._key(key), ExtraArgs=extra
        )

    def put_dir(self, directory: Path, prefix: str, pattern: str = "*") -> int:
        count = 0
        for path in sorted(directory.glob(pattern)):
            if path.is_file():
                self.put_file(path, f"{prefix.rstrip('/')}/{path.name}")
                count += 1
        return count

    def get_file(self, key: str, path: Path) -> bool:
        from botocore.exceptions import ClientError

        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._client.download_file(self.bucket, self._key(key), str(path))
        except ClientError:
            return False
        return True

    def delete_prefix(self, prefix: str) -> int:
        paginator = self._client.get_paginator("list_objects_v2")
        deleted = 0
        for page in paginator.paginate(
            Bucket=self.bucket, Prefix=self._key(prefix.rstrip("/") + "/")
        ):
            objects = [{"Key": o["Key"]} for o in page.get("Contents", [])]
            if objects:
                self._client.delete_objects(
                    Bucket=self.bucket, Delete={"Objects": objects}
                )
                deleted += len(objects)
        return deleted


def create_object_store(
    backend: str, bucket: str | None, prefix: str, region: str | None
) -> ObjectStore:
    if backend == "s3":
        return S3ObjectStore(bucket or "", prefix=prefix, region=region)
    if backend == "local":
        return LocalObjectStore()
    raise ValueError(f"Unknown storage backend: {backend!r} (use 'local' or 's3')")
