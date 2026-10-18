"""Object-store tests (S3 is exercised through a fake boto3 client)."""

import sys
import types

import pytest

from backend.app.storage import (
    LocalObjectStore,
    S3ObjectStore,
    create_object_store,
)


class FakeS3:
    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def upload_file(self, filename, bucket, key, ExtraArgs=None):
        with open(filename, "rb") as handle:
            self.objects[f"{bucket}/{key}"] = handle.read()

    def get_paginator(self, _name):
        outer = self

        class Paginator:
            def paginate(self, Bucket, Prefix):
                keys = [
                    k.split("/", 1)[1]
                    for k in outer.objects
                    if k.startswith(f"{Bucket}/{Prefix}")
                ]
                yield {"Contents": [{"Key": k} for k in keys]}

        return Paginator()

    def delete_objects(self, Bucket, Delete):
        for obj in Delete["Objects"]:
            self.objects.pop(f"{Bucket}/{obj['Key']}", None)


@pytest.fixture
def fake_s3(monkeypatch):
    client = FakeS3()
    module = types.SimpleNamespace(client=lambda *_a, **_k: client)
    monkeypatch.setitem(sys.modules, "boto3", module)
    return client


def test_local_store_is_noop(tmp_path):
    store = LocalObjectStore()
    f = tmp_path / "a.jpg"
    f.write_bytes(b"x")
    store.put_file(f, "k")
    assert store.put_dir(tmp_path, "p") == 0
    assert store.delete_prefix("p") == 0


def test_s3_put_dir_and_delete_prefix(fake_s3, tmp_path):
    store = S3ObjectStore("bkt", prefix="vista")
    (tmp_path / "evt_1.jpg").write_bytes(b"1")
    (tmp_path / "evt_2.jpg").write_bytes(b"2")
    (tmp_path / "note.txt").write_bytes(b"n")

    assert store.put_dir(tmp_path, "sessions/s1/evidence", "*.jpg") == 2
    assert set(fake_s3.objects) == {
        "bkt/vista/sessions/s1/evidence/evt_1.jpg",
        "bkt/vista/sessions/s1/evidence/evt_2.jpg",
    }
    assert store.delete_prefix("sessions/s1/evidence") == 2
    assert fake_s3.objects == {}


def test_s3_requires_bucket(fake_s3):
    with pytest.raises(ValueError):
        create_object_store("s3", None, "vista", None)


def test_unknown_backend():
    with pytest.raises(ValueError):
        create_object_store("ftp", None, "vista", None)
