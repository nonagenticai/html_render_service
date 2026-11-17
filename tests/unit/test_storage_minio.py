"""MinIO storage backend (Phase 2), exercised with a fake client — no server.

Verifies the round-trip and that the recorded path stays an absolute path under
the local root (so the artifact routes / _publicize stay backend-agnostic) while
the bytes actually live in the bucket under the derived object key.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from src.services.storage_service import StorageService


class _FakeResponse:
    def __init__(self, data: bytes) -> None:
        self._data = data

    async def read(self) -> bytes:
        return self._data

    def close(self) -> None:
        pass

    async def release(self) -> None:
        pass


class _FakeMinio:
    """Implements the subset of the ``miniopy-async`` client the facade calls."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    async def put_object(
        self, bucket: str, key: str, data: Any, length: int, content_type: str | None = None
    ) -> None:
        self.objects[(bucket, key)] = data.read()

    async def fput_object(
        self, bucket: str, key: str, file_path: str, content_type: str | None = None
    ) -> None:
        # Streaming upload path used by the facade's put_file().
        self.objects[(bucket, key)] = Path(file_path).read_bytes()

    async def get_object(self, bucket: str, key: str) -> _FakeResponse:
        if (bucket, key) not in self.objects:
            raise FileNotFoundError(key)
        return _FakeResponse(self.objects[(bucket, key)])

    async def remove_object(self, bucket: str, key: str) -> None:
        self.objects.pop((bucket, key), None)


def _minio_storage(root: Path) -> tuple[StorageService, _FakeMinio]:
    with patch("src.services.storage_service.get_settings") as mock_settings:
        st = mock_settings.return_value.storage
        st.backend = "minio"
        st.local_path = str(root)
        st.minio_bucket = "html-render-service"
        st.minio_endpoint = "minio.test:8389"
        st.minio_access_key = "k"  # pragma: allowlist secret
        st.minio_secret_key = "s"  # pragma: allowlist secret
        st.minio_secure = False
        st.minio_external_endpoint = None
        st.presign_expiry_seconds = 3600
        s = StorageService()
    fake = _FakeMinio()
    # Inject the fake client into the shared facade the wrapper delegates to.
    s._storage._minio_client = fake
    return s, fake


async def test_store_and_retrieve_html_round_trips(tmp_path) -> None:
    s, fake = _minio_storage(tmp_path)
    path = await s.store_html("job1", "<h1>hi</h1>", "redesign.html")

    # Recorded path is an absolute path under the local root (format unchanged)…
    assert path == str(tmp_path / "job1" / "redesign.html")
    # …no local file is written for the minio backend…
    assert not (tmp_path / "job1" / "redesign.html").exists()
    # …and the bytes live in the bucket under the derived key.
    assert ("html-render-service", "job1/redesign.html") in fake.objects
    assert await s.retrieve_text(path) == "<h1>hi</h1>"


async def test_store_bytes_round_trips(tmp_path) -> None:
    s, _ = _minio_storage(tmp_path)
    path = await s.store_bytes("job2", b"\x89PNG\r\n", "shot.png")
    assert await s.retrieve(path) == b"\x89PNG\r\n"


async def test_persist_paths_uploads_local_files(tmp_path) -> None:
    s, fake = _minio_storage(tmp_path)
    # Simulate Playwright having written a screenshot to local scratch.
    shot = tmp_path / "job3" / "screenshots" / "desktop.png"
    shot.parent.mkdir(parents=True, exist_ok=True)
    shot.write_bytes(b"PNGDATA")

    out = await s.persist_paths({"desktop": str(shot)})
    assert out == {"desktop": str(shot)}  # paths returned unchanged
    assert fake.objects[("html-render-service", "job3/screenshots/desktop.png")] == b"PNGDATA"
    assert await s.retrieve(str(shot)) == b"PNGDATA"


async def test_retrieve_missing_returns_none(tmp_path) -> None:
    s, _ = _minio_storage(tmp_path)
    assert await s.retrieve(str(tmp_path / "nope" / "x.png")) is None
    assert await s.retrieve_text(str(tmp_path / "nope" / "x.html")) is None


async def test_traversal_outside_root_is_rejected(tmp_path) -> None:
    s, _ = _minio_storage(tmp_path)
    with pytest.raises(ValueError, match="outside storage root"):
        await s.store_bytes("..", b"x", "../../etc/passwd")


async def test_delete_removes_object(tmp_path) -> None:
    s, fake = _minio_storage(tmp_path)
    path = await s.store_bytes("job4", b"data", "a.png")
    assert ("html-render-service", "job4/a.png") in fake.objects
    assert await s.delete(path) is True
    assert ("html-render-service", "job4/a.png") not in fake.objects


async def test_minio_init_failure_does_not_crashloop(tmp_path, monkeypatch) -> None:
    """A MinIO blip at boot must be swallowed, not raised (anti-crashloop).

    This is the regression the prior facade migration missed — a wrapper that
    raised at boot under STORAGE_BACKEND=minio crashlooped the prod pod, and
    per-PR CI never caught it.
    """
    s, _ = _minio_storage(tmp_path)

    async def _boom() -> None:
        raise ConnectionError("minio unreachable at boot")

    monkeypatch.setattr(s._storage, "initialize", _boom)
    await s.initialize()  # must not raise
