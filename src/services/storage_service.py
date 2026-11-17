"""File storage for design outputs (HTML, screenshots, images).

Two backends behind one interface:

- ``local`` (default): writes under ``STORAGE_LOCAL_PATH``; unchanged behavior.
- ``minio``: writes to a shared MinIO bucket so any replica can serve any job's
  artifacts — the prerequisite for ``replicas > 1``.

The recorded path FORMAT is identical for both backends: an absolute path under
the local root (e.g. ``/app/data/<job_id>/screenshots/desktop.png``). For the
MinIO backend that path is also the source of the object key (its location
relative to the local root), so the DB rows, the artifact-serving routes, and
``_publicize`` are all backend-agnostic — they keep proxy-serving via
``retrieve()`` and the absolute-URL contract is unchanged. Playwright/PIL always
write to a local scratch dir first (they need a real file); ``persist_paths``
then uploads those files to MinIO.

Object I/O for the minio backend is delegated to the shared
``w2v_common.storage.StorageService`` facade (the fleet-wide consolidation of the
per-service MinIO clients). This wrapper keeps the html-render-domain concerns
the facade does not know about: the local-scratch dirs, the abs-path↔object-key
mapping (``_object_key``), the traversal guard, and content-type guessing.
"""

from __future__ import annotations

import logging
import mimetypes
from pathlib import Path

from w2v_common.storage import StorageService as CommonStorage

from src.config import get_settings

logger = logging.getLogger(__name__)


class StorageService:
    """Manages file storage for design outputs (local filesystem or MinIO)."""

    def __init__(self) -> None:
        settings = get_settings()
        self._backend = str(settings.storage.backend).strip().lower()
        self._local_path = Path(settings.storage.local_path)
        self._minio_bucket = settings.storage.minio_bucket
        self._storage = CommonStorage(settings.storage)

    @property
    def backend_type(self) -> str:
        """Return the configured backend ('local' or 'minio')."""
        return self._backend

    async def initialize(self) -> None:
        """Initialize the configured storage backend."""
        # The local root is always used as the scratch dir Playwright/PIL write
        # into, even under the minio backend, so create it either way.
        self._local_path.mkdir(parents=True, exist_ok=True)
        if self._backend == "local":
            logger.info("Local storage initialized at %s", self._local_path)
            return
        if self._backend == "minio":
            # Best-effort: a MinIO blip at boot must not crashloop the pod — serve
            # non-artifact routes and let the bucket be created on first write.
            try:
                await self._storage.initialize()
                logger.info("MinIO storage initialized: bucket=%s", self._minio_bucket)
            except Exception as exc:  # noqa: BLE001 — startup must tolerate any minio failure
                logger.warning(
                    "MinIO bucket probe failed at startup (%s: %s) — proceeding "
                    "without verification; bucket will be created on first write.",
                    type(exc).__name__,
                    exc,
                )
            return
        raise ValueError(f"Unsupported storage backend: {self._backend}")

    def get_job_dir(self, job_id: str) -> Path:
        """Get the local scratch directory for a job (Playwright/PIL output)."""
        job_dir = self._local_path / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    # ------------------------------------------------------------------
    # path/key resolution
    # ------------------------------------------------------------------

    def _object_key(self, path: str) -> str | None:
        """Map a recorded (absolute-or-relative) path to its MinIO object key.

        The key is the path's location relative to the local root, e.g.
        ``/app/data/<job>/screenshots/desktop.png`` → ``<job>/screenshots/desktop.png``.
        Applies the same traversal guard as the local backend: anything that
        resolves outside the root returns None (treated as "not found").
        """
        root = self._local_path.resolve()
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self._local_path / path
        try:
            resolved = candidate.resolve()
        except OSError, RuntimeError:
            logger.warning("Could not resolve storage path: %s", path)
            return None
        if not resolved.is_relative_to(root):
            logger.warning("Rejected path outside storage root: %s", path)
            return None
        return resolved.relative_to(root).as_posix()

    def _safe_local_path(self, raw: str) -> Path | None:
        """Resolve ``raw`` and reject anything outside ``self._local_path``."""
        root = self._local_path.resolve()
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self._local_path / raw
        try:
            resolved = candidate.resolve()
        except OSError, RuntimeError:
            logger.warning("Could not resolve storage path: %s", raw)
            return None
        if not resolved.is_relative_to(root):
            logger.warning("Rejected path outside storage root: %s", raw)
            return None
        return resolved

    @staticmethod
    def _content_type(name: str) -> str:
        return mimetypes.guess_type(name)[0] or "application/octet-stream"

    def _minio_key_or_raise(self, path: str) -> str:
        key = self._object_key(path)
        if key is None:
            raise ValueError(f"Refusing to store path outside storage root: {path}")
        return key

    # ------------------------------------------------------------------
    # store
    # ------------------------------------------------------------------

    async def store_html(self, job_id: str, html: str, filename: str = "redesign.html") -> str:
        """Store HTML and return its recorded path (absolute, under the root)."""
        file_path = self._local_path / job_id / filename
        if self._backend == "minio":
            key = self._minio_key_or_raise(str(file_path))
            await self._storage.put_bytes(
                key, html.encode("utf-8"), content_type="text/html; charset=utf-8"
            )
            logger.info("Stored minio HTML: %s/%s (%d chars)", self._minio_bucket, key, len(html))
        else:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(html, encoding="utf-8")
            logger.info("Stored HTML: %s (%d chars)", file_path, len(html))
        return str(file_path)

    async def store_bytes(self, job_id: str, data: bytes, filename: str) -> str:
        """Store binary data and return its recorded path."""
        file_path = self._local_path / job_id / filename
        if self._backend == "minio":
            key = self._minio_key_or_raise(str(file_path))
            await self._storage.put_bytes(key, data, content_type=self._content_type(filename))
            logger.info("Stored minio file: %s/%s (%d bytes)", self._minio_bucket, key, len(data))
        else:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(data)
            logger.info("Stored file: %s (%d bytes)", file_path, len(data))
        return str(file_path)

    async def persist_file(self, path: str) -> str:
        """Upload a locally-written file to MinIO (no-op on the local backend).

        Playwright/PIL write screenshots and comparison images straight to the
        local scratch dir; on the minio backend this lifts each of them into the
        shared bucket so other replicas can serve them. Returns ``path`` unchanged
        so callers keep recording the same (backend-agnostic) reference. Uploads
        via the facade's streaming ``put_file`` (constant memory) so a large
        full-page screenshot never lands in RAM.
        """
        if self._backend != "minio":
            return path
        local = self._safe_local_path(path)
        if local is None or not local.exists():
            logger.warning("persist_file: nothing to upload at %s", path)
            return path
        key = self._object_key(path)
        if key is None:
            return path
        await self._storage.put_file(key, local, content_type=self._content_type(local.name))
        logger.info("Persisted minio://%s/%s", self._minio_bucket, key)
        return path

    async def persist_paths(self, paths: dict[str, str] | None) -> dict[str, str] | None:
        """Upload every value in a ``{viewport: path}`` dict; returns it unchanged."""
        if not paths or self._backend != "minio":
            return paths
        for value in paths.values():
            await self.persist_file(value)
        return paths

    async def presigned_url(self, path: str, expiry_seconds: int | None = None) -> str | None:
        """Return a presigned GET URL for a stored artifact (minio backend only).

        This is the fleet convention for artifacts a downstream service must
        fetch unauthenticated (cf. video_compiler, slide_service): the URL is a
        ``http://minio.test:8389/<bucket>/<key>?X-Amz...`` link that
        video_compiler can GET with no license header, and that the
        orchestrator's artifact proxy re-presigns on read for the browser
        (ignoring the baked signature, so expiry is a non-issue for the FE).

        Returns ``None`` on the local backend (no MinIO to sign) so callers
        fall back to the in-service authed proxy route — fine for dev/tests,
        which never cross the service boundary unauthenticated.
        """
        if self._backend != "minio":
            return None
        key = self._object_key(path)
        if key is None:
            return None
        return await self._storage.presign_get(key, expires=expiry_seconds)

    # ------------------------------------------------------------------
    # retrieve / delete
    # ------------------------------------------------------------------

    async def retrieve(self, path: str) -> bytes | None:
        """Retrieve file bytes from storage."""
        if self._backend == "minio":
            key = self._object_key(path)
            if key is None:
                return None
            return await self._storage.get_bytes(key)
        file_path = self._safe_local_path(path)
        if file_path is None:
            return None
        if file_path.exists():
            return file_path.read_bytes()
        logger.warning("File not found: %s", path)
        return None

    async def retrieve_text(self, path: str) -> str | None:
        """Retrieve a text file from storage."""
        if self._backend == "minio":
            data = await self.retrieve(path)
            return data.decode("utf-8") if data is not None else None
        file_path = self._safe_local_path(path)
        if file_path is None:
            return None
        if file_path.exists():
            return file_path.read_text(encoding="utf-8")
        logger.warning("File not found: %s", path)
        return None

    async def delete(self, path: str) -> bool:
        """Delete a file from storage. Returns True if it existed."""
        if self._backend == "minio":
            key = self._object_key(path)
            if key is None:
                return False
            return await self._storage.delete_object(key)
        file_path = self._safe_local_path(path)
        if file_path is None:
            return False
        if file_path.exists():
            file_path.unlink()
            return True
        return False
