"""Content-addressed disk cache for build artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import time
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

from gen_dsp.graph.models import Graph

_CACHE_SUBDIR = "dsp-graph"
_BUILDS_DIR = "builds"
_META_FILENAME = "meta.json"
_ARTIFACT_DIR = "artifact"
_MAX_AGE_SECONDS = 7 * 24 * 3600  # 7 days


def _default_cache_root() -> Path:
    """Return the platform-specific user cache directory."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / _CACHE_SUBDIR
    elif sys.platform == "win32":
        local = Path(
            __import__("os").environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")
        )
        return local / _CACHE_SUBDIR / "Cache"
    else:
        xdg = Path(__import__("os").environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        return xdg / _CACHE_SUBDIR


def cache_key(graph: Graph, platform: str) -> str:
    """Compute a deterministic cache key from graph content and platform.

    Key = sha256(gen_dsp_version | canonical_graph_json | platform).
    """
    gv = pkg_version("gen-dsp")
    canonical = json.dumps(graph.model_dump(), sort_keys=True, separators=(",", ":"))
    payload = f"{gv}|{canonical}|{platform}"
    return hashlib.sha256(payload.encode()).hexdigest()


class BuildCache:
    """Content-addressed build artifact cache with git-style shard directories."""

    def __init__(self, root: Path | None = None, max_age: int = _MAX_AGE_SECONDS) -> None:
        self._root = root or _default_cache_root()
        self._builds = self._root / _BUILDS_DIR
        self._max_age = max_age

    @property
    def cache_dir(self) -> Path:
        return self._root

    def _shard_dir(self, key: str) -> Path:
        return self._builds / key[:2] / key

    def get(self, key: str) -> tuple[bytes, str] | None:
        """Retrieve a cached artifact by key.

        Returns (artifact_bytes, filename) or None on miss.
        """
        shard = self._shard_dir(key)
        meta_path = shard / _META_FILENAME
        if not meta_path.exists():
            return None

        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

        # Check expiry
        created = meta.get("created", 0)
        if time.time() - created > self._max_age:
            shutil.rmtree(shard, ignore_errors=True)
            return None

        artifact_dir = shard / _ARTIFACT_DIR
        filename = meta.get("filename", "output")
        # The artifact is stored as a single file inside artifact_dir
        artifact_path = artifact_dir / filename
        if not artifact_path.exists():
            return None

        try:
            return (artifact_path.read_bytes(), filename)
        except OSError:
            return None

    def put(self, key: str, data: bytes, platform: str, filename: str) -> Path:
        """Store an artifact in the cache.

        Uses atomic rename to prevent partial-write visibility.
        Returns the path to the stored artifact.
        """
        self._evict_expired()

        shard = self._shard_dir(key)
        artifact_path = shard / _ARTIFACT_DIR / filename
        if artifact_path.exists():
            return artifact_path

        # Write to temp dir on same filesystem, then rename into place
        self._builds.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(dir=self._builds, prefix=".tmp_"))
        try:
            art_tmp = tmp / _ARTIFACT_DIR
            art_tmp.mkdir()
            (art_tmp / filename).write_bytes(data)

            meta: dict[str, Any] = {
                "created": time.time(),
                "platform": platform,
                "filename": filename,
                "size": len(data),
            }
            (tmp / _META_FILENAME).write_text(json.dumps(meta))

            # Atomic move: ensure parent shard dir exists
            shard.parent.mkdir(parents=True, exist_ok=True)
            try:
                tmp.rename(shard)
            except OSError:
                # Target appeared between check and rename (race); fine, use existing
                shutil.rmtree(tmp, ignore_errors=True)
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

        return shard / _ARTIFACT_DIR / filename

    def clear(self) -> int:
        """Remove all cached entries. Returns the number of entries removed."""
        count = 0
        if not self._builds.exists():
            return count
        for shard_prefix in self._builds.iterdir():
            if not shard_prefix.is_dir() or shard_prefix.name.startswith("."):
                continue
            for entry in shard_prefix.iterdir():
                if entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=True)
                    count += 1
            # Remove empty shard prefix dir
            try:
                shard_prefix.rmdir()
            except OSError:
                pass
        return count

    def size(self) -> tuple[int, int]:
        """Return (entry_count, total_bytes) for all cached artifacts."""
        count = 0
        total = 0
        if not self._builds.exists():
            return (0, 0)
        for shard_prefix in self._builds.iterdir():
            if not shard_prefix.is_dir() or shard_prefix.name.startswith("."):
                continue
            for entry in shard_prefix.iterdir():
                if not entry.is_dir():
                    continue
                count += 1
                for f in entry.rglob("*"):
                    if f.is_file():
                        total += f.stat().st_size
        return (count, total)

    def _evict_expired(self) -> None:
        """Lazily remove entries older than max_age."""
        if not self._builds.exists():
            return
        now = time.time()
        for shard_prefix in self._builds.iterdir():
            if not shard_prefix.is_dir() or shard_prefix.name.startswith("."):
                continue
            for entry in shard_prefix.iterdir():
                if not entry.is_dir():
                    continue
                meta_path = entry / _META_FILENAME
                try:
                    meta = json.loads(meta_path.read_text())
                    if now - meta.get("created", 0) > self._max_age:
                        shutil.rmtree(entry, ignore_errors=True)
                except (json.JSONDecodeError, OSError):
                    # Corrupt entry -- remove it
                    shutil.rmtree(entry, ignore_errors=True)
            # Clean up empty prefix dirs
            try:
                shard_prefix.rmdir()
            except OSError:
                pass


# Module-level singleton
_cache_instance: BuildCache | None = None


def get_cache(root: Path | None = None) -> BuildCache:
    """Return the lazily-initialized BuildCache singleton."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = BuildCache(root=root)
    return _cache_instance
