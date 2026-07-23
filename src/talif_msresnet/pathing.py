"""Portable references for paths stored in reproducibility artifacts."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _path_digest(path: Path) -> str:
    normalized = os.path.normcase(str(path))
    return hashlib.sha256(normalized.encode("utf-8", errors="surrogatepass")).hexdigest()


def artifact_path_reference(
    path: str | Path,
    project_root: str | Path,
    *,
    external_identifier: str | None = None,
) -> str:
    """Return a repository-relative POSIX path or an explicit external ID.

    Paths inside ``project_root`` are represented relative to that root.  An
    external regular file is represented by its content hash; an external
    directory or not-yet-created path is represented by an opaque normalized
    path hash.  Callers may provide a domain-specific external identifier when
    a content hash is already available.  No host path is returned.
    """

    root = Path(project_root).expanduser().resolve()
    candidate = Path(path).expanduser().resolve()
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        if external_identifier is not None:
            identifier = str(external_identifier).strip()
            if not identifier or any(char in identifier for char in ("/", "\\", "\r", "\n")):
                raise ValueError("external_identifier must be a non-empty path-free value")
            return f"external:{identifier}"
        if candidate.is_file():
            return f"external:file-sha256:{_sha256_file(candidate)}"
        return f"external:path-sha256:{_path_digest(candidate)}"
    return relative.as_posix() or "."
