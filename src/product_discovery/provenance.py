"""Run provenance recorded with every experiment artifact."""
from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_provenance() -> dict[str, Any]:
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    versions: dict[str, str] = {}
    for module in ("numpy", "faiss", "torch", "transformers", "peft", "sentence_transformers", "scipy"):
        loaded = sys.modules.get(module)
        if loaded is not None and hasattr(loaded, "__version__"):
            versions[module] = str(loaded.__version__)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_tracked_changes_uncommitted": bool(dirty),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "library_versions": versions,
    }
