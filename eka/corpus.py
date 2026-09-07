"""Manage the document corpus on disk (``data/documents/``).

Used by the Streamlit UI to add and remove source files without touching the
filesystem by hand. After any change the FAISS index must be rebuilt.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import settings
from .loaders import SUPPORTED_SUFFIXES, detect_kind

# extensions the file_uploader should offer
UPLOAD_TYPES = sorted(s.lstrip(".") for s in SUPPORTED_SUFFIXES)


@dataclass
class SourceFile:
    name: str
    path: Path
    size: int
    kind: str | None  # "pdf" | "docx" | "text" | None (unrecognised)

    @property
    def size_kb(self) -> float:
        return self.size / 1024

    @property
    def ingestable(self) -> bool:
        return self.kind is not None


def documents_dir() -> Path:
    directory = Path(settings.data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _safe_name(filename: str) -> str:
    """Reduce an arbitrary upload name to a bare, safe file name."""
    name = Path(filename).name.strip()
    if not name or name in {".", ".."} or name.startswith("."):
        raise ValueError(f"Unsafe file name: {filename!r}")
    return name


def list_source_files() -> list[SourceFile]:
    files: list[SourceFile] = []
    for path in sorted(documents_dir().iterdir(), key=lambda p: p.name.lower()):
        if not path.is_file():
            continue
        files.append(
            SourceFile(name=path.name, path=path, size=path.stat().st_size, kind=detect_kind(path))
        )
    return files


def save_upload(filename: str, data: bytes) -> Path:
    """Write an uploaded file into the corpus. Returns the saved path."""
    destination = documents_dir() / _safe_name(filename)
    destination.write_bytes(data)
    return destination


def delete_source_file(name: str) -> bool:
    """Delete one file from the corpus. Returns True if a file was removed."""
    target = documents_dir() / _safe_name(name)
    if target.is_file() and target.parent == documents_dir():
        target.unlink()
        return True
    return False
