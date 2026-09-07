"""Document ingestion - load a local folder into LangChain ``Document`` objects.

Supported formats: ``.pdf`` (per page), ``.docx``, ``.txt``, ``.md`` / ``.markdown``.
Files with a missing or wrong extension are sniffed by magic bytes, so an
extension-less PDF is still ingested. Loaders for optional formats are imported
lazily so a missing wheel (e.g. lxml for python-docx) only disables that format.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".text"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | {".pdf", ".docx"}


def detect_kind(path: Path) -> str | None:
    """Return 'pdf' | 'docx' | 'text' | None, from extension then file signature."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix == ".docx":
        return "docx"
    if suffix in TEXT_SUFFIXES:
        return "text"

    try:
        with open(path, "rb") as handle:
            head = handle.read(8)
    except OSError:
        return None
    if head.startswith(b"%PDF-"):
        return "pdf"
    if head.startswith(b"PK\x03\x04"):  # zip container - most likely .docx
        return "docx"
    return None


def _read_text_file(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf_pages(path: Path) -> list[tuple[int, str]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Reading PDF files requires 'pypdf' (pip install pypdf).") from exc

    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for number, page in enumerate(reader.pages, start=1):
        pages.append((number, page.extract_text() or ""))
    return pages


def _read_docx(path: Path) -> str:
    try:
        import docx  # python-docx
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Reading .docx files requires 'python-docx'.") from exc

    document = docx.Document(str(path))
    parts = [para.text for para in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text for cell in row.cells))
    return "\n".join(parts)


def load_documents(folder: str | Path) -> list[Document]:
    """Load every supported file under *folder* (recursively).

    Returns one ``Document`` per text/docx file and one per PDF page. Each carries
    ``metadata`` with at least ``source`` (file name) and ``path``; PDF pages also
    carry ``page``.
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Document folder does not exist: {folder.resolve()}")

    documents: list[Document] = []
    skipped: list[str] = []

    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        kind = detect_kind(path)
        if kind is None:
            continue

        try:
            if kind == "pdf":
                pages = _read_pdf_pages(path)
                if not any(text.strip() for _, text in pages):
                    skipped.append(f"{path.name}: no extractable text (scanned/image PDF - needs OCR)")
                    continue
                for page_number, text in pages:
                    if text.strip():
                        documents.append(
                            Document(
                                page_content=text,
                                metadata={"source": path.name, "path": str(path), "page": page_number},
                            )
                        )
            else:
                text = _read_docx(path) if kind == "docx" else _read_text_file(path)
                if text.strip():
                    documents.append(
                        Document(page_content=text, metadata={"source": path.name, "path": str(path)})
                    )
                else:
                    skipped.append(f"{path.name}: file contained no text")
        except Exception as exc:  # noqa: BLE001 - report and keep going
            skipped.append(f"{path.name}: {exc}")

    if skipped:
        print("[loaders] skipped some files:\n  - " + "\n  - ".join(skipped))

    return documents
