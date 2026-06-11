from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Iterable

from .text import Document, normalize_whitespace, relative_source


SUPPORTED_SUFFIXES = {".txt", ".md", ".csv", ".json", ".pdf"}


def load_documents(input_path: Path) -> list[Document]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    root = input_path if input_path.is_dir() else input_path.parent
    files = list(_iter_files(input_path))
    documents: list[Document] = []

    for file_path in files:
        suffix = file_path.suffix.lower()
        source = relative_source(file_path, root)
        if suffix == ".txt":
            documents.append(Document(source=source, text=file_path.read_text(encoding="utf-8", errors="ignore")))
        elif suffix == ".md":
            documents.append(Document(source=source, text=_read_markdown(file_path)))
        elif suffix == ".csv":
            documents.append(Document(source=source, text=_read_csv(file_path)))
        elif suffix == ".json":
            documents.append(Document(source=source, text=_read_json(file_path)))
        elif suffix == ".pdf":
            documents.extend(_read_pdf(file_path, source))

    return [doc for doc in documents if normalize_whitespace(doc.text)]


def _iter_files(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield input_path
        return

    for file_path in sorted(input_path.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield file_path


def _read_csv(file_path: Path) -> str:
    lines: list[str] = []
    with file_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames:
            for row_number, row in enumerate(reader, start=1):
                fields = [f"{key}: {value}" for key, value in row.items() if value]
                if fields:
                    lines.append(f"row {row_number}. " + "; ".join(fields))
        else:
            handle.seek(0)
            for row in csv.reader(handle):
                lines.append(" ".join(cell for cell in row if cell))
    return "\n".join(lines)


def _read_markdown(file_path: Path) -> str:
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    text = re.sub(r"^#{1,6}\s+.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text


def _read_json(file_path: Path) -> str:
    data = json.loads(file_path.read_text(encoding="utf-8", errors="ignore"))
    return _flatten_json(data)


def _flatten_json(value: object, prefix: str = "") -> str:
    if isinstance(value, dict):
        parts = []
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            parts.append(_flatten_json(child, child_prefix))
        return "\n".join(part for part in parts if part)
    if isinstance(value, list):
        return "\n".join(_flatten_json(item, prefix) for item in value)
    if value is None:
        return ""
    if prefix:
        if prefix.rsplit(".", maxsplit=1)[-1].lower() in {"body", "content", "text"}:
            return _as_sentence(value)
        return f"{prefix}: {_as_sentence(value)}"
    return _as_sentence(value)


def _as_sentence(value: object) -> str:
    text = str(value).strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text


def _read_pdf(file_path: Path, source: str) -> list[Document]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF ingestion requires pypdf. Install dependencies with: pip install -r requirements.txt") from exc

    reader = PdfReader(str(file_path))
    documents: list[Document] = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if normalize_whitespace(text):
            documents.append(Document(source=source, page=page_index, text=text))
    return documents
