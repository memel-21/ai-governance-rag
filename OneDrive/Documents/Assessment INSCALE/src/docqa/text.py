from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-']*")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "over",
    "should",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


@dataclass(frozen=True)
class Document:
    source: str
    text: str
    page: int | None = None


@dataclass(frozen=True)
class Chunk:
    id: int
    source: str
    text: str
    page: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(
            id=int(data["id"]),
            source=str(data["source"]),
            text=str(data["text"]),
            page=data.get("page"),
        )


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return [
        token
        for match in TOKEN_RE.finditer(text)
        if (token := normalize_token(match.group(0).lower())) not in STOPWORDS
    ]


def normalize_token(token: str) -> str:
    replacements = {
        "documented": "document",
        "documentation": "document",
        "documents": "document",
        "governance": "governance",
    }
    if token in replacements:
        return replacements[token]
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def split_sentences(text: str) -> list[str]:
    compact = normalize_whitespace(text)
    if not compact:
        return []
    return [sentence.strip() for sentence in SENTENCE_RE.split(compact) if sentence.strip()]


def chunk_documents(
    documents: Iterable[Document],
    chunk_words: int = 220,
    overlap_words: int = 50,
) -> list[Chunk]:
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    if overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("overlap_words must be non-negative and smaller than chunk_words")

    chunks: list[Chunk] = []
    next_id = 1
    for document in documents:
        words = normalize_whitespace(document.text).split()
        if not words:
            continue

        step = chunk_words - overlap_words
        for start in range(0, len(words), step):
            window = words[start : start + chunk_words]
            if not window:
                continue
            chunks.append(
                Chunk(
                    id=next_id,
                    source=document.source,
                    page=document.page,
                    text=" ".join(window),
                )
            )
            next_id += 1
            if start + chunk_words >= len(words):
                break
    return chunks


def score_sentence(sentence: str, query_terms: Counter[str]) -> float:
    terms = tokenize(sentence)
    if not terms:
        return 0.0
    counts = Counter(terms)
    overlap = sum(min(counts[term], query_terms[term]) for term in query_terms)
    density = overlap / math.sqrt(len(terms))
    exact_bonus = 0.2 if any(term in counts for term in query_terms) else 0.0
    return density + exact_bonus


def relative_source(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
