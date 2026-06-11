from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .text import Chunk, tokenize


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        if not chunks:
            raise ValueError("Cannot build an index with zero chunks")

        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.term_frequencies: list[Counter[str]] = []
        self.document_frequency: Counter[str] = Counter()
        self.doc_lengths: list[int] = []

        for chunk in chunks:
            counts = Counter(tokenize(chunk.text))
            self.term_frequencies.append(counts)
            self.document_frequency.update(counts.keys())
            self.doc_lengths.append(sum(counts.values()))

        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)
        self.inverted_index = self._build_inverted_index()

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query_terms = Counter(tokenize(query))
        if not query_terms:
            return []

        scores: defaultdict[int, float] = defaultdict(float)
        total_docs = len(self.chunks)

        for term, query_weight in query_terms.items():
            matching_docs = self.inverted_index.get(term)
            if not matching_docs:
                continue
            df = self.document_frequency[term]
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))

            for doc_index in matching_docs:
                term_frequency = self.term_frequencies[doc_index][term]
                doc_length = self.doc_lengths[doc_index] or 1
                denominator = term_frequency + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
                scores[doc_index] += idf * (term_frequency * (self.k1 + 1) / denominator) * query_weight

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [SearchResult(chunk=self.chunks[index], score=score) for index, score in ranked]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "k1": self.k1,
            "b": self.b,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        payload = json.loads(path.read_text(encoding="utf-8"))
        chunks = [Chunk.from_dict(chunk) for chunk in payload["chunks"]]
        return cls(chunks=chunks, k1=float(payload.get("k1", 1.5)), b=float(payload.get("b", 0.75)))

    def _build_inverted_index(self) -> dict[str, list[int]]:
        postings: defaultdict[str, list[int]] = defaultdict(list)
        for doc_index, counts in enumerate(self.term_frequencies):
            for term in counts:
                postings[term].append(doc_index)
        return dict(postings)
