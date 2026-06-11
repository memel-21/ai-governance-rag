from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .index import SearchResult
from .text import score_sentence, split_sentences, tokenize


@dataclass(frozen=True)
class Answer:
    text: str
    references: list[SearchResult]


def build_answer(query: str, results: list[SearchResult], max_sentences: int = 2) -> Answer:
    if not results:
        return Answer(
            text="I could not find enough relevant information in the indexed documents to answer this question.",
            references=[],
        )

    query_terms = Counter(tokenize(query))
    candidates: list[tuple[float, int, str]] = []
    for result_index, result in enumerate(results):
        for sentence in split_sentences(result.chunk.text):
            sentence_score = score_sentence(sentence, query_terms)
            if sentence_score > 0:
                candidates.append((sentence_score + result.score * 0.05, result_index, sentence))

    if not candidates:
        fallback = results[0].chunk.text
        return Answer(text=fallback, references=results[:1])

    candidates.sort(key=lambda item: item[0], reverse=True)
    minimum_score = candidates[0][0] * 0.5
    selected: list[tuple[int, str]] = []
    seen = set()
    for candidate_score, result_index, sentence in candidates:
        if candidate_score < minimum_score:
            break
        normalized = sentence.lower()
        if normalized in seen:
            continue
        selected.append((result_index, sentence))
        seen.add(normalized)
        if len(selected) >= max_sentences:
            break

    selected.sort(key=lambda item: item[0])
    used_result_indexes = sorted({result_index for result_index, _ in selected})
    references = [results[index] for index in used_result_indexes]
    answer_text = " ".join(sentence for _, sentence in selected)
    return Answer(text=answer_text, references=references)


def format_answer(answer: Answer, show_context: bool = False) -> str:
    lines = [answer.text]
    if answer.references:
        lines.append("")
        lines.append("References:")
        for ref_number, result in enumerate(answer.references, start=1):
            location = result.chunk.source
            if result.chunk.page is not None:
                location += f", page {result.chunk.page}"
            location += f", chunk {result.chunk.id}"
            lines.append(f"[{ref_number}] {location} (score: {result.score:.3f})")
            if show_context:
                lines.append(f"    {result.chunk.text}")
    return "\n".join(lines)
