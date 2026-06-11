import unittest
from pathlib import Path

from docqa.answer import build_answer
from docqa.index import BM25Index
from docqa.loaders import load_documents
from docqa.text import chunk_documents


class DocQATests(unittest.TestCase):
    def test_sample_corpus_answers_with_references(self) -> None:
        docs = load_documents(Path("sample_docs"))
        chunks = chunk_documents(docs, chunk_words=80, overlap_words=10)
        index = BM25Index(chunks)

        question = "What should AI governance programs document?"
        results = index.search(question, top_k=3)
        answer = build_answer(question, results)

        self.assertTrue(answer.references)
        self.assertTrue("document" in answer.text.lower() or "records" in answer.text.lower())

    def test_no_result_response_is_explicit(self) -> None:
        answer = build_answer("unrelated term", [])

        self.assertIn("could not find", answer.text.lower())
        self.assertEqual(answer.references, [])


if __name__ == "__main__":
    unittest.main()
