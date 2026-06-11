from __future__ import annotations

import argparse
from pathlib import Path

from .answer import build_answer, format_answer
from .index import BM25Index
from .loaders import load_documents
from .text import chunk_documents


def main() -> None:
    parser = argparse.ArgumentParser(prog="docqa", description="Grounded document question-answering CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Build a local retrieval index from documents")
    ingest_parser.add_argument("input", type=Path, help="Document file or directory to ingest")
    ingest_parser.add_argument("--index", type=Path, default=Path("storage/index.json"), help="Output index path")
    ingest_parser.add_argument("--chunk-words", type=int, default=220, help="Words per chunk")
    ingest_parser.add_argument("--overlap-words", type=int, default=50, help="Overlapping words between chunks")

    ask_parser = subparsers.add_parser("ask", help="Ask a question over an existing index")
    ask_parser.add_argument("question", help="Question to answer")
    ask_parser.add_argument("--index", type=Path, default=Path("storage/index.json"), help="Index path")
    ask_parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    ask_parser.add_argument("--sentences", type=int, default=2, help="Maximum answer sentences")
    ask_parser.add_argument("--show-context", action="store_true", help="Print retrieved context snippets")

    shell_parser = subparsers.add_parser("shell", help="Start an interactive question-answering shell")
    shell_parser.add_argument("--index", type=Path, default=Path("storage/index.json"), help="Index path")
    shell_parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    shell_parser.add_argument("--sentences", type=int, default=2, help="Maximum answer sentences")
    shell_parser.add_argument("--show-context", action="store_true", help="Print retrieved context snippets")

    args = parser.parse_args()
    if args.command == "ingest":
        ingest(args)
    elif args.command == "ask":
        ask(args)
    elif args.command == "shell":
        shell(args)


def ingest(args: argparse.Namespace) -> None:
    documents = load_documents(args.input)
    chunks = chunk_documents(documents, chunk_words=args.chunk_words, overlap_words=args.overlap_words)
    index = BM25Index(chunks)
    index.save(args.index)
    print(f"Indexed {len(documents)} document sections into {len(chunks)} chunks at {args.index}")


def ask(args: argparse.Namespace) -> None:
    index = BM25Index.load(args.index)
    results = index.search(args.question, top_k=args.top_k)
    answer = build_answer(args.question, results, max_sentences=args.sentences)
    print(format_answer(answer, show_context=args.show_context))


def shell(args: argparse.Namespace) -> None:
    index = BM25Index.load(args.index)
    print("Grounded Doc QA shell. Type 'exit' or press Ctrl+C to quit.")
    while True:
        try:
            question = input("\nquestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if question.lower() in {"exit", "quit"}:
            return
        if not question:
            continue
        results = index.search(question, top_k=args.top_k)
        answer = build_answer(question, results, max_sentences=args.sentences)
        print(format_answer(answer, show_context=args.show_context))


if __name__ == "__main__":
    main()
