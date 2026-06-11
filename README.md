# Grounded Document QA

A small retrieval-based question answering system for document corpora. It ingests local documents, builds a lightweight BM25-style index, and answers questions with extractive, cited responses.

This is designed for the ML Engineer assessment prompt: correctness, grounding, retrieval quality, and readable code are prioritized over UI polish.

## How It Works

1. **Load documents** from a local folder (`.txt`, `.md`, `.csv`, `.json`, and `.pdf` when `pypdf` is installed).
2. **Chunk text** into overlapping passages so retrieval has enough context without making each unit too broad.
3. **Index chunks** with a simple BM25-style lexical retriever implemented in Python.
4. **Answer questions** by retrieving top passages, selecting the most relevant sentences, and returning citations with source file, page, and chunk id when available.

The answer generation is intentionally extractive. That makes the system less fluent than a full LLM RAG pipeline, but it is easier to audit and much less likely to invent unsupported claims.

## Key Decisions

- **Extractive answers over generated answers:** improves grounding and makes citations meaningful.
- **BM25-style retrieval:** fast, dependency-light, and strong for policy/governance documents with named concepts and legal terminology.
- **JSON index format:** easy to inspect, portable, and sufficient for an assessment-scale project.
- **No large dataset committed:** use the sample docs for smoke tests, and download the full Kaggle corpus locally when needed.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

If you do not want editable install, you can also run with:

```bash
python -m docqa.cli --help
```

from the repository root after setting `PYTHONPATH=src`.

## Data

The suggested corpus is:

https://www.kaggle.com/datasets/umerhaddii/ai-governance-documents-data

Do not commit the full dataset. Download it locally, unzip it into `data/ai_governance_documents/`, then build the index:

```bash
docqa ingest data/ai_governance_documents --index storage/index.json
```

For a quick local smoke test, use the included tiny sample corpus:

```bash
docqa ingest sample_docs --index storage/sample_index.json
```

## Usage

Ask a question:

```bash
docqa ask "What should AI governance programs document?" --index storage/sample_index.json
```

Interactive mode:

```bash
docqa shell --index storage/sample_index.json
```

Tune retrieval:

```bash
docqa ask "How should high risk AI systems be monitored?" --index storage/sample_index.json --top-k 5 --sentences 4
```

## Example Queries

### Query

```text
What should AI governance programs document?
```

### Example Response

```text
AI governance programs should document intended purpose, foreseeable risks, evaluation results, oversight responsibilities, and post-deployment monitoring plans.

References:
[1] sample_docs/governance_notes.md
[2] sample_docs/risk_management.txt
```

### Query

```text
How does risk management continue after deployment?
```

### Example Response

```text
Risk management continues after deployment through monitoring, incident review, feedback collection, and periodic reassessment when the system or operating context changes.

References:
[1] sample_docs/risk_management.txt
```

The exact wording may differ slightly because the CLI extracts the highest-scoring supporting sentences from retrieved chunks.

## Limitations

- Lexical retrieval can miss semantically similar wording. A production version should add embeddings and hybrid retrieval.
- PDF extraction quality depends on the source PDFs. Scanned PDFs require OCR, which is not included.
- Answers are extractive summaries, so they may be less natural than LLM-generated responses.
- The system does not perform claim verification beyond requiring retrieved supporting text.

## What I Would Improve Next

- Add dense embeddings and reranking for better recall.
- Add optional LLM synthesis constrained to retrieved context.
- Add evaluation scripts for retrieval recall and citation faithfulness on a small labeled question set.
- Add OCR support for scanned governance PDFs.
