# AI engineering portfolio — Bharat Ankaraju

[![ci](https://github.com/bankaraju/ai-engineering-portfolio/actions/workflows/ci.yml/badge.svg)](https://github.com/bankaraju/ai-engineering-portfolio/actions/workflows/ci.yml)

Three stages of work on one problem: **getting reliable answers out of financial documents**.
It starts with retrieval-augmented generation over a scanned book, grows into a document-intelligence
platform for bank and company filings, then moves to LLM-based claim scoring. It ends where the
numbers are computed deterministically and the language model only narrates.

The flagship project, where that last step lives, is
**[anka-governed-research](https://github.com/bankaraju/anka-governed-research)**: a governed
equity-research system in which every number carries its source, method and reporting basis, or is
refused by name.

This repository holds cleaned extracts of the earlier stages, so a reader can see how the design got
there.

## Timeline

| When | Stage | What it is | Main techniques |
|---|---|---|---|
| Apr–Jun 2025 | [01 · Book RAG](01-book-rag-2025/) | Scanned book → structured text → vector search → Q&A, topic clusters, generated questions, persona rewrites | PyMuPDF layout analysis, OCR pipelines, BGE embeddings, PostgreSQL + pgvector, LlamaIndex, TF-IDF + KMeans, OpenAI / Claude / local Ollama models |
| 2025 | [02 · Document intelligence](02-document-intelligence-2025/) | Project-based platform: ingest annual reports and filings, extract tables, answer analyst questions | Google Document AI, table-extraction cascade (camelot, tabula, pdfplumber, OCR), semantic chunking, LlamaIndex multi-index routing, LangChain RetrievalQA, spaCy NER, FinBERT sentiment, DBSCAN, FastAPI |
| Apr 2026 | [03 · Trust Score](03-trust-score-2026/) | Extract management's forward guidance from annual reports with an LLM, score it against what was later reported, grade credibility | Structured LLM extraction, JSON repair, deterministic filters, golden-set example, quality gates, pytest with a mocked LLM |
| 2026 | [anka-governed-research](https://github.com/bankaraju/anka-governed-research) | Numbers computed from filed data ahead of time, checked, basis-labelled; the LLM writes only around proven facts | Contract-driven design, refusal codes, mutation testing, CI |

## Why the design moved

- **01 → 02.** Retrieval over one book worked. Retrieval over hundreds of filings needed better
  ingestion: real table extraction, document classification, several indexes and query routing.
- **02 → 03.** Retrieval plus an LLM found the right passage but could not guarantee the number in
  the answer. So I narrowed the model's job to one it does well: finding and structuring
  *claims*, then checking those claims against reported figures.
- **03 → Anka.** Even there, the grade was read from the model's own summary. The saved example in
  `03-trust-score-2026/golden_set/` shows that summary disagreeing with its own rows, and a test
  pins it. The fix became a rule in the next system: anything numeric is computed from the rows and
  the filings, never taken from the model's say-so.

## Skills, with where to see them

| Area | Evidence |
|---|---|
| Document AI and OCR | `01/src/extraction/`, `02/src/document_ingestion_pipeline.py`, `02/src/table_extraction_cascade.py` |
| RAG: chunking, embeddings, vector stores | `01/src/indexing/`, `01/src/rag/`, `02/src/semantic_document_pipeline.py`, `02/src/multi_index_retrieval.py`, `02/src/langchain_banking_rag.py` |
| Classical ML and NLP | TF-IDF + KMeans (`01/src/clustering/`); spaCy NER, FinBERT, DBSCAN (`02/src/narrative_pattern_mining.py`); scikit-learn regression (`02/src/banking_platform_api.py`) |
| LLM engineering | Several providers (OpenAI, Anthropic, Gemini, local Ollama), prompt design for structured JSON, repair of malformed replies (`03/run_trust_score.py`) |
| Evaluation and testing | Golden set and quality gates (`03/`); mutation testing and CI ([anka-governed-research](https://github.com/bankaraju/anka-governed-research)) |
| Services | FastAPI services (`01/src/rag/`, `02/src/banking_platform_api.py`) |
| Deterministic, auditable pipelines | [anka-governed-research](https://github.com/bankaraju/anka-governed-research) |

Also prototyped, not included here: dashboard front ends built with an AI app builder (React/Vite),
and an MCP server that exposed the analysis API as tools for AI agents.

## Run something

```bash
git clone https://github.com/bankaraju/ai-engineering-portfolio && cd ai-engineering-portfolio

# 01: offline demo on a generated PDF (no API keys, no network)
pip install PyMuPDF numpy pandas scikit-learn tqdm
cd 01-book-rag-2025 && python demo.py && cd ..

# 03: trust-score tests, LLM mocked
pip install -r 03-trust-score-2026/requirements.txt
cd 03-trust-score-2026 && python -m pytest -v tests
```

02 has no offline command. Every file needs external services (PostgreSQL + pgvector, Google
Document AI / Drive, a Claude API key); its README lists what each file needs. It is here to be read.

CI runs both on Python 3.10–3.12 and compiles every file in the repo.

## Honest notes

- **Preserved, not maintained.** This is earlier work, lightly cleaned: credentials, hosts and paths
  moved to environment variables, personal paths removed, a few bugs fixed so the demos run. Each
  section's README says what runs offline and what needs external services (PostgreSQL + pgvector,
  Google Document AI, an LLM API key).
- **What is left out.** The scanned book's text and images (copyrighted), client and bulk data,
  credentials, and working notes.
- **Known weaknesses are left visible,** not hidden. Each section README lists them: unimplemented
  gates, a JSON-repair bug recorded as an expected test failure, and rough edges in the question
  generator.
- **Tools.** The design and engineering decisions are mine. Throughout, I used AI coding assistants
  and several language models as part of the workflow.
