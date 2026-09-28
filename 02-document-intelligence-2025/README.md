# 2025 — Document intelligence & RAG platform (project-based work)

## Problem

Analysts covering Indian banks and listed companies work from long annual
reports, quarterly results, investor presentations and exchange filings, much
of it in PDFs with dense tables. The goal of this 2025 platform was to ingest
those documents, structure them (sections, tables, narrative), index them, and
let an analyst ask questions such as "how has the working-capital cycle changed
from FY22 to FY24?" or "what did management guide on margins?" and get an
answer grounded in the filings.

## Architecture (as supported by the code in `src/`)

- **Ingestion:** Google Drive folder scanning and download, Google Document AI
  for structured extraction, PyMuPDF for page/visual extraction; a table
  extraction cascade over camelot (lattice/stream), tabula, pdfplumber and OCR
  fallbacks (docTR, EasyOCR, Tesseract).
- **Classification:** regex/keyword rules to classify documents (annual report,
  quarterly results, presentation, …), detect fiscal year / quarter from file
  names, and separate text from tables.
- **Chunking:** semantic, structural and hybrid chunking strategies using BGE
  sentence embeddings, plus a chunk similarity graph linking semantically
  related chunks that are not adjacent in the document.
- **Embeddings:** BGE (`BAAI/bge-large-en-v1.5` / `bge-base-en-v1.5`) via
  sentence-transformers and HuggingFace embedding wrappers.
- **Vector storage:** PostgreSQL + pgvector (LlamaIndex `PGVectorStore`,
  LangChain `PGVector`); a FAISS store in the semantic pipeline. `chromadb` is
  imported in the semantic pipeline but not wired to anything.
- **Retrieval:** LlamaIndex multi-index setup (separate financial-statement,
  narrative and notes indices, metadata filters, HyDE query transform,
  post-processors) and LangChain `RetrievalQA`.
- **LLM:** Claude via LangChain (`ChatAnthropic`) and the Anthropic SDK; the
  model id is a single setting (`ANTHROPIC_MODEL`).
- **Analytics:** market and macro data via yfinance; regression of bank
  valuations on macro factors with scikit-learn `LinearRegression`; scenario
  stress tests with sector betas; Excel export via xlsxwriter.
- **NLP pattern mining:** spaCy NER, transformer sentiment (FinBERT), BGE
  embeddings and DBSCAN clustering to find recurring management-narrative
  patterns (guidance, risk language) across periods.
- **Delivery:** FastAPI service with a chat-style HTML front end.

## Files

| File | What it demonstrates |
|---|---|
| `src/document_ingestion_pipeline.py` | Drive scanning, Document AI extraction, hierarchical section extraction, PyMuPDF visual extraction, audit trail of every operation |
| `src/table_extraction_cascade.py` | Ordered fallback cascade of table/OCR extractors with per-tool stats and a validation step |
| `src/semantic_document_pipeline.py` | Semantic / structural / hybrid chunking with BGE, chunk similarity graph, LLM-built table of contents, FAISS retrieval chain |
| `src/multi_index_retrieval.py` | LlamaIndex hierarchical multi-index retrieval over pgvector with metadata filters and HyDE |
| `src/langchain_banking_rag.py` | Compact LangChain RetrievalQA over PGVector with a grounded-answer prompt |
| `src/narrative_pattern_mining.py` | Narrative extraction (regex patterns, spaCy NER, FinBERT sentiment) and DBSCAN pattern discovery |
| `src/banking_platform_api.py` | FastAPI banking analysis service: query routing, pgvector retrieval, Claude analysis, macro regression, stress tests, Excel export, HTML UI |
| `src/config/setup.py`, `src/ingestion/google_drive_monitor.py`, `src/vertex_ai_client.py` | Small stand-ins for modules of the original codebase that are not published here, so local imports resolve |

## Status

Preserved as written in 2025, with credentials, hosts and file paths moved to
environment variables (see `.env.example`) and a few cosmetic edits. Not
maintained. Running it needs external services: PostgreSQL with pgvector,
Google Document AI and Drive credentials, and an Anthropic API key. The files
were written against different LlamaIndex versions (see `requirements.txt`),
so they do not all import in one environment.

## What I learned / why I moved on

Retrieval plus an LLM was good at finding the right passage and summarising
narrative, but it could not guarantee that the numbers in an answer were
correct. For number-heavy financial work that is the part that matters most:
a ratio off by a period, a figure in the wrong unit, or a value taken from the
wrong table reads as plausible and is hard to catch. That led to a different
design — compute the numbers deterministically from filed data ahead of time,
check them, and let the language model only narrate facts it is handed — which
is the approach in
[anka-governed-research](https://github.com/bankaraju/anka-governed-research).
