# 2025 — Book RAG: from scanned pages to a question-answering platform

## Problem

In April–June 2025 I built a pipeline to turn a scanned investing book into
something a reader could question: recover the text and layout from page
images, rebuild the chapter/section structure, chunk and embed it, store it in
PostgreSQL with pgvector, and answer questions with retrieval plus an LLM.
Around it sat topic clustering, question generation and persona-styled
rewrites for a learning app. In late May 2025 the same service was repointed
at bank annual reports and earnings material.

This folder is a cleaned subset of that private repository.

## Pipeline

1. **OCR and layout.** The original repo used Tesseract (`pytesseract`),
   OpenCV, PyMuPDF, Google Document AI and Adobe PDF Extract across several
   experiments. Only the PyMuPDF-based code is here
   (`src/extraction/`); the Tesseract/OpenCV, Document AI and Adobe
   scripts remained in the private original because they are tied to
   credentials, cloud projects and the book's page images.
2. **Structure recovery.** `DocumentStructureAnalyzer` reads PyMuPDF span
   font sizes and bold flags, finds chapter headings (pattern + size relative
   to the median), sections inside each chapter, figures, and table-like
   blocks. `smart_extraction.py` scores each page for table/figure
   likelihood (regex cues, ruled lines, aligned rows, colour diversity) and
   extracts only from candidate pages.
3. **Chunking.** LlamaIndex `SentenceSplitter` (1024 tokens, 100 overlap)
   in the RAG service here; other scripts in the original used LangChain
   `RecursiveCharacterTextSplitter` (not copied).
4. **Embeddings.** BGE models: `bge-small-en-v1.5` (384-d) for the book
   index, `bge-base-en-v1.5` for the bank-table backfill,
   `bge-large-en-v1.5` in the chat service, and `bge-m3` via Ollama in the
   annual-reports scaffold.
5. **Vector store.** PostgreSQL + pgvector. The book indexer and the
   embedding backfill create an `ivfflat` index with
   `vector_cosine_ops`; the chat service's LlamaIndex store is configured
   with HNSW parameters.
6. **Retrieval and answers.** Top-k retrieval, then a rule-based filter that
   prefers chunks whose metadata flags (management commentary, financial
   metrics) match the query type, then an answer from Claude via the Anthropic
   API, plus rule-based follow-up questions. Across the original repo, OpenAI
   and Anthropic APIs and local Ollama models (DeepSeek-R1 distill, Qwen 2.5)
   were used; the local-model path is in `src/annual_reports_scaffold/`.
7. **Topic clustering.** TF-IDF + KMeans over sub-topic titles, with a
   keyword-count cluster namer; a separate step splits oversized clusters to a
   maximum size. (The original redistributor imported `silhouette_score`
   but never used it — the rebalancing is size-based. The demo computes a
   silhouette score only as a reported check.)
8. **Generation for the learning app.** Rule-based heading-to-question
   conversion and a rule-based persona rewrite (term-to-metaphor substitution
   for a "sweet-maker" or "chess" voice). Neither calls an LLM.
9. **May 2025 pivot.** From late May the service was pointed at Indian bank
   annual reports and earnings presentations: filename-derived metadata
   (document type, quarter, fiscal year), content flags, and a
   management-focused prompt. That line of work continues in
   [../02-document-intelligence-2025](../02-document-intelligence-2025).

## Files

| File | What it demonstrates |
|---|---|
| `src/extraction/document_structure.py` | Chapter/section/figure/table recovery from PyMuPDF font metrics |
| `src/extraction/smart_extraction.py` | Page-level table/figure likelihood scoring, targeted table extraction |
| `src/indexing/semantic_indexer_postgres.py` | LlamaIndex + BGE-small + pgvector table with ivfflat cosine index |
| `src/indexing/generate_bge_embeddings.py` | Batch BGE embedding backfill into pgvector tables + ivfflat index per table |
| `src/rag/annual_report_rag_chat.py` | Core RAG service: SentenceSplitter chunking, filename metadata, query-type filtering, retrieval, Claude answer, follow-ups (FastAPI + CLI) |
| `src/clustering/semantic_topic_clustering.py` | TF-IDF + KMeans topic clustering and cluster naming |
| `src/clustering/cluster_redistributor.py` | Size-capped split of oversized clusters |
| `src/generation/question_generator.py` | Heading-to-question rules |
| `src/generation/persona_transformer.py` | Rule-based persona rewrite |
| `src/annual_reports_scaffold/` | **Scaffold** (May 2025): local-model RAG over annual reports (Ollama bge-m3 + DeepSeek), Ollama model manager, and a processor stub whose visual extractor is not included and whose text/embedding steps were never written |
| `demo.py` | Offline end-to-end demo on a synthetic PDF |

## Run the demo

Needs only PyMuPDF, numpy, pandas, scikit-learn and tqdm; no keys, no
database, no network:

```
pip install PyMuPDF numpy pandas scikit-learn tqdm
python demo.py
```

`demo.py` writes a 4-page synthetic PDF (original neutral text about reading
an annual report, three chapters, one small ruled table) to a temp directory
and runs the extraction, clustering and generation code on it. Captured output
(stdout; progress bars go to stderr):

```
[1] Built synthetic PDF: 4 pages
[2] DocumentStructureAnalyzer: 3 chapters, 6 sections, 0 tables
    Ch 1 (p1): Reading the Annual Report
        - All Numbers Need Context
        - Reading the Auditor Report
    Ch 2 (p2): Understanding the Financial Statements
        - Cash vs. Profit
        - Reading the Balance Sheet
    Ch 3 (p4): Comparing Companies
        - Choosing a Peer Group
        - Quality Control in Comparisons
[3] smart_extraction table likelihood per page: [0.4, 0.4, 1.0, 0.2]
    detect_tables_from_text(p3): 5 rows x 4 cols; header ['Metric', 'FY2023', 'FY2024', 'FY2025']
[4] TF-IDF + KMeans on 12 paragraphs, k=3:
    cluster 0 (6 paragraphs): key terms = 'the a and'
    cluster 1 (2 paragraphs): key terms = 'growth revenue means'
    cluster 2 (4 paragraphs): key terms = 'and profit cash'
    silhouette (cosine, demo check): 0.073
    ClusterRedistributor(max=4): sizes [6, 4, 2] -> 4 clusters, max size 4
[5] Questions from section headings:
    'All Numbers Need Context'         -> What numbers need context?
    'Reading the Auditor Report'       -> How do you read  the auditor report?
    'Cash vs. Profit'                  -> What is the relationship between cash and  profit?
    'Reading the Balance Sheet'        -> How do you read  the balance sheet?
    'Choosing a Peer Group'            -> How do you choos  a peer group?
    'Quality Control in Comparisons'   -> How was quality controlled in comparisons?
    persona='chess' rewrite of one paragraph:
      ♟️ Looking at this from a chess master's perspective... 
      ♟️ Size matters too: margins, gambit and valuation multiples often differ between large and small companies competing in the same board.
      👑 As in chess, the best investors think several moves ahead, maintain strategic patience, and know when to convert a small advantage into a winning position.
(intermediate files written to a temporary directory)
```

What the output shows, including the rough edges of the 2025 code (kept as-is):

- Structure recovery finds all 3 chapters and 6 sections. Its own table
  heuristic finds 0 tables here because each cell of the synthetic table is a
  separate PyMuPDF block; `smart_extraction` scores the table page 1.0 and
  recovers the 5×4 table.
- The cluster namer counts raw words without removing stop words, so one
  cluster is labelled `the a and`. The silhouette score on 12 short
  paragraphs is low (0.07); this is a toy corpus.
- The heading-to-question rules strip "ing" naively ("choos", double spaces).

The pgvector indexer, embedding backfill and RAG chat service need
PostgreSQL with pgvector, model downloads and (for the chat service) an
Anthropic API key, so they are not run by the demo. The scaffold needs a
local Ollama server.

## Edits made when copying

- CRLF → LF; files renamed/placed under `src/` in snake_case.
- Hard-coded database names, users, passwords, hosts, data directories and
  model ids replaced with environment variables (`.env.example` lists the
  names) or CLI arguments; personal and machine-specific paths removed.
- Removed an import-time `nltk.download` (network) and the unused NLTK
  import; log files no longer written to the working directory.
- Small bug fixes needed for the code to run on current PyMuPDF: chapter
  headings now store their font size (the section step read a missing key);
  span x-positions read from `bbox` (spans have no `x0` key); drawing line
  items unpacked as two points.
- `semantic_topic_clustering.py`: core TF-IDF+KMeans step factored into
  `cluster_titles()` so it can run on in-memory data.
- `hdfc_complete_platform_langchain.py` (≈1,200 lines) trimmed to
  `annual_report_rag_chat.py` (≈430 lines): removed the embedded HTML UI,
  live market-data lookups, a hard-coded bank list and the canned demo-mode
  answer; the issuer name and data directory are now configuration.
- Example comments in `question_generator.py` replaced with neutral
  headings.

## What's not here and why

- **The book's content.** Extracted text, page images, layout-analysis
  images, generated Q&A, MCQs and paragraph datasets are copyrighted
  material and are excluded. No book text appears in this folder.
- **Credentials.** Service-account files, `.env` files and API keys are
  excluded; only variable names are listed in `.env.example`.
- **Bulk data and artefacts.** Databases, spreadsheets, checkpoints,
  processed outputs, virtual environments and logs are excluded.
- **Other experiments.** Tesseract/OpenCV OCR, Google Document AI and Adobe
  PDF Extract scripts, the LangChain splitter scripts and the web front ends
  stayed in the private original.

## Status

Preserved 2025 code, lightly cleaned so it compiles and the offline demo
runs. Not maintained.
