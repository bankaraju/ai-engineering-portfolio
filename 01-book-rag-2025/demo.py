#!/usr/bin/env python3
"""
Offline demo: no API keys, no database, no network.

1. Builds a small synthetic 3-chapter PDF with PyMuPDF (neutral original text
   about reading an annual report, plus one small table).
2. Runs DocumentStructureAnalyzer on it (chapters / sections / tables).
3. Scores pages for table likelihood with smart_extraction.
4. Clusters the PDF's paragraphs with TF-IDF + KMeans (the same core used by
   semantic_topic_clustering), reports a silhouette score, and runs the
   size-capped ClusterRedistributor.
5. Turns section headings into questions and applies one persona rewrite.

The RAG / pgvector / LLM code under src/indexing and src/rag needs PostgreSQL,
model downloads and API keys, so it is not exercised here.
"""
import json
import logging
import os
import sys
import tempfile

import fitz  # PyMuPDF

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from src.extraction.document_structure import DocumentStructureAnalyzer  # noqa: E402
from src.extraction.smart_extraction import is_table_likely, detect_tables_from_text  # noqa: E402
from src.clustering.semantic_topic_clustering import cluster_titles  # noqa: E402
from src.clustering.cluster_redistributor import ClusterRedistributor  # noqa: E402
from src.generation.question_generator import generate_question_from_heading  # noqa: E402
from src.generation.persona_transformer import transform_content  # noqa: E402

logging.disable(logging.CRITICAL)  # keep demo output to the printed summary

# ---------------------------------------------------------------------------
# Synthetic document (original neutral text written for this demo)
# ---------------------------------------------------------------------------
CHAPTERS = [
    ("CHAPTER 1: Reading the Annual Report", [
        ("All Numbers Need Context", [
            "An annual report mixes audited statements with management commentary. "
            "Read the statements first and treat the commentary as a claim to be checked.",
            "Revenue growth means little on its own. Compare it with the growth in receivables "
            "and inventory to see whether sales are turning into cash.",
        ]),
        ("Reading the Auditor Report", [
            "The auditor report states whether the accounts give a true and fair view. "
            "Look for qualifications, emphasis of matter paragraphs and key audit matters.",
            "A change of auditor, or a qualified opinion, is a reason to read the notes to the "
            "accounts slowly before trusting any headline number.",
        ]),
    ]),
    ("CHAPTER 2: Understanding the Financial Statements", [
        ("Cash vs. Profit", [
            "Profit is an accounting measure while cash flow records money that actually moved. "
            "Operating cash flow that lags profit for several years deserves investigation.",
            "Working capital changes explain most gaps between profit and operating cash flow, "
            "especially receivables, inventory and payables.",
        ]),
        ("Reading the Balance Sheet", [
            "The balance sheet shows what the company owns and owes at one date. "
            "Debt, lease liabilities and contingent liabilities all belong in the picture.",
            "Rising borrowings alongside flat revenue can signal that growth is being funded "
            "rather than earned.",
        ]),
    ]),
    ("CHAPTER 3: Comparing Companies", [
        ("Choosing a Peer Group", [
            "Peers should share a business model, not just an industry label. "
            "A lender and a manufacturer in the same index are not comparable.",
            "Size matters too: margins, risk and valuation multiples often differ between large "
            "and small companies competing in the same market.",
        ]),
        ("Quality Control in Comparisons", [
            "Use the same accounting basis for every peer, consolidated or standalone, and the "
            "same fiscal period, before comparing any ratio.",
            "Return on capital, margins and cash conversion travel well across companies; "
            "absolute profit figures do not.",
        ]),
    ]),
]

TABLE = [
    ["Metric", "FY2023", "FY2024", "FY2025"],
    ["Revenue", "1,000", "1,120", "1,260"],
    ["Operating profit", "150", "171", "196"],
    ["Operating margin %", "15.0", "15.3", "15.6"],
    ["Cash from operations", "130", "160", "180"],
]


def build_pdf(path):
    doc = fitz.open()
    for ch_title, sections in CHAPTERS:
        page = doc.new_page()
        y = 72
        page.insert_text((72, y), ch_title, fontname="hebo", fontsize=18)
        y += 36
        for sec_title, paras in sections:
            page.insert_text((72, y), sec_title, fontname="hebo", fontsize=13)
            y += 20
            for para in paras:
                rect = fitz.Rect(72, y, 523, y + 70)
                page.insert_textbox(rect, para, fontname="helv", fontsize=10.5)
                y += 70
            y += 10
        if ch_title.startswith("CHAPTER 2"):
            page = doc.new_page()
            page.insert_text((72, 72), "Table 1: Illustrative five-line summary",
                             fontname="helv", fontsize=10.5)
            ty = 100
            for row in TABLE:
                for c, cell in enumerate(row):
                    page.insert_text((72 + c * 110, ty), cell, fontname="helv", fontsize=10)
                ty += 16
            # grid lines, as a real table would have
            for i in range(len(TABLE) + 1):
                page.draw_line((68, 88 + i * 16), (512, 88 + i * 16))
            for c in range(5):
                page.draw_line((68 + c * 111, 88), (68 + c * 111, 88 + len(TABLE) * 16))
    doc.save(path)
    doc.close()


def main():
    work = tempfile.mkdtemp(prefix="book_rag_demo_")
    pdf_path = os.path.join(work, "synthetic_report_guide.pdf")
    build_pdf(path=pdf_path)
    print(f"[1] Built synthetic PDF: {fitz.open(pdf_path).page_count} pages")

    # 2. Structure recovery
    analyzer = DocumentStructureAnalyzer(pdf_path, os.path.join(work, "structure"))
    structure = analyzer.analyze()
    print(f"[2] DocumentStructureAnalyzer: {len(structure['chapters'])} chapters, "
          f"{sum(len(c['sections']) for c in structure['chapters'])} sections, "
          f"{len(structure['tables'])} tables")
    for ch in structure["chapters"]:
        print(f"    Ch {ch['number']} (p{ch['start_page']}): {ch['title']}")
        for s in ch["sections"]:
            print(f"        - {s['title']}")
    for t in structure["tables"]:
        print(f"    table p{t['page']}: {t['rows']} rows x {t['columns']} cols; "
              f"first row {t['data'][0]}")

    # 3. Smart extraction scoring
    doc = fitz.open(pdf_path)
    scores = [round(is_table_likely(p.get_text(), p), 2) for p in doc]
    print(f"[3] smart_extraction table likelihood per page: {scores}")
    best = max(range(len(scores)), key=scores.__getitem__)
    tables = detect_tables_from_text(doc[best])
    if tables:
        print(f"    detect_tables_from_text(p{best + 1}): {tables[0]['row_count']} rows x "
              f"{tables[0]['col_count']} cols; header {tables[0]['rows'][0]}")

    # 4. Topic clustering on the PDF's own paragraphs (text blocks)
    paragraphs = {}
    for pno, page in enumerate(doc):
        for b in page.get_text("blocks"):
            text = " ".join(b[4].split())
            if len(text.split()) >= 15:  # body paragraphs, not headings/table cells
                paragraphs[f"p{pno + 1}_b{b[5]}"] = text
    k = 3
    clusters = cluster_titles(paragraphs, n_clusters=k)
    print(f"[4] TF-IDF + KMeans on {len(paragraphs)} paragraphs, k={k}:")
    for cid, info in clusters.items():
        ids = [s["id"] for s in info["subtopics"]]
        print(f"    cluster {cid} ({len(ids)} paragraphs): "
              f"key terms = {info['cluster_name'].split(': ', 1)[1]!r}")

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import silhouette_score
    X = TfidfVectorizer(stop_words="english").fit_transform(list(paragraphs.values()))
    labels = {s["id"]: cid for cid, info in clusters.items() for s in info["subtopics"]}
    sil = silhouette_score(X, [labels[i] for i in paragraphs], metric="cosine")
    print(f"    silhouette (cosine, demo check): {sil:.3f}")

    assign_path = os.path.join(work, "clusters.json")
    emb_path = os.path.join(work, "embeddings.json")
    json.dump(labels, open(assign_path, "w"))
    json.dump({}, open(emb_path, "w"))
    red = ClusterRedistributor(emb_path, assign_path, work, max_cluster_size=4)
    counts, oversized = red.analyze_cluster_distribution()
    new = red.redistribute_clusters()
    print(f"    ClusterRedistributor(max=4): sizes {sorted(counts.values(), reverse=True)} -> "
          f"{len(set(new.values()))} clusters, max size "
          f"{max(list(new.values()).count(c) for c in set(new.values()))}")

    # 5. Question generation + persona rewrite
    print("[5] Questions from section headings:")
    for ch in structure["chapters"]:
        for s in ch["sections"]:
            print(f"    {s['title']!r:34} -> {generate_question_from_heading(s['title'])}")
    sample = CHAPTERS[2][1][0][1][1]
    rewritten = transform_content(sample, persona="chess")
    print("    persona='chess' rewrite of one paragraph:")
    for line in rewritten.splitlines():
        if line.strip():
            print(f"      {line}")
    print("(intermediate files written to a temporary directory)")


if __name__ == "__main__":
    main()
