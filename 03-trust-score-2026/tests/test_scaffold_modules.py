"""The pipeline/narrative modules are April-2026 scaffolds (TODO bodies).
These tests pin what they actually do today."""
from pipeline.narrative.claim_extractor import ManagementClaim, extract_claims, _extract_from_text
from pipeline.narrative.promise_scorer import PromiseResult, score_promises, detect_quietly_dropped


def test_management_claim_fields():
    c = ManagementClaim(
        quarter="FY2023-24", source="annual_report", page_or_timestamp="Page 25",
        claim_text="first BEV to be launched in 2025", category="product_launch",
        target_metric="BEV launch", target_value="1", target_timeline="2025",
    )
    assert c.confidence == 0.0
    assert c.target_timeline == "2025"


def test_extract_from_empty_text_returns_nothing():
    assert _extract_from_text("", "FY24", "annual_report") == []


def test_extract_claims_scaffold_returns_empty_list():
    # LLM call is a TODO in the scaffold, so no claims are produced.
    out = extract_claims([{"md_a_text": "We target Rs 5000 Cr revenue by FY26", "year": "FY24"}],
                         [{"text": "", "quarter": "Q3FY25"}])
    assert out == []


def test_score_promises_scaffold():
    assert score_promises([object()], []) == []
    assert detect_quietly_dropped([], []) == []


def test_promise_result_fields():
    r = PromiseResult("FY23", "x", "revenue", "15%", "FY25", None, "missed", "AR p.10", None)
    assert r.status == "missed"
