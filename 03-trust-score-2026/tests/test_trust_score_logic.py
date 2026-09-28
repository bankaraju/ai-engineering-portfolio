"""Offline tests for the deterministic parts of run_trust_score.py."""
import json
from pathlib import Path

import pytest

import run_trust_score as rts

GOLD = Path(__file__).resolve().parents[1] / "golden_set"
STATUSES = {"DELIVERED", "EXCEEDED", "PARTIALLY_DELIVERED", "MISSED",
            "QUIETLY_DROPPED", "TOO_EARLY", "UNVERIFIABLE"}


def test_extract_json_payload_strips_fences_and_trailing_prose():
    resp = '```json\n{"a": 1, "b": [2, 3]}\n```\nExecutive summary: fine.'
    assert json.loads(rts._extract_json_payload(resp)) == {"a": 1, "b": [2, 3]}


def test_extract_json_payload_leading_prose():
    assert json.loads(rts._extract_json_payload('Here you go: {"x": "y"} thanks')) == {"x": "y"}


def test_clean_json_expressions_evaluates_math():
    text = '{"variance_pct": (5.51 - 6.1) / 6.1 * 100, "n": NaN, "m": 1}'
    out = json.loads(rts._clean_json_expressions(text))
    assert out["variance_pct"] == pytest.approx(-9.67, abs=0.01)
    assert out["n"] is None


def test_filter_vague_guidance():
    items = [
        {"target_value": "Rs 5,000 Cr", "exact_quote": "Revenue of Rs 5,000 Cr by FY26", "category": "revenue"},
        {"target_value": "higher", "exact_quote": "we aim higher", "category": "revenue"},
        {"target_value": "8%", "exact_quote": "GDP is projected to grow at 8%", "category": "revenue"},
        {"target_value": "7%", "exact_quote": "x", "category": "macro_economic"},
        {"target_value": "net zero by 2040", "exact_quote": "net zero by 2040", "category": "esg"},
        {"target_value": "net zero by 2045", "exact_quote": "net-zero plants by 2045", "category": "esg"},
        {"target_value": "", "exact_quote": "anything", "category": "capex"},
    ]
    kept = rts._filter_vague_guidance(items)
    assert [k["target_value"] for k in kept] == ["Rs 5,000 Cr", "net zero by 2040"]


def test_golden_set_rows_are_well_formed():
    for name in ("MARUTI_gold_scorecard.json", "MARUTI_exemplar_compact.json"):
        data = json.loads((GOLD / name).read_text(encoding="utf-8"))
        assert data["scorecard"]
        for row in data["scorecard"]:
            assert row["status"] in STATUSES
            assert row["guidance_quote"] and row["target_value"] is not None


def test_golden_exemplar_is_injected_into_scoring_prompt():
    assert rts.GOLD_EXEMPLAR_JSON != "{}"
    assert "__GOLD_EXEMPLAR__" in rts.SCORING_PROMPT_TEMPLATE


def test_insufficient_data_below_five_scoreable():
    out = rts.calculate_trust_score({"summary": {"total_guidance_items": 6, "unverifiable": 2}}, {})
    assert out["verdict"] == "INSUFFICIENT_DATA" and out["trust_score_grade"] == "?"


def test_grade_is_70_30_blend_of_critical_and_overall_rate():
    scoring = {"summary": {"total_guidance_items": 10, "critical_items": 4,
                           "critical_delivery_rate_pct": 50.0, "delivery_rate_pct": 90.0}}
    out = rts.calculate_trust_score(scoring, {})
    assert out["trust_score_pct"] == 62.0  # 0.7*50 + 0.3*90
    assert out["trust_score_grade"] == "B"


def test_grade_is_taken_from_model_summary_not_rows():
    """Characterises an April-2026 property: the rates come from the model's own
    'summary' block; the scorecard rows are not recounted. The MARUTI gold file's
    summary (7 partial / 6 unverifiable, 53.85%) does not match its rows (8 / 5)."""
    gold = json.loads((GOLD / "MARUTI_gold_scorecard.json").read_text(encoding="utf-8"))
    rows = gold["scorecard"]
    assert sum(r["status"] == "PARTIALLY_DELIVERED" for r in rows) == 8
    assert gold["summary"]["partially_delivered"] == 7
    out = rts.calculate_trust_score({**gold, "scorecard": []}, {})
    assert out["trust_score_pct"] == round(0.0 * 0.7 + 53.85 * 0.3, 1)


def test_score_promises_with_mocked_llm(monkeypatch, tmp_path):
    monkeypatch.setattr(rts, "ARTIFACTS", tmp_path)
    captured = {}

    def fake_llm(prompt, max_tokens=4096, role="extraction"):
        captured["prompt"] = prompt
        return 'Result: {"scorecard": [], "summary": {"variance_pct": (90 - 100) / 100 * 100}}'

    monkeypatch.setattr(rts, "call_llm", fake_llm)
    narratives = [{"source_year": "FY2023-2024",
                   "guidance": [{"metric": "BEV launch", "target_value": "1", "exact_quote": "first BEV in 2025"}],
                   "actuals_reported": {"revenue_cr": 100}}]
    out = rts.score_promises("DEMO", {"metrics": {}}, narratives)
    assert out["summary"]["variance_pct"] == -10.0
    assert "first BEV in 2025" in captured["prompt"]
    assert rts.score_promises("DEMO", {}, []) == {"error": "No narratives to score"}


@pytest.mark.xfail(strict=True, reason="known April-2026 gap: a fenced reply that also contains a "
                   "math expression keeps the closing fence, so json.loads fails")
def test_fenced_reply_with_math_expression():
    resp = '```json\n{"variance_pct": (90 - 100) / 100 * 100}\n```'
    text = rts._clean_json_expressions(rts._extract_json_payload(resp))
    assert json.loads(text)["variance_pct"] == -10.0


@pytest.mark.llm
def test_live_llm_roundtrip():
    out = rts.call_llm('Return the JSON object {"ok": true} and nothing else.', max_tokens=50)
    assert json.loads(rts._extract_json_payload(out))["ok"] is True
