from types import SimpleNamespace

from pipeline.narrative.promise_scorer import PromiseResult
from pipeline.quality_gates.validator import run_all_gates


def _ctx(**kw):
    base = dict(bse_scrip="500000", nse_symbol="ABC", cin="L00000", annual_reports=[{}] * 5,
                transcripts=[{}] * 8, ratios={"roce": 20.0}, promise_delivery=[], trust_score=0.0)
    base.update(kw)
    return SimpleNamespace(**base)


def _missed():
    return PromiseResult("FY23", "q", "m", "1", "FY25", "0", "missed", "AR", -100.0)


def test_seven_gates_in_order():
    res = run_all_gates(_ctx())
    assert [g.gate for g in res] == [1, 2, 3, 4, 5, 6, 7]


def test_gate_outcomes_full_context():
    res = {g.gate: g for g in run_all_gates(_ctx(promise_delivery=[_missed()], trust_score=-3.5))}
    assert res[1].passed and res[2].passed and res[3].passed
    assert not res[4].passed and res[4].detail == "Not yet implemented"
    assert res[5].passed
    assert not res[6].passed
    assert res[7].passed and "-3.5%" in res[7].detail


def test_gate_failures_on_thin_context():
    res = {g.gate: g for g in run_all_gates(_ctx(nse_symbol="", annual_reports=[{}] * 2,
                                                 transcripts=[], ratios={}))}
    assert not res[1].passed and "NSE=MISSING" in res[1].detail
    assert not res[2].passed and "annual_reports: 2/5" in res[2].detail
    assert not res[3].passed
    assert not res[5].passed  # no missed/dropped promises surfaced
    assert not res[7].passed
