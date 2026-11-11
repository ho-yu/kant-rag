"""evaluation/retrieval/retrieval_scores.py 테스트 (발제 STEP 6 검색 지표)."""

from __future__ import annotations

import importlib.util

import pytest

from rag_service.storage.corpus import repo_root


def _load(rel: str, name: str):
    path = repo_root() / "evaluation" / rel
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rs = _load("retrieval/retrieval_scores.py", "retrieval_scores")
quality = _load("answer/quality.py", "answer_quality")


def ctx(*book_ids):
    return [{"sid": f"S{i}", "book_id": b, "page": 1, "text": "본문"} for i, b in enumerate(book_ids, start=1)]


def rec(*book_ids, status="answered"):
    return {"question": "질문", "status": status, "contexts": ctx(*book_ids), "error": None}


GOLDEN = {"evidence_groups": [[{"book_id": "SB01", "page": 3}], [{"book_id": "SB02", "page": 1}, {"book_id": "SB01", "page": 4}]]}


def test_gold_and_found_docs_dedupe_in_order():
    assert rs.gold_docs(GOLDEN) == ["SB01", "SB02"]
    assert rs.found_docs(rec("SB03", "sb01", "SB01", "SB02"), k=3) == ["SB03", "SB01"]


def test_example_from_assignment_hit_1_recall_half():
    r = rec("SB09", "SB01", "SB01")
    assert rs.hit_at_k(r, GOLDEN) == 1.0
    assert rs.recall_at_k(r, GOLDEN) == 0.5
    assert rs.mrr_at_k(r, GOLDEN) == 0.5


def test_only_top_k_counts():
    r = rec("SB09", "SB08", "SB01")
    assert rs.hit_at_k(r, GOLDEN, k=2) == 0.0
    assert rs.mrr_at_k(r, GOLDEN, k=2) == 0.0
    assert rs.mrr_at_k(r, GOLDEN, k=3) == pytest.approx(1 / 3)


def test_empty_search_result_scores_zero_not_excluded():
    scores = rs.score_retrieval(rec(), GOLDEN)
    assert [s["state"] for s in scores] == ["scored"] * 3
    assert [s["value"] for s in scores] == [0.0, 0.0, 0.0]


def test_score_retrieval_names_and_shape():
    scores = rs.score_retrieval(rec("SB01", "SB02"), GOLDEN, k=3)
    assert [s["name"] for s in scores] == ["hit_at_3", "recall_at_3", "mrr_at_3"]
    assert [s["value"] for s in scores] == [1.0, 1.0, 1.0]
    assert set(scores[0]) == {"name", "value", "state", "cause", "reason"}


@pytest.mark.parametrize(
    "r, golden, state, cause",
    [
        (rec(status="error"), GOLDEN, "unavailable", "service_error"),
        (rec("SB01"), None, "unavailable", "no_golden"),
        (rec("SB01"), {"evidence_groups": []}, "na", "no_evidence"),
        (rec("SB01"), {"evidence_groups": [[]]}, "na", "no_evidence"),
    ],
)
def test_not_scored_cases_are_not_zero(r, golden, state, cause):
    for fn in (rs.hit_at_k, rs.recall_at_k, rs.mrr_at_k):
        assert fn(r, golden) is None
    scores = rs.score_retrieval(r, golden)
    assert all(s["state"] == state and s["cause"] == cause and s["value"] is None for s in scores)


def test_same_score_shape_as_answer_quality():
    args = ("faithfulness", 0.5, "scored", None, "이유")
    assert rs.scoring.score(*args) == quality.score(*args)
    args = ("faithfulness", None, "na", "x", "이유")
    assert rs.scoring.score(*args) == quality.score(*args)


def test_scores_summarize_with_answer_quality_summary():
    rows = [("answered", rs.score_retrieval(rec("SB01"), GOLDEN)), ("abstained", rs.score_retrieval(rec(), {"evidence_groups": []}))]
    summary = quality.summarize(rows)
    assert summary["hit_at_5"]["mean"] == 1.0
    assert summary["recall_at_5"]["mean"] == 0.5
    assert summary["hit_at_5"]["n_na"] == 1
