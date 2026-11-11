"""evaluation/retrieval/retrieval_scores.py 테스트 (검색 판정 자리).

지금은 계산하지 않는 함수들이 오류 없이 불리고, Langfuse에 올릴 점수를 만들지 않는지만 본다.
"""

from __future__ import annotations

import importlib.util

from rag_service.storage.corpus import repo_root


def _load():
    path = repo_root() / "evaluation" / "retrieval" / "retrieval_scores.py"
    spec = importlib.util.spec_from_file_location("retrieval_scores", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rs = _load()

REC = {
    "question": "질문",
    "status": "answered",
    "contexts": [{"sid": "S1", "book_id": "SB01", "page": 3, "text": "본문"}],
    "error": None,
}
GOLDEN = {"evidence_groups": [[{"book_id": "SB01", "page": 3, "quote": "가"}]]}


def test_functions_run_without_error_and_score_nothing():
    for fn in (rs.hit_at_k, rs.recall_at_k, rs.mrr_at_k):
        assert fn(REC, GOLDEN) is None
        assert fn(REC, None, k=3) is None
    assert rs.score_retrieval(REC, GOLDEN) == []
    assert rs.score_retrieval({"status": "error"}, None, k=rs.K) == []
