"""retrievers.py 테스트.

기본 테스트: get_store를 메모리 Qdrant로 바꿔 끼워서 팀 Qdrant·모델 없이 돈다.
integration 테스트: 팀 Qdrant(읽기 전용)와 bge-m3로 chain.py와 같은 방식으로 부른다.

실행 (레포 맨 위에서)
    uv run pytest tests/retrieval_b/test_retrievers.py -q
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rag_service.retrieval_b import (
    build_hybrid_retriever,
    build_naive_retriever,
    retrievers,
)
from rag_service.retrieval_b.bm25 import BM25Index
from rag_service.storage.corpus import repo_root

load_dotenv(repo_root() / ".env")

QUERY = {"q": [1.0, 0.0]}


def meta(chunk_id: str, level: str, page: int) -> dict:
    return {"chunk_id": chunk_id, "book_id": "SB01", "page": page, "level": level}


# 질문 q=[1,0]과의 코사인: page 1.0, table 0.9950, para 0.7071
ROWS = [
    ("p", [1.0, 0.0], meta("SB01-p1", "page", 1)),
    ("t", [1.0, 0.1], meta("SB01-p2-t1", "table", 2)),
    ("a", [1.0, 1.0], meta("SB01-p3-c1", "para", 3)),
]


@pytest.fixture
def fake_get_store(make_store, monkeypatch):
    """get_store를 메모리 Qdrant로 바꾸고, 불린 (domain, difficulty)를 기록한다."""
    store = make_store(ROWS, QUERY)
    calls = []

    def _get_store(domain, difficulty):
        calls.append((domain, difficulty))
        return store

    monkeypatch.setattr(retrievers, "get_store", _get_store)
    return calls


def ids(docs) -> list[str]:
    return [d.metadata["chunk_id"] for d in docs]


def test_is_langchain_retriever(fake_get_store):
    assert isinstance(build_naive_retriever(), BaseRetriever)


def test_default_levels_skip_page(fake_get_store):
    """기본 levels는 para+table → page 청크는 점수가 제일 높아도 안 나온다."""
    docs = build_naive_retriever().invoke("q")
    assert ids(docs) == ["SB01-p2-t1", "SB01-p3-c1"]


def test_invoke_keeps_score_and_chain_fields(fake_get_store):
    for doc in build_naive_retriever().invoke("q"):
        for key in ("chunk_id", "book_id", "page", "level", "score"):
            assert key in doc.metadata


def test_k_and_levels_are_passed(fake_get_store):
    retriever = build_naive_retriever(k=1, levels=["page", "para", "table"])
    assert ids(retriever.invoke("q")) == ["SB01-p1"]


def test_domain_and_difficulty_go_to_get_store(fake_get_store):
    build_naive_retriever(domain="shipping", difficulty="advanced")
    assert fake_get_store == [("shipping", "advanced")]


def test_chain_style_call(fake_get_store):
    """chain.py처럼 키워드 domain=, difficulty=만 줘도 만들어진다."""
    retriever = build_naive_retriever(domain="shipping", difficulty="basic")
    assert retriever.k == 5
    assert retriever.levels == ["para", "table"]


def test_levels_are_copied(fake_get_store):
    """넘긴 리스트나 기본값 리스트를 나중에 바꿔도 검색기 설정은 그대로여야 한다."""
    mine = ["page"]
    retriever = build_naive_retriever(levels=mine)
    mine.append("para")
    assert retriever.levels == ["page"]
    build_naive_retriever().levels.append("page")
    assert retrievers.DEFAULT_LEVELS == ["para", "table"]


def test_empty_question_raises(fake_get_store):
    with pytest.raises(ValueError):
        build_naive_retriever().invoke("   ")


# ---------- integration: 팀 Qdrant (읽기 전용) + bge-m3 ----------


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("QDRANT_URL"), reason=".env에 QDRANT_URL 없음")
def test_team_naive_retriever_like_chain():
    retriever = build_naive_retriever(domain="shipping", difficulty="basic")
    docs = retriever.invoke("부산에서 LA로 컨테이너를 예약할 때 필요한 정보는?")
    assert len(docs) == 5
    assert docs[0].metadata["book_id"] == "SB01"
    assert all(d.metadata["level"] in ("para", "table") for d in docs)


# ---------- Hybrid: Dense + BM25 → RRF ----------

# 질문 "gamma"=[1,0]. Dense 순위: A(1.0) > B(0.8) > C(0.0) ...
# BM25(띄어쓰기 토큰): "gamma"가 들어간 건 B뿐 → B가 두 검색기 모두에 나와 1위가 된다.
HYBRID_ROWS = [
    ("alpha beta", [1.0, 0.0], meta("SB01-p1-c1", "para", 1)),
    ("beta gamma", [0.8, 0.6], meta("SB01-p2-c1", "para", 2)),
    ("delta", [0.0, 1.0], meta("SB01-p3-c1", "para", 3)),
    ("epsilon", [0.0, 1.0], meta("SB01-p4-c1", "para", 4)),
    ("zeta", [0.0, 1.0], meta("SB01-p5-c1", "para", 5)),
    ("page all", [1.0, 0.0], meta("SB01-p1", "page", 1)),
]


@pytest.fixture
def fake_hybrid(make_store, monkeypatch):
    """get_store는 메모리 Qdrant, get_bm25_index는 작은 색인으로 바꾼다."""
    store = make_store(HYBRID_ROWS, {"gamma": [1.0, 0.0]})
    calls = []

    def _get_bm25_index(domain, difficulty, levels):
        calls.append((domain, difficulty, levels))
        docs = [
            Document(page_content=text, metadata=m)
            for text, _, m in HYBRID_ROWS
            if m["level"] in levels
        ]
        return BM25Index(docs, tokenizer=str.split)

    monkeypatch.setattr(retrievers, "get_store", lambda domain, difficulty: store)
    monkeypatch.setattr(retrievers, "get_bm25_index", _get_bm25_index)
    return calls


def test_hybrid_is_langchain_retriever(fake_hybrid):
    assert isinstance(build_hybrid_retriever(), BaseRetriever)


def test_hybrid_fuses_dense_and_bm25(fake_hybrid):
    docs = build_hybrid_retriever(k=2).invoke("gamma")
    assert ids(docs) == ["SB01-p2-c1", "SB01-p1-c1"]
    top = docs[0].metadata
    assert (top["dense_rank"], top["bm25_rank"]) == (2, 1)
    assert docs[1].metadata["bm25_rank"] is None
    for key in ("chunk_id", "book_id", "page", "level", "score"):
        assert key in top


def test_hybrid_uses_same_levels_for_both(fake_hybrid):
    docs = build_hybrid_retriever(k=10).invoke("gamma")
    assert all(d.metadata["level"] == "para" for d in docs)
    assert fake_hybrid == [("shipping", "basic", ("para", "table"))]


def test_hybrid_candidate_k_never_below_k(fake_hybrid):
    retriever = build_hybrid_retriever(k=4, candidate_k=1)
    assert len(retriever.invoke("gamma")) == 4


def test_hybrid_chain_style_defaults(fake_hybrid):
    retriever = build_hybrid_retriever(domain="shipping", difficulty="basic")
    assert (retriever.k, retriever.candidate_k, retriever.rrf_k) == (5, 30, 60)
    assert retriever.levels == ["para", "table"]


# ---------- __main__: 명령줄 확인 도구 ----------


def test_main_prints_ranked_lines(fake_get_store, capsys):
    from rag_service.retrieval_b.__main__ import main

    main(["q", "--k", "1", "--levels", "page", "para", "table"])
    out = capsys.readouterr().out.strip().splitlines()
    assert out == ["1. SB01-p1 SB01 p1 1.0000 p"]


def test_main_hybrid_mode(fake_hybrid, capsys):
    from rag_service.retrieval_b.__main__ import main

    main(["gamma", "--mode", "hybrid", "--k", "1"])
    out = capsys.readouterr().out.strip().splitlines()
    assert out == ["1. SB01-p2-c1 SB01 p2 0.0325 beta gamma"]
