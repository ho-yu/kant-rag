"""retrievers.py 테스트.

기본 테스트: get_store를 메모리 Qdrant로 바꿔 끼워서 팀 Qdrant·모델 없이 돈다.
integration 테스트: 팀 Qdrant(읽기 전용)와 bge-m3로 chain.py와 같은 방식으로 부른다.

실행 (레포 맨 위에서)
    uv run pytest tests/retrieval_b/test_retrievers.py -q

[2026-11-10 리팩토링] 이유: domain·difficulty가 필수가 되고 chain.py 호환·입력 검증이 추가됨 → 효과: chain.py와 같은 호출, 잘못된 입력 차단, 인자>설정 우선순위가 테스트로 보장됨.
"""

from __future__ import annotations

import os
import warnings

import pytest
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rag_service.retrieval_b import (
    build_bm25_retriever,
    build_hybrid_retriever,
    build_naive_retriever,
    build_rerank_retriever,
    retrievers,
)
from rag_service.retrieval_b.bm25 import BM25Index
from rag_service.retrieval_b.settings import Settings
from rag_service.storage.corpus import repo_root

load_dotenv(repo_root() / ".env")

# chain.py는 create_chain(domain, difficulty)로 받은 값을 늘 넘긴다
SCOPE = {"domain": "shipping", "difficulty": "basic"}

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
    assert isinstance(build_naive_retriever(**SCOPE), BaseRetriever)


def test_default_levels_skip_page(fake_get_store):
    """기본 levels는 para+table → page 청크는 점수가 제일 높아도 안 나온다."""
    docs = build_naive_retriever(**SCOPE).invoke("q")
    assert ids(docs) == ["SB01-p2-t1", "SB01-p3-c1"]


def test_invoke_keeps_score_and_chain_fields(fake_get_store):
    for doc in build_naive_retriever(**SCOPE).invoke("q"):
        for key in ("chunk_id", "book_id", "page", "level", "score"):
            assert key in doc.metadata


def test_k_and_levels_are_passed(fake_get_store):
    retriever = build_naive_retriever(**SCOPE, k=1, levels=["page", "para", "table"])
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
    """넘긴 리스트나 설정값을 나중에 바꿔도 검색기 설정은 그대로여야 한다."""
    mine = ["page"]
    retriever = build_naive_retriever(**SCOPE, levels=mine)
    mine.append("para")
    assert retriever.levels == ["page"]
    build_naive_retriever(**SCOPE).levels.append("page")
    assert build_naive_retriever(**SCOPE).levels == ["para", "table"]


def test_empty_question_raises(fake_get_store):
    with pytest.raises(ValueError):
        build_naive_retriever(**SCOPE).invoke("   ")


# ---------- chain.py 호환과 입력 검사 ----------

BUILDERS = [
    build_naive_retriever,
    build_bm25_retriever,
    build_hybrid_retriever,
    build_rerank_retriever,
]


@pytest.mark.parametrize("build", BUILDERS)
def test_exact_chain_call_with_profile_none(fake_hybrid, build):
    """chain.py: build_*(domain=, difficulty=, profile=None). 경고 없이 만들어져야 한다."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        retriever = build(domain="shipping", difficulty="basic", profile=None)
    assert isinstance(retriever, BaseRetriever)


@pytest.mark.parametrize("build", BUILDERS)
def test_profile_is_accepted_but_warns(fake_hybrid, build):
    with pytest.warns(UserWarning, match="profile"):
        build(**SCOPE, profile="bge-m3")


@pytest.mark.parametrize("build", BUILDERS)
def test_domain_and_difficulty_are_required(fake_hybrid, build):
    """기본값이 없어서 빠뜨리면 엉뚱한 컬렉션을 보지 않고 바로 오류가 난다."""
    with pytest.raises(TypeError):
        build(difficulty="basic")
    with pytest.raises(TypeError):
        build(domain="shipping")


@pytest.mark.parametrize("build", BUILDERS)
@pytest.mark.parametrize(
    "domain, difficulty, error",
    [
        ("shiping", "basic", ValueError),
        ("shipping", "expert", ValueError),
        ("", "basic", ValueError),
        (None, "basic", TypeError),
    ],
)
def test_bad_scope_raises_before_connecting(
    fake_hybrid, build, domain, difficulty, error
):
    with pytest.raises(error):
        build(domain=domain, difficulty=difficulty)
    assert fake_hybrid == []  # 색인·Qdrant까지 가기 전에 멈춘다


def test_scope_is_normalized(fake_get_store):
    build_naive_retriever(domain=" Shipping ", difficulty="BASIC")
    assert fake_get_store == [("shipping", "basic")]


def test_arguments_are_keyword_only(fake_get_store):
    with pytest.raises(TypeError):
        build_naive_retriever("shipping", "basic")


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"k": 0}, ValueError),
        ({"k": -1}, ValueError),
        ({"levels": []}, ValueError),
        ({"levels": ["chapter"]}, ValueError),
        ({"k": True}, TypeError),
        ({"k": "5"}, TypeError),
        ({"levels": "para"}, TypeError),
    ],
)
def test_bad_tuning_values_raise_at_build(fake_get_store, kwargs, error):
    with pytest.raises(error):
        build_naive_retriever(**SCOPE, **kwargs)


def test_settings_are_used_when_not_passed(fake_hybrid, default_settings):
    default_settings(k=7, levels=("para",), candidate_k=40, rrf_k=10, rerank_n=60)
    hybrid = build_hybrid_retriever(**SCOPE)
    assert (hybrid.k, hybrid.levels, hybrid.candidate_k, hybrid.rrf_k) == (
        7,
        ["para"],
        40,
        10,
    )
    rr = build_rerank_retriever(**SCOPE, score_fn=lambda q, t: [0.0] * len(t))
    assert (rr.k, rr.hybrid.k) == (7, 60)
    assert build_bm25_retriever(**SCOPE).k == 7


def test_arguments_beat_settings(fake_hybrid, default_settings):
    default_settings(k=7, candidate_k=40)
    hybrid = build_hybrid_retriever(
        **SCOPE, k=2, candidate_k=3, levels=["page", "para"]
    )
    assert (hybrid.k, hybrid.candidate_k, hybrid.levels) == (2, 3, ["page", "para"])


def test_retrievers_ask_for_the_same_index_key(fake_hybrid):
    """chain.py처럼 여러 검색기를 만들어도 BM25 색인 키가 같아서 lru_cache로 한 번만 만든다."""
    build_hybrid_retriever(**SCOPE)
    build_rerank_retriever(**SCOPE, score_fn=lambda q, t: [0.0] * len(t))
    build_bm25_retriever(**SCOPE)
    assert len(fake_hybrid) == 3
    assert set(fake_hybrid) == {("shipping", "basic", ("para", "table"), "qdrant")}


def test_default_settings_match_previous_constants():
    """설정 파일로 옮기기 전 상수(DEFAULT_K, CANDIDATE_K, RRF_K, RERANK_N …)와 같은 값."""
    s = Settings()
    assert (s.k, s.levels, s.candidate_k, s.rrf_k, s.rerank_n) == (
        5,
        ("para", "table"),
        30,
        60,
        50,
    )
    assert (s.rerank_max_length, s.rerank_batch_size) == (1024, 16)


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

    def _get_bm25_index(domain, difficulty, levels, source):
        calls.append((domain, difficulty, levels, source))
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
    assert isinstance(build_hybrid_retriever(**SCOPE), BaseRetriever)


def test_hybrid_fuses_dense_and_bm25(fake_hybrid):
    docs = build_hybrid_retriever(**SCOPE, k=2).invoke("gamma")
    assert ids(docs) == ["SB01-p2-c1", "SB01-p1-c1"]
    top = docs[0].metadata
    assert (top["dense_rank"], top["bm25_rank"]) == (2, 1)
    assert docs[1].metadata["bm25_rank"] is None
    for key in ("chunk_id", "book_id", "page", "level", "score"):
        assert key in top


def test_hybrid_uses_same_levels_for_both(fake_hybrid):
    docs = build_hybrid_retriever(**SCOPE, k=10).invoke("gamma")
    assert all(d.metadata["level"] == "para" for d in docs)
    assert fake_hybrid == [("shipping", "basic", ("para", "table"), "qdrant")]


def test_bm25_source_reaches_index_key(fake_hybrid, default_settings):
    """인자로 준 bm25_source가 색인 키까지 간다. 안 주면 설정값을 쓴다."""
    build_hybrid_retriever(**SCOPE, bm25_source="jsonl")
    build_rerank_retriever(
        **SCOPE, bm25_source="jsonl", score_fn=lambda q, t: [0.0] * len(t)
    )
    default_settings(bm25_source="jsonl")
    build_bm25_retriever(**SCOPE)
    assert [call[3] for call in fake_hybrid] == ["jsonl", "jsonl", "jsonl"]


def test_hybrid_candidate_k_never_below_k(fake_hybrid):
    retriever = build_hybrid_retriever(**SCOPE, k=4, candidate_k=1)
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
