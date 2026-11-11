"""rerank.py와 build_rerank_retriever 테스트.

기본 테스트: 리랭커 점수를 가짜 함수로 바꿔서 모델 없이 돈다.
integration 테스트: 실제 bge-reranker-v2-m3를 내려받아 점수가 0~1로 나오는지 본다.

실행 (레포 맨 위에서)
    uv run pytest tests/retrieval_b/test_rerank_b.py -q

[2026-11-10 리팩토링] 이유: 리랭커 숫자 설정이 Settings로 옮겨지고 domain·difficulty가 필수가 됨 → 효과: 설정값 적용과 rerank_n ≥ k 검사가 설정을 거쳐도 지켜지는지 확인됨.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rag_service.retrieval_b import build_rerank_retriever, retrievers
from rag_service.retrieval_b import rerank as rerank_mod
from rag_service.retrieval_b.bm25 import BM25Index
from rag_service.retrieval_b.rerank import (
    RERANKER_MODEL,
    model_name_from_env,
    rerank,
    rerank_text,
)


def doc(cid: str, text: str = "본문", **extra) -> Document:
    meta = {"chunk_id": cid, "book_id": cid[:4], "page": 1, "level": "para", **extra}
    return Document(page_content=text, metadata=meta)


def ids(docs) -> list[str]:
    return [d.metadata["chunk_id"] for d in docs]


def fixed_scores(values):
    """후보 순서대로 values를 돌려주는 가짜 리랭커."""

    def score(question, texts):
        return list(values)[: len(texts)]

    return score


# ---------- rerank_text: 리랭커 입력 ----------


def test_rerank_text_matches_embedding_text_shape():
    d = doc(
        "SB01-p3-c1",
        "서류 마감은 출항 2일 전",
        doc_title="부킹 안내",
        section_title="서류",
        page=3,
        context="부산→LA 일반 건화물",
    )
    assert (
        rerank_text(d)
        == "[부킹 안내 | 서류 | p3]\n부산→LA 일반 건화물\n서류 마감은 출항 2일 전"
    )


def test_rerank_text_without_context():
    d = doc("SB01-p3-c1", "본문만", doc_title="부킹 안내", page=3)
    assert rerank_text(d) == "[부킹 안내 | p3]\n본문만"


# ---------- rerank: 점수 순으로 다시 세우기 ----------

CANDS = [
    doc("SB01-p1-c1", score=0.03),
    doc("SB02-p1-c1", score=0.02),
    doc("SB03-p1-c1"),
]


def test_order_follows_scores_and_keeps_prev_rank():
    out = rerank("q", CANDS, k=3, score_fn=fixed_scores([0.1, 0.9, 0.5]))
    assert ids(out) == ["SB02-p1-c1", "SB03-p1-c1", "SB01-p1-c1"]
    assert [d.metadata["prev_rank"] for d in out] == [2, 3, 1]
    assert [d.metadata["rerank_score"] for d in out] == [0.9, 0.5, 0.1]
    assert [d.metadata["score"] for d in out] == [0.9, 0.5, 0.1]


def test_hybrid_score_is_kept():
    out = rerank("q", CANDS, k=3, score_fn=fixed_scores([0.1, 0.9, 0.5]))
    assert [d.metadata["hybrid_score"] for d in out] == [0.02, None, 0.03]


def test_k_cuts_the_list():
    out = rerank("q", CANDS, k=1, score_fn=fixed_scores([0.1, 0.9, 0.5]))
    assert ids(out) == ["SB02-p1-c1"]


def test_ties_are_broken_by_chunk_id():
    cands = [doc("SB09-p1-c1"), doc("SB01-p1-c1")]
    out = rerank("q", cands, k=2, score_fn=fixed_scores([0.5, 0.5]))
    assert ids(out) == ["SB01-p1-c1", "SB09-p1-c1"]


def test_cannot_find_what_is_not_a_candidate():
    """7장 3강 §4: 후보에 없는 청크는 리랭커도 못 찾는다."""
    out = rerank("q", CANDS[1:], k=3, score_fn=fixed_scores([0.9, 0.9]))
    assert "SB01-p1-c1" not in ids(out)


def test_score_fn_gets_question_and_rerank_text():
    seen = {}

    def score(question, texts):
        seen["q"], seen["texts"] = question, texts
        return [0.0] * len(texts)

    d = doc("SB01-p1-c1", "원문", doc_title="제목", page=1)
    out = rerank("질문", [d], k=1, score_fn=score)
    assert seen == {"q": "질문", "texts": ["[제목 | p1]\n원문"]}
    assert out[0].page_content == "원문"  # 결과 본문은 원문 그대로


def test_inputs_are_not_changed():
    before = [dict(d.metadata) for d in CANDS]
    rerank("q", CANDS, k=3, score_fn=fixed_scores([0.1, 0.9, 0.5]))
    assert [d.metadata for d in CANDS] == before


def test_empty_candidates_skip_the_model():
    def boom(question, texts):
        raise AssertionError("후보가 없으면 모델을 부르지 않는다")

    assert rerank("q", [], k=5, score_fn=boom) == []


def test_bad_arguments():
    with pytest.raises(ValueError):
        rerank("q", CANDS, k=0, score_fn=fixed_scores([0.1, 0.9, 0.5]))
    with pytest.raises(ValueError):
        rerank("q", CANDS, k=3, score_fn=fixed_scores([0.1]))


def test_model_name_from_env(monkeypatch):
    monkeypatch.delenv("RERANKER_MODEL", raising=False)
    assert model_name_from_env() == RERANKER_MODEL == "BAAI/bge-reranker-v2-m3"
    monkeypatch.setenv("RERANKER_MODEL", "my/model")
    assert model_name_from_env() == "my/model"
    assert model_name_from_env("given/model") == "given/model"


# ---------- build_rerank_retriever: Hybrid 후보 → 리랭커 ----------

# 질문 "gamma". Hybrid 순서: B(p2) > A(p1) > C(p3) > D(p4) > E(p5) (test_retrievers와 같은 데이터)
ROWS = [
    ("alpha beta", [1.0, 0.0], "SB01-p1-c1", 1),
    ("beta gamma", [0.8, 0.6], "SB01-p2-c1", 2),
    ("delta", [0.0, 1.0], "SB01-p3-c1", 3),
    ("epsilon", [0.0, 1.0], "SB01-p4-c1", 4),
    ("zeta", [0.0, 1.0], "SB01-p5-c1", 5),
]


@pytest.fixture
def fake_hybrid(make_store, monkeypatch):
    rows = [
        (t, v, {"chunk_id": c, "book_id": "SB01", "page": p, "level": "para"})
        for t, v, c, p in ROWS
    ]
    store = make_store(rows, {"gamma": [1.0, 0.0]})
    docs = [Document(page_content=t, metadata=m) for t, _, m in rows]
    monkeypatch.setattr(retrievers, "get_store", lambda domain, difficulty: store)
    monkeypatch.setattr(
        retrievers,
        "get_bm25_index",
        lambda domain, difficulty, levels: BM25Index(docs, tokenizer=str.split),
    )


SCOPE = {"domain": "shipping", "difficulty": "basic"}


def zeta_first(question, texts):
    """본문에 zeta가 있으면 1점, 아니면 0점."""
    return [1.0 if "zeta" in t else 0.0 for t in texts]


def test_rerank_retriever_reorders_hybrid_candidates(fake_hybrid):
    retriever = build_rerank_retriever(**SCOPE, k=3, score_fn=zeta_first)
    assert isinstance(retriever, BaseRetriever)
    docs = retriever.invoke("gamma")
    # zeta(Hybrid 5위)가 1위로, 나머지는 0점 동점이라 chunk_id 순
    assert ids(docs) == ["SB01-p5-c1", "SB01-p1-c1", "SB01-p2-c1"]
    assert [d.metadata["prev_rank"] for d in docs] == [5, 2, 1]
    assert docs[2].metadata["bm25_rank"] == 1  # Hybrid가 붙인 값도 남는다


def test_rerank_retriever_asks_hybrid_for_rerank_n(fake_hybrid):
    retriever = build_rerank_retriever(**SCOPE, k=2, rerank_n=4, score_fn=zeta_first)
    assert (retriever.k, retriever.hybrid.k) == (2, 4)
    # 후보가 4개(B, A, C, D)뿐이라 zeta(5위)는 리랭커에 안 간다
    assert "SB01-p5-c1" not in ids(retriever.invoke("gamma"))


def test_rerank_n_below_k_raises(fake_hybrid):
    with pytest.raises(ValueError):
        build_rerank_retriever(**SCOPE, k=5, rerank_n=3, score_fn=zeta_first)


def test_rerank_n_below_settings_k_raises(fake_hybrid, default_settings):
    """k를 설정(RETRIEVAL_B_K)에서 가져와도 rerank_n과 비교한다."""
    default_settings(k=8, rerank_n=10)
    with pytest.raises(ValueError):
        build_rerank_retriever(**SCOPE, rerank_n=6, score_fn=zeta_first)


def test_chain_style_defaults_do_not_load_model(fake_hybrid, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("만들 때는 모델을 불러오지 않는다")

    monkeypatch.setattr(rerank_mod, "load_cross_encoder", boom)
    retriever = build_rerank_retriever(
        domain="shipping", difficulty="basic", profile=None
    )
    assert (retriever.k, retriever.hybrid.k) == (5, 50)
    assert retriever.hybrid.levels == ["para", "table"]


def test_scorer_uses_settings_for_max_length_and_batch(monkeypatch, default_settings):
    """cross_encoder_scorer()를 인자 없이 부르면 (rerank_diagnose.py) 설정값을 쓴다."""
    default_settings(rerank_max_length=512, rerank_batch_size=4)
    seen = {}

    class FakeModel:
        def predict(self, pairs, batch_size, **kwargs):
            seen["batch_size"] = batch_size
            return [0.5] * len(pairs)

    def fake_load(name, max_length):
        seen["max_length"] = max_length
        return FakeModel()

    monkeypatch.setattr(rerank_mod, "load_cross_encoder", fake_load)
    assert rerank_mod.cross_encoder_scorer()("q", ["a", "b"]) == [0.5, 0.5]
    assert seen == {"max_length": 512, "batch_size": 4}
    rerank_mod.cross_encoder_scorer(max_length=256, batch_size=2)("q", ["a"])
    assert seen == {"max_length": 256, "batch_size": 2}


def test_scorer_rejects_bad_values():
    with pytest.raises(ValueError):
        rerank_mod.cross_encoder_scorer(max_length=0)
    with pytest.raises(ValueError):
        rerank_mod.cross_encoder_scorer(batch_size=-1)


def test_main_rerank_mode(fake_hybrid, monkeypatch, capsys):
    from rag_service.retrieval_b.__main__ import main

    monkeypatch.setattr(
        retrievers, "cross_encoder_scorer", lambda *args, **kwargs: zeta_first
    )
    main(["gamma", "--mode", "rerank", "--k", "1"])
    out = capsys.readouterr().out.strip().splitlines()
    assert out == ["1. SB01-p5-c1 SB01 p5 1.0000 (hybrid #5) zeta"]


# ---------- integration: 실제 리랭커 ----------


@pytest.mark.integration
def test_real_reranker_scores_relevant_text_higher():
    score = rerank_mod.cross_encoder_scorer()
    scores = score(
        "서류 제출 마감은 언제인가요?",
        [
            "서류 제출 마감은 출항 2일 전 17시입니다.",
            "컨테이너 크기는 20피트와 40피트가 있습니다.",
        ],
    )
    assert len(scores) == 2
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores[0] > scores[1]
