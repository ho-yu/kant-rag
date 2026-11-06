"""retrieval_a Hybrid + Reranker — 네트워크·모델 없이 확인할 수 있는 부분."""

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rag_service.retrieval_a import rerank
from rag_service.retrieval_a.rerank import HybridRerankRetriever, build_rerank_retriever


def _doc(cid, text="", **meta):
    return Document(page_content=text, metadata={"chunk_id": cid, "book_id": cid.split("-")[0], "page": 1, **meta})


class _Fixed(BaseRetriever):
    docs: list[Document]

    def _get_relevant_documents(self, query, *, run_manager):
        return self.docs


def test_reranker_reorders_hybrid_candidates():
    cands = [_doc("A-1", "부작용 총정리"), _doc("B-1", "졸음 부작용이 있다"), _doc("C-1", "부작용 없는 감기약")]
    fake_scores = {"부작용 총정리": 0.40, "졸음 부작용이 있다": 0.03, "부작용 없는 감기약": 0.96}
    seen = {}

    def score_fn(query, texts):
        seen["query"], seen["texts"] = query, texts
        return [fake_scores[t] for t in texts]

    r = HybridRerankRetriever(hybrid=_Fixed(docs=cands), score_fn=score_fn, top_k=2, use_context=False)
    got = r.invoke("부작용이 없는 감기약")
    assert [d.metadata["chunk_id"] for d in got] == ["C-1", "A-1"]
    assert got[0].metadata["rerank_score"] == 0.96 and got[0].metadata["hybrid_rank"] == 3
    assert seen["query"] == "부작용이 없는 감기약" and len(seen["texts"]) == 3  # 후보 전부를 채점
    assert "rerank_score" not in cands[2].metadata  # 원본 Document는 건드리지 않음


def test_reranker_input_includes_title_when_use_context():
    cands = [_doc("A-1", "원문", doc_title="공지", section_title="SI 마감", context="맥락")]
    seen = {}

    def score_fn(query, texts):
        seen["texts"] = texts
        return [0.5]

    HybridRerankRetriever(hybrid=_Fixed(docs=cands), score_fn=score_fn, use_context=True).invoke("q")
    assert seen["texts"] == ["[공지 | SI 마감 | p1]\n맥락\n원문"]


def test_empty_candidates():
    def score_fn(query, texts):
        raise AssertionError("후보가 없으면 리랭커를 부르지 않는다")

    assert HybridRerankRetriever(hybrid=_Fixed(docs=[]), score_fn=score_fn).invoke("q") == []


def test_chain_style_call(monkeypatch):
    """chain.py처럼 build_rerank_retriever(domain=..., difficulty=...)로 부르면 Hybrid 후보 50개 → 최종 top_k."""
    seen = {}

    def fake_build_hybrid(**kwargs):
        seen.update(kwargs)
        return _Fixed(docs=[_doc("A-1", "a")])

    monkeypatch.setattr(rerank, "build_hybrid_retriever", fake_build_hybrid)
    r = build_rerank_retriever(domain="shipping", difficulty="intermediate")
    assert seen["domain"] == "shipping" and seen["difficulty"] == "intermediate"
    assert seen["top_k"] == 50
    assert seen["config_path"].name == "hybrid.yaml"
    assert r.top_k == rerank.load_config()["search"]["top_k"]
