import numpy as np
from langchain_core.documents import Document

from rag_service.retrieval_a import hybrid
from rag_service.retrieval_a.hybrid import (
    HybridRetriever,
    _token_weights,
    build_hybrid_retriever,
    doc_sparse_weights,
    fuse,
    index_text,
    min_max,
    ranks_of,
    sparse_score,
)


def _doc(cid, text="", **meta):
    return Document(page_content=text, metadata={"chunk_id": cid, "book_id": cid.split("-")[0], "page": 1, **meta})


def test_token_weights_keeps_max_and_skips_special():
    w = _token_weights([0, 5, 7, 5, 2], np.array([0.9, 0.1, 0.3, 0.4, 0.8]), skip_ids={0, 2})
    assert w == {"5": 0.4, "7": 0.3}


def test_sparse_score_only_shared_tokens():
    assert sparse_score({"1": 0.3, "2": 0.28, "3": 0.05}, {"1": 0.25, "2": 0.31, "9": 0.12}) == 0.3 * 0.25 + 0.28 * 0.31
    assert sparse_score({"1": 1.0}, {"2": 1.0}) == 0


def test_min_max_and_ranks():
    assert min_max(np.array([2.0, 4.0, 3.0])).tolist() == [0.0, 1.0, 0.5]
    assert min_max(np.array([1.0, 1.0])).tolist() == [0.0, 0.0]
    assert ranks_of(np.array([0.1, 0.9, 0.5])).tolist() == [3, 1, 2]


def test_fuse_minmax_and_rrf():
    dense = np.array([0.80, 0.70, 0.60])
    sparse = np.array([0.00, 0.30, 0.10])
    f = fuse(dense, sparse, method="minmax", dense_weight=0.7, sparse_weight=0.3)
    assert np.argsort(-f).tolist() == [0, 1, 2]
    f2 = fuse(dense, sparse, method="minmax", dense_weight=0.3, sparse_weight=0.7)
    assert int(np.argmax(f2)) == 1
    r = fuse(dense, sparse, method="rrf", dense_weight=0.5, sparse_weight=0.5, rrf_k=60)
    assert int(np.argmax(r)) == 1


def test_index_text_includes_title_and_context():
    d = _doc("SB11-p1-c1", "원문", doc_title="변경 공지", section_title="SI 마감", context="맥락 한 줄")
    assert index_text(d) == "[변경 공지 | SI 마감 | p1]\n맥락 한 줄\n원문"
    assert index_text(d, use_context=False) == "원문"


def test_doc_sparse_weights_uses_cache(tmp_path):
    calls = []

    class FakeEncoder:
        def encode(self, texts, batch_size=16):
            calls.append(len(texts))
            return np.zeros((len(texts), 2)), [{"1": float(len(t))} for t in texts]

    docs = [_doc("A-p1-c1", "aa"), _doc("B-p1-c1", "bbb")]
    path = tmp_path / "cache.json"
    first = doc_sparse_weights(docs, FakeEncoder, path, use_context=False, batch_size=4)
    again = doc_sparse_weights(docs, FakeEncoder, path, use_context=False, batch_size=4)
    assert first == again == [{"1": 2.0}, {"1": 3.0}]
    assert calls == [2]
    docs[1] = _doc("B-p1-c1", "changed")
    doc_sparse_weights(docs, FakeEncoder, path, use_context=False, batch_size=4)
    assert calls == [2, 1]


def _retriever(**kw):
    docs = [_doc("A-1", "a"), _doc("B-1", "b"), _doc("C-1", "c")]
    dense_matrix = np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype=np.float32)
    doc_sparse = [{}, {"7": 1.0}, {}]

    def encode_query(q):
        return np.array([1.0, 0.0]), {"7": 0.5}

    return HybridRetriever(
        docs=docs, dense_matrix=dense_matrix, doc_sparse=doc_sparse, encode_query=encode_query, **kw
    )


def test_hybrid_retriever_scores_and_metadata():
    got = _retriever(top_k=2, method="minmax", dense_weight=0.5, sparse_weight=0.5).invoke("q")
    assert [d.metadata["chunk_id"] for d in got] == ["B-1", "A-1"]
    m = got[0].metadata
    assert m["dense_rank"] == 2 and m["sparse_rank"] == 1 and m["book_id"] == "B"
    only_dense = _retriever(top_k=1, method="minmax", dense_weight=1.0, sparse_weight=0.0).invoke("q")
    assert only_dense[0].metadata["chunk_id"] == "A-1"


def test_chain_style_call(monkeypatch, tmp_path):
    seen = {}

    def fake_connect(name):
        seen["name"] = name
        return object()

    def fake_load(client, name, levels):
        seen["levels"] = levels
        return [_doc("A-1", "a")], np.array([[1.0, 0.0]], dtype=np.float32)

    def fake_sparse(docs, encoder_fn, cache_path, *, use_context, batch_size):
        seen["cache"] = cache_path.name
        return [{}]

    monkeypatch.setattr(hybrid, "_connect_client", fake_connect)
    monkeypatch.setattr(hybrid, "load_points", fake_load)
    monkeypatch.setattr(hybrid, "doc_sparse_weights", fake_sparse)
    r = build_hybrid_retriever(domain="shipping", difficulty="intermediate")
    assert seen["name"] == "kant_shipping_intermediate_bge-m3"
    assert seen["levels"] == ["para", "table"]
    assert seen["cache"] == "kant_shipping_intermediate_bge-m3_para-table.json"
    assert r.top_k == hybrid.load_config(hybrid.CONFIG_PATH)["search"]["top_k"]
    assert (r.method, r.dense_weight, r.sparse_weight) == ("minmax", 0.7, 0.3)
