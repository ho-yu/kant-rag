from rag_service.retrieval_a import basic_bge_m3


def test_bge_m3_file_fixes_its_model(monkeypatch):
    seen = []

    def fake(**kwargs):
        seen.append(kwargs)
        return "retriever"

    for name in ("build_naive_retriever", "build_hybrid_retriever", "build_rerank_retriever"):
        monkeypatch.setattr(basic_bge_m3, name, fake)
    for mode in ("naive", "hybrid", "rerank"):
        assert basic_bge_m3.BUILDERS[mode](top_k=3) == "retriever"
    assert all(k["profile"] == "basic" and k["difficulty"] == "basic" and k["top_k"] == 3 for k in seen)
