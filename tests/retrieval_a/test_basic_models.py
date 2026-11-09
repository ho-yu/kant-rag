import pytest

from rag_service.retrieval_a import basic_bge_m3, basic_qwen3


@pytest.mark.parametrize("module, profile", [(basic_bge_m3, "basic"), (basic_qwen3, "qwen3")])
def test_each_file_fixes_its_model(monkeypatch, module, profile):
    seen = []

    def fake(**kwargs):
        seen.append(kwargs)
        return "retriever"

    for name in ("build_naive_retriever", "build_hybrid_retriever", "build_rerank_retriever"):
        monkeypatch.setattr(module, name, fake)
    for mode in ("naive", "hybrid", "rerank"):
        assert module.BUILDERS[mode](top_k=3) == "retriever"
    assert all(k["profile"] == profile and k["difficulty"] == "basic" and k["top_k"] == 3 for k in seen)
