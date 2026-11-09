import pytest
from qdrant_client import models

from rag_service.retrieval_a import naive
from rag_service.retrieval_a.naive import (
    build_naive_retriever,
    collection_name,
    level_filter,
    load_config,
    model_profile,
    model_slug,
)


def test_config_loads_and_names_collection():
    cfg = load_config()
    assert collection_name(cfg) == "kant_shipping_basic_bge-m3"
    with pytest.raises(ValueError):
        collection_name(cfg, difficulty="advanced")
    assert collection_name(cfg, domain=" Shipping ", difficulty="BASIC") == "kant_shipping_basic_bge-m3"


def test_collection_name_rejects_unknown_values():
    cfg = load_config()
    with pytest.raises(ValueError):
        collection_name(cfg, difficulty="expert")
    with pytest.raises(ValueError):
        collection_name(cfg, domain="medical")


def test_level_filter_matches_metadata_level():
    f = level_filter(["para", "table"])
    cond = f.must[0]
    assert cond.key == "metadata.level"
    assert isinstance(cond.match, models.MatchAny)
    assert cond.match.any == ["para", "table"]


def test_chain_style_call(monkeypatch):
    seen = {}

    class FakeStore:
        def as_retriever(self, **kwargs):
            seen["kwargs"] = kwargs
            return "retriever"

    def fake_connect(cfg, name, profile):
        seen["name"], seen["model"] = name, profile["embedding"]
        return FakeStore()

    monkeypatch.setattr(naive, "_connect_store", fake_connect)
    assert build_naive_retriever(domain="shipping", difficulty="basic") == "retriever"
    assert seen["name"] == "kant_shipping_basic_bge-m3" and seen["model"] == "BAAI/bge-m3"
    assert seen["kwargs"]["search_kwargs"]["k"] == load_config()["search"]["top_k"]
    assert seen["kwargs"]["search_kwargs"]["filter"].must[0].match.any == ["para", "table"]

    with pytest.raises(TypeError):
        build_naive_retriever("shipping", "basic")


def test_profiles_pick_model_and_collection_per_difficulty(tmp_path):
    cfg = load_config()
    assert collection_name(cfg, difficulty="intermediate") == "kant_shipping_intermediate_bge-m3"
    assert model_profile("intermediate")["embedding"] == "BAAI/bge-m3"
    assert model_slug("Qwen/Qwen3-Embedding-0.6B") == "qwen3-embedding-0.6b"
    assert model_profile("basic")["sparse"] == "bge-m3"
    bad = tmp_path / "models.yaml"
    bad.write_text(
        "basic:\n  embedding: Qwen/Qwen3-Embedding-0.6B\n  sparse: bge-m3\n  reranker: x\n", encoding="utf-8"
    )
    with pytest.raises(ValueError):
        model_profile("basic", bad)


def test_profile_overrides_difficulty_model(monkeypatch):
    seen = {}

    class FakeStore:
        def as_retriever(self, **kwargs):
            return "retriever"

    def fake_connect(cfg, name, profile):
        seen["name"], seen["model"] = name, profile["embedding"]
        return FakeStore()

    monkeypatch.setattr(naive, "_connect_store", fake_connect)
    build_naive_retriever(domain="shipping", difficulty="basic", profile="qwen3")
    assert seen["name"] == "kant_shipping_basic_qwen3-embedding-0.6b"
    assert seen["model"] == "Qwen/Qwen3-Embedding-0.6B"
    assert collection_name(load_config(), difficulty="basic") == "kant_shipping_basic_bge-m3"
    with pytest.raises(ValueError):
        model_profile("없는설정")


def test_connect_client_falls_back_to_local(monkeypatch, tmp_path):
    class FakeClient:
        def __init__(self, names):
            self.names = names

        def collection_exists(self, name):
            return name in self.names

    remote, local = FakeClient({"remote_col"}), FakeClient({"local_col"})
    monkeypatch.setenv("QDRANT_URL", "http://example")
    monkeypatch.setattr(naive, "_remote_client", lambda url, key: remote)
    monkeypatch.setattr(naive, "local_client", lambda: local)
    monkeypatch.setattr(naive, "LOCAL_QDRANT_PATH", tmp_path)
    assert naive._connect_client("remote_col") is remote
    assert naive._connect_client("local_col") is local
    with pytest.raises(RuntimeError):
        naive._connect_client("none")
