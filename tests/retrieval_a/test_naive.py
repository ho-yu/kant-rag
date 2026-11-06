import pytest
from qdrant_client import models

from rag_service.retrieval_a import naive
from rag_service.retrieval_a.naive import (
    build_naive_retriever,
    collection_name,
    level_filter,
    load_config,
)


def test_config_loads_and_names_collection():
    cfg = load_config()
    assert collection_name(cfg) == "kant_shipping_basic_bge-m3"
    assert collection_name(cfg, domain="shipping", difficulty="advanced") == "kant_shipping_advanced_bge-m3"
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

    def fake_connect(cfg, name):
        seen["name"] = name
        return FakeStore()

    monkeypatch.setattr(naive, "_connect_store", fake_connect)
    assert build_naive_retriever(domain="shipping", difficulty="intermediate") == "retriever"
    assert seen["name"] == "kant_shipping_intermediate_bge-m3"
    assert seen["kwargs"]["search_kwargs"]["k"] == load_config()["search"]["top_k"]
    assert seen["kwargs"]["search_kwargs"]["filter"].must[0].match.any == ["para", "table"]

    with pytest.raises(TypeError):
        build_naive_retriever("shipping", "basic")
