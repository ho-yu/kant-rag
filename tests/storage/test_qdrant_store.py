"""Qdrant 공통 기능(storage/qdrant_store.py) 테스트: 팀 검색 코드가 쓰는 읽기 쪽 기능을 확인한다.
로컬 Qdrant(폴더) + 가짜 임베딩이라 bge-m3 다운로드·Qdrant Cloud 없이 돈다.

실행: uv run pytest tests/storage -q
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest

from rag_service.ingestion import build_corpus
from rag_service.storage import qdrant_store
from rag_service.storage.corpus import load_config

ROOT = Path(__file__).resolve().parents[2]
BASIC = ROOT / "data/basic/shipping/documents"
NAME = "kant_shipping_basic_bge-m3"
needs_pdf = pytest.mark.skipif(not (BASIC / "SB01.pdf").exists(), reason="data PDF 없음")


class FakeEmbedder:
    calls = 0

    def __init__(self, *args, **kwargs):
        self.dim = 8

    def encode(self, texts):
        FakeEmbedder.calls += len(texts)
        return [[b / 255 for b in hashlib.sha256(t.encode()).digest()[:8]] for t in texts]


class FakeSparse:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, progress=False):
        return [{int(b): 1.0 for b in hashlib.sha256(t.encode()).digest()[:4]} for t in texts]


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='t'\n", encoding="utf-8")
    (tmp_path / "config").mkdir(parents=True)
    shutil.copy(ROOT / "config/chunking.yaml", tmp_path / "config/chunking.yaml")
    docs = tmp_path / "data/basic/shipping/documents"
    docs.mkdir(parents=True)
    for b in ("SB01", "SB02"):
        shutil.copy(BASIC / f"{b}.pdf", docs / f"{b}.pdf")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.setattr(qdrant_store, "Embedder", FakeEmbedder)
    monkeypatch.setattr(qdrant_store, "SparseEncoder", FakeSparse)   # sparse 벡터도 올라간 컬렉션에서 확인
    FakeEmbedder.calls = 0
    return tmp_path


def run(*extra) -> int:
    return build_corpus.main(["--difficulty", "basic", "--no-context", *extra])


def client():
    return qdrant_store.get_client(load_config())


@needs_pdf
def test_langchain_vectorstore_reads_our_collection(repo):
    """팀 체인(langchain-qdrant)이 컬렉션을 그대로 읽어 Document(book_id, page)를 얻는지."""
    pytest.importorskip("langchain_qdrant")
    from langchain_core.embeddings import DeterministicFakeEmbedding
    from langchain_qdrant import QdrantVectorStore

    assert run() == 0
    c = client()
    try:
        store = QdrantVectorStore(client=c, collection_name=NAME, embedding=DeterministicFakeEmbedding(size=8))
        docs = store.similarity_search("예약 확정", k=5, filter=qdrant_store.level_filter(["para"]))
        assert len(docs) == 5
        for d in docs:
            assert d.page_content and d.metadata["level"] == "para"
            assert d.metadata["book_id"] in {"SB01", "SB02"} and isinstance(d.metadata["page"], int)
    finally:
        c.close()


def test_device_falls_back_to_cpu_without_gpu():
    pytest.importorskip("torch")
    import torch

    if torch.cuda.is_available():
        assert qdrant_store.resolve_device("cuda") == "cuda"
    else:
        assert qdrant_store.resolve_device("cuda") == "cpu"     # GPU 없으면 CPU로
    assert qdrant_store.resolve_device("cpu") == "cpu"


def test_collection_name_follows_topic_and_difficulty():
    cfg = load_config()
    assert qdrant_store.collection_name(cfg, "basic") == "kant_shipping_basic_bge-m3"
    assert qdrant_store.collection_name(cfg, "intermediate", "shipping") == "kant_shipping_intermediate_bge-m3"


def test_level_filter_keeps_only_given_levels():
    f = qdrant_store.level_filter(["para", "table"])
    assert f.must[0].key == "metadata.level" and set(f.must[0].match.any) == {"para", "table"}
    assert qdrant_store.level_filter(["page"], negate=True).must_not
