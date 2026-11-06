"""Qdrant 동기화(ingestion/index_sync.py) 테스트: 로컬 Qdrant(폴더) + 가짜 임베딩으로 흐름만 확인한다.
bge-m3 다운로드·Qdrant Cloud 없이 돈다.

실행: uv run pytest tests/ingestion/test_index_sync.py -q
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest

from rag_service.ingestion import build_corpus, index_sync
from rag_service.storage import qdrant_store
from rag_service.storage.corpus import load_chunks, load_config

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
    FakeEmbedder.calls = 0
    return tmp_path


def run(*extra) -> int:
    return build_corpus.main(["--difficulty", "basic", "--no-context", *extra])


def client():
    return qdrant_store.get_client(load_config())


def levels_in_qdrant() -> set:
    c = client()
    try:
        return {p.payload["metadata"]["level"] for p in c.scroll(NAME, limit=10_000)[0]}
    finally:
        c.close()


@needs_pdf
def test_one_command_and_only_changes_are_synced(repo):
    # 1) 첫 실행: 청킹 + jsonl + Qdrant
    assert run() == 0
    first = FakeEmbedder.calls
    c = client()
    try:
        assert c.count(NAME, exact=True).count == first > 0
        pts, _ = c.scroll(NAME, limit=3, with_payload=True)
        assert pts[0].payload["page_content"]                                 # LangChain 형식
        assert {"book_id", "page", "chunk_id", "level"} <= set(pts[0].payload["metadata"])
    finally:
        c.close()
    assert levels_in_qdrant() == {"page", "para"}                            # row 없음 (Basic엔 표도 없음)

    # 2) 바뀐 게 없으면 임베딩 0
    FakeEmbedder.calls = 0
    assert run() == 0
    assert FakeEmbedder.calls == 0

    # 3) SB02만 바뀜: SB02 청크만 임베딩, 개수는 최신 jsonl과 같음, 컬렉션은 하나
    shutil.copy(BASIC / "SB03.pdf", repo / "data/basic/shipping/documents/SB02.pdf")
    FakeEmbedder.calls = 0
    assert run() == 0
    sb02 = [c for c in load_chunks("basic", levels=["page", "para", "table"]) if c.book_id == "SB02"]
    assert FakeEmbedder.calls == len(sb02)
    c = client()
    try:
        latest = list(load_chunks("basic", levels=["page", "para", "table"]))
        assert c.count(NAME, exact=True).count == len(latest)
        assert [x.name for x in c.get_collections().collections] == [NAME]
    finally:
        c.close()


@needs_pdf
def test_shifted_ids_reuse_vectors_and_only_page_changes(repo):
    """3쪽 맨 앞에 문단이 끼어들어 번호가 밀려도, 내용이 같은 청크는 벡터를 복사한다."""
    assert run() == 0
    chunks = list(load_chunks("basic", levels=["page", "para"]))
    p3 = [c for c in chunks if c.book_id == "SB01" and c.page == 3 and c.level == "para"]
    others = [c for c in chunks if not (c.book_id == "SB01" and c.page == 3 and c.level == "para")]
    inserted = p3[0].model_copy(update={"text": "새로 끼어든 문단이다."})
    shifted = [inserted] + [c.model_copy(update={"chunk_id": f"SB01-p3-c{i + 2}"}) for i, c in enumerate(p3)]
    shifted = [c.model_copy(update={"chunk_id": "SB01-p3-c1"}) if c is inserted else c for c in shifted]

    FakeEmbedder.calls = 0
    cl = client()
    try:
        s = index_sync.sync_points(cl, NAME, others + shifted, FakeEmbedder)
        assert FakeEmbedder.calls == 1                     # 끼어든 문단 하나만 임베딩
        assert s["embedded"] == 1 and s["reused"] == len(p3)
        assert s["upserted"] == len(p3) + 1 and s["deleted"] == 0   # ID가 하나 늘었을 뿐 지울 것 없음
        assert s["count"] == len(others) + len(shifted)
    finally:
        cl.close()


@needs_pdf
def test_index_only_and_level_change(repo):
    assert run("--no-index") == 0
    assert run("--index-only") == 0
    full = len(list(load_chunks("basic", levels=["page", "para", "table"])))
    # 레벨을 줄여 동기화 → page 포인트만 지우고 임베딩 없음
    FakeEmbedder.calls = 0
    s = index_sync.sync("basic", levels=["para"])
    assert FakeEmbedder.calls == 0 and s["deleted"] == full - s["count"] and levels_in_qdrant() == {"para"}
    # 원래대로 → 지워졌던 page만 새로 임베딩
    s = index_sync.sync("basic")
    assert s["count"] == full and s["embedded"] == full - len(list(load_chunks("basic", levels=["para"])))


@needs_pdf
def test_restore_syncs_qdrant(repo):
    assert run() == 0
    shutil.copy(BASIC / "SB03.pdf", repo / "data/basic/shipping/documents/SB02.pdf")
    assert run() == 0
    FakeEmbedder.calls = 0
    assert run("--restore", "SB02@v1") == 0
    c = client()
    try:
        latest = list(load_chunks("basic", levels=["page", "para", "table"]))
        assert c.count(NAME, exact=True).count == len(latest)
        texts = {p.payload["page_content"] for p in c.scroll(NAME, limit=10_000, with_payload=True)[0]}
        assert {x.text for x in latest} == texts
    finally:
        c.close()
