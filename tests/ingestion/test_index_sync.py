"""Qdrant 동기화(ingestion/index_sync.py) 테스트: 로컬 Qdrant(폴더) + 가짜 임베딩·가짜 sparse로 흐름만 확인한다.
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


class FakeSparse:
    calls = 0

    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, progress=False):
        FakeSparse.calls += len(texts)
        return [{int(b): 1.0 + i for i, b in enumerate(hashlib.sha256(t.encode()).digest()[:4])} for t in texts]


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
    monkeypatch.setattr(qdrant_store, "SparseEncoder", FakeSparse)
    FakeEmbedder.calls = 0
    FakeSparse.calls = 0
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


def set_sparse(repo, value: str) -> None:
    """config index.sparse 값을 바꾼다. 예) "[m3, bm25]", "[m3]", "[]" """
    path = repo / "config/chunking.yaml"
    lines = path.read_text(encoding="utf-8").split("\n")
    lines = [f"  sparse: {value}" if ln.startswith("  sparse: ") else ln for ln in lines]
    path.write_text("\n".join(lines), encoding="utf-8")


@needs_pdf
def test_sparse_vectors_are_uploaded_and_searchable(repo):
    """sparse 켜짐: 포인트마다 dense + m3 + bm25가 같이 올라가고, Qdrant에 바로 hybrid 후보를 물어볼 수 있다."""
    assert run() == 0
    c = client()
    try:
        assert qdrant_store.has_sparse(c, NAME, ["m3", "bm25"])
        total = c.count(NAME, exact=True).count
        assert FakeSparse.calls == total                                   # m3: 청크마다 한 번
        p = c.scroll(NAME, scroll_filter=qdrant_store.level_filter(["para"]), limit=1,
                     with_vectors=True, with_payload=True)[0][0]
        assert set(p.vector) == {"", "m3", "bm25"} and p.payload["sparse_kinds"] == ["m3", "bm25"]
        dense, m3 = p.vector[""], FakeSparse().encode([qdrant_store.sparse_text(p.payload)])[0]
        hits = qdrant_store.search_dense_sparse(c, NAME, dense, m3, kind="m3", levels=["para"], limit=5)
        assert hits and all(h["payload"]["metadata"]["level"] == "para" for h in hits)
        same = [h for h in hits if h["payload"]["metadata"]["chunk_id"] == p.payload["metadata"]["chunk_id"]]
        assert same and same[0]["dense_score"] is not None and same[0]["sparse_score"] is not None  # 자기 자신은 양쪽 1위

        # bm25: 그 문단의 원문으로 물으면 그 문단이 BM25 1위
        q = qdrant_store.bm25_query(p.payload["page_content"])
        hits = qdrant_store.search_dense_sparse(c, NAME, None, q, kind="bm25", levels=["para"], limit=3)
        best = max(hits, key=lambda h: h["sparse_score"])
        assert best["payload"]["metadata"]["chunk_id"] == p.payload["metadata"]["chunk_id"]
        # page 레벨에도 sparse가 있다 (page 확장 검색용)
        pg = c.scroll(NAME, scroll_filter=qdrant_store.level_filter(["page"]), limit=1, with_vectors=True)[0][0]
        assert set(pg.vector) == {"", "m3", "bm25"}
    finally:
        c.close()
    FakeSparse.calls = 0
    assert run() == 0 and FakeSparse.calls == 0                              # 바뀐 게 없으면 sparse 계산도 0


@needs_pdf
def test_old_collection_without_sparse_is_rebuilt_reusing_dense(repo):
    """sparse 없이 올린 기존 컬렉션 → sparse 켜고 동기화하면 새 구성으로 다시 만들고 dense는 재사용한다."""
    set_sparse(repo, "[]")
    assert run() == 0
    c = client()
    try:
        assert not qdrant_store.collection_sparse_kinds(c, NAME)
        total = c.count(NAME, exact=True).count
    finally:
        c.close()
    set_sparse(repo, "[m3, bm25]")
    FakeEmbedder.calls = FakeSparse.calls = 0
    s = index_sync.sync("basic")
    assert s["rebuilt_for_sparse"] and s["count"] == total
    assert FakeEmbedder.calls == 0 and s["embedded"] == 0                   # dense는 기존 값 그대로
    assert FakeSparse.calls == total == s["sparse"]["m3"] == s["sparse"]["bm25"]
    c = client()
    try:
        assert qdrant_store.has_sparse(c, NAME, ["m3", "bm25"])
    finally:
        c.close()


@needs_pdf
def test_adding_bm25_keeps_existing_m3(repo):
    """m3만 있던 컬렉션에 bm25를 추가 → 다시 만들 때 dense·m3는 재사용, bm25만 계산."""
    set_sparse(repo, "[m3]")
    assert run() == 0
    c = client()
    try:
        assert qdrant_store.collection_sparse_kinds(c, NAME) == {"m3"}
        total = c.count(NAME, exact=True).count
    finally:
        c.close()
    FakeEmbedder.calls = FakeSparse.calls = 0
    assert run("--index-only", "--sparse", "m3,bm25") == 0                   # 명령줄 --sparse가 config보다 우선
    c = client()
    try:
        assert qdrant_store.collection_sparse_kinds(c, NAME) == {"m3", "bm25"}
        assert c.count(NAME, exact=True).count == total
    finally:
        c.close()
    assert FakeEmbedder.calls == 0 and FakeSparse.calls == 0


def test_bm25_tokens_keep_codes_whole_and_ids_are_stable():
    pytest.importorskip("kiwipiepy")
    toks = qdrant_store.bm25_tokens("예약 HG2030-00362·장비 HL-40GP-00362 건의 보관료")
    assert "hg2030-00362" in toks and "hl-40gp-00362" in toks          # 번호·코드는 통째로 한 단어
    assert not {"hl", "gp", "00362", "2030-00362"} & set(toks)            # 조각은 남기지 않음
    assert "보관" in toks or "보관료" in toks
    assert qdrant_store.bm25_tokens("장비 N1-20-F09118, 예약 GR27-F02861은") [:2] == ["n1-20-f09118", "gr27-f02861"]
    assert set(qdrant_store.bm25_tokens("S2/40GP 장비")) >= {"s2", "40gp"}       # 슬래시는 나눔
    assert "20gp" in qdrant_store.bm25_tokens("20GP 한 대")
    assert "b/l" in qdrant_store.bm25_tokens("B/L 발행")
    money = qdrant_store.bm25_tokens("청구 USD12, 18100kg, 머리말 p116")
    assert not {"usd12", "18100kg", "p116"} & set(money)                  # 쪽 번호·금액·무게는 묶지 않음
    assert "2029-09-11" in qdrant_store.bm25_tokens("2029-09-11 양하")     # 날짜는 Kiwi 일련번호(W_SERIAL)
    many = qdrant_store.bm25_tokens_many(["예약 HG2030-00362 건", "S2/40GP 장비"])
    assert many == [qdrant_store.bm25_tokens("예약 HG2030-00362 건"), qdrant_store.bm25_tokens("S2/40GP 장비")]
    assert qdrant_store.bm25_token_id("보관료") == qdrant_store.bm25_token_id("보관료") < 2**32
    text = "선박 선박 반납"
    q = qdrant_store.bm25_query(text)                                        # {토큰 번호: 질문 속 횟수}
    counts = {}
    for t in qdrant_store.bm25_tokens(text):
        counts[qdrant_store.bm25_token_id(t)] = counts.get(qdrant_store.bm25_token_id(t), 0) + 1.0
    assert q == counts and q[qdrant_store.bm25_token_id("선박")] == 2.0


def test_bm25_encoder_weights_follow_bm25():
    pytest.importorskip("kiwipiepy")
    enc = qdrant_store.BM25Encoder({"para": 4.0, "page": 40.0}, k1=1.5, b=0.75)
    assert qdrant_store.bm25_tokens("선박 선박") == ["선박", "선박"]
    w_para, w_page = enc.encode(["선박 선박", "선박 선박"], ["para", "page"])
    tid = qdrant_store.bm25_token_id("선박")
    # 같은 글이라도 page 평균 길이가 길면 상대적으로 짧은 글이라 값이 크다
    assert w_page[tid] > w_para[tid] > 0
    assert abs(w_para[tid] - 2 * 2.5 / (2 + 1.5 * (0.25 + 0.75 * 2 / 4))) < 1e-9


def test_sparse_kinds_from_config():
    assert qdrant_store.sparse_kinds({"index": {"sparse": ["bm25", "m3"]}}) == ["m3", "bm25"]
    assert qdrant_store.sparse_kinds({"index": {"sparse": "none"}}) == []
    assert qdrant_store.sparse_kinds({"index": {"sparse": True}}) == ["m3"]
    with pytest.raises(ValueError):
        qdrant_store.sparse_kinds({"index": {"sparse": ["splade"]}})
