"""bm25.py와 build_bm25_retriever 테스트.

기본 테스트: 작은 문서 목록 + 띄어쓰기 토큰화라서 결과가 손으로 확인된다.
Qdrant 입력 테스트: 가짜 Qdrant 클라이언트로 돈다 (팀 Qdrant에 접속하지 않는다).
실제 jsonl 테스트: 레포에 있는 artifacts/corpus/basic/shipping으로 색인을 만든다 (Qdrant 안 씀, 약 10초).

실행 (레포 맨 위에서)
    uv run pytest tests/retrieval_b/test_bm25.py -q

[2026-11-10 리팩토링] 이유: build_bm25_retriever가 domain·difficulty를 꼭 받게 바뀜 → 효과: 색인 캐시 테스트도 실제 호출과 같은 인자로 확인함.
[2026-11-11 기능 추가] 이유: BM25가 Dense처럼 팀 Qdrant에서도 청크를 읽게 됨 → 효과: Qdrant 입력이 jsonl과 같은 청크·순서가 되는지, 난이도별 컬렉션을 고르는지 확인함.
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from qdrant_client.http.exceptions import ResponseHandlingException

from rag_service.contracts.chunk import Chunk
from rag_service.retrieval_b import bm25, build_bm25_retriever, retrievers
from rag_service.retrieval_b.bm25 import (
    BM25Index,
    as_levels,
    chunk_from_payload,
    file_order,
    load_team_chunks,
    tokenize,
)
from rag_service.storage.corpus import corpus_dir, load_chunks


def doc(chunk_id: str, text: str, level: str = "para") -> Document:
    page = int(chunk_id.split("-p")[1].split("-")[0])
    meta = {"chunk_id": chunk_id, "book_id": chunk_id[:4], "page": page, "level": level}
    return Document(page_content=text, metadata=meta)


# rank-bm25는 문서 절반 이상에 나오는 단어의 idf를 0 근처로 깎는다.
# 그래서 "예약"·"서류"는 6개 중 2개에만 넣는다.
DOCS = [
    doc("SB01-p1-c1", "예약 마감 안내"),
    doc("SB01-p2-c1", "서류 제출 방법"),
    doc("SB02-p1-c1", "반입 장소 지도"),
    doc("SB03-p1-c1", "예약 변경 서류"),
    doc("SB04-p1-c1", "운임 표"),
    doc("SB05-p1-c1", "담당 부서 연락처"),
]


def small_index(docs=DOCS) -> BM25Index:
    return BM25Index(docs, tokenizer=str.split)


def ids(docs) -> list[str]:
    return [d.metadata["chunk_id"] for d in docs]


# ---------- tokenize: Kiwi ----------


def test_tokenize_keeps_container_codes_and_drops_particles():
    assert tokenize("20GP와 40GP의 SI 마감은 언제인가요?") == [
        "20gp",
        "40gp",
        "si",
        "마감",
        "언제",
    ]


def test_tokenize_lowercases_english():
    assert tokenize("VGM 접수 경로를 확인했다") == ["vgm", "접수", "경로", "확인"]


# ---------- BM25Index.search ----------


def test_only_matching_docs_are_returned():
    assert ids(small_index().search("마감", k=5)) == ["SB01-p1-c1"]


def test_more_matching_words_rank_higher():
    # "예약 서류": SB03은 둘 다, SB01은 예약만, SB01-p2는 서류만
    docs = small_index().search("예약 서류", k=5)
    assert ids(docs)[0] == "SB03-p1-c1"
    assert set(ids(docs)) == {"SB03-p1-c1", "SB01-p1-c1", "SB01-p2-c1"}


def test_no_matching_word_returns_empty():
    assert small_index().search("냉동 컨테이너", k=5) == []


def test_k_limits_result_count():
    assert len(small_index().search("예약 서류", k=1)) == 1


def test_scores_are_floats_and_descending():
    docs = small_index().search("예약 서류", k=5)
    scores = [d.metadata["score"] for d in docs]
    assert all(isinstance(s, float) and s > 0 for s in scores)
    assert scores == sorted(scores, reverse=True)


def test_ties_are_broken_by_chunk_id():
    twins = [
        doc("SB09-p1-c1", "마감 안내"),
        doc("SB02-p1-c1", "마감 안내"),
        doc("SB05-p1-c1", "반입 장소"),
        doc("SB06-p1-c1", "서류 방법"),
        doc("SB07-p1-c1", "지도 보기"),
    ]
    docs = BM25Index(twins, tokenizer=str.split).search("마감", k=5)
    assert ids(docs) == ["SB02-p1-c1", "SB09-p1-c1"]


def test_search_does_not_touch_index_documents():
    index = small_index()
    index.search("마감", k=5)
    assert all("score" not in d.metadata for d in index.docs)


def test_index_text_can_differ_from_page_content():
    """색인은 embedding_text(제목·context 포함)로, 결과 본문은 원문으로."""
    index = BM25Index(
        DOCS, texts=["가온루트 예약", "b", "c", "d", "e", "f"], tokenizer=str.split
    )
    [hit] = index.search("가온루트", k=5)
    assert hit.page_content == "예약 마감 안내"


@pytest.mark.parametrize(("question", "k"), [("", 3), ("  ", 3), ("마감", 0)])
def test_invalid_input_raises(question, k):
    with pytest.raises(ValueError):
        small_index().search(question, k=k)


def test_bad_index_input_raises():
    with pytest.raises(ValueError):
        BM25Index([])
    with pytest.raises(ValueError):
        BM25Index(DOCS, texts=["하나뿐"])


# ---------- get_bm25_index / build_bm25_retriever ----------


class FakeChunk:
    def __init__(self, document: Document, index_text: str):
        self.document = document
        self.index_text = index_text

    def payload(self) -> dict:
        return {
            "page_content": self.document.page_content,
            "metadata": self.document.metadata,
        }

    def embedding_text(self) -> str:
        return self.index_text


@pytest.fixture
def fake_chunks(monkeypatch):
    """load_chunks를 가짜로 바꾸고, 불린 인자를 기록한다."""
    calls = []

    def _load_chunks(difficulty, domain="shipping", levels=None):
        calls.append((domain, difficulty, levels))
        return [FakeChunk(d, f"[제목] {d.page_content}") for d in DOCS]

    monkeypatch.setattr(bm25, "load_chunks", _load_chunks)
    bm25.get_bm25_index.cache_clear()
    yield calls
    bm25.get_bm25_index.cache_clear()


def test_build_bm25_retriever_uses_team_chunks(fake_chunks):
    retriever = build_bm25_retriever(
        domain="shipping", difficulty="basic", k=2, bm25_source="jsonl"
    )
    docs = retriever.invoke("예약")
    assert len(docs) == 2
    assert fake_chunks == [("shipping", "basic", ("para", "table"))]
    for key in ("chunk_id", "book_id", "page", "level", "score"):
        assert key in docs[0].metadata


def test_index_is_built_once_per_setting(fake_chunks):
    scope = {"domain": "shipping", "difficulty": "basic"}
    first = build_bm25_retriever(**scope, levels=["table", "para"], bm25_source="jsonl")
    second = build_bm25_retriever(
        **scope, levels=["para", "table"], bm25_source="jsonl"
    )
    assert first.index is second.index
    assert len(fake_chunks) == 1


def test_qdrant_source_builds_separate_index(fake_chunks, monkeypatch):
    """bm25_source마다 색인을 따로 만든다. 기본(qdrant)은 load_chunks를 부르지 않는다."""
    monkeypatch.setattr(
        bm25,
        "load_qdrant_chunks",
        lambda domain, difficulty, levels: [
            FakeChunk(d, d.page_content) for d in DOCS[:3]
        ],
    )
    scope = {"domain": "shipping", "difficulty": "basic"}
    from_jsonl = build_bm25_retriever(**scope, bm25_source="jsonl")
    from_qdrant = build_bm25_retriever(**scope)  # 기본값 = qdrant
    assert from_jsonl.index is not from_qdrant.index
    assert len(from_qdrant.index.docs) == 3
    assert len(fake_chunks) == 1  # jsonl은 한 번만 읽었다


def test_unknown_source_stops_before_reading(fake_chunks):
    with pytest.raises(ValueError, match="bm25_source"):
        build_bm25_retriever(domain="shipping", difficulty="basic", bm25_source="csv")
    with pytest.raises(ValueError, match="source"):
        load_team_chunks("shipping", "basic", ("para",), "csv")
    assert fake_chunks == []


def test_as_levels_sorts_and_dedupes():
    assert as_levels(["table", "para", "para"]) == ("para", "table")


def test_retriever_is_exported():
    assert retrievers.build_bm25_retriever is build_bm25_retriever


# ---------- Qdrant에서 청크 읽기 (가짜 클라이언트) ----------


def chunk(chunk_id: str, level: str = "para", text: str = "본문") -> Chunk:
    book_id, rest = chunk_id.split("-p", 1)
    return Chunk(
        chunk_id=chunk_id,
        parent_id=None if level == "page" else f"{book_id}-p{rest.split('-')[0]}",
        level=level,
        chunk_type={"page": "page", "para": "text", "table": "table"}[level],
        book_id=book_id,
        file_name=f"{book_id}.pdf",
        page=int(rest.split("-")[0]),
        doc_title="예약 안내",
        section_title="마감",
        difficulty="intermediate",
        domain="shipping",
        text=text,
        context="맥락 한 줄",
    )


class FakeQdrant:
    """scroll만 흉내 낸다. 받은 인자를 기록하고, payload를 limit개씩 나눠 준다."""

    def __init__(self, payloads: list[dict], fail_times: int = 0):
        self.payloads = payloads
        self.fail_times = fail_times
        self.calls: list[dict] = []

    def scroll(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_times:
            self.fail_times -= 1
            raise ResponseHandlingException(ConnectionResetError("끊김"))
        levels = set(kwargs["scroll_filter"].must[0].match.any)
        rows = [p for p in self.payloads if p["metadata"]["level"] in levels]
        start = kwargs["offset"] or 0
        end = start + kwargs["limit"]
        points = [SimpleNamespace(payload=p) for p in rows[start:end]]
        return points, (end if end < len(rows) else None)


@pytest.fixture
def fake_qdrant(monkeypatch):
    """bm25.get_store를 가짜로 바꾼다. 부른 (domain, difficulty)를 기록한다."""
    state = {"client": FakeQdrant([]), "calls": []}

    def _get_store(domain, difficulty):
        state["calls"].append((domain, difficulty))
        return SimpleNamespace(
            client=state["client"],
            collection_name=f"kant_{domain}_{difficulty}_bge-m3",
        )

    monkeypatch.setattr(bm25, "get_store", _get_store)
    monkeypatch.setattr(bm25, "SCROLL_LIMIT", 2)  # 여러 번 나눠 받는 경우도 확인
    monkeypatch.setattr(bm25.time, "sleep", lambda seconds: None)
    bm25.get_bm25_index.cache_clear()
    yield state
    bm25.get_bm25_index.cache_clear()


def qdrant_payload(c: Chunk) -> dict:
    """청킹 담당이 올리는 모양 그대로: payload() + content_hash."""
    return {**c.payload(), "content_hash": "hash"}


def test_payload_round_trip_keeps_chunk_and_index_text():
    original = chunk("IS01-p3-c2")
    restored = chunk_from_payload(qdrant_payload(original))
    assert restored == original
    assert restored.embedding_text() == original.embedding_text()
    assert Document(**restored.payload()) == Document(**original.payload())


def test_file_order_matches_jsonl_order():
    ids = [
        "IS01-p1",
        "IS01-p1-c1",
        "IS01-p1-c2",
        "IS01-p1-c10",
        "IS01-p2",
        "IS01-p2-c1",
        "IS02-p1-c1",
    ]
    chunks = [chunk(i, level="page" if "-c" not in i else "para") for i in ids]
    shuffled = random.Random(0).sample(chunks, len(chunks))
    assert [c.chunk_id for c in sorted(shuffled, key=file_order)] == ids


def test_qdrant_chunks_follow_difficulty_levels_and_file_order(fake_qdrant):
    ids = ["IS01-p1-c1", "IS01-p1-c2", "IS01-p2-c1", "IS02-p1-c1", "IS02-p1-c3"]
    rows = [chunk(i) for i in ids] + [chunk("IS01-p1", level="page")]
    fake_qdrant["client"] = FakeQdrant(
        [qdrant_payload(c) for c in random.Random(1).sample(rows, len(rows))]
    )
    chunks = bm25.load_qdrant_chunks("shipping", "intermediate", ("para", "table"))
    assert [c.chunk_id for c in chunks] == ids  # page는 빠지고 jsonl 순서로
    assert fake_qdrant["calls"] == [("shipping", "intermediate")]
    calls = fake_qdrant["client"].calls
    assert len(calls) == 3  # 5개를 2개씩: 2 + 2 + 1
    assert {c["collection_name"] for c in calls} == {
        "kant_shipping_intermediate_bge-m3"
    }
    assert all(c["with_vectors"] is False for c in calls)  # 벡터는 받지 않는다


def test_qdrant_and_jsonl_give_same_index(fake_qdrant, monkeypatch):
    rows = [
        chunk(f"IS01-p{p}-c{n}", text=f"예약 마감 {p}{n}")
        for p in (1, 2)
        for n in (1, 2)
    ]
    monkeypatch.setattr(
        bm25, "load_chunks", lambda difficulty, domain, levels: iter(rows)
    )
    fake_qdrant["client"] = FakeQdrant([qdrant_payload(c) for c in reversed(rows)])
    scope = ("shipping", "intermediate", ("para", "table"))
    assert load_team_chunks(*scope, "qdrant") == load_team_chunks(*scope, "jsonl")
    a = bm25.get_bm25_index(*scope, "jsonl")
    b = bm25.get_bm25_index(*scope, "qdrant")
    assert a.docs == b.docs
    assert a.search("예약 마감", k=3) == b.search("예약 마감", k=3)


def test_qdrant_scroll_retries_then_succeeds(fake_qdrant):
    fake_qdrant["client"] = FakeQdrant(
        [qdrant_payload(chunk("IS01-p1-c1"))], fail_times=2
    )
    chunks = bm25.load_qdrant_chunks("shipping", "basic", ("para",))
    assert [c.chunk_id for c in chunks] == ["IS01-p1-c1"]
    assert len(fake_qdrant["client"].calls) == 3


def test_qdrant_scroll_gives_up_with_hint(fake_qdrant):
    fake_qdrant["client"] = FakeQdrant([], fail_times=bm25.SCROLL_RETRIES)
    with pytest.raises(RuntimeError, match="--bm25-source jsonl"):
        bm25.load_qdrant_chunks("shipping", "basic", ("para",))


def test_empty_qdrant_levels_is_error(fake_qdrant):
    fake_qdrant["client"] = FakeQdrant([qdrant_payload(chunk("IS01-p1", level="page"))])
    with pytest.raises(RuntimeError, match="kant_shipping_basic_bge-m3"):
        bm25.load_qdrant_chunks("shipping", "basic", ("para", "table"))


# ---------- 실제 jsonl (Qdrant 없이) ----------

needs_corpus = pytest.mark.skipif(
    not any(corpus_dir("basic").glob("*.jsonl")), reason="basic jsonl 없음"
)


@needs_corpus
def test_real_basic_corpus():
    bm25.get_bm25_index.cache_clear()
    retriever = build_bm25_retriever(
        domain="shipping", difficulty="basic", bm25_source="jsonl"
    )
    assert len(retriever.index.docs) == 1600  # para 1600, Basic엔 table 없음
    docs = retriever.invoke("부산에서 LA로 컨테이너를 예약할 때 필요한 정보는?")
    assert len(docs) == 5
    assert docs[0].metadata["chunk_id"] == "SB01-p1-c1"  # 11/9 클라우드 실측 13.195
    assert all(d.metadata["level"] == "para" for d in docs)


@needs_corpus
def test_real_basic_payload_round_trip_and_order():
    """jsonl 청크를 Qdrant에 올린 모양으로 바꿨다가 되돌리면, 섞어도 같은 청크·같은 순서가 된다."""
    chunks = list(load_chunks("basic", domain="shipping", levels=("para", "table")))
    payloads = [{**c.payload(), "content_hash": "h"} for c in chunks]
    random.Random(2).shuffle(payloads)
    restored = sorted((chunk_from_payload(p) for p in payloads), key=file_order)
    assert restored == chunks
