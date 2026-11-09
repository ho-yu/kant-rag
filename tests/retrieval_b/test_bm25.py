"""bm25.py와 build_bm25_retriever 테스트.

기본 테스트: 작은 문서 목록 + 띄어쓰기 토큰화라서 결과가 손으로 확인된다.
실제 jsonl 테스트: 레포에 있는 artifacts/corpus/basic/shipping으로 색인을 만든다 (Qdrant 안 씀, 약 10초).

실행 (레포 맨 위에서)
    uv run pytest tests/retrieval_b/test_bm25.py -q
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from rag_service.retrieval_b import bm25, build_bm25_retriever, retrievers
from rag_service.retrieval_b.bm25 import BM25Index, as_levels, tokenize
from rag_service.storage.corpus import corpus_dir


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
    retriever = build_bm25_retriever(domain="shipping", difficulty="basic", k=2)
    docs = retriever.invoke("예약")
    assert len(docs) == 2
    assert fake_chunks == [("shipping", "basic", ("para", "table"))]
    for key in ("chunk_id", "book_id", "page", "level", "score"):
        assert key in docs[0].metadata


def test_index_is_built_once_per_setting(fake_chunks):
    first = build_bm25_retriever(levels=["table", "para"])
    second = build_bm25_retriever(levels=["para", "table"])
    assert first.index is second.index
    assert len(fake_chunks) == 1


def test_as_levels_sorts_and_dedupes():
    assert as_levels(["table", "para", "para"]) == ("para", "table")


def test_retriever_is_exported():
    assert retrievers.build_bm25_retriever is build_bm25_retriever


# ---------- 실제 jsonl (Qdrant 없이) ----------

needs_corpus = pytest.mark.skipif(
    not any(corpus_dir("basic").glob("*.jsonl")), reason="basic jsonl 없음"
)


@needs_corpus
def test_real_basic_corpus():
    bm25.get_bm25_index.cache_clear()
    retriever = build_bm25_retriever(domain="shipping", difficulty="basic")
    assert len(retriever.index.docs) == 1600  # para 1600, Basic엔 table 없음
    docs = retriever.invoke("부산에서 LA로 컨테이너를 예약할 때 필요한 정보는?")
    assert len(docs) == 5
    assert docs[0].metadata["chunk_id"] == "SB01-p1-c1"  # 11/9 클라우드 실측 13.195
    assert all(d.metadata["level"] == "para" for d in docs)
