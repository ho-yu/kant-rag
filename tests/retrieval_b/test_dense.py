"""dense.py 테스트.

기본 테스트: 가짜 벡터 + 메모리 Qdrant라서 모델 다운로드도, 팀 Qdrant 접속도 없다.
integration 테스트: 팀 Qdrant(읽기 전용)와 bge-m3를 실제로 쓴다. .env에 QDRANT_URL이 없으면 건너뛴다.

실행 (레포 맨 위에서)
    uv run pytest tests/retrieval_b/test_dense.py -q                  # 전부
    uv run pytest tests/retrieval_b/test_dense.py -q -m "not integration"   # 팀 push 전 점검과 같은 범위
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv
from langchain_core.documents import Document
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from rag_service.retrieval_b import dense
from rag_service.retrieval_b.dense import dense_search, get_store
from rag_service.storage.corpus import repo_root
from rag_service.storage.qdrant_store import level_filter

load_dotenv(repo_root() / ".env")

ALL_LEVELS = ["page", "para", "table"]
QUERY = {"q": [1.0, 0.0]}


def meta(chunk_id: str, level: str = "page", page: int = 1) -> dict:
    return {"chunk_id": chunk_id, "book_id": "SB01", "page": page, "level": level}


# 개선판 4장 2강 §6 손계산: 질문 q=[1,0]과의 코사인 → b 1.0, a 0.7071, c -1.0
HAND_ROWS = [
    ("a", [1.0, 1.0], meta("SB01-p1-c1")),
    ("b", [3.0, 0.0], meta("SB01-p2-c1", page=2)),
    ("c", [-1.0, 0.0], meta("SB01-p3-c1", level="para", page=3)),
]


def ids(docs) -> list[str]:
    return [d.metadata["chunk_id"] for d in docs]


# ---------- dense_search: 기본 동작 ----------


def test_hand_calculation_order_and_scores(make_store):
    store = make_store(HAND_ROWS, QUERY)
    docs = dense_search(store, "q", k=3, levels=ALL_LEVELS)
    assert [d.page_content for d in docs] == ["b", "a", "c"]
    assert [d.metadata["score"] for d in docs] == pytest.approx(
        [1.0, 0.7071, -1.0], abs=1e-4
    )


def test_level_filter_keeps_only_given_levels(make_store):
    store = make_store(HAND_ROWS, QUERY)  # c만 para
    docs = dense_search(store, "q", k=3, levels=["page"])
    assert [d.page_content for d in docs] == ["b", "a"]
    assert {d.metadata["level"] for d in docs} == {"page"}


def test_k_larger_than_collection_returns_what_exists(make_store):
    store = make_store(HAND_ROWS, QUERY)
    assert len(dense_search(store, "q", k=10, levels=ALL_LEVELS)) == 3


def test_k_limits_result_count(make_store):
    store = make_store(HAND_ROWS, QUERY)
    assert ids(dense_search(store, "q", k=1, levels=ALL_LEVELS)) == ["SB01-p2-c1"]


def test_result_metadata_has_fields_for_chain(make_store):
    """chain.py의 merge_docs(chunk_id)와 to_pages(book_id, page)가 쓰는 키가 남아 있어야 한다."""
    store = make_store(HAND_ROWS, QUERY)
    for doc in dense_search(store, "q", k=3, levels=ALL_LEVELS):
        for key in ("chunk_id", "book_id", "page", "level", "score"):
            assert key in doc.metadata
        assert isinstance(doc.metadata["score"], float)


def test_page_content_is_original_text(make_store):
    store = make_store(HAND_ROWS, QUERY)
    assert {
        d.page_content for d in dense_search(store, "q", k=3, levels=ALL_LEVELS)
    } == {"a", "b", "c"}


# ---------- dense_search: 정렬 규칙 (-점수, chunk_id) ----------


class ListStore:
    """정해 둔 (Document, 점수) 목록을 순서 그대로 돌려주는 가짜 검색 창구."""

    def __init__(self, pairs):
        self.pairs = pairs

    def similarity_search_with_score(self, question, k, filter=None):
        return self.pairs[:k]


def test_results_are_sorted_by_score_then_chunk_id():
    """검색 엔진이 동점을 아무 순서로 돌려줘도 결과는 (-점수, chunk_id)로 고정돼야 한다."""
    pairs = [
        (Document(page_content="x2", metadata=meta("SB02-p1-c1")), 0.5),
        (Document(page_content="top", metadata=meta("SB09-p1-c1")), 0.9),
        (Document(page_content="x1", metadata=meta("SB01-p9-c1")), 0.5),
    ]
    docs = dense_search(ListStore(pairs), "q", k=3, levels=ALL_LEVELS)
    assert ids(docs) == ["SB09-p1-c1", "SB01-p9-c1", "SB02-p1-c1"]


def test_ties_are_broken_by_chunk_id(make_store):
    rows = [
        ("x2", [0.0, 1.0], meta("SB02-p1-c1")),
        ("x1", [0.0, 1.0], meta("SB01-p9-c1")),  # x2와 같은 벡터 → 같은 점수
        ("top", [1.0, 0.0], meta("SB09-p1-c1")),
    ]
    store = make_store(rows, QUERY)
    docs = dense_search(store, "q", k=3, levels=ALL_LEVELS)
    assert ids(docs) == ["SB09-p1-c1", "SB01-p9-c1", "SB02-p1-c1"]
    assert docs[1].metadata["score"] == pytest.approx(docs[2].metadata["score"])


def test_scores_are_descending(make_store):
    store = make_store(HAND_ROWS, QUERY)
    scores = [
        d.metadata["score"] for d in dense_search(store, "q", k=3, levels=ALL_LEVELS)
    ]
    assert scores == sorted(scores, reverse=True)


# ---------- dense_search: 입력 검사 ----------


@pytest.mark.parametrize(
    ("question", "k"),
    [("", 3), ("   ", 3), ("\n\t", 3), ("q", 0), ("q", -1)],
)
def test_invalid_input_raises_before_search(question, k):
    class ExplodingStore:
        def similarity_search_with_score(self, *args, **kwargs):
            raise AssertionError("입력이 잘못됐는데 검색을 불렀다")

    with pytest.raises(ValueError):
        dense_search(ExplodingStore(), question, k=k, levels=ALL_LEVELS)


def test_error_message_shows_bad_k(make_store):
    store = make_store(HAND_ROWS, QUERY)
    with pytest.raises(ValueError, match="0"):
        dense_search(store, "q", k=0, levels=ALL_LEVELS)


# ---------- get_embeddings: 모델 이름과 캐시 (진짜 모델은 안 불러온다) ----------


class FakeHF:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def fake_hf(monkeypatch):
    monkeypatch.setattr(dense, "HuggingFaceEmbeddings", FakeHF)
    dense.get_embeddings.cache_clear()
    yield
    dense.get_embeddings.cache_clear()


def test_embedding_model_from_env_is_stripped(fake_hf, monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "  my/model  ")
    emb = dense.get_embeddings()
    assert emb.kwargs["model_name"] == "my/model"
    assert emb.kwargs["encode_kwargs"] == {"normalize_embeddings": True}


def test_empty_embedding_model_falls_back_to_bge_m3(fake_hf, monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "   ")
    assert dense.get_embeddings().kwargs["model_name"] == "BAAI/bge-m3"


def test_get_embeddings_is_cached(fake_hf, monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "")
    assert dense.get_embeddings() is dense.get_embeddings()


# ---------- get_store: 연결 오류 안내 (팀 Qdrant에 접속하지 않는다) ----------


class FakeClient:
    """collection_exists만 흉내 내는 가짜 Qdrant 클라이언트."""

    def __init__(self, exists=True, error=None):
        self.exists = exists
        self.error = error

    def collection_exists(self, name):
        if self.error is not None:
            raise self.error
        return self.exists


@pytest.fixture
def store_env(monkeypatch):
    """get_store 캐시를 비우고, 가짜 URL·키를 넣는다. 진짜 .env 값은 쓰지 않는다."""
    monkeypatch.setenv("QDRANT_URL", "http://fake-qdrant")
    monkeypatch.setenv("QDRANT_API_KEY", "fake-secret-key")
    dense.get_store.cache_clear()
    yield monkeypatch
    dense.get_store.cache_clear()


def use_client(monkeypatch, client):
    monkeypatch.setattr(dense, "get_client", lambda cfg: client)


def test_empty_qdrant_url_tells_how_to_fix(store_env):
    # delenv가 아니라 빈 값: 지우면 load_dotenv가 .env에서 다시 채운다
    store_env.setenv("QDRANT_URL", "  ")
    use_client(store_env, FakeClient(error=AssertionError("URL이 없는데 접속했다")))
    with pytest.raises(RuntimeError, match=r"\.env"):
        get_store("shipping", "basic")


def test_missing_collection_names_the_collection(store_env):
    use_client(store_env, FakeClient(exists=False))
    with pytest.raises(RuntimeError, match="kant_shipping_advanced_bge-m3"):
        get_store("shipping", "advanced")


@pytest.mark.parametrize("status", [401, 403])
def test_rejected_api_key_says_check_key(store_env, status):
    error = UnexpectedResponse(status, "Forbidden", b"", None)
    use_client(store_env, FakeClient(error=error))
    with pytest.raises(RuntimeError, match="QDRANT_API_KEY") as caught:
        get_store("shipping", "basic")
    assert "fake-secret-key" not in str(caught.value)  # 키 값은 화면에 안 나온다


def test_unreachable_server_says_check_url(store_env):
    error = ResponseHandlingException(ConnectionRefusedError("refused"))
    use_client(store_env, FakeClient(error=error))
    with pytest.raises(RuntimeError, match="QDRANT_URL") as caught:
        get_store("shipping", "basic")
    assert "fake-qdrant" not in str(caught.value)  # URL 값도 안 나온다


def test_other_server_errors_are_not_hidden(store_env):
    error = UnexpectedResponse(500, "Server Error", b"", None)
    use_client(store_env, FakeClient(error=error))
    with pytest.raises(UnexpectedResponse):
        get_store("shipping", "basic")


# ---------- integration: 팀 Qdrant (읽기 전용) + bge-m3 ----------

needs_team_qdrant = pytest.mark.skipif(
    not os.getenv("QDRANT_URL"), reason=".env에 QDRANT_URL 없음"
)


@pytest.mark.integration
@needs_team_qdrant
def test_team_collection_counts():
    store = get_store("shipping", "basic")
    assert store.collection_name == "kant_shipping_basic_bge-m3"
    assert store.client.count(store.collection_name).count == 1960
    page_only = store.client.count(
        store.collection_name, count_filter=level_filter(["page"])
    )
    assert page_only.count == 360
    assert get_store("shipping", "basic") is store


@pytest.mark.integration
@needs_team_qdrant
def test_team_search_finds_booking_guide():
    store = get_store("shipping", "basic")
    docs = dense_search(
        store,
        "부산에서 LA로 컨테이너를 예약할 때 필요한 정보는?",
        k=5,
        levels=["para", "table"],
    )
    assert len(docs) == 5
    assert docs[0].metadata["book_id"] == "SB01"  # 11/9 Windows 실측: SB01-p1-c1 0.7072
    assert all(d.metadata["level"] in ("para", "table") for d in docs)
    scores = [d.metadata["score"] for d in docs]
    assert scores == sorted(scores, reverse=True)
