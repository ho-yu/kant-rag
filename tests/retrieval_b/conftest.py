"""retrieval_b 테스트 공용 도구.

- TableEmbeddings: 진짜 모델 대신 "글자 → 벡터" 표를 그대로 돌려주는 가짜 임베딩
- make_store: 메모리 Qdrant(":memory:")에 가짜 문서를 넣은 검색 창구. 팀 Qdrant는 건드리지 않는다
- integration 마커: 팀 Qdrant(읽기 전용)와 bge-m3가 필요한 테스트. 팀 push 전 점검은 -m "not integration"
- default_settings: 모든 테스트에서 검색기가 기본 Settings를 쓰게 한다 (.env의 RETRIEVAL_B_* 무시)

[2026-11-10 리팩토링] 이유: 설정을 환경변수에서 읽게 되어 개인 .env에 따라 테스트 결과가 달라질 수 있음 → 효과: 누구 PC에서든 같은 기본 설정으로 테스트가 돎.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore

from rag_service.retrieval_b import rerank, retrievers
from rag_service.retrieval_b.settings import Settings


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: 팀 Qdrant·실제 모델이 필요한 테스트"
    )


@pytest.fixture(autouse=True)
def default_settings(monkeypatch):
    """검색기·리랭커가 읽는 설정을 기본값으로 고정한다. 바꾸려면 set_settings(k=7, ...)."""

    def set_settings(**values) -> Settings:
        s = Settings(**values)
        monkeypatch.setattr(retrievers, "get_settings", lambda: s)
        monkeypatch.setattr(rerank, "get_settings", lambda: s)
        return s

    set_settings()
    return set_settings


class TableEmbeddings(Embeddings):
    """표에 적힌 벡터를 그대로 돌려준다.

    langchain-qdrant는 연결할 때 embed_documents(["dummy_text"])로 차원을 잰다.
    그래서 표에 없는 글자에도 같은 길이의 벡터를 돌려준다.
    """

    def __init__(self, table: dict[str, list[float]]):
        self.table = table
        self.dim = len(next(iter(table.values())))

    def _vector(self, text: str) -> list[float]:
        return list(self.table.get(text, [0.0] * (self.dim - 1) + [1.0]))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


@pytest.fixture
def make_store():
    """rows = [(본문, 벡터, metadata), ...], queries = {질문: 벡터} → 메모리 Qdrant 검색 창구."""
    stores = []

    def _make(rows, queries):
        table = {text: vector for text, vector, _ in rows}
        table.update(queries)
        docs = [Document(page_content=text, metadata=meta) for text, _, meta in rows]
        store = QdrantVectorStore.from_documents(
            docs,
            TableEmbeddings(table),
            location=":memory:",
            collection_name="test_dense",
        )
        stores.append(store)
        return store

    yield _make
    for store in stores:
        store.client.close()
