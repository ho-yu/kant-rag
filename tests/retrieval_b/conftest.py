"""retrieval_b 테스트 공용 도구.

- TableEmbeddings: 진짜 모델 대신 "글자 → 벡터" 표를 그대로 돌려주는 가짜 임베딩
- make_store: 메모리 Qdrant(":memory:")에 가짜 문서를 넣은 검색 창구. 팀 Qdrant는 건드리지 않는다
- integration 마커: 팀 Qdrant(읽기 전용)와 bge-m3가 필요한 테스트. 팀 push 전 점검은 -m "not integration"
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: 팀 Qdrant·실제 모델이 필요한 테스트"
    )


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
