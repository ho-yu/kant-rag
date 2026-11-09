"""chain.py가 쓰는 검색기(retriever)를 만든다. 지금은 Naive(dense)와 BM25가 있다.

chain.py는 build_naive_retriever(domain=, difficulty=)로 검색기를 만들고
.invoke(질문)을 불러 Document 목록(chunk_id, book_id, page, score 포함)을 받는다.
"""

from __future__ import annotations

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_qdrant import QdrantVectorStore

from rag_service.retrieval_b.bm25 import BM25Index, as_levels, get_bm25_index
from rag_service.retrieval_b.dense import dense_search, get_store

DEFAULT_K = 5
DEFAULT_LEVELS = ["para", "table"]


class DenseRetriever(BaseRetriever):
    """dense_search를 .invoke(질문) → Document 목록 모양으로 감싼다."""

    store: QdrantVectorStore
    k: int = DEFAULT_K
    levels: list[str] = DEFAULT_LEVELS

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return dense_search(self.store, query, k=self.k, levels=self.levels)


def build_naive_retriever(
    domain: str = "shipping",
    difficulty: str = "basic",
    k: int = DEFAULT_K,
    levels: list[str] | None = None,
) -> DenseRetriever:
    """Naive RAG 검색기. chain.py가 build_naive_retriever(domain=, difficulty=)로 부른다."""
    return DenseRetriever(
        store=get_store(domain, difficulty),
        k=k,
        levels=list(DEFAULT_LEVELS if levels is None else levels),
    )


class BM25Retriever(BaseRetriever):
    """BM25Index.search를 .invoke(질문) → Document 목록 모양으로 감싼다."""

    index: BM25Index
    k: int = DEFAULT_K

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self.index.search(query, k=self.k)


def build_bm25_retriever(
    domain: str = "shipping",
    difficulty: str = "basic",
    k: int = DEFAULT_K,
    levels: list[str] | None = None,
) -> BM25Retriever:
    """BM25 단독 검색기. 만들 때 jsonl을 읽어 색인까지 끝낸다 (Naive와 같은 인자)."""
    levels = DEFAULT_LEVELS if levels is None else levels
    return BM25Retriever(
        index=get_bm25_index(domain, difficulty, as_levels(levels)),
        k=k,
    )
