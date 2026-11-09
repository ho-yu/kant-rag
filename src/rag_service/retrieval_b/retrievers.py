"""chain.py가 쓰는 검색기(retriever)를 만든다. Naive(dense), BM25, Hybrid(RRF), Rerank가 있다.

chain.py는 build_naive_retriever(domain=, difficulty=)로 검색기를 만들고
.invoke(질문)을 불러 Document 목록(chunk_id, book_id, page, score 포함)을 받는다.
"""

from __future__ import annotations

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_qdrant import QdrantVectorStore
from pydantic import ConfigDict

from rag_service.retrieval_b.bm25 import BM25Index, as_levels, get_bm25_index
from rag_service.retrieval_b.dense import dense_search, get_store
from rag_service.retrieval_b.fusion import RRF_K, rrf_fuse
from rag_service.retrieval_b.rerank import (
    MAX_LENGTH,
    RERANK_N,
    ScoreFn,
    cross_encoder_scorer,
    rerank,
)

DEFAULT_K = 5
DEFAULT_LEVELS = ["para", "table"]
# Hybrid가 Dense·BM25에서 각각 먼저 뽑는 후보 수 (k보다 작으면 k를 쓴다)
CANDIDATE_K = 30


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


class HybridRetriever(BaseRetriever):
    """Dense와 BM25에서 후보를 candidate_k개씩 뽑고 RRF로 합쳐 상위 k개를 돌려준다."""

    store: QdrantVectorStore
    index: BM25Index
    k: int = DEFAULT_K
    levels: list[str] = DEFAULT_LEVELS
    candidate_k: int = CANDIDATE_K
    rrf_k: int = RRF_K

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        n = max(self.candidate_k, self.k)
        rankings = {
            "dense": dense_search(self.store, query, k=n, levels=self.levels),
            "bm25": self.index.search(query, k=n),
        }
        return rrf_fuse(rankings, k=self.k, rrf_k=self.rrf_k)


def build_hybrid_retriever(
    domain: str = "shipping",
    difficulty: str = "basic",
    k: int = DEFAULT_K,
    levels: list[str] | None = None,
    candidate_k: int = CANDIDATE_K,
    rrf_k: int = RRF_K,
) -> HybridRetriever:
    """Hybrid(Dense + BM25, RRF) 검색기. Dense와 BM25는 같은 levels의 청크만 본다."""
    levels = list(DEFAULT_LEVELS if levels is None else levels)
    return HybridRetriever(
        store=get_store(domain, difficulty),
        index=get_bm25_index(domain, difficulty, as_levels(levels)),
        k=k,
        levels=levels,
        candidate_k=candidate_k,
        rrf_k=rrf_k,
    )


class RerankRetriever(BaseRetriever):
    """Hybrid 후보 rerank_n개를 리랭커로 다시 세워 상위 k개를 돌려준다."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    hybrid: HybridRetriever
    score_fn: ScoreFn
    k: int = DEFAULT_K

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        candidates = self.hybrid.invoke(
            query, config={"callbacks": run_manager.get_child()}
        )
        return rerank(query, candidates, k=self.k, score_fn=self.score_fn)


def build_rerank_retriever(
    domain: str = "shipping",
    difficulty: str = "basic",
    k: int = DEFAULT_K,
    levels: list[str] | None = None,
    rerank_n: int = RERANK_N,
    candidate_k: int = CANDIDATE_K,
    rrf_k: int = RRF_K,
    model_name: str | None = None,
    max_length: int = MAX_LENGTH,
    score_fn: ScoreFn | None = None,
) -> RerankRetriever:
    """Hybrid + Rerank 검색기. Hybrid에서 rerank_n개(기본 50)를 받아 리랭커로 k개를 고른다.

    rerank_n < k 이면 ValueError. 모델은 첫 검색 때 불러온다.
    score_fn을 넘기면 실제 모델 대신 그 함수를 쓴다 (테스트용).
    """
    if rerank_n < k:
        raise ValueError(f"rerank_n({rerank_n})은 k({k}) 이상이어야 합니다.")
    hybrid = build_hybrid_retriever(
        domain=domain,
        difficulty=difficulty,
        k=rerank_n,
        levels=levels,
        candidate_k=candidate_k,
        rrf_k=rrf_k,
    )
    return RerankRetriever(
        hybrid=hybrid,
        score_fn=score_fn or cross_encoder_scorer(model_name, max_length),
        k=k,
    )
