"""chain.py가 쓰는 검색기(retriever)를 만든다. Naive(dense), BM25, Hybrid(RRF), Rerank가 있다.

chain.py와의 약속 (retrieval_a와 같은 모양이라 import 한 줄만 바꾸면 갈아 끼울 수 있다)
    build_naive_retriever(domain=, difficulty=, profile=)   ← create_chain이 받은 값을 그대로 넘긴다
    retriever.invoke(질문) → Document 목록 (metadata에 chunk_id, book_id, page, level, score)

값을 정하는 곳 (settings.py 참고)
- domain, difficulty: 호출하는 쪽이 꼭 넘긴다. 기본값이 없어서 빠뜨리면 바로 오류가 난다.
- k, levels, candidate_k, rrf_k, rerank_n, bm25_source …: 인자로 넘기면 그 값, 안 넘기면 Settings 기본값.
  평가 스크립트는 명령줄 옵션(--k, --rerank-n 등)으로 받은 값을 인자로 넘긴다.
- profile: retrieval_a 호환용. retrieval_b는 모델 하나(EMBEDDING_MODEL)만 써서 값은 쓰지 않는다.

무거운 자원(Qdrant 연결, BM25 색인, 리랭커)은 (domain, difficulty, levels[, bm25_source])마다 한 번만 만들고
검색기끼리 같이 쓴다. 그래서 chain.py가 검색기 3개를 한꺼번에 만들어도 색인은 한 번만 만든다.

[2026-11-10 리팩토링] 이유: profile 인자가 없어 chain.py와 호환되지 않고 domain·difficulty 누락 시 shipping/basic으로 조용히 검색함 → 효과: import 한 줄로 retrieval_a와 교체 가능, 잘못된 입력은 Qdrant 접속 전에 차단.
[2026-11-11 기능 추가] 이유: BM25만 로컬 jsonl을 읽어 Dense(팀 Qdrant)와 다른 곳의 청크를 볼 수 있었음 → 효과: BM25도 기본으로 팀 Qdrant에서 청크를 읽음 (chain의 hybrid·hybrid_rerank 포함). --bm25-source jsonl로 로컬 파일도 고를 수 있음 (검색 계산은 그대로).
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_qdrant import QdrantVectorStore
from pydantic import ConfigDict

from rag_service.retrieval_b.bm25 import BM25Index, as_levels, get_bm25_index
from rag_service.retrieval_b.dense import dense_search, get_store
from rag_service.retrieval_b.fusion import rrf_fuse
from rag_service.retrieval_b.rerank import ScoreFn, cross_encoder_scorer, rerank
from rag_service.retrieval_b.settings import (
    BM25_SOURCES,
    Settings,
    check_int,
    check_levels,
    get_settings,
    validate_scope,
)


def _pick(name: str, value: int | None, default: int, minimum: int = 1) -> int:
    """넘긴 값이 있으면 그 값을, 없으면 설정값을 쓴다. 어느 쪽이든 범위를 검사한다."""
    return check_int(name, default if value is None else value, minimum)


def _pick_levels(levels: Iterable[str] | None, s: Settings) -> list[str]:
    """새 list로 돌려줘서 넘긴 목록을 나중에 바꿔도 검색기 설정은 그대로다."""
    return list(s.levels if levels is None else check_levels(levels))


def _pick_source(source: str | None, s: Settings) -> str:
    """BM25가 청크를 읽는 곳. 넘긴 값이 있으면 그 값, 없으면 설정값 (jsonl 또는 qdrant)."""
    source = s.bm25_source if source is None else source
    if source not in BM25_SOURCES:
        raise ValueError(f"bm25_source는 {BM25_SOURCES} 중 하나여야 합니다: {source!r}")
    return source


def _ignore_profile(profile: str | None) -> None:
    if profile is not None:
        warnings.warn(
            f"retrieval_b는 profile을 쓰지 않습니다 (받은 값: {profile!r}). "
            "모델은 EMBEDDING_MODEL, RERANKER_MODEL 환경변수로 정합니다.",
            stacklevel=3,
        )


class DenseRetriever(BaseRetriever):
    """dense_search를 .invoke(질문) → Document 목록 모양으로 감싼다."""

    store: QdrantVectorStore
    k: int
    levels: list[str]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return dense_search(self.store, query, k=self.k, levels=self.levels)


def build_naive_retriever(
    *,
    domain: str,
    difficulty: str,
    profile: str | None = None,
    k: int | None = None,
    levels: list[str] | None = None,
) -> DenseRetriever:
    """Naive RAG 검색기. chain.py가 build_naive_retriever(domain=, difficulty=, profile=)로 부른다."""
    domain, difficulty = validate_scope(domain, difficulty)
    _ignore_profile(profile)
    s = get_settings()
    return DenseRetriever(
        store=get_store(domain, difficulty),
        k=_pick("k", k, s.k),
        levels=_pick_levels(levels, s),
    )


class BM25Retriever(BaseRetriever):
    """BM25Index.search를 .invoke(질문) → Document 목록 모양으로 감싼다."""

    index: BM25Index
    k: int

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self.index.search(query, k=self.k)


def build_bm25_retriever(
    *,
    domain: str,
    difficulty: str,
    profile: str | None = None,
    k: int | None = None,
    levels: list[str] | None = None,
    bm25_source: str | None = None,
) -> BM25Retriever:
    """BM25 단독 검색기. 만들 때 청크를 읽어 색인까지 끝낸다 (Naive와 같은 인자 + bm25_source)."""
    domain, difficulty = validate_scope(domain, difficulty)
    _ignore_profile(profile)
    s = get_settings()
    return BM25Retriever(
        index=get_bm25_index(
            domain,
            difficulty,
            as_levels(_pick_levels(levels, s)),
            _pick_source(bm25_source, s),
        ),
        k=_pick("k", k, s.k),
    )


class HybridRetriever(BaseRetriever):
    """Dense와 BM25에서 후보를 candidate_k개씩 뽑고 RRF로 합쳐 상위 k개를 돌려준다."""

    store: QdrantVectorStore
    index: BM25Index
    k: int
    levels: list[str]
    candidate_k: int
    rrf_k: int

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
    *,
    domain: str,
    difficulty: str,
    profile: str | None = None,
    k: int | None = None,
    levels: list[str] | None = None,
    candidate_k: int | None = None,
    rrf_k: int | None = None,
    bm25_source: str | None = None,
) -> HybridRetriever:
    """Hybrid(Dense + BM25, RRF) 검색기. Dense와 BM25는 같은 levels의 청크만 본다."""
    domain, difficulty = validate_scope(domain, difficulty)
    _ignore_profile(profile)
    s = get_settings()
    levels = _pick_levels(levels, s)
    return HybridRetriever(
        store=get_store(domain, difficulty),
        index=get_bm25_index(
            domain, difficulty, as_levels(levels), _pick_source(bm25_source, s)
        ),
        k=_pick("k", k, s.k),
        levels=levels,
        candidate_k=_pick("candidate_k", candidate_k, s.candidate_k),
        rrf_k=_pick("rrf_k", rrf_k, s.rrf_k, minimum=0),
    )


class RerankRetriever(BaseRetriever):
    """Hybrid 후보 rerank_n개를 리랭커로 다시 세워 상위 k개를 돌려준다."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    hybrid: HybridRetriever
    score_fn: ScoreFn
    k: int

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        candidates = self.hybrid.invoke(
            query, config={"callbacks": run_manager.get_child()}
        )
        return rerank(query, candidates, k=self.k, score_fn=self.score_fn)


def build_rerank_retriever(
    *,
    domain: str,
    difficulty: str,
    profile: str | None = None,
    k: int | None = None,
    levels: list[str] | None = None,
    rerank_n: int | None = None,
    candidate_k: int | None = None,
    rrf_k: int | None = None,
    model_name: str | None = None,
    max_length: int | None = None,
    score_fn: ScoreFn | None = None,
    bm25_source: str | None = None,
) -> RerankRetriever:
    """Hybrid + Rerank 검색기. Hybrid에서 rerank_n개(기본 50)를 받아 리랭커로 k개를 고른다.

    rerank_n < k 이면 ValueError. 모델은 첫 검색 때 불러온다.
    score_fn을 넘기면 실제 모델 대신 그 함수를 쓴다 (테스트용).
    """
    domain, difficulty = validate_scope(domain, difficulty)
    _ignore_profile(profile)
    s = get_settings()
    k = _pick("k", k, s.k)
    rerank_n = _pick("rerank_n", rerank_n, s.rerank_n)
    if rerank_n < k:
        raise ValueError(f"rerank_n({rerank_n})은 k({k}) 이상이어야 합니다.")
    hybrid = build_hybrid_retriever(
        domain=domain,
        difficulty=difficulty,
        k=rerank_n,
        levels=levels,
        candidate_k=candidate_k,
        rrf_k=rrf_k,
        bm25_source=bm25_source,
    )
    if score_fn is None:
        score_fn = cross_encoder_scorer(
            model_name,
            max_length=_pick("max_length", max_length, s.rerank_max_length),
            batch_size=s.rerank_batch_size,
        )
    return RerankRetriever(hybrid=hybrid, score_fn=score_fn, k=k)
