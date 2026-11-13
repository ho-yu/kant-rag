"""검색기 구현 변경 없이 체인에서 필요한 검색기 준비와 Hybrid 자료 공유 관리."""

from threading import RLock

from rag_service.retrieval_a.hybrid import CONFIG_PATH as HYBRID_CONFIG
from rag_service.retrieval_a.hybrid import build_hybrid_retriever
from rag_service.retrieval_a.naive import (
    REPO_ROOT,
    build_naive_retriever,
    load_config,
    model_profile,
    resolve_scope,
)
from rag_service.retrieval_a.rerank import CONFIG_PATH as RERANK_CONFIG
from rag_service.retrieval_a.rerank import HybridRerankRetriever, make_scorer

# 난이도·체인이 공유하는 모델/캐시의 최초 준비만 보호한다. 검색 실행에는 사용하지 않는다.
RESOURCE_INIT_LOCK = RLock()


def warmup_retriever(retriever, mode):
    """지연 로딩과 토크나이저 설정을 완료한 뒤 검색기를 요청에 공개한다."""
    docs = retriever.invoke("해상 운송")
    if mode == "hybrid_rerank" and not docs:
        retriever.score_fn("해상 운송", ["해상 운송"])


class RetrieverPool:
    """한 체인에서 사용하는 검색기의 최초 요청 시 생성 및 자원 재사용."""

    def __init__(self, domain, difficulty, profile=None, top_k=None):
        self.scope = {"domain": domain, "difficulty": difficulty, "profile": profile}
        self.top_k = top_k  # 최종 문맥 수 덮어쓰기 (None이면 설정 파일 값)
        # 검색 방식별 실행 객체와 설정 파일별 Hybrid 원본의 별도 보관
        self.retrievers, self.hybrids = {}, {}

    def _hybrid(self, path, top_k=None):
        """동일 설정의 원문과 벡터 및 Sparse 점수 함수 공유."""
        path = path.resolve()
        if path not in self.hybrids:
            self.hybrids[path] = build_hybrid_retriever(**self.scope, config_path=path)
        base = self.hybrids[path]
        # 깊은 복사와 기존 top_k 변경 없이 후보 개수만 다른 얕은 복사 생성
        # 원문 목록과 벡터 행렬 및 Sparse 점수 함수의 동일 객체 참조 유지
        return base if top_k is None else base.model_copy(update={"top_k": top_k}, deep=False)

    def get(self, mode):
        """준비된 검색기는 즉시 반환하고 최초 준비만 한 번 수행한다."""
        if mode in self.retrievers:
            return self.retrievers[mode]
        with RESOURCE_INIT_LOCK:
            if mode not in self.retrievers:
                if mode == "naive":
                    retriever = build_naive_retriever(**self.scope, top_k=self.top_k)
                elif mode == "hybrid":
                    retriever = self._hybrid(HYBRID_CONFIG, self.top_k)
                elif mode == "hybrid_rerank":
                    cfg = load_config(RERANK_CONFIG)
                    hybrid_path = REPO_ROOT / cfg["hybrid_config"]
                    _, difficulty = resolve_scope(
                        load_config(hybrid_path), self.scope["domain"], self.scope["difficulty"]
                    )
                    # 기존 Rerank 클래스와 점수 함수를 그대로 사용하고 Hybrid만 공유 객체로 주입
                    retriever = HybridRerankRetriever(
                        hybrid=self._hybrid(hybrid_path, cfg["search"]["candidate_k"]),
                        score_fn=make_scorer(
                            cfg, model_profile(self.scope["profile"] or difficulty)["reranker"]
                        ),
                        top_k=self.top_k or cfg["search"]["top_k"],
                        use_context=cfg["reranker"].get("use_context", True),
                    )
                else:
                    raise ValueError(f"지원하지 않는 검색 방식: {mode}")
                warmup_retriever(retriever, mode)
                self.retrievers[mode] = retriever
            return self.retrievers[mode]

    def warmup(self, modes):
        """답변 생성 없이 검색 실행으로 지연 로딩 모델과 검색 자료의 사전 준비."""
        for mode in modes:
            self.get(mode)


class RetrievalBPool(RetrieverPool):
    """retrieval_b 검색기의 최초 요청 시 생성 및 재사용. 워밍업은 RetrieverPool과 같음."""

    def __init__(self, domain, difficulty, k=None):
        super().__init__(domain, difficulty)
        self.scope = {"domain": domain, "difficulty": difficulty}
        self.k = k  # 최종 문맥 수 (None이면 retrieval_b Settings 기본값)

    def get(self, mode):
        if mode in self.retrievers:
            return self.retrievers[mode]
        with RESOURCE_INIT_LOCK:
            if mode not in self.retrievers:
                from rag_service import retrieval_b

                builders = {
                    "naive": retrieval_b.build_naive_retriever,
                    "hybrid": retrieval_b.build_hybrid_retriever,
                    "hybrid_rerank": retrieval_b.build_rerank_retriever,
                }
                if mode not in builders:
                    raise ValueError(f"지원하지 않는 검색 방식: {mode}")
                extra = {"k": self.k} if self.k else {}
                retriever = builders[mode](**self.scope, **extra)
                warmup_retriever(retriever, mode)
                self.retrievers[mode] = retriever
            return self.retrievers[mode]
