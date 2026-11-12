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

# 난이도별 체인에서도 공유하는 로컬 검색 모델의 초기화와 실행 보호
RETRIEVAL_LOCK = RLock()


class RetrieverPool:
    """한 체인에서 사용하는 검색기의 최초 요청 시 생성 및 자원 재사용."""

    def __init__(self, domain, difficulty, profile=None, top_k=None):
        self.scope = {"domain": domain, "difficulty": difficulty, "profile": profile}
        self.top_k = top_k  # 최종 문맥 수 덮어쓰기 (None이면 설정 파일 값)
        # 검색 방식별 실행 객체와 설정 파일별 Hybrid 원본의 별도 보관
        self.retrievers, self.hybrids = {}, {}
        self.lock = RLock()

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
        """선택한 검색기만 준비 및 동시 최초 요청의 중복 생성 방지."""
        with self.lock:
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
                self.retrievers[mode] = retriever
            return self.retrievers[mode]

    def warmup(self, modes):
        """답변 생성 없이 검색 실행으로 지연 로딩 모델과 검색 자료의 사전 준비."""
        # 실제 요청과 같은 잠금과 검색 객체 사용 및 워밍업 중 중복 초기화 방지
        with RETRIEVAL_LOCK:
            for mode in modes:
                retriever = self.get(mode)
                docs = retriever.invoke("해상 운송")
                # 검색 결과가 없어도 첫 점수 계산 때 로드되는 리랭커의 사전 준비
                if mode == "hybrid_rerank" and not docs:
                    retriever.score_fn("해상 운송", ["해상 운송"])
