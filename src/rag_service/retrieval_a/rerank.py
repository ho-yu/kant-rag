"""검색 구현 A — Hybrid + Reranker.

1단계: Hybrid 검색기(BGE-M3 dense + sparse)로 후보 candidate_k개(기본 50)를 넓게 추린다.
2단계: bge-reranker-v2-m3(cross-encoder)가 (질문, 후보) 쌍을 함께 읽고 관련도 0~1 점수를 매긴다.
       질문과 문서를 처음부터 함께 읽으므로 부정·조건·관계까지 반영해 더 정확하지만 느리다.
       그래서 전체가 아니라 1단계 후보에만 쓴다.
3단계: 리랭커 점수순으로 상위 top_k개를 돌려준다.

반환: LangChain 검색기. `retriever.invoke(question)` → list[Document]
  - page_content: 청크 원문, metadata["book_id"], metadata["page"]
  - metadata["rerank_score"]: 리랭커 점수(0~1), ["hybrid_rank"]: 1단계 순위
    (hybrid가 붙인 hybrid_score·dense_rank·sparse_rank도 그대로 남는다)

직접 실행:
  uv run python -m rag_service.retrieval_a.rerank "VGM은 언제까지 내야 하나요?" --k 5
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from rag_service.retrieval_a.hybrid import build_hybrid_retriever, index_text
from rag_service.retrieval_a.naive import REPO_ROOT, _resolve_device, load_config

CONFIG_PATH = REPO_ROOT / "config" / "retrieval_a" / "rerank.yaml"


@lru_cache(maxsize=2)
def _cross_encoder(model_name: str, device: str, fp16: bool, max_length: int):
    """리랭커 모델은 무거우니 한 번만 올린다."""
    import torch
    from sentence_transformers import CrossEncoder

    device = _resolve_device(device)
    model_kwargs = {"dtype": torch.float16} if fp16 and device.startswith("cuda") else {}
    return CrossEncoder(
        model_name,
        device=device,
        max_length=max_length,
        model_kwargs=model_kwargs,
        activation_fn=torch.nn.Sigmoid(),  # 로짓 → 0~1 (FlagReranker의 normalize=True와 같음)
    )


def make_scorer(cfg: dict[str, Any]):
    """(질문, [텍스트]) → [점수] 함수를 만든다. 모델은 처음 부를 때 올린다."""
    rr = cfg["reranker"]

    def score(query: str, texts: list[str]) -> list[float]:
        model = _cross_encoder(rr["model_name"], rr["device"], rr.get("fp16", True), rr.get("max_length", 1024))
        scores = model.predict(
            [(query, t) for t in texts],
            batch_size=rr.get("batch_size", 16),
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [float(s) for s in scores]

    return score


class HybridRerankRetriever(BaseRetriever):
    """Hybrid 후보를 cross-encoder로 다시 줄 세우는 검색기."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    hybrid: BaseRetriever  # top_k = candidate_k 로 만든 Hybrid 검색기
    score_fn: Any  # (query, texts) -> list[float]
    top_k: int = 5
    use_context: bool = True

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        candidates = self.hybrid.invoke(query, config={"callbacks": run_manager.get_child()})
        if not candidates:
            return []
        scores = self.score_fn(query, [index_text(d, self.use_context) for d in candidates])
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        out = []
        for i in order[: self.top_k]:
            doc = candidates[i]
            meta = {**doc.metadata, "rerank_score": round(scores[i], 6), "hybrid_rank": i + 1}
            out.append(Document(page_content=doc.page_content, metadata=meta))
        return out


def build_rerank_retriever(
    *,
    domain: str | None = None,
    difficulty: str | None = None,
    top_k: int | None = None,
    candidate_k: int | None = None,
    levels: list[str] | None = None,
    config_path: Path = CONFIG_PATH,
) -> BaseRetriever:
    """Hybrid + Reranker 검색기 (공통 체인용).

    chain.py에서 `build_rerank_retriever(domain=domain, difficulty=difficulty)`로 한 번 만들고,
    질문마다 `retriever.invoke(question)`으로 list[Document]를 받는다.
    1단계 Hybrid 설정은 rerank.yaml의 hybrid_config(기본 hybrid.yaml)를 따른다.
    """
    cfg = load_config(config_path)
    search = cfg["search"]
    hybrid = build_hybrid_retriever(
        domain=domain,
        difficulty=difficulty,
        top_k=candidate_k or search["candidate_k"],
        levels=levels,
        config_path=REPO_ROOT / cfg["hybrid_config"],
    )
    return HybridRerankRetriever(
        hybrid=hybrid,
        score_fn=make_scorer(cfg),
        top_k=top_k or search["top_k"],
        use_context=cfg["reranker"].get("use_context", True),
    )


def _main() -> None:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(description="retrieval_a Hybrid + Reranker 검색 결과 확인")
    parser.add_argument("question")
    parser.add_argument("--domain", default=None)
    parser.add_argument("--difficulty", default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--candidates", type=int, default=None, help="리랭커로 넘길 Hybrid 후보 개수")
    args = parser.parse_args()

    retriever = build_rerank_retriever(
        domain=args.domain, difficulty=args.difficulty, top_k=args.k, candidate_k=args.candidates
    )
    for i, doc in enumerate(retriever.invoke(args.question), start=1):
        m = doc.metadata
        print(
            f"[S{i}] {m.get('book_id')}.pdf {m.get('page')}쪽 ({m.get('level')}, {m.get('chunk_id')}) "
            f"rerank={m['rerank_score']:.3f} hybrid#{m['hybrid_rank']} "
            f"dense#{m.get('dense_rank')} sparse#{m.get('sparse_rank')}"
        )
        print("    " + doc.page_content[:200].replace("\n", " "))


if __name__ == "__main__":
    _main()
