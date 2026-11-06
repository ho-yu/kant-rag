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
    import torch
    from sentence_transformers import CrossEncoder

    device = _resolve_device(device)
    model_kwargs = {"dtype": torch.float16} if fp16 and device.startswith("cuda") else {}
    return CrossEncoder(
        model_name,
        device=device,
        max_length=max_length,
        model_kwargs=model_kwargs,
        activation_fn=torch.nn.Sigmoid(),
    )


def make_scorer(cfg: dict[str, Any]):
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
    model_config = ConfigDict(arbitrary_types_allowed=True)

    hybrid: BaseRetriever
    score_fn: Any
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
