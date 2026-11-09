from __future__ import annotations

import sys

from langchain_core.retrievers import BaseRetriever

from rag_service.retrieval_a.basic_bge_m3 import DIFFICULTY, DOMAIN, run
from rag_service.retrieval_a.hybrid import build_hybrid_retriever
from rag_service.retrieval_a.naive import build_naive_retriever
from rag_service.retrieval_a.rerank import build_rerank_retriever

PROFILE = "qwen3"


def build_index(recreate: bool = False) -> dict:
    from rag_service.retrieval_a.index_local import build_local_index

    return build_local_index(PROFILE, domain=DOMAIN, difficulty=DIFFICULTY, recreate=recreate)


def naive(top_k: int | None = None) -> BaseRetriever:
    return build_naive_retriever(domain=DOMAIN, difficulty=DIFFICULTY, profile=PROFILE, top_k=top_k)


def hybrid(top_k: int | None = None) -> BaseRetriever:
    return build_hybrid_retriever(domain=DOMAIN, difficulty=DIFFICULTY, profile=PROFILE, top_k=top_k)


def rerank(top_k: int | None = None, candidate_k: int | None = None) -> BaseRetriever:
    return build_rerank_retriever(
        domain=DOMAIN, difficulty=DIFFICULTY, profile=PROFILE, top_k=top_k, candidate_k=candidate_k
    )


BUILDERS = {"naive": naive, "hybrid": hybrid, "rerank": rerank}


if __name__ == "__main__":
    if sys.argv[1:] == ["--build-index"]:
        import json

        print(json.dumps(build_index(), ensure_ascii=False, indent=2))
    else:
        run(BUILDERS, "Qwen3")
