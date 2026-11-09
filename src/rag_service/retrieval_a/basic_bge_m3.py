from __future__ import annotations

import argparse

from langchain_core.retrievers import BaseRetriever

from rag_service.retrieval_a.hybrid import build_hybrid_retriever
from rag_service.retrieval_a.naive import REPO_ROOT, build_naive_retriever
from rag_service.retrieval_a.rerank import build_rerank_retriever

DOMAIN = "shipping"
DIFFICULTY = "basic"
PROFILE = "basic"
MODES = ("naive", "hybrid", "rerank")


def naive(top_k: int | None = None) -> BaseRetriever:
    return build_naive_retriever(domain=DOMAIN, difficulty=DIFFICULTY, profile=PROFILE, top_k=top_k)


def hybrid(top_k: int | None = None) -> BaseRetriever:
    return build_hybrid_retriever(domain=DOMAIN, difficulty=DIFFICULTY, profile=PROFILE, top_k=top_k)


def rerank(top_k: int | None = None, candidate_k: int | None = None) -> BaseRetriever:
    return build_rerank_retriever(
        domain=DOMAIN, difficulty=DIFFICULTY, profile=PROFILE, top_k=top_k, candidate_k=candidate_k
    )


BUILDERS = {"naive": naive, "hybrid": hybrid, "rerank": rerank}


def describe(doc) -> str:
    m = doc.metadata
    scores = []
    if "rerank_score" in m:
        scores.append(f"rerank={m['rerank_score']:.3f} hybrid#{m['hybrid_rank']}")
    if "hybrid_score" in m:
        scores.append(f"dense#{m['dense_rank']} sparse#{m['sparse_rank']}")
    return f"{m.get('book_id')}.pdf {m.get('page')}쪽 ({m.get('chunk_id')}) " + " ".join(scores)


def run(builders: dict, title: str) -> None:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(description=f"basic 자료 {title} 검색 (naive / hybrid / rerank)")
    parser.add_argument("question")
    parser.add_argument("--mode", default="all", choices=["all", *MODES])
    parser.add_argument("--k", type=int, default=None)
    args = parser.parse_args()

    for mode in MODES if args.mode == "all" else [args.mode]:
        print(f"\n=== {title} · {mode} ===")
        for i, doc in enumerate(builders[mode](top_k=args.k).invoke(args.question), start=1):
            print(f"[S{i}] {describe(doc)}")
            print("    " + doc.page_content[:150].replace("\n", " "))


if __name__ == "__main__":
    run(BUILDERS, "bge-m3")
