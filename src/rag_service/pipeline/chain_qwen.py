from __future__ import annotations

import argparse

from rag_service.pipeline.chain import create_chain
from rag_service.retrieval_a.naive import REPO_ROOT, model_profile

PROFILE = "qwen3"
MODES = ("naive", "hybrid", "hybrid_rerank")


def trace_metadata(domain: str, difficulty: str, mode: str | None = None) -> dict:
    m = model_profile(PROFILE)
    meta = {
        "domain": domain,
        "difficulty": difficulty,
        "profile": PROFILE,
        "embedding": m["embedding"],
        "sparse": m["sparse"],
        "reranker": m["reranker"],
    }
    if mode:
        meta["retrieval_mode"] = mode
    return meta


def create_qwen_chain(domain: str = "shipping", difficulty: str = "basic"):
    chain = create_chain(domain=domain, difficulty=difficulty, profile=PROFILE)
    return chain.with_config(run_name="rag-answer-qwen3", metadata=trace_metadata(domain, difficulty))


def ask(question: str, *, domain: str = "shipping", difficulty: str = "basic", mode: str = "hybrid_rerank", chain=None):
    from langfuse import get_client

    chain = chain or create_qwen_chain(domain, difficulty)
    langfuse = get_client()
    with langfuse.start_as_current_observation(
        as_type="span",
        name="user-question-qwen3",
        input={"question": question},
        metadata=trace_metadata(domain, difficulty, mode),
    ) as span:
        result = chain.invoke({"question": question, "retrieval_mode": mode})
        span.update(output=result.model_dump())
    return result


def _main() -> None:
    from dotenv import load_dotenv
    from langfuse import get_client

    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(description="Qwen3 임베딩으로 RAG 답변 생성 (Langfuse 기록)")
    parser.add_argument("question")
    parser.add_argument("--domain", default="shipping")
    parser.add_argument("--difficulty", default="basic")
    parser.add_argument("--mode", default="hybrid_rerank", choices=MODES)
    args = parser.parse_args()

    try:
        result = ask(args.question, domain=args.domain, difficulty=args.difficulty, mode=args.mode)
        print(result.answer)
        print(f"\nstatus={result.status} sources={result.source_ids}")
    finally:
        get_client().shutdown()


if __name__ == "__main__":
    _main()
