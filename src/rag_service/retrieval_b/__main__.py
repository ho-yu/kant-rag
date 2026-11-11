"""질문 하나로 retrieval_b 검색기를 바로 돌려 본다.

실행 (레포 맨 위에서)
    uv run python -m rag_service.retrieval_b "부산에서 LA로 보내는 일반 건화물의 서류 제출 마감은 언제인가요?"
    uv run python -m rag_service.retrieval_b "질문" --k 3 --levels page
    uv run python -m rag_service.retrieval_b "질문" --mode hybrid
    uv run python -m rag_service.retrieval_b "질문" --mode rerank   (처음엔 모델 내려받기)
    uv run python -m rag_service.retrieval_b "질문" --mode rerank --rerank-n 20 --cand 50

[2026-11-10 리팩토링] 이유: --k·--levels 기본값이 삭제된 상수(DEFAULT_K·DEFAULT_LEVELS)에 묶여 있었음 → 효과: 옵션을 생략하면 서버 설정(RETRIEVAL_B_*)을 그대로 따름.
[2026-11-11 리팩토링] 이유: 튜닝 값을 .env에서 읽지 않게 됨 → 효과: 평가 스크립트와 같은 튜닝 옵션(--cand, --rrf, --rerank-n, --rerank-len)을 씀.
"""

from __future__ import annotations

import argparse
import sys

from rag_service.retrieval_b.retrievers import (
    build_bm25_retriever,
    build_hybrid_retriever,
    build_naive_retriever,
    build_rerank_retriever,
)
from rag_service.retrieval_b.settings import (
    add_settings_args,
    retriever_kwargs,
    settings_from_args,
)

BUILDERS = {
    "naive": build_naive_retriever,
    "bm25": build_bm25_retriever,
    "hybrid": build_hybrid_retriever,
    "rerank": build_rerank_retriever,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="retrieval_b 검색 결과 확인")
    parser.add_argument("question")
    parser.add_argument("--mode", choices=sorted(BUILDERS), default="naive")
    parser.add_argument("--domain", default="shipping")
    parser.add_argument("--difficulty", default="basic")
    add_settings_args(parser)
    args = parser.parse_args(argv)
    settings = settings_from_args(args)

    retriever = BUILDERS[args.mode](
        domain=args.domain,
        difficulty=args.difficulty,
        **retriever_kwargs(settings, args.mode),
    )
    for rank, doc in enumerate(retriever.invoke(args.question), 1):
        m = doc.metadata
        text = doc.page_content[:80].replace("\n", " ")
        # rerank는 리랭크 전 Hybrid 순위를 같이 보여 준다
        prev = f" (hybrid #{m['prev_rank']})" if "prev_rank" in m else ""
        print(
            f"{rank}. {m['chunk_id']} {m['book_id']} p{m['page']} {m['score']:.4f}"
            f"{prev} {text}"
        )


if __name__ == "__main__":
    # Windows 터미널 기본(cp949)으로는 본문의 "–" 같은 글자를 못 찍는다
    sys.stdout.reconfigure(encoding="utf-8")
    main()
