"""질문 하나로 retrieval_b 검색기를 바로 돌려 본다.

실행 (레포 맨 위에서)
    uv run python -m rag_service.retrieval_b "부산에서 LA로 보내는 일반 건화물의 서류 제출 마감은 언제인가요?"
    uv run python -m rag_service.retrieval_b "질문" --k 3 --levels page
"""

from __future__ import annotations

import argparse
import sys

from rag_service.retrieval_b.retrievers import (
    DEFAULT_K,
    DEFAULT_LEVELS,
    build_naive_retriever,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="retrieval_b 검색 결과 확인")
    parser.add_argument("question")
    parser.add_argument("--mode", choices=["naive"], default="naive")
    parser.add_argument("--domain", default="shipping")
    parser.add_argument("--difficulty", default="basic")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--levels", nargs="+", default=DEFAULT_LEVELS)
    args = parser.parse_args(argv)

    retriever = build_naive_retriever(
        domain=args.domain, difficulty=args.difficulty, k=args.k, levels=args.levels
    )
    for rank, doc in enumerate(retriever.invoke(args.question), 1):
        m = doc.metadata
        text = doc.page_content[:80].replace("\n", " ")
        print(
            f"{rank}. {m['chunk_id']} {m['book_id']} p{m['page']} {m['score']:.4f} {text}"
        )


if __name__ == "__main__":
    # Windows 터미널 기본(cp949)으로는 본문의 "–" 같은 글자를 못 찍는다
    sys.stdout.reconfigure(encoding="utf-8")
    main()
