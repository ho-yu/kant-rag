"""검색 방식별 Hit@k·Recall@k 비교 실행기.

같은 골든셋, 같은 k, 같은 레벨로 여러 검색기를 채점하고 결과를 JSON으로 저장한다.
retrieval_b(내 구현)와 retrieval_a(최완위 님 구현)를 같은 기준으로 채점할 수 있다.

실행 (레포 맨 위에서)
    uv run python evaluation/retrieval/retrieval_b_compare.py
    uv run python evaluation/retrieval/retrieval_b_compare.py --modes naive bm25 a_naive
    uv run python evaluation/retrieval/retrieval_b_compare.py --split additional --levels page

결과: results/comparison/retrieval_b/{난이도}_{split}_{levels}.json
      같은 이름의 .html도 만든다 (브라우저로 여는 결과 페이지, --open이면 바로 열림).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from rag_service.retrieval_b.metrics import (
    DEFAULT_KS,
    SPLITS,
    evaluate,
    golden_path,
    load_golden,
)
from rag_service.retrieval_b.report import page_link, write_html
from rag_service.storage.corpus import repo_root

WARMUP_QUESTION = "부산에서 LA로 컨테이너를 예약할 때 필요한 정보는?"
OUT_DIR = Path("results") / "comparison" / "retrieval_b"

Builder = Callable[[str, str, int, list[str]], Any]


def _b_naive(domain: str, difficulty: str, k: int, levels: list[str]):
    from rag_service.retrieval_b import build_naive_retriever

    return build_naive_retriever(
        domain=domain, difficulty=difficulty, k=k, levels=levels
    )


def _b_bm25(domain: str, difficulty: str, k: int, levels: list[str]):
    from rag_service.retrieval_b import build_bm25_retriever

    return build_bm25_retriever(
        domain=domain, difficulty=difficulty, k=k, levels=levels
    )


def _a_naive(domain: str, difficulty: str, k: int, levels: list[str]):
    from rag_service.retrieval_a import build_naive_retriever

    return build_naive_retriever(
        domain=domain, difficulty=difficulty, top_k=k, levels=levels
    )


def _a_hybrid(domain: str, difficulty: str, k: int, levels: list[str]):
    from rag_service.retrieval_a import build_hybrid_retriever

    return build_hybrid_retriever(
        domain=domain, difficulty=difficulty, top_k=k, levels=levels
    )


def _a_rerank(domain: str, difficulty: str, k: int, levels: list[str]):
    from rag_service.retrieval_a import build_rerank_retriever

    return build_rerank_retriever(
        domain=domain, difficulty=difficulty, top_k=k, levels=levels
    )


# 이름 → 검색기 만드는 함수. Hybrid·Rerank(retrieval_b)를 만들면 여기에 한 줄씩 추가한다.
MODES: dict[str, Builder] = {
    "naive": _b_naive,
    "bm25": _b_bm25,
    "a_naive": _a_naive,
    "a_hybrid": _a_hybrid,
    "a_rerank": _a_rerank,
}


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def print_table(results: dict[str, Any], ks: Sequence[int]) -> None:
    head = f"{'mode':<10} {'k':>3} {'n':>3} {'docHit':>7} {'pageHit':>8} {'pageRec':>8} {'pageGrp':>8} {'pageMRR':>8}"
    print(head)
    print("-" * len(head))
    for mode, res in results.items():
        if "error" in res:
            print(f"{mode:<10} 실패: {res['error']}")
            continue
        for k in ks:
            s = res["summary"][str(k)]
            cells = [
                s["doc_hit"],
                s["page_hit"],
                s["page_recall"],
                s["page_group_recall"],
                s.get("page_mrr"),
            ]
            text = " ".join(
                f"{'-' if v is None else f'{v:.3f}':>{w}}"
                for v, w in zip(cells, (7, 8, 8, 8, 8), strict=True)
            )
            print(f"{mode:<10} {k:>3} {s['n']:>3} {text}")
        lat = res["latency_ms"]
        print(f"{'':<10} 검색 시간 평균 {lat['mean']} ms, 오류 {res['n_errors']}건")


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="검색 방식별 Hit@k·Recall@k 비교")
    parser.add_argument("--domain", default="shipping")
    parser.add_argument("--difficulty", default="basic")
    parser.add_argument("--split", choices=SPLITS, default="development")
    parser.add_argument("--modes", nargs="+", choices=sorted(MODES), default=["naive"])
    parser.add_argument("--levels", nargs="+", default=["para", "table"])
    parser.add_argument("--ks", nargs="+", type=int, default=list(DEFAULT_KS))
    parser.add_argument(
        "--limit", type=int, default=None, help="앞에서 N개 질문만 (빠른 확인용)"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="저장 파일 경로 (기본: results/...)"
    )
    parser.add_argument(
        "--open", action="store_true", help="끝나면 결과 페이지를 브라우저로 바로 연다"
    )
    args = parser.parse_args(argv)

    load_dotenv(repo_root() / ".env")
    ks = sorted(set(args.ks))
    items = load_golden(
        golden_path(args.split, domain=args.domain, difficulty=args.difficulty)
    )
    if args.limit:
        items = items[: args.limit]

    results: dict[str, Any] = {}
    for mode in args.modes:
        print(f"[{mode}] 검색기 준비 중 ...", flush=True)
        try:
            started = time.perf_counter()
            retriever = MODES[mode](
                args.domain, args.difficulty, ks[-1], list(args.levels)
            )
            # 모델 로딩·첫 호출 지연을 검색 시간에서 뺀다
            retriever.invoke(WARMUP_QUESTION)
            build_s = round(time.perf_counter() - started, 1)
        except Exception as e:  # noqa: BLE001 — 한 방식이 실패해도 나머지는 비교한다
            results[mode] = {"error": f"{type(e).__name__}: {e}"}
            print(f"[{mode}] 준비 실패: {results[mode]['error']}", flush=True)
            continue
        print(f"[{mode}] 질문 {len(items)}개 채점 중 ...", flush=True)
        res = evaluate(retriever, items, ks)
        res["build_seconds"] = build_s
        results[mode] = res

    levels_tag = "-".join(args.levels)
    out = (
        args.out
        or repo_root() / OUT_DIR / f"{args.difficulty}_{args.split}_{levels_tag}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "settings": {
            "domain": args.domain,
            "difficulty": args.difficulty,
            "split": args.split,
            "levels": list(args.levels),
            "ks": ks,
            "limit": args.limit,
            "modes": list(args.modes),
        },
        "results": results,
    }
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    page = write_html(report, out.with_suffix(".html"), items)

    print()
    print_table(results, ks)
    print(f"\n저장: {out}")
    print(f"결과 페이지 (Ctrl+클릭): {page_link(page, open_browser=args.open)}")
    return out


if __name__ == "__main__":
    # Windows 터미널 기본(cp949)으로는 본문의 "–" 같은 글자를 못 찍는다
    sys.stdout.reconfigure(encoding="utf-8")
    main()
