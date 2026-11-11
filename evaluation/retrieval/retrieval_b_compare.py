"""검색 방식별 Hit@k·Recall@k 비교 실행기.

같은 골든셋, 같은 k, 같은 레벨로 여러 검색기를 채점하고 결과를 JSON으로 저장한다.
retrieval_b(내 구현)와 retrieval_a(팀원 구현)를 같은 기준으로 채점할 수 있다.

실행 (레포 맨 위에서)
    uv run python evaluation/retrieval/retrieval_b_compare.py
    uv run python evaluation/retrieval/retrieval_b_compare.py --modes naive bm25 hybrid a_hybrid
    uv run python evaluation/retrieval/retrieval_b_compare.py --split additional --levels page
    uv run python evaluation/retrieval/retrieval_b_compare.py --modes rerank --rerank-n 20   # 튜닝 실험

결과: results/comparison/retrieval_b/{난이도}_{split}_{levels}{설정}{k}.json  (--exp 이름이면 results/experiments/<이름>/comparison/retrieval_b/)
      같은 이름의 .html도 만든다 (브라우저로 여는 결과 페이지, --open이면 바로 열림).
      {설정}은 명령줄로 준 튜닝 값(--cand, --rrf, --rerank-n, --rerank-len)이 기본값과 다를 때만 붙는다 (예: _n20, _cand50_rrf20).
      {k}는 --ks가 기본 목록과 다를 때 붙는다: k 하나면 _k5, 여러 개면 _ks3-7-9. --out을 주면 그 이름이 우선.
      그래서 설정을 바꿔 돌려도 기본 설정 결과를 덮어쓰지 않는다.
      실제로 쓴 튜닝 값은 JSON의 settings.retrieval_b에 남는다.
      로컬에 실험 비교판(evaluation/reporting/experiment_board.py)이 있으면 끝난 뒤 다시 만든다. --no-board면 건너뛴다.

[2026-11-10 추가] 이유: 환경변수로 바꾼 후보 수·RRF 상수·리랭커 후보 수가 결과에 남지 않고,
같은 이름으로 덮어써져 실험끼리 비교할 수 없었음 → 효과: 설정이 결과 파일 이름과 JSON에 남는다.
[2026-11-11 리팩토링] 이유: 튜닝 값을 .env(RETRIEVAL_B_*)에서 읽지 않게 됨 → 효과: 명령줄 옵션으로 받아 b 방식 검색기에 직접 넘긴다.
"""

from __future__ import annotations

import argparse
import importlib.util
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
from rag_service.retrieval_b.settings import (
    Settings,
    add_settings_args,
    retriever_kwargs,
    settings_from_args,
)
from rag_service.storage.corpus import repo_root

WARMUP_QUESTION = "부산에서 LA로 컨테이너를 예약할 때 필요한 정보는?"
OUT_DIR = Path("results") / "comparison" / "retrieval_b"

# (domain, difficulty, k, 설정) → 검색기. k는 --ks 중 최댓값이고, 설정에서는 levels·튜닝 값을 쓴다.
Builder = Callable[[str, str, int, Settings], Any]

# retrieval_b 튜닝 값 → 파일 이름에 붙일 짧은 이름. k(--ks)·levels는 이름에 따로 들어간다.
# 배치 크기는 점수를 바꾸지 않으므로 기록만 하고 이름에는 안 붙인다.
TUNING_TAGS: dict[str, str | None] = {
    "candidate_k": "cand",
    "rrf_k": "rrf",
    "rerank_n": "n",
    "rerank_max_length": "len",
    "rerank_batch_size": None,
}


def ks_tag(ks: Sequence[int]) -> str:
    """--out이 없을 때 파일 이름 꼬리. 기본 k 목록이면 없음, k 하나면 _k5, 여러 개면 _ks3-7-9."""
    ks = sorted(set(ks))
    if ks == sorted(DEFAULT_KS):
        return ""
    if len(ks) == 1:
        return f"_k{ks[0]}"
    return "_ks" + "-".join(str(k) for k in ks)


def tuning_values(settings: Settings) -> dict[str, int]:
    """retrieval_b 검색기가 이번 실행에서 실제로 쓰는 튜닝 값 (명령줄 옵션 반영)."""
    return {name: getattr(settings, name) for name in TUNING_TAGS}


def tuning_tag(values: dict[str, Any]) -> str:
    """기본값과 다른 튜닝 값만 이어 붙인 파일 이름 꼬리. 모두 기본값이면 빈 문자열."""
    default = Settings()
    return "".join(
        f"_{short}{values[name]}"
        for name, short in TUNING_TAGS.items()
        if short and name in values and values[name] != getattr(default, name)
    )


def _paths():
    """evaluation/experiment_paths.py를 불러온다 (실험 이름 규칙·결과 폴더가 거기 한 곳에 있다)."""
    path = Path(__file__).resolve().parents[1] / "experiment_paths.py"
    spec = importlib.util.spec_from_file_location("experiment_paths", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def results_base(exp: str | None = None) -> Path:
    """결과 최상위 폴더. --exp가 없으면 results/, 있으면 results/experiments/<이름>/."""
    return _paths().results_dir(repo_root(), exp)


def check_exp_arg(parser: argparse.ArgumentParser, exp: str | None) -> str | None:
    """--exp 값 검사. 틀리면 argparse 오류로 멈추고, 맞으면 "[실험] ..." 한 줄을 찍는다."""
    paths = _paths()
    try:
        exp = paths.check_exp(exp)
    except ValueError as e:
        parser.error(str(e))
    print(paths.exp_banner(exp), flush=True)
    return exp


def refresh_board(exp: str | None = None) -> Path | None:
    """실험 비교판(로컬 도구)이 있으면 그 실험의 비교판을 다시 만든다. 없으면 조용히 건너뛴다."""
    return _paths().refresh_board(repo_root(), exp)


def _b_kwargs(s: Settings, mode: str, k: int) -> dict[str, Any]:
    """b 검색기 인자: 설정의 levels·튜닝 값에 k만 --ks 최댓값으로 바꾼다."""
    return {**retriever_kwargs(s, mode), "k": k}


def _b_naive(domain: str, difficulty: str, k: int, s: Settings):
    from rag_service.retrieval_b import build_naive_retriever

    return build_naive_retriever(
        domain=domain, difficulty=difficulty, **_b_kwargs(s, "naive", k)
    )


def _b_bm25(domain: str, difficulty: str, k: int, s: Settings):
    from rag_service.retrieval_b import build_bm25_retriever

    return build_bm25_retriever(
        domain=domain, difficulty=difficulty, **_b_kwargs(s, "bm25", k)
    )


def _b_hybrid(domain: str, difficulty: str, k: int, s: Settings):
    from rag_service.retrieval_b import build_hybrid_retriever

    return build_hybrid_retriever(
        domain=domain, difficulty=difficulty, **_b_kwargs(s, "hybrid", k)
    )


def _b_rerank(domain: str, difficulty: str, k: int, s: Settings):
    from rag_service.retrieval_b import build_rerank_retriever

    return build_rerank_retriever(
        domain=domain, difficulty=difficulty, **_b_kwargs(s, "rerank", k)
    )


# a 방식은 levels만 이 설정에서 쓰고, 나머지 튜닝 값은 config/retrieval_a 설정을 쓴다.
def _a_naive(domain: str, difficulty: str, k: int, s: Settings):
    from rag_service.retrieval_a import build_naive_retriever

    return build_naive_retriever(
        domain=domain, difficulty=difficulty, top_k=k, levels=list(s.levels)
    )


def _a_hybrid(domain: str, difficulty: str, k: int, s: Settings):
    from rag_service.retrieval_a import build_hybrid_retriever

    return build_hybrid_retriever(
        domain=domain, difficulty=difficulty, top_k=k, levels=list(s.levels)
    )


def _a_rerank(domain: str, difficulty: str, k: int, s: Settings):
    from rag_service.retrieval_a import build_rerank_retriever

    return build_rerank_retriever(
        domain=domain, difficulty=difficulty, top_k=k, levels=list(s.levels)
    )


# 이름 → 검색기 만드는 함수
MODES: dict[str, Builder] = {
    "naive": _b_naive,
    "bm25": _b_bm25,
    "hybrid": _b_hybrid,
    "rerank": _b_rerank,
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
    parser.add_argument(
        "--ks",
        nargs="+",
        type=int,
        default=list(DEFAULT_KS),
        help="채점할 k 목록 (검색은 가장 큰 k로 한 번)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="앞에서 N개 질문만 (빠른 확인용)"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="저장 파일 경로 (기본: results/...)"
    )
    parser.add_argument(
        "--open", action="store_true", help="끝나면 결과 페이지를 브라우저로 바로 연다"
    )
    parser.add_argument(
        "--no-board",
        action="store_true",
        help="끝난 뒤 실험 비교판을 다시 만들지 않는다 (비교판이 있을 때만 의미 있음)",
    )
    parser.add_argument(
        "--exp",
        default=None,
        help="실험 이름. 주면 results/experiments/<이름>/에 따로 저장",
    )
    add_settings_args(parser, skip=("k",))  # k는 --ks로 정한다
    args = parser.parse_args(argv)
    exp = check_exp_arg(parser, args.exp)
    settings = settings_from_args(args)

    load_dotenv(repo_root() / ".env")
    tuning = tuning_values(settings)
    tag = tuning_tag(tuning)
    print(
        "[설정] "
        + " ".join(f"{name}={value}" for name, value in tuning.items())
        + (f" (기본값과 다름: {tag[1:]})" if tag else " (모두 기본값)"),
        flush=True,
    )
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
            retriever = MODES[mode](args.domain, args.difficulty, ks[-1], settings)
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

    levels_tag = "-".join(settings.levels)
    out = args.out or (
        results_base(exp)
        / OUT_DIR.relative_to("results")
        / f"{args.difficulty}_{args.split}_{levels_tag}{tag}{ks_tag(ks)}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "settings": {
            "domain": args.domain,
            "difficulty": args.difficulty,
            "split": args.split,
            "levels": list(settings.levels),
            "ks": ks,
            "limit": args.limit,
            "modes": list(args.modes),
            "exp": exp,
            # b 방식(naive·bm25·hybrid·rerank)이 쓴 값. a_ 방식은 config/retrieval_a 설정을 쓴다.
            "retrieval_b": tuning,
        },
        "results": results,
    }
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    page = write_html(report, out.with_suffix(".html"), items)

    print()
    print_table(results, ks)
    print(f"\n저장: {out}")
    print(f"결과 페이지 (Ctrl+클릭): {page_link(page, open_browser=args.open)}")
    if not args.no_board:
        refresh_board(exp)
    return out


if __name__ == "__main__":
    # Windows 터미널 기본(cp949)으로는 본문의 "–" 같은 글자를 못 찍는다
    sys.stdout.reconfigure(encoding="utf-8")
    main()
