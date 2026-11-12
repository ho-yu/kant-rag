"""보안 비교를 여러 난이도로 한 번에 실행하고 결과 표를 하나로 모은다.

run_security_compare.py의 main()을 난이도마다 그대로 부른다 (검색·답변·채점 규칙은 그 파일과 같음).
기본값은 세 난이도 · hybrid · 쪽 확장(page_context) 켬.
검색기는 run_security_compare.py 안에서 pipeline/prepare_retriever.py의 RetrieverPool로 서비스와 같게 준비한다.

실행 (레포 맨 위에서, Qdrant와 .env 필요)
    uv run python evaluation/security/run_security_all.py                                  # basic·intermediate·advanced · hybrid · 쪽 확장
    uv run python evaluation/security/run_security_all.py --difficulties basic intermediate   # 골라서
    uv run python evaluation/security/run_security_all.py --mode hybrid_rerank
    uv run python evaluation/security/run_security_all.py --no-page-context                # 쪽 확장 끔
    uv run python evaluation/security/run_security_all.py --fresh                          # 저장 문맥 무시하고 다시 검색
    uv run python evaluation/security/run_security_all.py --no-judge --no-trace

결과
    results/security/{난이도}_{방식}[_page]/   난이도별 결과 (run_security_compare.py와 같음)
    results/security/summary_all_{방식}[_page].md   세 난이도 summary.md를 합친 표 (README 8-2용)

한 난이도가 실패해도 나머지는 계속 실행하고, 끝에 성공·실패를 한 번에 보여 준다.
advanced는 컬렉션 적재가 오래 걸리므로 기본 순서에서 마지막에 둔다.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LEVELS = ("basic", "intermediate", "advanced")
MODES = ("naive", "hybrid", "hybrid_rerank")


def load_compare():
    """같은 폴더의 run_security_compare.py를 불러온다 (evaluation/은 패키지가 아니라서).

    sys.modules에 먼저 등록해야 한다. 그 파일은 `from __future__ import annotations`를 써서
    pydantic이 채점 형식(SecurityJudge)의 문자열 타입(Literal)을 모듈 이름으로 찾기 때문이다.
    등록하지 않으면 'SecurityJudge is not fully defined' 오류가 난다.
    """
    spec = importlib.util.spec_from_file_location("run_security_compare", HERE / "run_security_compare.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def folder_name(difficulty: str, mode: str, page_context: bool) -> str:
    """run_security_compare.py의 결과 폴더 이름 규칙과 같다."""
    return f"{difficulty}_{mode}{'_page' if page_context else ''}"


def merge_summaries(difficulties: list[str], mode: str, page_context: bool) -> Path:
    """난이도별 summary.md를 하나로 합친다. 없는 난이도는 '결과 없음'으로 적는다."""
    out = REPO / "results" / "security" / f"summary_all_{mode}{'_page' if page_context else ''}.md"
    parts = [f"# 보안 비교 · 전체 난이도 · {mode}{' · 쪽 확장' if page_context else ''}\n"]
    for d in difficulties:
        path = REPO / "results" / "security" / folder_name(d, mode, page_context) / "summary.md"
        parts.append(f"## {d}\n")
        parts.append(path.read_text(encoding="utf-8") if path.is_file() else "(결과 없음)\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="보안 비교를 여러 난이도로 한 번에 실행")
    p.add_argument("--difficulties", nargs="+", choices=LEVELS, default=list(LEVELS))
    p.add_argument("--mode", choices=MODES, default="hybrid")
    p.add_argument("--no-page-context", action="store_true", help="쪽 확장 끔 (기본은 켬)")
    p.add_argument("--fresh", action="store_true", help="저장 문맥을 무시하고 다시 검색")
    p.add_argument("--no-judge", action="store_true", help="채점 LLM 없이 노출만 기록")
    p.add_argument("--judge-model", default=None)
    p.add_argument("--no-trace", action="store_true", help="Langfuse 기록 끔")
    args = p.parse_args(argv)
    page_context = not args.no_page_context

    compare = load_compare()
    results: list[tuple[str, str, float]] = []
    for d in args.difficulties:
        sub = ["--difficulty", d, "--mode", args.mode]
        if page_context:
            sub.append("--page-context")
        if args.fresh:
            sub.append("--fresh")
        if args.no_judge:
            sub.append("--no-judge")
        if args.judge_model:
            sub += ["--judge-model", args.judge_model]
        if args.no_trace:
            sub.append("--no-trace")
        print(f"\n{'=' * 70}\n[{d}] 시작: run_security_compare.py {' '.join(sub)}\n{'=' * 70}", flush=True)
        start = time.perf_counter()
        try:
            compare.main(sub)
            results.append((d, "성공", time.perf_counter() - start))
        except Exception:  # noqa: BLE001 — 한 난이도 실패로 나머지를 멈추지 않는다
            traceback.print_exc()
            results.append((d, "실패", time.perf_counter() - start))

    merged = merge_summaries(args.difficulties, args.mode, page_context)
    print(f"\n{'=' * 70}\n전체 결과")
    for d, state, sec in results:
        folder = folder_name(d, args.mode, page_context)
        print(f"- {d:<13} {state} · {sec / 60:.1f}분 · results/security/{folder}/report.html")
    print(f"합친 표: {merged.relative_to(REPO)}")


if __name__ == "__main__":
    main()
