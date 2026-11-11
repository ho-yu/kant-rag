"""답변 평가 2단계: RAGAS Faithfulness. 답변의 주장을 실제 최종 문맥이 뒷받침하는지 본다.

발제 STEP 6 "RAGAS와 LLM-as-a-Judge 실행 방법" 2번: 질문 → user_input, 생성 답변 → response,
실제 최종 문맥 → retrieved_contexts. 1단계 answers_{방식}.jsonl을 읽어 채점만 한다 (답변을 새로 만들지 않는다).

결과 상태 (발제 STEP 6 "해당 없음과 오류")
- scored       0~1 점수. 0점도 평균에 넣는다
- na           해당 없음. 답변에 사실 주장이 없다 (RAGAS가 주장을 0개로 나누면 NaN을 돌려준다)
- unavailable  평가 불가. 서비스 오류로 답변이 없음 / 문맥이 비어 RAGAS가 받지 않음 / 채점 호출 실패
na와 unavailable은 평균에서 빼고 건수를 따로 센다. 빈 값을 0이나 1로 바꾸지 않는다.

주의: 되묻기 답변도 RAGAS가 문장을 "주장"으로 나누면 점수가 나온다. 대표 사례는 사람이 확인한다.

API: ragas 0.4.3의 ragas.metrics.collections.Faithfulness (발제가 링크한 공식 예제와 같은 방식).
gpt-6-luna는 RAGAS가 추론 모델로 보고 temperature 1.0, max_completion_tokens로 바꿔 보낸다.
구조화 출력이 잘리지 않게 max_tokens를 넉넉히 준다.

실행 (레포 맨 위에서, 1단계를 먼저 실행)
    uv run python evaluation/answer/faithfulness.py --modes naive --limit 1
    uv run python evaluation/answer/faithfulness.py
    uv run python evaluation/answer/faithfulness.py --summary-only        # 채점 없이 요약만 다시

    uv run python evaluation/answer/faithfulness.py --run basic_development_k3   # 폴더를 직접 고르기

결과 ({폴더}는 run_answers.py와 같은 규칙: {난이도}_{split}{바꾼 설정}{_tag})
    results/answer/retrieval_b/{폴더}/faithfulness_{방식}.jsonl   한 줄 = 문항 하나의 점수·상태
    results/answer/retrieval_b/{폴더}/grading_usage.jsonl       채점 호출 토큰 (서비스 지표와 따로)
    results/answer/retrieval_b/{폴더}/faithfulness_summary.json 방식별 평균과 건수
    results/answer/retrieval_b/{폴더}/faithfulness_per_question.csv
폴더는 run_answers.py를 돌릴 때와 같은 RETRIEVAL_B_* 환경변수·--tag로 실행하면 알아서 찾는다.
환경변수를 이미 지웠다면 --run에 폴더 이름을 그대로 준다. 못 찾으면 있는 폴더 목록을 보여 주고 멈춘다.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import importlib.util
import json
import math
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from dotenv import load_dotenv

from rag_service.storage.corpus import repo_root

MODEL = "gpt-6-luna"
TIMEOUT_S = 120
MAX_RETRIES = 2
MAX_TOKENS = 4096
WORKERS = 4
MODES = ("naive", "hybrid", "hybrid_rerank")
SPLITS = ("development", "additional", "expansion_validation")
OUT_DIR = Path("results") / "answer" / "retrieval_b"

Record = dict[str, Any]


# ---------- 파일 ----------


def run_name(difficulty: str, split: str, *, tag: str = "") -> str:
    """run_answers.py와 같은 규칙의 폴더 이름. 같은 RETRIEVAL_B_* 환경변수로 실행해야 같은 폴더가 된다."""
    from rag_service.retrieval_b.settings import get_settings, settings_tag

    name = f"{difficulty}_{split}{settings_tag(get_settings())}"
    return f"{name}_{tag}" if tag else name


def out_dir_for(
    difficulty: str, split: str, *, run: str = "", tag: str = "", exp: str | None = None
) -> Path:
    """--run으로 폴더 이름을 직접 주면 그 폴더, 아니면 지금 설정으로 이름을 만든다 (--exp면 그 실험 폴더 안에서)."""
    base = results_base(exp) / OUT_DIR.relative_to("results")
    return base / (run or run_name(difficulty, split, tag=tag))


def available_runs(difficulty: str, split: str, exp: str | None = None) -> list[str]:
    """같은 난이도·split의 답변 폴더 이름들 (폴더를 못 찾았을 때 안내용)."""
    base = results_base(exp) / OUT_DIR.relative_to("results")
    if not base.is_dir():
        return []
    prefix = f"{difficulty}_{split}"
    return sorted(
        p.name
        for p in base.iterdir()
        if p.is_dir() and p.name.startswith(prefix) and any(p.glob("answers_*.jsonl"))
    )


def read_jsonl(path: Path) -> list[Record]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


# ---------- 채점 호출 토큰 세기 ----------


class UsageCounter:
    """채점 호출의 토큰을 모은다. 서비스 지표(1단계 tokens)와 섞지 않으려고 따로 센다."""

    def __init__(self) -> None:
        self.calls = 0
        self.input = 0
        self.output = 0
        self.unknown = 0  # 응답에 usage가 없던 호출 수

    def add(self, usage: Any) -> None:
        self.calls += 1
        if usage is None:
            self.unknown += 1
            return
        self.input += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.output += int(getattr(usage, "completion_tokens", 0) or 0)

    def as_dict(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "input": self.input,
            "output": self.output,
            "unknown": self.unknown,
        }


def count_usage(client: Any, counter: UsageCounter) -> Any:
    """client.chat.completions.create를 감싸 응답의 usage를 센다.

    RAGAS(instructor)가 이 create를 불러 쓰므로, llm_factory에 넘기기 전에 감싼다.
    """
    completions = client.chat.completions
    original = completions.create

    async def create(*args: Any, **kwargs: Any) -> Any:
        response = await original(*args, **kwargs)
        counter.add(getattr(response, "usage", None))
        return response

    completions.create = create
    return client


def make_scorer(
    model: str = MODEL, counter: UsageCounter | None = None, client: Any = None
) -> Any:
    """평가 LLM을 붙인 Faithfulness 객체. 객체를 만들 때는 API를 부르지 않는다.

    client: 테스트에서 가짜 응답을 주는 AsyncOpenAI를 넣는다. 없으면 .env의 OPENAI_API_KEY로 만든다.
    """
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import Faithfulness

    if client is None:
        client = AsyncOpenAI(timeout=TIMEOUT_S, max_retries=MAX_RETRIES)
    if counter is not None:
        count_usage(client, counter)
    llm = llm_factory(model, client=client, max_tokens=MAX_TOKENS)
    return Faithfulness(llm=llm)


# ---------- 문항 하나 채점 ----------


def _result(
    rec: Record,
    value: float | None,
    state: str,
    cause: str | None,
    reason: str,
    start: float,
) -> Record:
    return {
        "id": rec.get("id"),
        "mode": rec.get("mode"),
        "status": rec.get("status"),
        "faithfulness": value,
        "state": state,
        "cause": cause,
        "reason": reason,
        "eval_ms": round((time.perf_counter() - start) * 1000, 1),
    }


def precheck(rec: Record) -> tuple[str, str, str] | None:
    """채점 호출 없이 상태가 정해지는 경우. (state, cause, reason)을 돌려주고, 채점해야 하면 None."""
    if rec.get("status") == "error":
        why = f"서비스 오류로 답변이 없음 ({rec.get('error') or '이유 없음'})"
        return "unavailable", "service_error", why
    if not (rec.get("answer") or "").strip():
        return "unavailable", "empty_answer", "답변이 비어 있음"
    if not [c for c in rec.get("contexts") or [] if (c.get("text") or "").strip()]:
        why = "실제 최종 문맥이 비어 있음 (RAGAS는 빈 문맥을 받지 않음)"
        return "unavailable", "empty_context", why
    return None


async def score_one(rec: Record, scorer: Any) -> Record:
    start = time.perf_counter()
    pre = precheck(rec)
    if pre is not None:
        return _result(rec, None, *pre, start)
    contexts = [c["text"] for c in rec["contexts"] if (c.get("text") or "").strip()]
    try:
        result = await scorer.ascore(
            user_input=rec["question"],
            response=rec["answer"],
            retrieved_contexts=contexts,
        )
    except Exception as e:  # noqa: BLE001 — 채점 실패는 평가 불가로 남긴다
        why = f"채점 실패: {type(e).__name__}: {e}"
        return _result(rec, None, "unavailable", "grading_failed", why, start)
    value = getattr(result, "value", result)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        why = "답변에 사실 주장이 없음 (RAGAS가 주장을 0개로 나눔)"
        return _result(rec, None, "na", "no_claims", why, start)
    return _result(rec, float(value), "scored", None, "", start)


# ---------- 한 방식 전체 ----------


async def score_records(
    records: Sequence[Record],
    scorer: Any,
    path: Path,
    *,
    workers: int = WORKERS,
    log: Callable[[str], None] = print,
) -> list[Record]:
    """아직 채점 안 한 기록만 workers개씩 동시에 채점하고, 끝나는 대로 한 줄씩 저장한다."""
    done = {r["id"] for r in read_jsonl(path) if r.get("id")}
    todo = [r for r in records if r.get("id") not in done]
    log(f"저장됨 {len(records) - len(todo)}, 새로 채점 {len(todo)}")
    if not todo:
        return []
    path.parent.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(max(1, workers))

    async def guarded(rec: Record) -> Record:
        async with sem:
            return await score_one(rec, scorer)

    new: list[Record] = []
    with path.open("a", encoding="utf-8") as fh:
        for i, fut in enumerate(
            asyncio.as_completed([guarded(r) for r in todo]), start=1
        ):
            row = await fut
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            new.append(row)
            shown = "-" if row["faithfulness"] is None else f"{row['faithfulness']:.2f}"
            log(f"  {i}/{len(todo)} {row['id']} {row['state']} {shown}")
    return new


def append_usage(
    out_dir: Path,
    *,
    step: str,
    mode: str,
    model: str,
    records: int,
    counter: UsageCounter,
) -> None:
    row = {
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "step": step,
        "mode": mode,
        "model": model,
        "records": records,
        **counter.as_dict(),
    }
    with (out_dir / "grading_usage.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------- 요약 (평균 옆에 대상 수·해당 없음·평가 불가·오류 건수) ----------


def summarize(rows: Sequence[Record], n_answers: int | None = None) -> dict[str, Any]:
    """한 방식의 Faithfulness 요약. 해당 없음·평가 불가는 평균에서 빼고 0점은 넣는다.

    n_answers: 1단계 답변 수. 아직 채점하지 않은 문항(missing)을 세려고 받는다.
    """
    scored = [r["faithfulness"] for r in rows if r.get("state") == "scored"]
    by_status: dict[str, Any] = {}
    for status in ("answered", "needs_clarification", "abstained"):
        vals = [
            r["faithfulness"]
            for r in rows
            if r.get("state") == "scored" and r.get("status") == status
        ]
        by_status[status] = {
            "mean": round(mean(vals), 4) if vals else None,
            "n": len(vals),
        }
    return {
        "mean": round(mean(scored), 4) if scored else None,
        "n_scored": len(scored),
        "n_zero": sum(1 for v in scored if v == 0),
        "n_na": sum(1 for r in rows if r.get("state") == "na"),
        "n_unavailable": sum(1 for r in rows if r.get("state") == "unavailable"),
        "n_service_error": sum(1 for r in rows if r.get("cause") == "service_error"),
        "n_grading_failed": sum(1 for r in rows if r.get("cause") == "grading_failed"),
        "n_missing": None if n_answers is None else max(0, n_answers - len(rows)),
        "by_status": by_status,
    }


def grading_usage(out_dir: Path, mode: str) -> dict[str, int]:
    """grading_usage.jsonl에서 이 방식의 Faithfulness 채점 토큰을 합친다."""
    total = {"calls": 0, "input": 0, "output": 0, "unknown": 0}
    for row in read_jsonl(out_dir / "grading_usage.jsonl"):
        if row.get("step") == "faithfulness" and row.get("mode") == mode:
            for key in total:
                total[key] += int(row.get(key) or 0)
    return total


def write_summary(out_dir: Path, modes: Sequence[str], *, model: str) -> dict[str, Any]:
    """faithfulness_summary.json, faithfulness_per_question.csv를 쓰고 요약을 돌려준다."""
    summary: dict[str, Any] = {
        "metric": "ragas.metrics.collections.Faithfulness (ragas 0.4.3)",
        "judge_model": model,
        "inputs": "user_input=질문, response=생성 답변, retrieved_contexts=실제 최종 문맥(청크 본문)",
        "modes": {},
    }
    csv_rows: list[Record] = []
    for mode in modes:
        answers = read_jsonl(out_dir / f"answers_{mode}.jsonl")
        if not answers:
            continue
        rows = read_jsonl(out_dir / f"faithfulness_{mode}.jsonl")
        summary["modes"][mode] = {
            **summarize(rows, n_answers=len(answers)),
            "grading_usage": grading_usage(out_dir, mode),
        }
        by_id = {r["id"]: r for r in rows}
        for a in answers:
            f = by_id.get(a["id"], {})
            csv_rows.append(
                {
                    "id": a["id"],
                    "mode": mode,
                    "status": a.get("status"),
                    "faithfulness": f.get("faithfulness"),
                    "state": f.get("state", "missing"),
                    "reason": f.get("reason", ""),
                    "question": a.get("question"),
                }
            )
    (out_dir / "faithfulness_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out_dir / "faithfulness_per_question.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as fh:
        fields = ["id", "mode", "status", "faithfulness", "state", "reason", "question"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    return summary


def report(summary: dict[str, Any]) -> str:
    """비교표에 붙일 마크다운 표."""
    lines = [
        "| 방식 | Faithfulness 평균 | 평가 대상 | 해당 없음 | 평가 불가 (서비스 오류 / 채점 실패) | 채점 토큰 입력 / 출력 |",
        "|---|---|---|---|---|---|",
    ]
    for mode, m in summary["modes"].items():
        avg = "-" if m["mean"] is None else f"{m['mean']:.3f}"
        u = m["grading_usage"]
        lines.append(
            f"| {mode} | {avg} | {m['n_scored']} | {m['n_na']} | "
            f"{m['n_unavailable']} ({m['n_service_error']} / {m['n_grading_failed']}) | "
            f"{u['input']:,} / {u['output']:,} |"
        )
    return "\n".join(lines)


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


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv(repo_root() / ".env")
    p = argparse.ArgumentParser(description="답변 평가 2단계: RAGAS Faithfulness")
    p.add_argument("--split", default="development", choices=SPLITS)
    p.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    p.add_argument("--difficulty", default="basic")
    p.add_argument("--model", default=MODEL, help="평가 LLM (세 방식에 같게)")
    p.add_argument("--workers", type=int, default=WORKERS)
    p.add_argument("--limit", type=int, default=0, help="앞에서 N문항만 (0이면 전부)")
    p.add_argument(
        "--summary-only", action="store_true", help="채점 없이 저장된 결과로 요약만"
    )
    p.add_argument(
        "--run", default="", help="채점할 답변 폴더 이름 (예: basic_development_k3)"
    )
    p.add_argument("--tag", default="", help="run_answers.py에 준 --tag와 같게")
    p.add_argument(
        "--no-board",
        action="store_true",
        help="끝난 뒤 실험 비교판을 다시 만들지 않는다",
    )
    p.add_argument(
        "--exp",
        default=None,
        help="실험 이름. 주면 results/experiments/<이름>/에 따로 저장",
    )
    args = p.parse_args(argv)
    exp = check_exp_arg(p, args.exp)

    out_dir = out_dir_for(
        args.difficulty, args.split, run=args.run, tag=args.tag, exp=exp
    )
    rel = out_dir.relative_to(repo_root())
    if not any(out_dir.glob("answers_*.jsonl")):
        runs = available_runs(args.difficulty, args.split, exp)
        raise SystemExit(
            f"답변 폴더 {rel}가 없습니다. run_answers.py와 같은 RETRIEVAL_B_* 설정으로 실행하거나 "
            f"--run으로 폴더를 고르세요. 있는 폴더: {', '.join(runs) or '없음'}"
        )
    print(f"[답변 폴더] {rel}", flush=True)
    if not args.summary_only:
        for mode in args.modes:
            records = read_jsonl(out_dir / f"answers_{mode}.jsonl")
            if not records:
                print(
                    f"[{mode}] answers_{mode}.jsonl이 없습니다. run_answers.py를 먼저 실행하세요."
                )
                continue
            if args.limit:
                records = records[: args.limit]
            counter = UsageCounter()
            scorer = make_scorer(args.model, counter)
            print(f"[{mode}] Faithfulness 채점")
            new = asyncio.run(
                score_records(
                    records,
                    scorer,
                    out_dir / f"faithfulness_{mode}.jsonl",
                    workers=args.workers,
                )
            )
            if new:
                append_usage(
                    out_dir,
                    step="faithfulness",
                    mode=mode,
                    model=args.model,
                    records=len(new),
                    counter=counter,
                )
                print(
                    f"[{mode}] 채점 호출 {counter.calls}회, "
                    f"입력 {counter.input:,} / 출력 {counter.output:,} 토큰"
                )
    summary = write_summary(out_dir, args.modes, model=args.model)
    print()
    print(report(summary))
    print(f"\n저장: {(out_dir / 'faithfulness_summary.json').relative_to(repo_root())}")
    if not args.no_board:
        refresh_board(exp)


if __name__ == "__main__":
    main()
