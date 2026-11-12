"""이미 돌린 비교 실험 문항에 RAGAS 4개만 계산해 올린다 (답변은 다시 만들지 않는다).

langfuse_experiment.py가 지금은 RAGAS를 실험 때 직접 계산하지만(e659014), 그 전에 돌린 실험에는 RAGAS 점수가 없다.
실험 문항의 저장된 답변 기록(질문·상태·답변·실제 최종 문맥)과 로컬 정답 파일로 quality.QualityScorer(Judge 없이)를 돌려
같은 이름(faithfulness 등, 채점 불가면 <이름>_state)으로 trace에 올린다.
이미 faithfulness(또는 faithfulness_state)가 있는 trace는 건너뛴다 → 중간에 끊겨도 다시 실행하면 이어서 한다.

사용: python ragas_backfill.py [--limit N] [--concurrency 8] [--skip "advanced:B"]
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import time
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
load_dotenv(ROOT / ".env")

_spec = importlib.util.spec_from_file_location("lx", ROOT / "evaluation" / "langfuse_experiment.py")
lx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lx)
quality = lx.quality


def golden_by_question(difficulty: str) -> dict[str, dict]:
    out = {}
    for split in ("development", "additional", "expansion_validation"):
        path = ROOT / "data" / difficulty / "shipping" / "evaluation" / f"{split}.json"
        if path.exists():
            out.update({q["question"].strip(): q for q in json.loads(path.read_text(encoding="utf-8-sig"))})
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--skip", nargs="*", default=["advanced:B"], help="난이도:구성 (다시 돌린 실행은 RAGAS가 이미 있음)")
    ap.add_argument("--retry-unavailable", action="store_true",
                    help="채점 호출 실패로 평가 불가(unavailable)만 남은 지표가 있는 문항을 다시 채점한다")
    args = ap.parse_args()

    from datetime import datetime, timedelta, timezone

    from langfuse import get_client

    lf = get_client()
    data = json.loads((HERE / "collected.json").read_text(encoding="utf-8"))
    skip = {tuple(s.split(":")) for s in args.skip}
    since = datetime.now(timezone.utc) - timedelta(days=2)
    todo = []
    for run in data["runs"]:
        if (run["difficulty"], run["config"]) in skip:
            continue
        gold = golden_by_question(run["difficulty"])
        items = lf.api.experiments.list_items(from_start_time=since, experiment_id=run["experiment_id"], limit=100,
                                              fields="core,io").data
        for it in items:
            d = it.model_dump()
            rec = d.get("output")
            rec = json.loads(rec) if isinstance(rec, str) else (rec or {})
            inp = d.get("input")
            inp = json.loads(inp) if isinstance(inp, str) else (inp or {})
            question = (inp.get("question") or rec.get("question") or "").strip()
            rec.setdefault("question", question)
            todo.append((run["experiment"], d["trace_id"], rec, gold.get(question)))
    print(f"[대상] 문항 {len(todo)}개 (건너뛴 구성 {sorted(skip)})", flush=True)

    scorer = quality.QualityScorer.create(lx.quality.MODEL, with_judge=False)
    sem = asyncio.Semaphore(args.concurrency)
    done = {"scored": 0, "skipped": 0, "failed": 0}
    t0 = time.time()

    async def one(exp, trace_id, rec, golden):
        async with sem:
            rows = await asyncio.to_thread(lambda: [x.model_dump() for x in lf.api.scores.get_many(trace_id=trace_id, limit=50).data])
            names = {x["name"] for x in rows}
            if args.retry_unavailable:
                numeric = {x["name"] for x in rows if x["name"] in quality.RAGAS_METRICS}
                failed = {x["name"].removesuffix("_state") for x in rows
                          if x["name"].endswith("_state") and str(x.get("value")) == "unavailable"}
                redo = (failed - numeric) and rec.get("status") != "error"
                if not redo:
                    done["skipped"] += 1
                    return
            elif all(n in names or f"{n}_state" in names for n in quality.RAGAS_METRICS):  # 4개 모두 있을 때만 건너뜀
                done["skipped"] += 1
                return
            try:
                scores = await scorer.score(rec, golden)
            except Exception as e:  # noqa: BLE001
                done["failed"] += 1
                print(f"[실패] {exp} {trace_id[:10]} {type(e).__name__}: {e}", flush=True)
                return
            for ev in lx.to_evaluations(scores, status=rec.get("status")):
                lf.create_score(trace_id=trace_id, name=ev.name, value=ev.value, data_type=ev.data_type,
                                comment=ev.comment, metadata=ev.metadata)
            done["scored"] += 1
            n = sum(done.values())
            if n % 25 == 0:
                rate = n / (time.time() - t0)
                print(f"[진행] {n}/{len(todo)} 채점 {done['scored']} 건너뜀 {done['skipped']} 실패 {done['failed']} · "
                      f"남은 시간 약 {(len(todo) - n) / rate / 60:.0f}분", flush=True)
                lf.flush()

    work = todo[: args.limit] if args.limit else todo
    await asyncio.gather(*(one(*w) for w in work))
    lf.flush()
    usage = scorer.grading_usage()
    print(f"[끝] {done} · {time.time() - t0:.0f}초 · 채점 토큰 입력 {usage['total']['input']:,} 출력 {usage['total']['output']:,} "
          f"(호출 {usage['total']['calls']})", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
