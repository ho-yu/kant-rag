"""ragas_backfill.py가 trace에 붙인 RAGAS 점수를, 실험 평가기가 붙이는 자리(문항의 루트 observation)로 옮긴다.
실험 화면에서 k8·k10(실험 중 채점, Observation)과 k5 등 예전 실험(나중 채점, Trace)의 RAGAS가 같은 열에 보이게 하려는 것.
같은 값·설명으로 observation 점수를 만들고(score_id 고정이라 다시 돌려도 덮어씀), trace에 있던 원래 점수는 지운다.
대상: collected.json의 실험 (모두 ragas_backfill로 채점됨)
사용: python move_ragas_to_observation.py [--limit N] [--keep]
"""
import json
import sys
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).parent
RAGAS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
NAMES = set(RAGAS) | {f"{n}_state" for n in RAGAS}


def main(limit, keep):
    load_dotenv(HERE.parents[2] / ".env")
    from langfuse import get_client

    lf = get_client()
    runs = json.loads((HERE / "collected.json").read_text(encoding="utf-8"))["runs"]
    since = datetime.now(timezone.utc) - timedelta(days=7)
    work = []  # (trace_id, 실험 문항 observation id, [trace에 붙은 RAGAS 점수])
    for r in runs:
        for it in lf.api.experiments.list_items(from_start_time=since, experiment_id=r["experiment_id"], limit=100,
                                                fields="core,scores", score_limit=50).data:
            d = it.model_dump()
            obs = next((x["subject"]["id"] for x in d["scores"] if (x.get("subject") or {}).get("kind") == "observation"), None)
            trace_scores = [x for x in d["scores"] if x["name"] in NAMES and (x.get("subject") or {}).get("kind") == "trace"]
            work.append((d["trace_id"], obs, trace_scores))
    if limit:
        work = work[:limit]
    stat = {"moved": 0, "deleted": 0, "no_obs": 0}

    def one(w):
        tid, obs, scores = w
        if not obs:
            stat["no_obs"] += 1
            return
        for s in scores:
            lf.create_score(trace_id=tid, observation_id=obs, score_id=f"{obs}-{s['name']}", name=s["name"], value=s["value"],
                            data_type=s["data_type"], comment=s.get("comment"), metadata=s.get("metadata"))
            stat["moved"] += 1
            if not keep:
                lf.api.scores.delete(s["id"])
                stat["deleted"] += 1

    with ThreadPoolExecutor(8) as ex:
        for i, _ in enumerate(ex.map(one, work), 1):
            if i % 200 == 0:
                print(f"{i}/{len(work)} {stat}", flush=True)
    lf.flush()
    print("items", len(work), stat)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[a.index("--limit") + 1]) if "--limit" in a else 0, "--keep" in a)
