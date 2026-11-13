"""collect.py · collect_k.py로 모은 실험마다 실험 단위 요약 점수를 Langfuse 실험(dataset run)에 올린다.
실험 목록 화면에서 이름(검색기·모드·난이도·데이터·k·LLM) 옆에 최종 점수가 보이게 하려는 것.

올리는 점수 (score_id를 실험id-이름으로 고정해 다시 돌려도 덮어쓴다)
- judge_*_mean   : 서버 Judge 4항목 평균 ([해당 없음]·[평가 불가]는 뺌, server_judge_summary 규칙)  ※ --judge일 때만
- accuracy_mean  : 정답률 = Judge 4항목을 모두 통과한 문항 비율
- tokens_in_mean · tokens_out_mean : 문항당 답변 LLM 토큰 (입력은 캐시 토큰 포함)
- latency_mean · latency_median    : 문항당 응답 시간(초)
사용: python upload_run_summary.py collected_k.json --judge
"""
import json
import sys
from pathlib import Path
from statistics import mean, median

from dotenv import load_dotenv

HERE = Path(__file__).parent
JUDGE = ("judge_facts", "judge_decision", "judge_evidence", "judge_conditions")


def summary(items):
    out, meta = {}, {}
    for n in JUDGE:
        v = [it["scores"][n] for it in items if n in it["scores"]]
        if v:
            out[f"{n}_mean"] = mean(v)
            meta[f"{n}_mean"] = {"n_scored": len(v), "n_not_scored": len(items) - len(v)}
    judged = [it for it in items if any(n in it["scores"] for n in JUDGE)]
    if judged:
        out["accuracy_mean"] = mean(1.0 if all(it["scores"].get(n, 1.0) == 1.0 for n in JUDGE) else 0.0 for it in judged)
        meta["accuracy_mean"] = {"n_items": len(judged)}
    toks = [it["tokens"] for it in items if it["tokens"]["with_usage"]]
    if toks:
        out["tokens_in_mean"] = mean(t["input"] for t in toks)
        out["tokens_out_mean"] = mean(t["output"] for t in toks)
    lat = [it["latency_ms"] / 1000 for it in items if it.get("latency_ms")]
    if lat:
        out["latency_mean"] = mean(lat)
        out["latency_median"] = median(lat)
    return out, meta


COMMENT = {
    "accuracy_mean": "정답률: 서버 Judge 4항목(사실·결론·근거·조건)을 모두 통과한 문항 비율",
    "tokens_in_mean": "문항당 답변 LLM 입력 토큰 (캐시 포함)",
    "tokens_out_mean": "문항당 답변 LLM 출력 토큰",
    "latency_mean": "문항당 응답 시간 평균(초)",
    "latency_median": "문항당 응답 시간 중앙값(초)",
}


def main(path, with_judge):
    load_dotenv(HERE.parents[2] / ".env")
    from langfuse import get_client

    lf = get_client()
    runs = json.loads((HERE / path).read_text(encoding="utf-8"))["runs"]
    n = 0
    for r in runs:
        vals, meta = summary(r["items"])
        for name, v in vals.items():
            if name.startswith("judge_") and not with_judge:
                continue
            lf.create_score(name=name, value=float(v), dataset_run_id=r["experiment_id"], data_type="NUMERIC",
                            score_id=f"{r['experiment_id']}-{name}",
                            comment=COMMENT.get(name, "서버 Judge 점수를 [해당 없음]·[평가 불가] 규칙(server_judge_summary.py)으로 집계"),
                            metadata={"source": "upload_run_summary", **meta.get(name, {})})
            n += 1
        print(f"{r['experiment']:60} " + " ".join(f"{k}={v:.3g}" for k, v in vals.items()))
    lf.flush()
    print("올린 점수:", n)


if __name__ == "__main__":
    main(sys.argv[1], "--judge" in sys.argv)
