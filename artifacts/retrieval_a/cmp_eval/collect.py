"""retrieval_a(기존·Qdrant sparse) / retrieval_b 비교 실험 결과를 Langfuse에서 모은다.

실험 이름(langfuse_experiment.py --tag)으로 구성을 찾는다.
    A-기존   : *_cmp_local   (RETRIEVAL_A_SPARSE_SOURCE=local, hybrid·hybrid_rerank)
    A-Qdrant : *_cmp_qdrant  (naive·hybrid·hybrid_rerank. naive는 sparse를 안 써서 A-기존 naive로도 쓴다)
    B        : b_*_cmp

문항마다 답변 기록(상태·응답 시간·문맥), 점수(검색: 스크립트, 답변: 서버 evaluator), 서비스 토큰(GENERATION usage)을 모은다.
실행: python artifacts/retrieval_a/cmp_eval/collect.py  → collected.json
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).with_name("collected.json")
MODES = ("naive", "hybrid", "hybrid_rerank")
CONFIGS = {"A-기존": r"^(naive|hybrid|hybrid_rerank)_\d{8}-\d{4}_cmp_local(?:_[a-z_]+)?$",
           "A-Qdrant": r"^(naive|hybrid|hybrid_rerank)_\d{8}-\d{4}_cmp_qdrant(?:_[a-z_]+)?$",
           "B": r"^b_(naive|hybrid|hybrid_rerank)_\d{8}-\d{4}_cmp(?:_[a-z_]+)?$"}
DIFFICULTIES = ("basic", "intermediate", "advanced")
INPUT_KEYS = ("input", "input_cache_read", "input_cache_creation")  # 캐시로 나뉜 입력 토큰을 합친다


SPLITS = ("development", "additional", "expansion_validation")


def _load_judge_summary():
    """서버 Judge 점수의 [해당 없음]·[평가 불가] 표시를 읽는 팀 규칙 (evaluation/answer/server_judge_summary.py)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("server_judge_summary", ROOT / "evaluation" / "answer" / "server_judge_summary.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def question_ids(difficulty: str) -> dict[str, str]:
    """질문 문장 → 문항 번호 (API가 실험 문항 메타데이터를 돌려주지 않아 로컬 정답 파일로 찾는다)."""
    out = {}
    for split in SPLITS:
        path = ROOT / "data" / difficulty / "shipping" / "evaluation" / f"{split}.json"
        if path.exists():
            out.update({q["question"].strip(): q["id"] for q in json.loads(path.read_text(encoding="utf-8-sig"))})
    return out


def split_of(e) -> str | None:
    ds = str((e.metadata or {}).get("dataset") or e.description or "")
    return next((sp for sp in SPLITS if f"-{sp.replace('_', '-')}-" in ds), None)


def difficulty_of(e) -> str | None:
    """실험 메타데이터의 데이터셋 이름(shipping-<난이도>-development-v1)에서 난이도."""
    ds = str((e.metadata or {}).get("dataset") or e.description or "")
    return next((d for d in DIFFICULTIES if f"-{d}-" in ds), None)


def to_number(score: dict) -> float | None:
    """NUMERIC / BOOLEAN / CATEGORICAL(pass·fail·na) → 숫자. 해당 없음·평가 불가는 None."""
    v = score.get("value")
    if score.get("data_type") == "BOOLEAN" or str(v) in ("True", "False"):
        return 1.0 if str(v) in ("True", "1", "1.0") else 0.0
    if isinstance(v, str):
        low = v.strip().lower()
        if low in ("pass", "yes", "true", "1"):
            return 1.0
        if low in ("fail", "no", "false", "0"):
            return 0.0
        try:
            return float(low)
        except ValueError:
            return None
    return None if v is None else float(v)


def main() -> None:
    load_dotenv(ROOT / ".env")
    from langfuse import get_client

    lf = get_client()
    sjs = _load_judge_summary()
    since = datetime.now(timezone.utc) - timedelta(days=2)
    exps = lf.api.experiments.list(from_start_time=since, limit=100).data
    runs = {}
    for cfg, pat in CONFIGS.items():
        for e in exps:
            m = re.match(pat, e.name)
            if not m:
                continue
            key = (difficulty_of(e), split_of(e), cfg, m.group(1))
            if key[0] is None or key[1] is None:
                continue
            if key not in runs or e.start_time > runs[key].start_time:
                runs[key] = e
    out = {"collected_at": datetime.now().isoformat(timespec="seconds"), "runs": []}
    for (diff, split, cfg, mode), e in sorted(runs.items()):
        qmap = question_ids(diff)
        items = lf.api.experiments.list_items(from_start_time=since, experiment_id=e.id, limit=100,
                                              fields="core,io,scores", score_limit=50).data
        rows = []
        for it in items:
            d = it.model_dump()
            rec = d.get("output")
            rec = json.loads(rec) if isinstance(rec, str) else (rec or {})
            inp = d.get("input")
            inp = json.loads(inp) if isinstance(inp, str) else (inp or {})
            question = (inp.get("question") or rec.get("question") or "").strip()
            scores, states = {}, {}
            # list_items는 서버 evaluator 점수(source EVAL)를 돌려주지 않아 trace 기준으로 모두 가져온다
            trace_scores = [x.model_dump() for x in lf.api.scores.get_many(trace_id=d["trace_id"], limit=100).data]
            judge_hits = {}
            for s in trace_scores or d.get("scores") or []:
                name = s.get("name", "")
                if name in sjs.JUDGE_NAMES:
                    judge_hits.setdefault(name, []).append(
                        {"value": s.get("value"), "comment": s.get("comment"), "timestamp": str(s.get("timestamp") or "")})
                    continue
                if name.endswith("_state"):
                    states[name.removesuffix("_state")] = s.get("value")
                    continue
                num = to_number(s)
                if num is None:
                    states[name] = s.get("value")
                else:
                    scores[name] = num
            for js in sjs.item_scores(rec.get("status") or "", judge_hits):
                if js["state"] == "scored":
                    scores[js["name"]] = float(js["value"])
                else:
                    states[js["name"]] = js["state"]
            tok_in = tok_out = 0
            cost = 0.0
            gens = lf.api.observations.get_many(trace_id=d["trace_id"], type="GENERATION", limit=50,
                                                fields="core,usage").data
            known = 0
            for g in gens:
                gd = g.model_dump()
                u = gd.get("usage_details") or {}
                if u:
                    known += 1
                    tok_in += sum(int(u.get(k) or 0) for k in INPUT_KEYS)
                    tok_out += int(u.get("output") or 0)
                    cost += float((gd.get("cost_details") or {}).get("total") or 0)
            rows.append({
                "question_id": qmap.get(question),
                "question": question,
                "status": rec.get("status"),
                "error": rec.get("error"),
                "latency_ms": rec.get("latency_ms"),
                "n_contexts": len(rec.get("contexts") or []),
                "context_books": [c.get("book_id") for c in rec.get("contexts") or []],
                "scores": scores,
                "states": states,
                "tokens": {"input": tok_in, "output": tok_out, "cost": cost, "generations": len(gens), "with_usage": known},
                "trace_id": d["trace_id"],
            })
        out["runs"].append({"difficulty": diff, "split": split, "config": cfg, "mode": mode, "experiment": e.name, "experiment_id": e.id,
                            "n_items": len(rows), "items": rows})
        print(f"{diff:12} {split:20} {cfg:9} {mode:14} {e.name:50} items={len(rows)}")
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved", OUT)


if __name__ == "__main__":
    main()
