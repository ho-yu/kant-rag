"""A · Qdrant의 후보 수(candidate_k) 비교: 검색 지표와 질문당 검색 시간만 (답변·LLM 호출 없음).

사용: python candidate_sweep.py [난이도 ...]   (기본: advanced)
결과: candidate_sweep.json
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean, median

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
CANDS = (100, 300, 500, 1000)
WEIGHTS = ((0.7, 0.3), (0.5, 0.5))
OUT = HERE / "candidate_sweep.json"
os.environ["RETRIEVAL_A_SPARSE_SOURCE"] = "qdrant"

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
_spec = importlib.util.spec_from_file_location("rs", ROOT / "evaluation" / "retrieval" / "retrieval_scores.py")
RS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RS)


def golden(difficulty):
    out = []
    for split in ("development", "additional", "expansion_validation"):
        path = ROOT / "data" / difficulty / "shipping" / "evaluation" / f"{split}.json"
        if path.exists():
            out += json.loads(path.read_text(encoding="utf-8-sig"))
    return [q for q in out if [g for g in q.get("evidence_groups") or [] if g]]


def main(diffs):
    from rag_service.retrieval_a import hybrid as hybrid_mod
    from rag_service.retrieval_a.naive import load_config, model_profile
    from rag_service.retrieval_a.rerank import CONFIG_PATH as RERANK_CONFIG
    from rag_service.retrieval_a.rerank import HybridRerankRetriever, make_scorer

    rr_cfg = load_config(RERANK_CONFIG)
    results = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else []
    for d in diffs:
        qs = golden(d)
        base = hybrid_mod.build_hybrid_retriever(domain="shipping", difficulty=d, top_k=5)
        score_fn = make_scorer(rr_cfg, model_profile(d)["reranker"])
        base.invoke(qs[0]["question"])  # 모델 로딩을 시간 측정에서 뺀다
        for w in WEIGHTS:
            for c in CANDS:
                upd = {"dense_weight": w[0], "sparse_weight": w[1], "candidate_k": c}
                h5 = base.model_copy(update={**upd, "top_k": 5})
                h50 = base.model_copy(update={**upd, "top_k": rr_cfg["search"]["candidate_k"]})
                rr = HybridRerankRetriever(hybrid=h50, score_fn=score_fn, top_k=5, use_context=rr_cfg["reranker"].get("use_context", True))
                for mode, retriever in (("hybrid", h5), ("hybrid_rerank", rr)):
                    per, secs = [], []
                    for q in qs:
                        t = time.perf_counter()
                        docs = retriever.invoke(q["question"])
                        secs.append(time.perf_counter() - t)
                        rec = {"status": "answered", "contexts": [{"book_id": x.metadata.get("book_id")} for x in docs]}
                        per.append({f"{m}@{k}": f(rec, q, k) for k in (1, 3, 5) for m, f in
                                    (("hit", RS.hit_at_k), ("recall", RS.recall_at_k), ("mrr", RS.mrr_at_k))})
                    agg = {k: mean(x[k] for x in per) for k in per[0]}
                    row = {"difficulty": d, "weights": list(w), "candidate_k": c, "mode": mode, "n": len(per),
                           "sec_mean": mean(secs), "sec_median": median(secs), **agg}
                    results.append(row)
                    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
                    print(f"{d} {w} cand={c:4} {mode:13} Hit@5 {agg['hit@5']:.3f} Recall@5 {agg['recall@5']:.3f} "
                          f"MRR@5 {agg['mrr@5']:.3f} | 검색 {mean(secs):.2f}s (중앙 {median(secs):.2f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["advanced"])
