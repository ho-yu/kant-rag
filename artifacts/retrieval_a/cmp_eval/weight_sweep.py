"""retrieval_a hybrid 가중치(dense, sparse) 비교: 검색 지표만 (답변·LLM 호출 없음).

A · 기존(local)과 A · Qdrant 각각, 세 난이도의 골든 질문 전체로 hybrid와 hybrid + rerank를 돌린다.
검색기는 구성·난이도마다 한 번만 만들고 가중치만 바꾼 복사본으로 검색한다.
지표: 발제 기준 문서 단위 Hit@k·Recall@k·MRR@k (k = 1, 3, 5, evaluation/retrieval/retrieval_scores.py)
결과: weight_sweep.json
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
WEIGHTS = ((0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (0.1, 0.9))
KS = (1, 3, 5)
SOURCES = ("qdrant", "local")
DIFFS = ("basic", "intermediate", "advanced")
OUT = HERE / "weight_sweep.json"

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
    return out


def scores(docs, gold):
    rec = {"status": "answered", "contexts": [{"book_id": d.metadata.get("book_id")} for d in docs]}
    out = {}
    for k in KS:
        out[f"hit@{k}"] = RS.hit_at_k(rec, gold, k)
        out[f"recall@{k}"] = RS.recall_at_k(rec, gold, k)
        out[f"mrr@{k}"] = RS.mrr_at_k(rec, gold, k)
    return out


def main():
    from rag_service.retrieval_a import hybrid as hybrid_mod
    from rag_service.retrieval_a.rerank import CONFIG_PATH as RERANK_CONFIG
    from rag_service.retrieval_a.rerank import HybridRerankRetriever, make_scorer
    from rag_service.retrieval_a.naive import load_config, model_profile

    rr_cfg = load_config(RERANK_CONFIG)
    results = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else []
    done = {(r["source"], r["difficulty"], r["mode"], tuple(r["weights"])) for r in results}
    for source in SOURCES:
        os.environ["RETRIEVAL_A_SPARSE_SOURCE"] = source
        for d in DIFFS:
            questions = [q for q in golden(d) if [g for g in q.get("evidence_groups") or [] if g]]
            t = time.time()
            base = hybrid_mod.build_hybrid_retriever(domain="shipping", difficulty=d, top_k=5)
            score_fn = make_scorer(rr_cfg, model_profile(d)["reranker"])
            for w in WEIGHTS:
                h5 = base.model_copy(update={"dense_weight": w[0], "sparse_weight": w[1], "top_k": 5})
                h50 = base.model_copy(update={"dense_weight": w[0], "sparse_weight": w[1], "top_k": rr_cfg["search"]["candidate_k"]})
                rr = HybridRerankRetriever(hybrid=h50, score_fn=score_fn, top_k=rr_cfg["search"]["top_k"],
                                           use_context=rr_cfg["reranker"].get("use_context", True))
                for mode, retriever in (("hybrid", h5), ("hybrid_rerank", rr)):
                    if (source, d, mode, w) in done:
                        continue
                    per_q = [scores(retriever.invoke(q["question"]), q) for q in questions]
                    agg = {k: mean(x[k] for x in per_q) for k in per_q[0]}
                    results.append({"source": source, "difficulty": d, "mode": mode, "weights": list(w), "n": len(per_q), **agg})
                    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
                    print(f"{source:6} {d:12} {mode:13} {w} n={len(per_q)} Hit@5 {agg['hit@5']:.3f} Recall@5 {agg['recall@5']:.3f} "
                          f"MRR@5 {agg['mrr@5']:.3f}", flush=True)
            print(f"  ({source} {d} {time.time() - t:.0f}초)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
