"""최적 k 찾기 1단계: 검색만으로 retrieval_a · b의 후보 설정과 최종 k를 바꿔 가며 비교한다 (LLM 호출 없음).

질문마다 상위 MAX_K개를 한 번 뽑고 앞에서 잘라 k = 1, 3, 5, 8, 10, 15의 지표를 계산한다 (순위가 정렬돼 있어 앞부분 = 상위 k).
지표 (난이도별)
- 문서 단위 Hit@k · Recall@k · MRR@k  (evaluation/retrieval/retrieval_scores.py, 발제 기준)
- 쪽 단위 Hit@k · Recall@k            (evidence_groups의 (book_id, page). 답변에 필요한 쪽을 가져왔나)
- 질문당 검색 시간
retrieval_a는 현재 설정(Qdrant sparse, hybrid.yaml)을 쓴다.
결과: k_sweep.json
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
KS = (1, 3, 5, 8, 10, 15)
MAX_K = max(KS)
OUT = HERE / "k_sweep.json"
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


def gold_pages(q):
    return {(str(e["book_id"]).upper(), int(e["page"])) for g in q["evidence_groups"] for e in g or [] if e.get("book_id")}


def metrics(docs, q):
    books = [{"book_id": d.metadata.get("book_id")} for d in docs]
    pages = [(str(d.metadata.get("book_id")).upper(), int(d.metadata.get("page") or -1)) for d in docs]
    gp = gold_pages(q)
    out = {}
    for k in KS:
        rec = {"status": "answered", "contexts": books[:k]}
        out[f"hit@{k}"] = RS.hit_at_k(rec, q, k)
        out[f"recall@{k}"] = RS.recall_at_k(rec, q, k)
        out[f"mrr@{k}"] = RS.mrr_at_k(rec, q, k)
        found = set(pages[:k]) & gp
        out[f"page_hit@{k}"] = 1.0 if found else 0.0
        out[f"page_recall@{k}"] = len(found) / len(gp) if gp else 0.0
    return out


def configs_a(d):
    from rag_service.retrieval_a import build_naive_retriever
    from rag_service.retrieval_a import hybrid as hybrid_mod
    from rag_service.retrieval_a.naive import load_config, model_profile
    from rag_service.retrieval_a.rerank import CONFIG_PATH as RERANK_CONFIG
    from rag_service.retrieval_a.rerank import HybridRerankRetriever, make_scorer

    rr_cfg = load_config(RERANK_CONFIG)
    base = hybrid_mod.build_hybrid_retriever(domain="shipping", difficulty=d, top_k=MAX_K)
    score_fn = make_scorer(rr_cfg, model_profile(d)["reranker"])
    yield "naive", "-", build_naive_retriever(domain="shipping", difficulty=d, top_k=MAX_K)
    for c in (100, 500):
        yield "hybrid", f"qdrant후보 {c}", base.model_copy(update={"candidate_k": c, "top_k": MAX_K})
    for n in (30, 50, 100):
        h = base.model_copy(update={"candidate_k": 500, "top_k": n})
        yield "hybrid_rerank", f"qdrant후보 500 · 리랭커 {n}", HybridRerankRetriever(
            hybrid=h, score_fn=score_fn, top_k=MAX_K, use_context=rr_cfg["reranker"].get("use_context", True))


def configs_b(d):
    from rag_service.retrieval_b import build_hybrid_retriever, build_naive_retriever, build_rerank_retriever

    yield "naive", "-", build_naive_retriever(domain="shipping", difficulty=d, k=MAX_K)
    for c in (30, 100):
        yield "hybrid", f"후보 {c}", build_hybrid_retriever(domain="shipping", difficulty=d, k=MAX_K, candidate_k=c)
    for c, n in ((30, 50), (100, 100)):
        yield "hybrid_rerank", f"후보 {c} · 리랭커 {n}", build_rerank_retriever(
            domain="shipping", difficulty=d, k=MAX_K, candidate_k=c, rerank_n=n)


def main(diffs):
    results = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else []
    done = {(r["impl"], r["difficulty"], r["mode"], r["setting"]) for r in results}
    for d in diffs:
        qs = golden(d)
        for impl, gen in (("retrieval_a", configs_a), ("retrieval_b", configs_b)):
            for mode, setting, retriever in gen(d):
                if (impl, d, mode, setting) in done:
                    continue
                retriever.invoke(qs[0]["question"])  # 모델·자료 준비를 시간 측정에서 뺀다
                per, secs = [], []
                for q in qs:
                    t = time.perf_counter()
                    docs = retriever.invoke(q["question"])[:MAX_K]
                    secs.append(time.perf_counter() - t)
                    per.append(metrics(docs, q))
                agg = {key: mean(x[key] for x in per) for key in per[0]}
                results.append({"impl": impl, "difficulty": d, "mode": mode, "setting": setting, "n": len(per),
                                "sec_mean": mean(secs), "sec_median": median(secs), **agg})
                OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
                ks = " ".join(f"k{k}:{agg[f'page_recall@{k}']:.2f}" for k in KS)
                print(f"{d:12} {impl:11} {mode:13} {setting:24} 쪽 Recall {ks} | {mean(secs):.2f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["basic", "intermediate", "advanced"])
