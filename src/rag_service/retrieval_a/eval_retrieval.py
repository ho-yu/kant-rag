from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import os
import re
from pathlib import Path

from rag_service.retrieval_a.naive import REPO_ROOT

log = logging.getLogger("eval_retrieval")

MODES = ("naive", "hybrid", "rerank")
PAIRS = (("naive", "hybrid"), ("naive", "rerank"), ("hybrid", "rerank"))
_STRIP = re.compile(r"[\s\W_]+", re.UNICODE)
_CLI_LINE = re.compile(r"^\[S(\d+)\]\s+(.+?)\.pdf\s+(\d+)쪽\s+\(([^)]*)\)")


def normalize(text: str) -> str:
    return _STRIP.sub("", text or "").lower()


def _pieces(text: str, size: int = 10, step: int = 5) -> list[str]:
    if len(text) <= size:
        return [text] if text else []
    out = [text[i : i + size] for i in range(0, len(text) - size + 1, step)]
    if (len(text) - size) % step:
        out.append(text[-size:])
    return out


def quote_in_chunk(chunk_text: str, quote: str, threshold: float = 0.6) -> bool:
    c, q = normalize(chunk_text), normalize(quote)
    if not c or not q:
        return False
    qp = _pieces(q)
    if sum(p in c for p in qp) / len(qp) >= threshold:
        return True
    if len(c) >= 30:
        cp = _pieces(c)
        if sum(p in q for p in cp) / len(cp) >= 0.8:
            return True
    return False


def same_page(hit: dict, ev: dict, tolerance: int = 0) -> bool:
    if str(hit.get("book_id", "")).lower() != str(ev.get("book_id", "")).lower():
        return False
    try:
        return abs(int(hit.get("page")) - int(ev.get("page"))) <= tolerance
    except (TypeError, ValueError):
        return False


def evidence_match(hit: dict, ev: dict, text: str | None, *, threshold: float, tolerance: int, use_quote: bool) -> bool:
    if not same_page(hit, ev, tolerance):
        return False
    if not use_quote:
        return True
    return text is not None and quote_in_chunk(text, ev.get("quote", ""), threshold)


def group_rank(
    hits: list[dict],
    groups: list[list[dict]],
    texts: dict[str, str],
    *,
    threshold: float = 0.6,
    tolerance: int = 0,
    use_quote: bool = True,
) -> int | None:
    worst = 0
    for group in groups:
        first = next(
            (
                h["rank"]
                for h in hits
                if any(
                    evidence_match(h, ev, texts.get(h["chunk_id"]), threshold=threshold, tolerance=tolerance, use_quote=use_quote)
                    for ev in group
                )
            ),
            None,
        )
        if first is None:
            return None
        worst = max(worst, first)
    return worst


def parse_cli_output(stdout: str) -> list[dict]:
    hits = []
    for line in stdout.splitlines():
        m = _CLI_LINE.match(line.strip())
        if m:
            hits.append(
                {
                    "rank": int(m.group(1)),
                    "book_id": m.group(2),
                    "page": int(m.group(3)),
                    "chunk_id": m.group(4).split(",")[-1].split()[-1],
                }
            )
    return hits


def load_golden(paths: list[str]) -> tuple[list[dict], list[str]]:
    items, skipped, seen = [], [], set()
    for path in paths:
        text = Path(path).read_text(encoding="utf-8-sig").strip()
        rows = json.loads(text) if text.startswith("[") else [json.loads(l) for l in text.splitlines() if l.strip()]
        for q in rows:
            if q["id"] in seen:
                continue
            seen.add(q["id"])
            groups = [g for g in q.get("evidence_groups") or [] if g]
            if not groups:
                skipped.append(q["id"])
                continue
            items.append({"id": q["id"], "question": q["question"], "groups": groups})
    return items, skipped


def load_corpus(pattern: str) -> dict[str, str]:
    texts = {}
    full = pattern if os.path.isabs(pattern) else str(REPO_ROOT / pattern)
    for f in sorted(glob.glob(full)):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    if "chunk_id" in rec:
                        texts[rec["chunk_id"]] = rec.get("text", "")
    return texts


def load_raw(path: Path) -> dict[str, list[dict]]:
    done = {}
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done[rec["id"]] = rec["hits"]
    return done


def _build(mode: str, k: int, profile: str = "bge-m3", difficulty: str = "basic", domain: str = "shipping"):
    from rag_service.retrieval_a import build_hybrid_retriever, build_naive_retriever, build_rerank_retriever

    builder = {"naive": build_naive_retriever, "hybrid": build_hybrid_retriever, "rerank": build_rerank_retriever}[mode]
    return builder(domain=domain, difficulty=difficulty, profile=profile, top_k=k)


def default_out(profile: str, difficulty: str) -> str:
    name = profile if difficulty == "basic" else f"{profile}_{difficulty}"
    return f"artifacts/retrieval_a/eval_{name}"


def retrieve_mode(
    mode: str, items: list[dict], k: int, raw_path: Path, fresh: bool, build=None
) -> dict[str, list[dict]]:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if fresh and raw_path.exists():
        raw_path.unlink()
    done = load_raw(raw_path)
    todo = [q for q in items if q["id"] not in done]
    log.info("[%s] 저장된 결과 %d문항, 새로 검색 %d문항", mode, len(items) - len(todo), len(todo))
    if todo:
        retriever = (build or _build)(mode, k)
        with open(raw_path, "a", encoding="utf-8") as fh:
            for i, q in enumerate(todo, start=1):
                docs = retriever.invoke(q["question"])
                hits = [
                    {
                        "rank": r,
                        "book_id": d.metadata.get("book_id"),
                        "page": d.metadata.get("page"),
                        "chunk_id": d.metadata.get("chunk_id"),
                    }
                    for r, d in enumerate(docs, start=1)
                ]
                fh.write(json.dumps({"id": q["id"], "hits": hits}, ensure_ascii=False) + "\n")
                fh.flush()
                done[q["id"]] = hits
                if i % 10 == 0 or i == len(todo):
                    log.info("[%s] %d/%d", mode, i, len(todo))
    return done


def metrics(rows: list[dict], k: int) -> dict:
    n = len(rows) or 1
    out = {}
    for cut in (1, 3, 5):
        if cut <= k:
            out[f"Hit@{cut}"] = sum(1 for r in rows if r["evidence_rank"] and r["evidence_rank"] <= cut) / n
    if k not in (1, 3, 5):
        out[f"Hit@{k}"] = sum(1 for r in rows if r["evidence_rank"]) / n
    out["MRR"] = sum(1 / r["evidence_rank"] for r in rows if r["evidence_rank"]) / n
    out["PageHit@1"] = sum(1 for r in rows if r["page_rank"] == 1) / n
    return out


def pair_table(per_mode: dict[str, dict[str, dict]], a: str, b: str, cut: int) -> dict:
    def ok(r):
        return r["evidence_rank"] is not None and r["evidence_rank"] <= cut

    counts = {"a_only": 0, "b_only": 0, "both": 0, "neither": 0}
    for qid, ra in per_mode[a].items():
        x, y = ok(ra), ok(per_mode[b][qid])
        counts["both" if x and y else "a_only" if x else "b_only" if y else "neither"] += 1
    return counts


def evaluate(args) -> dict:
    items, skipped = load_golden(args.golden)
    if args.limit:
        items = items[: args.limit]
    texts = load_corpus(args.corpus)
    log.info("청크 원문 %d개 불러옴", len(texts))
    log.info("채점 문항 %d개, 근거 없어 제외 %d개 %s", len(items), len(skipped), skipped)

    out_arg = args.out or default_out(args.profile, args.difficulty)
    out = Path(out_arg) if os.path.isabs(out_arg) else REPO_ROOT / out_arg
    log.info("모델 설정 %s · 난이도 %s · 결과 폴더 %s", args.profile, args.difficulty, out)

    def build(mode, k):
        return _build(mode, k, args.profile, args.difficulty, args.domain)

    per_mode: dict[str, dict[str, dict]] = {}
    missing = set()
    for mode in args.modes:
        raw = retrieve_mode(mode, items, args.k, out / "raw" / f"{mode}_k{args.k}.jsonl", args.fresh, build)
        per_mode[mode] = {}
        for q in items:
            hits = raw[q["id"]][: args.k]
            missing.update(h["chunk_id"] for h in hits if h["chunk_id"] not in texts)
            kw = {"threshold": args.threshold, "tolerance": args.page_tolerance}
            per_mode[mode][q["id"]] = {
                "id": q["id"],
                "mode": mode,
                "evidence_rank": group_rank(hits, q["groups"], texts, use_quote=True, **kw),
                "page_rank": group_rank(hits, q["groups"], texts, use_quote=False, **kw),
                "top1": hits[0]["chunk_id"] if hits else "",
                "top3": [h["chunk_id"] for h in hits[:3]],
                "question": q["question"],
            }
    if missing:
        log.warning("원문을 찾지 못한 청크 %d개: %s", len(missing), sorted(missing)[:10])

    summary = {
        "k": args.k,
        "profile": args.profile,
        "difficulty": args.difficulty,
        "questions": len(items),
        "skipped": len(skipped),
        "skipped_ids": skipped,
        "threshold": args.threshold,
        "page_tolerance": args.page_tolerance,
        "metrics": {m: metrics(list(per_mode[m].values()), args.k) for m in args.modes},
        "pairs": {
            f"{a}-{b}": {f"Hit@{cut}": pair_table(per_mode, a, b, cut) for cut in sorted({1, args.k})}
            for a, b in PAIRS
            if a in per_mode and b in per_mode
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out / "per_question.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "mode", "evidence_rank", "page_rank", "top1", "question"])
        for m in args.modes:
            for r in per_mode[m].values():
                w.writerow([r["id"], m, r["evidence_rank"] or "", r["page_rank"] or "", r["top1"], r["question"]])
    summary["_per_mode"] = per_mode
    summary["_out"] = str(out)
    return summary


def print_report(summary: dict, show_misses: bool) -> None:
    k, n = summary["k"], summary["questions"]
    names = list(next(iter(summary["metrics"].values())).keys())
    print(f"\n[{summary.get('profile')} · {summary.get('difficulty')}] 채점 {n}문항 (근거 없어 제외 {summary['skipped']}문항), top-{k}")
    print("| 모드 | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for mode, m in summary["metrics"].items():
        print(f"| {mode} | " + " | ".join(f"{m[c]:.3f}" for c in names) + " |")
    print(f"1문항 = 1/{n} = {1 / max(n, 1):.3f}")

    if summary["pairs"]:
        print("\n짝 비교 (A만 / B만 / 둘 다 / 둘 다 실패)")
        for pair, cuts in summary["pairs"].items():
            a, b = pair.split("-")
            for cut, c in cuts.items():
                print(f"  {a} vs {b} {cut}: {a}만 {c['a_only']} / {b}만 {c['b_only']} / 둘 다 {c['both']} / 둘 다 실패 {c['neither']}")

    if show_misses:
        for mode, rows in summary["_per_mode"].items():
            misses = [r for r in rows.values() if r["evidence_rank"] is None]
            print(f"\n[{mode}] top-{k} 실패 {len(misses)}문항")
            for r in misses:
                print(f"  {r['id']:<16} {r['question'][:40]}  → {', '.join(r['top3'])}")
    print(f"\n저장: {summary['_out']}")


def _main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("httpx", "huggingface_hub", "sentence_transformers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    p = argparse.ArgumentParser(description="검색 3방식 골든 데이터셋 채점 (LLM 없음)")
    p.add_argument("--golden", nargs="+", required=True)
    p.add_argument("--profile", default="bge-m3", help="models.yaml의 모델 설정 (bge-m3, bge-m3-bm25 …)")
    p.add_argument("--difficulty", default="basic")
    p.add_argument("--domain", default="shipping")
    p.add_argument("--corpus", default=None, help="기본: artifacts/corpus/<난이도>/*/*.jsonl")
    p.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.6)
    p.add_argument("--page-tolerance", type=int, default=0)
    p.add_argument("--out", default=None, help="기본: artifacts/retrieval_a/eval_<모델>[_<난이도>]")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--show-misses", action="store_true")
    args = p.parse_args()
    args.corpus = args.corpus or f"artifacts/corpus/{args.difficulty}/*/*.jsonl"
    print_report(evaluate(args), args.show_misses)

    import gc

    from rag_service.retrieval_a.naive import local_client

    if local_client.cache_info().currsize:
        local_client().close()
        local_client.cache_clear()
        gc.collect()


if __name__ == "__main__":
    _main()
