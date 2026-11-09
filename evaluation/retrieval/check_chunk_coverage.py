"""청킹 점검: 평가 JSON의 근거 인용(quote)이 청크 안에 온전히 남았는지 확인한다.

※ 평가 전용 스크립트다. quote·정답은 여기서 '청킹 결과를 검사'하는 데만 쓰고,
  색인·검색어·필터·프롬프트에는 절대 넣지 않는다 (발제문 규칙).

확인하는 것 (인용마다)
    page_exact   : 그 쪽(page 레코드) 원문에 띄어쓰기까지 똑같이 있음 (줄바꿈·연속 공백은 공백 하나로 봄)
    page_loose   : 공백을 무시하면 쪽 원문에 있음 (추출·줄잇기 문제 확인용)
    one_chunk    : 검색 레벨(index.levels 중 page 제외) 청크 하나 안에 통째로 있음 (공백 무시)

실행
    uv run python evaluation/retrieval/check_chunk_coverage.py --difficulty basic
    uv run python evaluation/retrieval/check_chunk_coverage.py --difficulty basic --split development additional
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict

from rag_service.storage.corpus import (
    load_chunks,
    load_config,
    repo_root,
)


def _norm(s: str) -> str:
    """공백과 표 칸 구분자(|)를 지운다."""
    return re.sub(r"[\s|]+", "", s)


def _collapse(s: str) -> str:
    """줄바꿈·연속 공백·표 칸 구분자(|)를 공백 하나로 바꾼다 (표 행 인용은 칸을 공백으로 이어 씀)."""
    return re.sub(r"[\s|]+", " ", s).strip()


def collect_quotes(eval_dir, splits: list[str]) -> list[dict]:
    out = []
    for split in splits:
        f = eval_dir / f"{split}.json"
        if not f.exists():
            print(f"  (없음) {f}")
            continue
        for item in json.loads(f.read_text(encoding="utf-8")):
            for group in item.get("evidence_groups", []):
                for ev in group:
                    out.append({"split": split, "id": item.get("id"), "book_id": ev["book_id"],
                                "page": int(ev["page"]), "quote": ev["quote"]})
    uniq = {}
    for q in out:
        uniq.setdefault((q["book_id"], q["page"], _norm(q["quote"])), q)
    return list(uniq.values())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty", required=True)
    ap.add_argument("--domain", default="shipping")
    ap.add_argument("--split", nargs="+", default=["development"])
    ap.add_argument("--levels", nargs="+", default=None, help="기본: config의 index.levels에서 page를 뺀 것")
    args = ap.parse_args(argv)

    cfg = load_config()
    levels = args.levels or [lv for lv in cfg["index"]["levels"] if lv != "page"]
    eval_dir = repo_root() / cfg["paths"]["data_dir"] / args.difficulty / args.domain / "evaluation"
    quotes = collect_quotes(eval_dir, args.split)

    pages = {}
    by_page = defaultdict(list)
    for c in load_chunks(args.difficulty, args.domain):
        if c.level == "page":
            pages[(c.book_id, c.page)] = c.text
        elif c.level in levels:
            by_page[(c.book_id, c.page)].append(_norm(c.text))

    stats = defaultdict(int)
    misses = []
    for q in quotes:
        key = (q["book_id"], q["page"])
        ptext = pages.get(key, "")
        nq = _norm(q["quote"])
        exact = _collapse(q["quote"]) in _collapse(ptext)
        loose = nq in _norm(ptext)
        one = any(nq in t for t in by_page.get(key, []))
        stats["page_exact"] += exact
        stats["page_loose"] += loose
        stats["one_chunk"] += one
        if not (loose and one):
            misses.append({**q, "page_loose": loose, "one_chunk": one, "quote": q["quote"][:80]})

    n = len(quotes)
    print(f"[{args.difficulty}/{args.domain}] 근거 인용 {n}개 (중복 제거, split={args.split}, levels={levels})")
    for k in ("page_exact", "page_loose", "one_chunk"):
        print(f"  {k:11s}: {stats[k]}/{n} ({stats[k] / max(n, 1):.0%})")
    if misses:
        print(f"  한 청크에 안 들어간 인용 {len(misses)}개 (여러 문단에 걸친 경우가 많음, 부모 쪽으로 보완):")
        for m in misses[:10]:
            print(f"   - {m['id']} {m['book_id']} p{m['page']} loose={m['page_loose']} one={m['one_chunk']} | {m['quote']}")
    report = repo_root() / "results" / "chunk_coverage" / f"{args.difficulty}_{args.domain}.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"splits": args.split, "levels": levels, "total": n, **stats, "misses": misses},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  저장: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
