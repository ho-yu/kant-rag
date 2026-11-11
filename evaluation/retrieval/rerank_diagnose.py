"""리랭크 진단: Rerank가 Hybrid보다 어느 질문에서, 왜 졌는지 질문별로 본다.

compare 표는 "평균 몇 점"만 알려준다. 이 스크립트는 질문마다 정답 쪽(gold page)이
Hybrid에서 몇 위였고, 리랭커가 순서를 바꾼 뒤 몇 위가 됐는지 따라가서 원인을 나눈다.
검색 코드는 바꾸지 않는다 (재기만 한다).

질문마다 보는 순위 (정답 쪽 = 골든셋 evidence_groups의 (book_id, page), 그중 가장 위의 것)
- hybrid:  compare의 "hybrid"와 같은 설정(후보 30개씩, RRF)으로 낸 순위
- 후보:    리랭커에 넘기는 Hybrid 후보 rerank_n개(기본 50) 안에서의 순위
- rerank:  리랭커가 다시 세운 순위
- 깊이:    Hybrid를 probe_n개(기본 100)까지 늘렸을 때의 순위 (후보 없음일 때 rerank_n을 늘리면 들어오는지)

분류 (상위 k개 기준, 쪽 단위)
- 둘 다 맞힘: Hybrid도 Rerank도 k 안에 정답 쪽
- 밀림:      Hybrid는 맞혔는데 리랭커가 k 밖으로 내림 → 리랭커 입력·모델 쪽 문제
- 이김:      Hybrid는 놓쳤는데 리랭커가 k 안으로 올림
- 후보 없음: 리랭커 후보 rerank_n개 안에 정답 쪽이 없음 → 리랭커 잘못 아님, 후보 수 문제
- 둘 다 놓침: 후보에는 있는데 Hybrid도 Rerank도 k 밖
- 정답 없음: 골든셋에 근거가 없는 질문 (거절 문항). 평균에서 뺀다
놓친 질문에는 "같은 책"도 표시한다: 리랭커 상위 k개에 정답 책(book_id)은 있는데 쪽이 다름.
이때는 그 쪽이 실제로 답이 되는지 직접 읽어 본다 (골든셋이 빠뜨린 정답일 수 있음, 11/6 회의).

실행 (레포 맨 위에서, 노트북에서. 팀 Qdrant 읽기 + 리랭커 모델이 필요하다)
    uv run python evaluation/retrieval/rerank_diagnose.py
    uv run python evaluation/retrieval/rerank_diagnose.py --k 3 --rerank-n 30
    uv run python evaluation/retrieval/rerank_diagnose.py --rerank-n 20 --cand 50
    uv run python evaluation/retrieval/rerank_diagnose.py --only 밀림 "후보 없음"

결과: 화면 표 + results/diagnose/retrieval_b/rerank_{난이도}_{split}_k{k}_n{rerank_n}{설정}.json
      {설정}은 --levels·--cand·--rrf·--rerank-len이 기본값과 다를 때만 붙는다 (예: _cand50).
평가 JSON(정답)은 채점에만 쓴다. 검색기와 리랭커에는 question만 넘긴다.

[2026-11-11 리팩토링] 이유: 튜닝 값을 .env에서 읽지 않게 됨 → 효과: 다른 평가 스크립트와 같은 명령줄 옵션
(--k, --levels, --cand, --rrf, --rerank-n, --rerank-len)으로 받아 Hybrid 후보와 리랭커에 그대로 넘긴다.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
import unicodedata
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document

from rag_service.retrieval_b.metrics import (
    SPLITS,
    gold_docs,
    gold_pages,
    golden_path,
    load_golden,
)
from rag_service.retrieval_b.settings import (
    Settings,
    add_settings_args,
    settings_from_args,
    settings_tag,
)
from rag_service.storage.corpus import repo_root

OUT_DIR = Path("results") / "diagnose" / "retrieval_b"
COMPARE_HYBRID_K = (
    10  # compare는 ks의 최대값(10)으로 Hybrid를 만든다 → 후보 max(30, 10)=30개씩
)
LABELS = ("둘 다 맞힘", "밀림", "이김", "후보 없음", "둘 다 놓침", "정답 없음")
SNIPPET = 70


# ---------- 순위 찾기와 분류 (모델 없이 도는 부분) ----------


def page_key(doc: Document) -> tuple[str, int] | None:
    m = doc.metadata
    if not m.get("book_id") or m.get("page") is None:
        return None
    return (m["book_id"], int(m["page"]))


def first_gold_rank(docs: Sequence[Document], gold: set) -> int | None:
    """정답 쪽이 처음 나온 순위 (1부터). 없으면 None."""
    for rank, doc in enumerate(docs, start=1):
        if page_key(doc) in gold:
            return rank
    return None


def within(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= k


def classify(
    has_gold: bool,
    hybrid_rank: int | None,
    cand_rank: int | None,
    rerank_rank: int | None,
    k: int,
) -> str:
    if not has_gold:
        return "정답 없음"
    hybrid_hit, rerank_hit = within(hybrid_rank, k), within(rerank_rank, k)
    if hybrid_hit and rerank_hit:
        return "둘 다 맞힘"
    if hybrid_hit:
        return "밀림"
    if rerank_hit:
        return "이김"
    if cand_rank is None:
        return "후보 없음"
    return "둘 다 놓침"


def snippet(text: str, n: int = SNIPPET) -> str:
    one_line = " ".join(text.split())
    return one_line if len(one_line) <= n else one_line[: n - 1] + "…"


def diagnose_question(
    item: dict[str, Any],
    hybrid_docs: Sequence[Document],
    candidates: Sequence[Document],
    probe_docs: Sequence[Document],
    reranked: Sequence[Document],
    k: int,
) -> dict[str, Any]:
    """질문 하나의 진단 행. reranked는 후보 전부를 리랭커 점수 순으로 세운 목록이다."""
    g_pages, g_docs = gold_pages(item), gold_docs(item)
    hybrid_rank = first_gold_rank(hybrid_docs, g_pages)
    cand_rank = first_gold_rank(candidates, g_pages)
    rerank_rank = first_gold_rank(reranked, g_pages)
    label = classify(bool(g_pages), hybrid_rank, cand_rank, rerank_rank, k)

    top = list(reranked[:k])
    same_book = label in ("밀림", "후보 없음", "둘 다 놓침") and any(
        d.metadata.get("book_id") in g_docs for d in top
    )
    gold_doc = reranked[rerank_rank - 1] if rerank_rank else None
    return {
        "id": item["id"],
        "question": item["question"],
        "expected_action": item.get("expected_action"),
        "gold_pages": sorted([list(p) for p in g_pages]),
        "label": label,
        "same_book": same_book,
        "hybrid_rank": hybrid_rank,
        "candidate_rank": cand_rank,
        "probe_rank": first_gold_rank(probe_docs, g_pages),
        "rerank_rank": rerank_rank,
        "gold_rerank_score": gold_doc.metadata["rerank_score"] if gold_doc else None,
        "gold_chunk_id": gold_doc.metadata["chunk_id"] if gold_doc else None,
        "top1_rerank_score": top[0].metadata["rerank_score"] if top else None,
        "rerank_top": [
            {
                "rank": i,
                "chunk_id": d.metadata.get("chunk_id"),
                "page": list(page_key(d) or []),
                "rerank_score": round(d.metadata["rerank_score"], 4),
                "hybrid_rank": d.metadata.get("prev_rank"),
                "is_gold": page_key(d) in g_pages,
                "text": snippet(d.page_content),
            }
            for i, d in enumerate(top, start=1)
        ],
        "hybrid_top": [d.metadata.get("chunk_id") for d in hybrid_docs[:k]],
    }


def summarize(rows: Sequence[dict[str, Any]], k: int) -> dict[str, Any]:
    scored = [r for r in rows if r["label"] != "정답 없음"]
    n = len(scored)
    counts = Counter(r["label"] for r in rows)
    return {
        "n": n,
        "labels": {name: counts.get(name, 0) for name in LABELS},
        "same_book": sum(1 for r in rows if r["same_book"]),
        # compare의 pageHit@k와 같아야 한다 (같은 설정 확인용)
        "hybrid_page_hit": round(
            sum(within(r["hybrid_rank"], k) for r in scored) / n, 4
        )
        if n
        else None,
        "rerank_page_hit": round(
            sum(within(r["rerank_rank"], k) for r in scored) / n, 4
        )
        if n
        else None,
        "candidate_page_hit": round(
            sum(r["candidate_rank"] is not None for r in scored) / n, 4
        )
        if n
        else None,
    }


def extra_tag(settings: Settings) -> str:
    """파일 이름 꼬리. k·rerank_n은 이름에 늘 들어가므로 뺀 나머지 바꾼 값만 (예: _cand50, _lvpage)."""
    default = Settings()
    return settings_tag(
        dataclasses.replace(settings, k=default.k, rerank_n=default.rerank_n)
    )


# ---------- 화면 출력 ----------


def _pad(text: str, width: int) -> str:
    """한글은 화면에서 두 칸을 차지하니까 그만큼 빼고 채운다."""
    shown = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
    return text + " " * max(width - shown, 0)


def _r(rank: int | None) -> str:
    return "-" if rank is None else str(rank)


def _s(score: float | None) -> str:
    return "-" if score is None else f"{score:.3f}"


def print_table(
    rows: Sequence[dict[str, Any]],
    summary: dict[str, Any],
    k: int,
    n: int,
    probe_n: int,
) -> None:
    head = (
        f"{'id':<11} {_pad('분류', 10)} {'hybrid':>6} {'후보':>3} {'rerank':>6} "
        f"{f'깊이{probe_n}':>7} {'정답점수':>7} {'1위점수':>7}  같은책"
    )
    print(head)
    print("-" * (len(head) + 6))
    for r in rows:
        print(
            f"{r['id']:<11} {_pad(r['label'], 10)} {_r(r['hybrid_rank']):>6} "
            f"{_r(r['candidate_rank']):>4} {_r(r['rerank_rank']):>6} "
            f"{_r(r['probe_rank']):>7} {_s(r['gold_rerank_score']):>7} "
            f"{_s(r['top1_rerank_score']):>7}  {'O' if r['same_book'] else ''}"
        )
    labels = ", ".join(f"{name} {c}" for name, c in summary["labels"].items() if c)
    print(
        f"\n요약 (k={k}, 후보 {n}개, 질문 {summary['n']}개): {labels}, 같은 책 {summary['same_book']}"
    )
    print(
        f"pageHit@{k}: hybrid {summary['hybrid_page_hit']}, rerank {summary['rerank_page_hit']}"
        f" / 후보 {n}개 안에 정답 쪽: {summary['candidate_page_hit']}"
    )


def print_details(rows: Sequence[dict[str, Any]], only: Sequence[str]) -> None:
    for r in rows:
        if r["label"] not in only:
            continue
        gold = ", ".join(f"{b} p{p}" for b, p in r["gold_pages"])
        print(f"\n[{r['label']}] {r['id']} {r['question']}")
        print(f"  정답 쪽: {gold}")
        if r["gold_chunk_id"]:
            print(
                f"  정답 청크 {r['gold_chunk_id']}: hybrid {_r(r['hybrid_rank'])}위"
                f" → rerank {r['rerank_rank']}위 (점수 {_s(r['gold_rerank_score'])})"
            )
        print("  리랭커 상위:")
        for t in r["rerank_top"]:
            mark = "*" if t["is_gold"] else " "
            print(
                f"  {mark}{t['rank']}. {t['chunk_id']} {t['rerank_score']:.3f}"
                f" (hybrid #{t['hybrid_rank']}) {t['text']}"
            )


# ---------- 실행 ----------


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


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Rerank vs Hybrid 질문별 진단")
    parser.add_argument("--domain", default="shipping")
    parser.add_argument("--difficulty", default="basic")
    parser.add_argument("--split", choices=SPLITS, default="development")
    parser.add_argument(
        "--probe-n", type=int, default=100, help="후보 밖 정답을 어디까지 찾아볼지"
    )
    parser.add_argument("--limit", type=int, default=None, help="앞에서 N개 질문만")
    parser.add_argument(
        "--only",
        nargs="*",
        default=["밀림", "후보 없음", "둘 다 놓침"],
        help="자세히 보여줄 분류 (아무것도 안 주면 상세 생략)",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--no-board",
        action="store_true",
        help="끝난 뒤 실험 비교판을 다시 만들지 않는다",
    )
    parser.add_argument(
        "--exp",
        default=None,
        help="실험 이름. 주면 results/experiments/<이름>/에 따로 저장",
    )
    add_settings_args(
        parser
    )  # --k는 몇 위까지를 맞힘으로 볼지, --rerank-n은 리랭커 후보 수
    args = parser.parse_args(argv)
    exp = check_exp_arg(parser, args.exp)
    s = settings_from_args(args)  # rerank_n < k 같은 잘못된 값은 여기서 멈춘다
    probe_n = max(args.probe_n, s.rerank_n)

    from rag_service.retrieval_b import build_hybrid_retriever
    from rag_service.retrieval_b.rerank import cross_encoder_scorer, rerank

    load_dotenv(repo_root() / ".env")
    items = load_golden(
        golden_path(args.split, domain=args.domain, difficulty=args.difficulty)
    )
    if args.limit:
        items = items[: args.limit]

    common = {
        "domain": args.domain,
        "difficulty": args.difficulty,
        "levels": list(s.levels),
        "candidate_k": s.candidate_k,
        "rrf_k": s.rrf_k,
        "bm25_source": s.bm25_source,
    }
    hybrid_k = max(COMPARE_HYBRID_K, s.k)
    hybrid = build_hybrid_retriever(k=hybrid_k, **common)
    cands = build_hybrid_retriever(k=s.rerank_n, **common)
    probe = build_hybrid_retriever(k=probe_n, **common)
    score_fn = cross_encoder_scorer(max_length=s.rerank_max_length)

    rows = []
    for n, item in enumerate(items, start=1):
        print(f"\r질문 {n}/{len(items)} 진단 중 ...", end="", flush=True)
        q = item["question"]
        candidates = cands.invoke(q)
        reranked = (
            rerank(q, candidates, k=len(candidates), score_fn=score_fn)
            if candidates
            else []
        )
        rows.append(
            diagnose_question(
                item, hybrid.invoke(q), candidates, probe.invoke(q), reranked, s.k
            )
        )
    print()

    summary = summarize(rows, s.k)
    print_table(rows, summary, s.k, s.rerank_n, probe_n)
    if args.only:
        print_details(rows, args.only)

    out = args.out or (
        results_base(exp)
        / OUT_DIR.relative_to("results")
        / f"rerank_{args.difficulty}_{args.split}_k{s.k}_n{s.rerank_n}{extra_tag(s)}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "settings": {
            **common,
            "split": args.split,
            "k": s.k,
            "rerank_n": s.rerank_n,
            "rerank_max_length": s.rerank_max_length,
            "probe_n": probe_n,
            "hybrid_k": hybrid_k,
            "limit": args.limit,
            "exp": exp,
        },
        "summary": summary,
        "items": rows,
    }
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {out}")
    if not args.no_board:
        refresh_board(exp)
    return out


if __name__ == "__main__":
    # Windows 터미널 기본(cp949)으로는 본문의 "–" 같은 글자를 못 찍는다
    sys.stdout.reconfigure(encoding="utf-8")
    main()
