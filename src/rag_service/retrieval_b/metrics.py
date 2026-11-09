"""검색 평가: 골든셋(evidence_groups)으로 Hit@k와 Recall@k를 계산한다.

정답 단위는 두 가지로 센다.
- 문서 기준: book_id (예: SB01)
- 쪽 기준: (book_id, page) (예: ("SB01", 1)). 같은 쪽의 청크가 여러 개 나와도 한 번만 센다.

Recall은 두 방식으로 센다. 대표 값은 팀과 정한다.
- 합집합 방식(recall): 모든 근거 후보를 정답 집합 하나로 보고, 그중 몇 개를 찾았는지
- 묶음 방식(group_recall): evidence_groups의 묶음 하나는 "이 중 하나면 충분"(OR)이다.
  묶음마다 후보를 하나라도 찾으면 그 묶음을 채운 것으로 보고, 채운 묶음의 비율을 센다.

MRR(doc_mrr, page_mrr): 상위 k개 안에서 처음 나온 정답의 순위 r로 1/r을 내고 평균한다.
Hit는 "있나"만 보고, MRR은 "얼마나 위에 있나"까지 본다 (k 안에 없으면 0).
eval_retrieval.py의 MRR은 "모든 묶음을 찾은 순위" 기준이라 값이 다를 수 있다.

주의: 튜터 답변(11/6)대로 골든셋에 없는 쪽이라도 실제 근거면 정답이다.
그래서 이 점수는 검색 성능을 낮게 볼 수 있다. 답변 단위 판정(LLM 채점)과 같이 본다.

평가 JSON(정답·quote)은 채점에만 쓴다. 검색기에는 question만 넘긴다.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from statistics import mean, median
from typing import Any

from langchain_core.documents import Document

from rag_service.storage.corpus import repo_root

SPLITS = ("development", "additional", "expansion_validation")
DEFAULT_KS = (1, 3, 5, 10)
METRICS = (
    "doc_hit",
    "doc_recall",
    "doc_group_recall",
    "page_hit",
    "page_recall",
    "page_group_recall",
    "doc_mrr",
    "page_mrr",
)

Page = tuple[str, int]


# ---------- 골든셋 읽기 ----------


def golden_path(
    split: str, *, domain: str = "shipping", difficulty: str = "basic"
) -> Path:
    """data/{난이도}/{주제}/evaluation/{split}.json (11/9 팀 main에서 intermediate → evaluation)"""
    if split not in SPLITS:
        raise ValueError(f"split은 {SPLITS} 중 하나여야 합니다: {split!r}")
    return repo_root() / "data" / difficulty / domain / "evaluation" / f"{split}.json"


def load_golden(path: str | Path) -> list[dict[str, Any]]:
    """골든셋 JSON(목록 하나)을 읽고, 채점에 필요한 키가 있는지 검사한다."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise TypeError(f"{path.name}: 질문 목록(list)이어야 합니다")
    seen: set[str] = set()
    for n, item in enumerate(items, start=1):
        for key in ("id", "question", "evidence_groups"):
            if key not in item:
                raise ValueError(f"{path.name} {n}번째 항목: {key!r}가 없습니다")
        if item["id"] in seen:
            raise ValueError(f"{path.name}: id가 겹칩니다: {item['id']!r}")
        seen.add(item["id"])
        for group in item["evidence_groups"]:
            for ev in group:
                if not ev.get("book_id") or not isinstance(ev.get("page"), int):
                    raise ValueError(
                        f"{path.name} {item['id']}: 근거에 book_id와 정수 page가 필요합니다"
                    )
    return items


# ---------- 정답과 검색 결과를 집합으로 ----------


def doc_groups(item: dict[str, Any]) -> list[set[str]]:
    return [
        {ev["book_id"] for ev in group} for group in item["evidence_groups"] if group
    ]


def page_groups(item: dict[str, Any]) -> list[set[Page]]:
    return [
        {(ev["book_id"], int(ev["page"])) for ev in group}
        for group in item["evidence_groups"]
        if group
    ]


def gold_docs(item: dict[str, Any]) -> set[str]:
    return set().union(*doc_groups(item))


def gold_pages(item: dict[str, Any]) -> set[Page]:
    return set().union(*page_groups(item))


def found_docs(docs: Iterable[Document]) -> set[str]:
    return {d.metadata["book_id"] for d in docs if d.metadata.get("book_id")}


def found_pages(docs: Iterable[Document]) -> set[Page]:
    return {
        (d.metadata["book_id"], int(d.metadata["page"]))
        for d in docs
        if d.metadata.get("book_id") and d.metadata.get("page") is not None
    }


# ---------- 점수 계산 ----------


def hit_recall(found: set, gold: set) -> tuple[int, float] | None:
    """Hit = 정답을 하나라도 찾았는지(0/1), Recall = 찾은 정답 수 / 정답 수.

    정답이 비어 있으면 None (평균에서 뺀다). 검색 결과가 비면 (0, 0.0).
    """
    if not gold:
        return None
    hit_count = len(found & gold)
    return int(hit_count > 0), hit_count / len(gold)


def group_recall(found: set, groups: Sequence[set]) -> float | None:
    """묶음마다 후보를 하나라도 찾았으면 채운 것으로 보고, 채운 묶음 비율을 돌려준다."""
    if not groups:
        return None
    return sum(1 for g in groups if found & g) / len(groups)


def reciprocal_rank(keys: Sequence[Any], gold: set) -> float | None:
    """처음 나온 정답의 순위 r로 1/r을 돌려준다 (1등 1.0, 2등 0.5, 없으면 0.0).

    keys는 상위 k개 청크를 순서대로 문서·쪽 키로 바꾼 목록이다. 정답이 비어 있으면 None.
    """
    if not gold:
        return None
    for rank, key in enumerate(keys, start=1):
        if key in gold:
            return 1 / rank
    return 0.0


def _doc_key(d: Document) -> str | None:
    return d.metadata.get("book_id") or None


def _page_key(d: Document) -> Page | None:
    m = d.metadata
    if not m.get("book_id") or m.get("page") is None:
        return None
    return (m["book_id"], int(m["page"]))


def score_question(
    item: dict[str, Any], docs: Sequence[Document], k: int
) -> dict[str, Any]:
    """질문 하나의 상위 k개를 채점한다. 청크를 먼저 k개로 자른 뒤 집합으로 바꾼다."""
    if k < 1:
        raise ValueError(f"k는 1 이상이어야 합니다: {k}")
    top = list(docs)[:k]
    f_docs, f_pages = found_docs(top), found_pages(top)
    g_pages = gold_pages(item)

    doc_hr = hit_recall(f_docs, gold_docs(item))
    page_hr = hit_recall(f_pages, g_pages)
    return {
        "doc_hit": None if doc_hr is None else doc_hr[0],
        "doc_recall": None if doc_hr is None else doc_hr[1],
        "doc_group_recall": group_recall(f_docs, doc_groups(item)),
        "page_hit": None if page_hr is None else page_hr[0],
        "page_recall": None if page_hr is None else page_hr[1],
        "page_group_recall": group_recall(f_pages, page_groups(item)),
        "doc_mrr": reciprocal_rank([_doc_key(d) for d in top], gold_docs(item)),
        "page_mrr": reciprocal_rank([_page_key(d) for d in top], g_pages),
        "found_chunk_ids": [d.metadata.get("chunk_id") for d in top],
        "missing_pages": sorted([list(p) for p in g_pages - f_pages]),
    }


def summarize(rows: Sequence[dict[str, Any]], ks: Sequence[int]) -> dict[str, Any]:
    """k별 평균. None인 질문(정답 없음)은 빼고, 평균에 들어간 질문 수 n을 같이 적는다."""
    out: dict[str, Any] = {}
    for k in ks:
        per_k = [r["scores"][str(k)] for r in rows]
        stats: dict[str, Any] = {}
        for name in METRICS:
            values = [s[name] for s in per_k if s[name] is not None]
            stats[name] = round(mean(values), 4) if values else None
        stats["n"] = sum(1 for s in per_k if s["page_hit"] is not None)
        out[str(k)] = stats
    return out


def evaluate(
    retriever: Any,
    items: Sequence[dict[str, Any]],
    ks: Sequence[int] = DEFAULT_KS,
) -> dict[str, Any]:
    """검색기(.invoke(질문) → Document 목록)를 골든셋으로 채점한다.

    질문마다 검색은 한 번만 하고, 그 결과를 앞에서 k개씩 잘라 모든 k를 채점한다.
    그래서 검색기는 max(ks)개 이상을 돌려주도록 만들어서 넘긴다.
    검색이 실패한 질문은 빈 결과(0점)로 채점하고 error에 이유를 남긴다.
    """
    ks = sorted(set(ks))
    if not ks or ks[0] < 1:
        raise ValueError(f"ks는 1 이상의 정수여야 합니다: {ks}")

    rows: list[dict[str, Any]] = []
    for item in items:
        error = None
        started = time.perf_counter()
        try:
            docs = list(retriever.invoke(item["question"]))
        except Exception as e:  # noqa: BLE001 — 한 질문의 실패로 전체 평가를 멈추지 않는다
            docs, error = [], f"{type(e).__name__}: {e}"
        latency_ms = (time.perf_counter() - started) * 1000
        rows.append(
            {
                "id": item["id"],
                "expected_action": item.get("expected_action"),
                "n_groups": len(page_groups(item)),
                "n_returned": len(docs),
                "latency_ms": round(latency_ms, 1),
                "error": error,
                "scores": {str(k): score_question(item, docs, k) for k in ks},
            }
        )

    latencies = [r["latency_ms"] for r in rows if r["error"] is None]
    answer_rows = [r for r in rows if r["expected_action"] == "answer"]
    return {
        "ks": ks,
        "n_questions": len(rows),
        "n_errors": sum(1 for r in rows if r["error"]),
        "short_results": sum(1 for r in rows if r["n_returned"] < ks[-1]),
        "latency_ms": {
            "mean": round(mean(latencies), 1) if latencies else None,
            "median": round(median(latencies), 1) if latencies else None,
            "max": round(max(latencies), 1) if latencies else None,
        },
        "summary": summarize(rows, ks),
        "summary_answer_only": summarize(answer_rows, ks),
        "items": rows,
    }
