"""검색 평가: 검색이 참고 근거 문서를 찾았는지 Hit@k, Recall@k, MRR로 판정한다.

evaluation/langfuse_experiment.py의 Langfuse 평가기(retrieval_judgement)가 문항마다 score_retrieval을 부르고,
돌려준 점수를 Langfuse에 올린다. 점수 dict 모양과 상태는 evaluation/scoring.py (답변 평가와 같음).

받는 값
    rec     답변 기록 한 건. contexts = 모델이 실제로 받은 최종 문맥 [{sid, book_id, page, chunk_id, text}] (S1부터 순서대로)
    golden  정답 파일 한 항목 (Langfuse 데이터셋의 expected_output). evidence_groups만 쓴다

계산 (발제 STEP 6 "검색 지표")
- 참고 문서: evidence_groups에 등록된 모든 book_id를 모아 중복을 뺀 목록
- 찾은 문서: 최종 문맥 상위 k개의 book_id. 같은 문서의 청크가 여러 개여도 한 번만 센다
- Hit@k     참고 문서를 하나라도 찾았으면 1, 없으면 0
- Recall@k  찾은 참고 문서 수 / 전체 참고 문서 수
- MRR@k     참고 문서가 처음 나온 순위 r의 1/r, 상위 k개에 없으면 0

상태
| 상황 | 상태 (cause) |
|---|---|
| 서비스 오류로 검색 결과를 알 수 없음 | 평가 불가 (service_error) |
| 정답 파일에 없는 문항 | 평가 불가 (no_golden) |
| evidence_groups가 비어 있음 (보류·되묻기 문항 등) | 해당 없음 (no_evidence) |
| 검색 결과가 비었지만 참고 문서는 있음 | 0점 (제외하지 않는다) |
"""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _load_scoring():
    path = Path(__file__).resolve().parents[1] / "scoring.py"
    spec = importlib.util.spec_from_file_location("evaluation_scoring", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scoring = _load_scoring()

Record = scoring.Record
K = 5
METRICS = ("hit", "recall", "mrr")
NO_EVIDENCE = "no_evidence"


def metric_name(metric: str, k: int = K) -> str:
    """Langfuse 점수 이름: hit_at_5, recall_at_5, mrr_at_5."""
    return f"{metric}_at_{k}"


def _norm(book_id: Any) -> str:
    return str(book_id or "").strip().upper()


def gold_docs(golden: Mapping[str, Any] | None) -> list[str]:
    """evidence_groups의 모든 book_id (처음 나온 순서, 중복 제거)."""
    out: list[str] = []
    for group in (golden or {}).get("evidence_groups") or []:
        for ev in group or []:
            doc = _norm(ev.get("book_id"))
            if doc and doc not in out:
                out.append(doc)
    return out


def found_docs(rec: Mapping[str, Any], k: int = K) -> list[str]:
    """최종 문맥 상위 k개의 book_id (처음 나온 순서, 중복 제거)."""
    out: list[str] = []
    for ctx in (rec.get("contexts") or [])[:k]:
        doc = _norm(ctx.get("book_id"))
        if doc and doc not in out:
            out.append(doc)
    return out


def precheck(rec: Mapping[str, Any], golden: Mapping[str, Any] | None) -> tuple[str, str, str] | None:
    """채점할 수 없으면 (state, cause, reason). 채점할 수 있으면 None."""
    if rec.get("status") == "error":
        return scoring.UNAVAILABLE, scoring.SERVICE_ERROR, scoring.service_error_reason(rec)
    if golden is None:
        return scoring.UNAVAILABLE, scoring.NO_GOLDEN, "정답 파일에 없는 문항"
    if not gold_docs(golden):
        return scoring.NA, NO_EVIDENCE, "evidence_groups에 참고 문서가 없음"
    return None


def hit_at_k(rec: Mapping[str, Any], golden: Mapping[str, Any] | None, k: int = K) -> float | None:
    """Hit@k. 채점할 수 없으면 None."""
    if precheck(rec, golden):
        return None
    return 1.0 if set(found_docs(rec, k)) & set(gold_docs(golden)) else 0.0


def recall_at_k(rec: Mapping[str, Any], golden: Mapping[str, Any] | None, k: int = K) -> float | None:
    """Recall@k. 채점할 수 없으면 None."""
    if precheck(rec, golden):
        return None
    gold = set(gold_docs(golden))
    return len(set(found_docs(rec, k)) & gold) / len(gold)


def mrr_at_k(rec: Mapping[str, Any], golden: Mapping[str, Any] | None, k: int = K) -> float | None:
    """MRR@k (문항 하나의 Reciprocal Rank). 채점할 수 없으면 None."""
    if precheck(rec, golden):
        return None
    gold = set(gold_docs(golden))
    for rank, ctx in enumerate((rec.get("contexts") or [])[:k], start=1):
        if _norm(ctx.get("book_id")) in gold:
            return 1.0 / rank
    return 0.0


def score_retrieval(
    rec: Mapping[str, Any], golden: Mapping[str, Any] | None, *, k: int = K
) -> list[Record]:
    """문항 하나의 검색 점수 목록 (hit_at_k, recall_at_k, mrr_at_k)."""
    names = [metric_name(m, k) for m in METRICS]
    skip = precheck(rec, golden)
    if skip:
        state, cause, reason = skip
        return [scoring.score(n, None, state, cause, reason) for n in names]
    gold, found = gold_docs(golden), found_docs(rec, k)
    hit = [d for d in gold if d in found]
    reason = f"참고 문서 {gold} 중 {hit or '없음'} 찾음 (상위 {k}개 문서 {found or '없음'})"
    values = (hit_at_k(rec, golden, k), recall_at_k(rec, golden, k), mrr_at_k(rec, golden, k))
    return [scoring.score(n, v, scoring.SCORED, None, reason) for n, v in zip(names, values, strict=True)]
