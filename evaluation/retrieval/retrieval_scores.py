"""검색 평가: 검색이 맞는 문서·쪽을 찾았는지 Hit@k, Recall@k, MRR로 판정한다.

evaluation/langfuse_experiment.py의 Langfuse 평가기(retrieval_judgement)가 문항마다 score_retrieval을 부르고,
돌려준 점수를 Langfuse에 올린다. 판정 기준과 계산 방법은 이 파일 담당이 정한다.

지금은 빈 함수라 점수를 돌려주지 않는다 (Langfuse에 검색 점수가 안 올라가고, 오류도 나지 않는다).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

Record = dict[str, Any]
K = 5


def hit_at_k(
    rec: Mapping[str, Any], golden: Mapping[str, Any] | None, k: int = K
) -> float | None:
    """Hit@k. 지금은 계산하지 않음 (None)."""
    return None


def recall_at_k(
    rec: Mapping[str, Any], golden: Mapping[str, Any] | None, k: int = K
) -> float | None:
    """Recall@k. 지금은 계산하지 않음 (None)."""
    return None


def mrr_at_k(
    rec: Mapping[str, Any], golden: Mapping[str, Any] | None, k: int = K
) -> float | None:
    """MRR@k (문항 하나의 Reciprocal Rank). 지금은 계산하지 않음 (None)."""
    return None


def score_retrieval(
    rec: Mapping[str, Any], golden: Mapping[str, Any] | None, *, k: int = K
) -> list[Record]:
    """문항 하나의 검색 판정 점수 목록. 지금은 빈 목록 (Langfuse에 올릴 검색 점수 없음)."""
    return []
