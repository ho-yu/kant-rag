
# score(name, value, state, cause, reason): 점수 dict 하나를 만들어요. 답변 평가의 quality.score()와 이름, 인자, 결과가 똑같아요. 테스트로 두 함수 결과가 같은지도 확인해요
# SCORED, NA, UNAVAILABLE : 점수 상태 이름
# SERVICE_ERROR, NO_GOLDEN : 두 평가가 같이 쓰는 사유 이름
# service_error_reason(rec) : 서비스 오류 안내 문구 
from __future__ import annotations

from typing import Any

Record = dict[str, Any]

SCORED = "scored"
NA = "na"
UNAVAILABLE = "unavailable"
STATES = (SCORED, NA, UNAVAILABLE)

# 두 평가가 같이 쓰는 사유 이름
SERVICE_ERROR = "service_error"
NO_GOLDEN = "no_golden"


def score(
    name: str,
    value: float | None,
    state: str,
    cause: str | None = None,
    reason: str = "",
) -> Record:
    """점수 dict 하나. scored일 때만 value가 있다 (quality.score와 같은 함수)."""
    if state not in STATES:
        raise ValueError(f"state는 {STATES} 중 하나여야 합니다: {state!r}")
    return {
        "name": name,
        "value": None if state != SCORED else float(value),
        "state": state,
        "cause": cause,
        "reason": reason,
    }


def service_error_reason(rec: Record) -> str:
    return f"서비스 오류로 답변이 없음 ({rec.get('error') or '이유 없음'})"
