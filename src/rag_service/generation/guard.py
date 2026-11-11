"""검사 모델의 판단 결과를 실제 실행 경로와 사용자 안내로 변환하는 함수 모음.

프롬프트와 모델의 연결 및 검사 실행 순서는 pipeline/chain.py에서 구성.
이 파일에서는 모델 호출 없이 검사 결과의 해석과 실패 시 처리만 담당.
"""

import logging
import re

from rag_service.contracts.schemas import QueryDecision
from rag_service.generation.rag_config import max_sub_questions

logger = logging.getLogger(__name__)

# 가드가 block을 냈을 때 사용자에게 보여 줄 안내문 (클라이언트·API에서 공통으로 사용)
BLOCKED_QUESTION = "죄송합니다. 이 질문에는 답변할 수 없습니다. 업무 문서와 관련된 질문을 해 주세요."
BLOCKED_ANSWER = "죄송합니다. 답변에 안내할 수 없는 내용이 포함되어 표시하지 않았습니다. 질문을 바꿔 다시 시도해 주세요."


def apply_query_decision(state):
    """사전 판단 결과를 정리하여 다음 LangChain 분기에서 사용할 route 추가."""
    decision = state["decision"]
    if decision is None:
        # 검사 호출 실패를 정상 통과와 구분하여 기록하고 기존 원문 검색 정책 유지
        logger.warning("query guard failed; using original question")
        state["guard_errors"].append("query_guard")
        decision = QueryDecision(
            action="rag", category="ok", search_question=state["request"].question,
            sub_questions=[], reason="",
        )
    decision = decision.model_copy(update={
        "search_question": decision.search_question.strip() or state["request"].question,
        "sub_questions": [q.strip() for q in decision.sub_questions if q.strip()][: max_sub_questions()],
    })
    return {**state, "decision": decision, "route": decision.action}


def blocked_question_result(state):
    """입력 차단 시 모델을 추가 호출하지 않고 반환할 안내문 구성."""
    return {"answer": BLOCKED_QUESTION, "status": "blocked", "reason": state["decision"].reason}


def apply_answer_check(state):
    """사후 검사 결과에 따른 기존 답변 유지 또는 차단 안내 교체."""
    check = state["check"]
    if check is None:
        logger.warning("answer guard failed; retaining generated answer")
        state["guard_errors"].append("answer_guard")
    elif check.verdict == "block":
        # 차단한 답변이 출처나 문맥 항목을 통해 다시 전달되지 않도록 결과 전체 교체
        return {**state, "route": "blocked_answer", "result": {
            "answer": BLOCKED_ANSWER, "status": "blocked", "reason": check.reason,
            "source_ids": [], "sources": [], "contexts": [], "invalid_source_ids": [],
        }}
    return state


# 사후 검사 기본값: LLM 호출 없이 답변 본문을 규칙으로만 확인 (보안 규칙 자체는 prompt.answer_prompt에 있음)
#   업무 문서에는 아래 형식이 나오지 않는 것을 확인함 (basic~advanced 청크 전체에서 0건)
ANSWER_RULES = [
    ("leak", re.compile(r"sk-[A-Za-z0-9_-]{16,}")),                                      # OpenAI 등 API 키
    ("leak", re.compile(r"(?i)(api[_ -]?key|secret|password|비밀번호)\s*[:=]\s*\S{6,}")),
    ("leak", re.compile(r"\[검색 자료를 읽는 규칙\]|\[답변에 넣지 않는 내용\]")),            # 시스템 프롬프트 노출
    ("personal_info", re.compile(r"(?<!\d)\d{6}-[1-4]\d{6}(?!\d)")),                     # 주민등록번호
    ("personal_info", re.compile(r"(?<!\d)01[016789]-?\d{3,4}-?\d{4}(?!\d)")),           # 휴대전화
]


def check_answer_rules(answer: str) -> str | None:
    """답변 본문이 규칙에 걸리면 이유(category), 아니면 None."""
    for category, pattern in ANSWER_RULES:
        if pattern.search(answer or ""):
            return category
    return None


def apply_answer_rules(state):
    """규칙 검사에 걸린 답변을 차단 안내로 교체 (LLM 사후 검사 대신 쓰는 기본 경로)."""
    category = check_answer_rules(state["result"].get("answer", ""))
    if category is None:
        return state
    logger.warning("answer blocked by rule: %s", category)
    return {**state, "route": "blocked_answer", "result": {
        "answer": BLOCKED_ANSWER, "status": "blocked", "reason": f"규칙 검사: {category}",
        "source_ids": [], "sources": [], "contexts": [], "invalid_source_ids": [],
    }}
