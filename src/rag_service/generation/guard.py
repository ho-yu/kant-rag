"""클라이언트용 가드: 질문 사전 판단과 답변 사후 검사 (RAG 체인과는 별개).

사용 예 (client.py)
    query_guard = create_query_guard()
    decision = query_guard.invoke({"domain": "해운 업무", "history": "(없음)", "question": "…"})
    # decision.action: "rag" | "direct" | "block", decision.search_question: 검색용 질문

    answer_guard = create_answer_guard()
    check = answer_guard.invoke({"question": "…", "answer": "…"})
    # check.verdict: "pass" | "block"
"""
from langchain_openai import ChatOpenAI

from rag_service.contracts.schemas import AnswerCheck, QueryDecision
from rag_service.generation.prompt import answer_guard_prompt, query_guard_prompt

# 가드가 block을 냈을 때 사용자에게 보여 줄 안내문 (클라이언트·API에서 공통으로 사용)
BLOCKED_QUESTION = "죄송합니다. 이 질문에는 답변할 수 없습니다. 업무 문서와 관련된 질문을 해 주세요."
BLOCKED_ANSWER = "죄송합니다. 답변에 안내할 수 없는 내용이 포함되어 표시하지 않았습니다. 질문을 바꿔 다시 시도해 주세요."


def create_query_guard():
    """질문 사전 판단. invoke({"domain", "history", "question"}) → QueryDecision"""
    return query_guard_prompt | ChatOpenAI(model="gpt-6-luna").with_structured_output(QueryDecision)


def create_answer_guard():
    """답변 사후 검사. invoke({"question", "answer"}) → AnswerCheck"""
    return answer_guard_prompt | ChatOpenAI(model="gpt-6-luna").with_structured_output(AnswerCheck)
