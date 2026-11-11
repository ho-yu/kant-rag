"""RAG와 직접 답변 및 검사 모델의 공통 생성 설정."""

import os

from langchain_openai import ChatOpenAI


def create_model(role="answer"):
    """답변 모델 기본값 공유 및 검사 모델만 선택적으로 재정의."""
    answer_model = os.getenv("OPENAI_MODEL") or "gpt-6-luna"
    # 검사 전용 모델 미지정 시 RAG와 직접 답변에 사용하는 모델명 공유
    model = (os.getenv("RAG_GUARD_MODEL") or answer_model) if role == "guard" else answer_model
    return ChatOpenAI(
        model=model,
        reasoning_effort=os.getenv("RAG_REASONING_EFFORT") or "low",
        timeout=float(os.getenv("RAG_MODEL_TIMEOUT") or "120"),  # 모델 요청의 응답 대기 제한 단위 초
        max_retries=int(os.getenv("RAG_MODEL_RETRIES") or "1"),  # 최초 호출 실패 후 추가 시도 횟수
        max_tokens=int(os.getenv("RAG_OUTPUT_TOKENS") or "4096"),  # 검색 문맥과 별개인 모델 출력 토큰 상한
    )
