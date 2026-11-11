import logging
import os
import sys
from contextlib import contextmanager

from langfuse import get_client as get_langfuse_client
from langfuse.langchain import CallbackHandler

logger = logging.getLogger(__name__)


@contextmanager
def optional_trace(question: str, request_id: str):
    """실행 기록 장애와 답변 처리의 분리 및 요청 재실행 방지."""
    manager, trace = None, None
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        try:
            manager = trace_request(question)
            trace = manager.__enter__()
        except Exception:
            manager = None
            logger.exception("trace start failed request_id=%s", request_id)
    try:
        # 답변 실행은 한 번만 진행하며 실제 처리 오류는 호출자에게 전파
        yield trace
    finally:
        if manager is not None:
            try:
                manager.__exit__(*sys.exc_info())
            except Exception:
                logger.exception("trace finish failed request_id=%s", request_id)


def update_trace(trace, **kwargs):
    """추적 정보 전송 실패의 별도 기록 및 사용자 응답 유지."""
    if trace is not None:
        try:
            trace.update(**kwargs)
        except Exception:
            logger.exception("trace update failed")

def with_langfuse(chain, *, domain=None, difficulty=None, run_name="rag-answer"):
    """Langfuse 추적 설정을 붙인 새 체인을 반환"""
    handler = CallbackHandler()

    return chain.with_config(
        callbacks=[handler], run_name=run_name,
        metadata={k: v for k, v in {"domain": domain, "difficulty": difficulty}.items() if v is not None}
    )

def create_traced_openai_client():
    """Langfuse 추적이 적용된 OpenAI 클라이언트를 반환"""

    # Langfuse의 OpenAI 래퍼는 프로세스 안의 OpenAI SDK 메서드를 감싸므로,
    # LangChain 내부의 OpenAI 호출에도 영향을 줄 수 있음
    # # 중복 기록을 줄이기 위한 설정
    os.environ.setdefault("LANGFUSE_OPENAI_SKIP_RAW_RESPONSES", "True")

    # 이 함수를 실제 호출할 때 래퍼가 적용되도록 하기 위함
    from langfuse.openai import OpenAI
    return OpenAI()


@contextmanager
def trace_request(question):
    """질문 한 건의 처리 과정을 하나의 실행 기록으로 관리"""
    langfuse = get_langfuse_client()

    with langfuse.start_as_current_observation(
        as_type="span",
        name="user-question",
        input={"question": question}
    ) as span:
        yield span
