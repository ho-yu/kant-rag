import os
from contextlib import contextmanager
from langfuse.langchain import CallbackHandler
from langfuse import get_client as get_langfuse_client

def with_langfuse(chain, *, domain, difficulty):
    """Langfuse 추적 설정을 붙인 새 체인을 반환"""
    handler = CallbackHandler()

    return chain.with_config(
        callbacks=[handler], run_name="rag-answer",
        metadata={"domain": domain, "difficulty": difficulty}
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