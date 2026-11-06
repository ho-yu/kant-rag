from dotenv import load_dotenv
from langfuse import get_client

from rag_service.pipeline import create_chain
from rag_service.observability import trace_request, with_langfuse

def main():
    load_dotenv()
    langfuse = get_client()

    try:
        domain="shipping"   # TODO 실제 사용자의 옵션 선택 변수로 대체
        difficulty="basic"  # TODO 실제 사용자의 옵션 선택 변수로 대체
        retrieval_mode = "naive"  # TODO 실제 사용자의 옵션 선택 변수로 대체
        
        chain = with_langfuse(
            create_chain(domain=domain, difficulty=difficulty),
            domain=domain,
            difficulty=difficulty
        )

        question = "부산에서 LA로 보내는 일반 건화물의 서류 제출 마감은 언제인가요?" # TODO 예시

        with trace_request(question) as trace:
            result = chain.invoke({
                "question": question,
                "retrieval_mode": retrieval_mode
            })
            trace.update(output=result.model_dump())

        print(result)

    finally:
        langfuse.shutdown()


if __name__ == "__main__":
    main()