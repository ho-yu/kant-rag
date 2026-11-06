import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langfuse import get_client

from rag_service.observability import trace_request, with_langfuse


def main():
    load_dotenv()

    langfuse = get_client()

    try:
        # 1. Langfuse 서버 주소와 API 키 확인
        if not langfuse.auth_check():
            raise RuntimeError("Langfuse 연결 또는 API 키를 확인해주세요.")

        # 2. 테스트할 체인 생성
        prompt = ChatPromptTemplate.from_messages([
            ("human", "{question}"),
        ])

        model = ChatOpenAI(model="gpt-6-luna")

        chain = prompt | model | StrOutputParser()

        # 3. 체인에 Langfuse 추적 설정 연결
        chain_with_langfuse = with_langfuse(
            chain,
            domain="shipping",
            difficulty="basic",
        )

        question = "안녕!"

        # 4. 질문 한 건의 부모 기록 안에서 체인 실행
        with trace_request(question) as trace:
            result = chain_with_langfuse.invoke({
                "question": question,
            })

            # 부모 기록에도 최종 답변 저장
            trace.update(output=result)

        print(result)

    finally:
        # 5. 남은 기록을 전송하고 클라이언트 종료
        langfuse.shutdown()


if __name__ == "__main__":
    main()