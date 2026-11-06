from dotenv import load_dotenv
from rag_service.pipeline.chain import create_chain

def main():
    load_dotenv()


    domain="shipping"   # TODO 실제 사용자의 옵션 선택 변수로 대체
    difficulty="basic"  # TODO 실제 사용자의 옵션 선택 변수로 대체
    retrieval_mode = "naive"  # TODO naive / hybrid / hybrid_rerank

    chain = create_chain(domain=domain, difficulty=difficulty)

    # TODO 실제 서비스에 맞게 변경 필요
    question = "부산에서 LA로 보내는 일반 건화물의 서류 제출 마감은 언제인가요?"

    result = chain.invoke({
        "question": question,
        "retrieval_mode": retrieval_mode
    })

    print(result)

if __name__ == "__main__":
    main()