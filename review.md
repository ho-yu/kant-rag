# code review

1. Starting with main.py 
``` python {cmd}

from dotenv import load_dotenv

load_dotenv()

"""
1.  .env 파일에 적어 둔 값을 환경 변수로 불러오는 함수이다. python-dotenv라는 라이브러리에 들어 있고, 보통 파일 맨 위에서 이렇게 가져온다. 

2. API 키나 비밀번호 같은 값을 코드에 직접 적으면, 코드를 git에 올리거나 남에게 보여줄 떄 키가 그대로 노출된다. 그래서 이런 값은 프로젝트 폴더의 .enc라는 별도 파일에 따로 적어 두고, 이 파일을 .gitignore에 넣어서 git 에 올라가지 않게 한다. 

3. 실행하면 .env 파일을 찾는다. 실행한 파일이 있는 폴더부터 시작해서 상위 폴더로 올라가며 찾는다. 그 다음 파일의 이름=값 줄을 하나씩 읽는다. 그 다음 읽은 값을 환경 변수에 등록한다. 그 다음 부터 import os / os.getenv() 를 사용해서 어디서든 꺼내 쓸 수 있다.
"""
from langfuse import get_client 

langfuse = get_client() 

"""

1. langfuse = get_client() 는 Langfuse에 기록을 보내는 "클라이언트" 객체를 가져와서 langfuse라는 변수에 담는 코드이다. 보통 파일 위쪽에 작성한다. 
    - langfuse 클라이언트: 우리 프로그램 안에서 Langfuse 서버에 기록을 보내 주는 담당자이다. 즉, 우리 프로그램 안에서 기록을 받아 적고 보관소로 보내는 객체이다. 
    - Langfuse 서버: 인터넷 어딘가에 있는 기록 보관소이다. 웹 화면에서 trace를 보는곳이다. 
    Ex. 우체국으로 비유

    Langfuse 서버 -> 우체국 (편지가 최종으로 도착하는곳)
    Langfuse 클라이언트 -> 우리 집 우체통 + 집배원
    기록(trace, span) -> 편지 
    API 키 -> 우리 집 주소와 인증서 

    1. 앞에서 load_dotenv()를 통해 .env의 키가 환경 변수로 먼저 올라가 있어야 get_client() 가 그걸 읽을 수 있다. 
    2. 환경 변수에서 접속 정보를 읽는다. LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, 그리고 서버 주소(LANGFUSE_HOST 또는 LANGFUSE_BASE_URL)예요.
    3. 그 정보로 Langfuse 서버와 통신할 클라이언트를 만들어서 돌려준다. 
    4. Langfuse는 기록을 하나 남길 때마다 바로 보내지 않고, 모아 두었다가 백그라운드에서 한꺼번에 보낸다. 그래서 프로그램이 너무 빨리 끝나면 마지막 기록이 전송되지 못하고 사라질 수도 있다. 이를 막을려고 보통 끝에서 이렇게 쓴다. 

    try: 
        ... # 실제 작업
    finally:
        langfuse.flush() # 남아 있는 기록을 서버로 전부 보내기 
    
    -> finally 라서 중간에 에러가 나도 flush() 는 꼭 실행된다. langfuse 변수를 미리 받아 두는 주된 이유가 바로 이것이다.

    5. 기록이 생길 때마다 바로 서버로 보내면, 보낼 때마다 인터넷 통신을 기다려야 해서 프로그램이 느려진다. 그래서 클라이언트는 기록을 우체통에 모아 뒀다가 몇 초마다 한꺼번에 보낸다. 

    6. 한 줄로 정리하면, 환경 변수의 키로 Langfuse 연결 객체를 (하나만) 준비해서, 나중에 기록 전송 등에 쓰려고 변수에 담아 두는 코드 예요. 

2. Langfuse는 LLM 앱이 실행되는 과정을 기록해 주는 도구이다. 어떤 질문이 들어왔고, 어떤 문서를 검색했고, 모델이 뭐라고 답했고, 시간이 얼마나 걸렸는지 등을 웹 화면에서 볼 수 있게 남겨준다. 이렇게 남긴 실행 기록 하나하나를 trace라고 부른다. 
"""

from rag_service.pipeline import create_chain
from rag_service.observability import trace_request, with_langfuse

try:
        domain="shipping"  
        difficulty="basic"  # "basic", "intermediate", "advancced"
        profile = "bge-m3"  # None = 난이도 기본 모델(bge-m3), "qwen3" = Qwen3 임베딩
        retrieval_mode = "hybrid_rerank"  # "naive", "hybrid", "hybrid_rerank"

        chain = with_langfuse(
            create_chain(domain=domain, difficulty=difficulty, profile=profile),
            domain=domain,
            difficulty=difficulty
        )

        question = "가온루트해운 부산–LA 안내에서 일반 건화물로 예약할 수 있는 컨테이너 종류는 무엇인가요?"
        with trace_request(question) as trace:
            result = chain.invoke({
                "question": question,
                "retrieval_mode": retrieval_mode
            })
            trace.update(output=result.model_dump())

        print(result)

    finally:
        langfuse.shutdown()

"""

1. try: / finally: - 끝에 꼭 할 일 보장하기 

try: 
        ... # 실제 작업
    finally:
        langfuse.shutdown()

- try 안에서 무슨일이 생기든 (정상 종료든 에러든) funally 안의 코드는 반드시 실행된다. 그래서 중간에 검색이나 GPT 호출이 실패해도 langfuse.shutdown() 이 실행되어 그때까지 쌓인 기록이 서버로 전송되고 클라이언트가 깔끔하게 닫힌다. 에러 직전까지의 기록이 남아 있어야 Langfuse 화면에서 "어디서 실패했는지" 볼 수 있으니 중요한 장치이다. 

2. 설정값 

domain="shipping"  # 문서 분야: 해운/ 금융 즉 주제 문서중 하나 고르기 
difficulty="basic"  # 난이도: 각 주제에는 basic/intermediate/advanced가 있음, 어떤 문서 묶음-Qdrant 컬렉션을 쓸지 
profile = "bge-m3"  # 모델 설정 묶음 이름
retrieval_mode = "hybrid_rerank"  # 검색방식: "naive", "hybrid", "hybrid_rerank"

3. 체인 만들기 - 함수 안에 함수 

chain = with_langfuse(
            create_chain(domain=domain, difficulty=difficulty, profile=profile),
            domain=domain,
            difficulty=difficulty
        )

- create_chain() 코드 먼저 분석하기: src/rag_service/pipeline/chain.py
"""
```



2. review chain.py to understand a function which is creat_chain()

``` python {cmd}

def create_chain(domain, difficulty, profile=None):
    model = ChatOpenAI(model="gpt-6-luna")

    retrievers = {
        "naive": build_naive_retriever(domain=domain, difficulty=difficulty, profile=profile),
        "hybrid": build_hybrid_retriever(domain=domain, difficulty=difficulty, profile=profile),
        "hybrid_rerank": build_rerank_retriever(domain=domain, difficulty=difficulty, profile=profile),
        }

    collection = collection_name(load_config(), domain=domain, difficulty=difficulty, profile=profile)  # page 확장에서 쪽을 꺼낼 컬렉션

    def retrieve_docs(request):
        """question 하나로 검색. sub_questions가 있으면 질문별로 검색해서 번갈아 합친다
        (여러 내용을 묻는 질문에서 한 내용의 자료가 검색 결과를 독차지하지 않게)."""
        search_question = request.get("question")
        mode = request.get("retrieval_mode", "naive")
        sub_questions = [q.strip() for q in request.get("sub_questions") or [] if q and q.strip()]
        if sub_questions:
            queries = list(dict.fromkeys([search_question, *sub_questions]))
            docs = merge_docs([retrievers[mode].invoke(q) for q in queries])
        else:
            docs = retrievers[mode].invoke(search_question)
        if request.get("page_context"):  # page 확장: 찾은 문단·표 → 그 쪽 전체 (같은 파일+page는 한 번만)
            docs = to_pages(docs, collection)
        return docs

    def retrieve_context(request):
        return format_docs(request["docs"])

    chain = (
            RunnablePassthrough.assign(
                docs=RunnableLambda(retrieve_docs)
            )
            | RunnablePassthrough.assign(
                context=RunnableLambda(retrieve_context)
            )
            | RunnablePassthrough.assign(
                answer=answer_prompt | model.with_structured_output(RagAnswer)
            )
            | RunnableLambda(
                lambda request: add_sources(request, domain=domain, difficulty=difficulty)
            )
        )

    return with_langfuse(chain, domain=domain, difficulty=difficulty)

"""

create_chain()은 "검색" -> "문맥정리" -> "GPT 답변" -> "출처 붙이기"를 하나로 이은 체인을 만들어 돌려주는 함수이다. 

1. 함수 정의롸 모델 준비
def create_chain(domain, difficulty, profile=None):
    model = ChatOpenAI(model="gpt-6-luna")

- profile = None은 기본값이 있는 매개변수이다. 호출할때 profile을 안 넘기면 None이 들어가고, 그 러면 난이도 이름으로 models.yaml을 찾는다. main.py는 profile = "bge-m3"을 넘겼으니 이 값이 쓰인다. 

- ChatOpenAI()는 LangChain에서 GPT를 부르는 객체이다. 여기서 만든 model 이 나중에 답변을 쓰는 역할을 한다. API키는 load_dotenv()로 올려 둔 OPENAI_API_KEY를 자동으로 읽는다. 


2. 검색기 세 개를 미리 만들어 딕셔너리에 담기 
retrievers = {
    "naive": build_naive_retriever(...),
    "hybrid": build_hybrid_retriever(...),
    "hybrid_rerank": build_rerank_retriever(...),
}

- 검색방식 이름을 키로, 실제 검색기 객체를 값으로 담은 딕셔너리이다. 나중에 retrievers["hybrid_rerank"] 처럼 이름으로 꺼낼 수 있다. 
- 키 이름이 "hybrid_rerank" 이다. 그래서 "rerank"를 넣으면 이 딕셔너리에 그런 키가 없어서 KeyError가 난다. 알려진 문제 1번이 바로 이 자리에서 생긴다. 
- 실제로는 하나만 쓰더라고 세 개를 모두 미리 만든다. 그래서 "naive"로만 실행해도 리랭커 모델까지 메모리에 올라가서 시작이 느리고 메모리를 더 쓴다. 대신 한 번 만든 체인으로 모드를 바꿔가며 실행할 수 있다는 장점이 있다. Streamlit 화면에서 검색방식을 바꿀 수 있는 것도 이 덕분이다. -> 이게 바로 trade-off : 속도가 느려지는 대신 모드를 바꿔가며 실행할 수 있음

3. page 확장용 컬랙션 이름 
collection = collection_name(load_config(), domain=..., difficulty=..., profile=...)

- load_config(): 설정 파일(yaml)을 읽어서 dict로 돌려준다. 컬렉션 이름 앞부분(prefix) 같은 값이 여기서 나온다. 
- collection_name(): 그 설정에 도메인, 난이도, 임베딩 모델 이름을 붙여서 Qdrant 컬랙션 이름을 만든다. 
- Qdrant에서 컬랙션은 DB의 "테이블" 같은거다. 도메인, 난이도, 모델마다 따로 있다. 
- load_config()로 설정을 읽고, 그걸로 Qdrant 컬랙션 이름(예: kant_shipping_basic_bge-m3)을 만들어 둔다. 주석처럼 page확장(찾은 문단이 들어 있는 쪽 전체를 꺼내오기)을 할 때 어느 컬렉션에서 꺼낼지 알아야 해서 미리 준비하는거다. 

Ex. 
1 단계: load_config()가 naive.yaml을 읽어서 dict로 돌려준다.

cfg = {
    "collection": {"prefix": "kant", "domain": "shipping", "difficulty": "basic", ...},
    "embedding": {...},
    "search": {"top_k": 5, ...},
}

2 단계: collection_name(cfg, domain="shipping", difficulty = "basic", profile = "qwen3") 안에서 일어나는 일 


4. 안쪽 함수 두개 (함수 중첩)
- 함수 안에 또 함수를 정의했다. 안쪽 함수는 바깥 함수의 변수(retrievers, collection, domain등)를 그대로 쓸 수 있다. 그래서 retrieve_docs가 request 하나만 받아도 검색기를 찾아 쓸 수 있다. 
"""
```