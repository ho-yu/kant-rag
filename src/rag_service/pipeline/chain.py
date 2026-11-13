"""개별 처리 블록을 연결하는 RAG 하위 체인과 전체 질문 처리 체인 구성."""

import os
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough

from rag_service.contracts.schemas import AnswerCheck, QueryDecision, RagAnswer
from rag_service.generation.guard import (
    apply_answer_check,
    apply_answer_rules,
    apply_query_decision,
    blocked_question_result,
)
from rag_service.generation.models import create_model
from rag_service.generation.prompt import (
    answer_guard_prompt,
    answer_prompt,
    prepare_answer_input,
    prepare_direct_input,
    prepare_query_input,
    query_guard_prompt,
)
from rag_service.generation.rag_config import max_docs
from rag_service.generation.sources import add_sources, build_answer_result
from rag_service.observability import with_langfuse
from rag_service.pipeline.prepare_retriever import RETRIEVAL_LOCK, RetrievalBPool, RetrieverPool
from rag_service.retrieval_a import (
    build_hybrid_retriever,
    build_naive_retriever,
    build_rerank_retriever,
)
from rag_service.retrieval_a.naive import collection_name, load_config
from rag_service.storage.qdrant_store import get_pages

MAX_DOCS = max_docs()  # 나눠 찾기(sub_questions)로 합친 검색 결과를 LLM에 넘기는 최대 개수 (config/generation/rag.yaml)
IMPLS = ("retrieval_a", "retrieval_b")  # 동일한 체인에서 선택 가능한 검색 구현
# 서비스(main.py)가 쓰는 검색 구현과 난이도별 최종 문맥 수 k (k 실험 결과, 2026-11-13)
SERVICE_IMPL = "retrieval_b"
SERVICE_TOP_K = {"basic": 10, "intermediate": 5, "advanced": 8}


def format_docs(docs: list[Document]) -> str:
    """자료 선택과 축약 없이 검색 원문에 인용 번호와 문서 위치를 붙이는 기본 포맷."""
    return "\n\n".join(
        f"[S{i}] {doc.metadata['book_id']}.pdf / page {doc.metadata['page']}\n{doc.page_content}"
        for i, doc in enumerate(docs, start=1)
    )


def merge_docs(results: list[list[Document]], limit: int = MAX_DOCS) -> list[Document]:
    """질문별 검색 결과를 순위대로 번갈아 합친다 (각 질문의 1위 → 2위 …). 같은 청크는 한 번만."""
    merged, seen = [], set()
    for rank in range(max((len(r) for r in results), default=0)):
        for docs in results:
            if rank < len(docs):
                doc = docs[rank]
                key = doc.metadata.get("chunk_id") or doc.page_content
                if key not in seen:
                    seen.add(key)
                    merged.append(doc)
    return merged[:limit]


def to_pages(docs: list[Document], collection: str) -> list[Document]:
    """page 확장: 찾은 문단·표 청크 → 그 쪽(page) 전체 원문으로 바꾼다.
    같은 파일+page는 한 번만 (처음 나온 순서), 쪽을 못 찾으면 원래 청크 그대로."""
    groups: dict[str, list[Document]] = {}  # 쪽 chunk_id(예: SB11-p5) → 그 쪽에서 찾은 청크들
    for doc in docs:
        groups.setdefault(f"{doc.metadata['book_id']}-p{doc.metadata['page']}", []).append(doc)
    pages = get_pages(collection, list(groups))
    out = []
    for page_id, hits in groups.items():
        payload = pages.get(page_id)
        if payload is None:
            out.extend(hits)
            continue
        meta = {**payload.get("metadata", {}), "hit_chunk_ids": [h.metadata.get("chunk_id") for h in hits]}
        out.append(Document(page_content=payload.get("page_content", ""), metadata=meta))
    return out


def retriever_builders(impl="retrieval_a"):
    """검색 구현별 naive와 hybrid 및 hybrid_rerank 생성 함수 반환."""
    if impl == "retrieval_a":
        return build_naive_retriever, build_hybrid_retriever, build_rerank_retriever
    if impl == "retrieval_b":
        from rag_service import retrieval_b

        return retrieval_b.build_naive_retriever, retrieval_b.build_hybrid_retriever, retrieval_b.build_rerank_retriever
    raise ValueError(f"impl은 {IMPLS} 중 하나여야 합니다: {impl!r}")


def create_chain(
    domain, difficulty, profile=None, impl="retrieval_a", *,
    include_context=False, trace=True, retriever_pool=None, top_k=None,
):
    """기본 RagAnswer 반환 및 API 요청 시 실제 모델 입력 자료의 추가 반환."""
    # 검색 구현 검증 및 객체 생성 전 검색 방식별 생성 함수 보관
    builders = dict(zip(("naive", "hybrid", "hybrid_rerank"), retriever_builders(impl), strict=True))
    # 기존 검색 후 답변 기능의 독립 실행 유지 및 전체 서비스 체인의 RAG 분기로 재사용
    answer_chain = RunnableLambda(prepare_answer_input) | answer_prompt | create_model().with_structured_output(RagAnswer)
    # A의 Hybrid 공유 유지 및 B의 선택한 검색기만 최초 호출 시 생성
    # 서버에서 사전 준비한 검색 자원 주입 및 기존 독립 호출의 자체 준비 유지
    # top_k: 답변에 넣을 최종 문맥 수 (None이면 설정 파일·Settings 기본값). k 비교 실험용
    pool = retriever_pool or (RetrieverPool(domain, difficulty, profile, top_k=top_k) if impl == "retrieval_a" else None)
    retrievers = {}

    def retrieve_docs(request):
        """선택한 검색기만 호출 및 기존 하위 질문별 결과 병합 유지."""
        # ainvoke에서는 이 동기 함수가 작업 스레드에서 실행되므로 이벤트 루프 대기 방지
        # 기존 검색기의 공유 모델과 캐시 파일 보호를 위해 검색 구간만 순차 실행
        with RETRIEVAL_LOCK:
            search_question = request.get("search_question") or request.get("question")
            mode = request.get("retrieval_mode", "naive")
            if mode not in builders:
                raise ValueError(f"지원하지 않는 검색 방식: {mode}")
            if pool is not None:
                retriever = pool.get(mode)
            else:
                if mode not in retrievers:
                    extra = {"k": top_k} if top_k else {}
                    retrievers[mode] = builders[mode](domain=domain, difficulty=difficulty, profile=profile, **extra)
                retriever = retrievers[mode]
            sub_questions = [q.strip() for q in request.get("sub_questions") or [] if q and q.strip()]
            if sub_questions:
                queries = list(dict.fromkeys([search_question, *sub_questions]))
                docs = merge_docs([retriever.invoke(q) for q in queries])
            else:
                docs = retriever.invoke(search_question)
            if request.get("page_context"):
                # 페이지 확장 요청에만 컬렉션 설정 확인
                collection = collection_name(load_config(), domain=domain, difficulty=difficulty, profile=profile)
                docs = to_pages(docs, collection)
            return docs

    def retrieve_context(request):
        return format_docs(request["docs"])

    def finish(request):
        if not include_context:
            return add_sources(request, domain=domain, difficulty=difficulty)
        return build_answer_result(request)

    # assign으로 기존 입력을 보존하면서 검색 자료와 프롬프트 문맥 및 답변을 순서대로 추가
    chain = (
        RunnablePassthrough.assign(docs=RunnableLambda(retrieve_docs))
        | RunnablePassthrough.assign(context=RunnableLambda(retrieve_context))
        | RunnablePassthrough.assign(answer=answer_chain)
        | RunnableLambda(finish)
    )

    # 단독 실행의 추적 유지 및 전체 서비스 체인에 연결할 때 상위 콜백 상속
    if trace:
        return with_langfuse(chain, domain=domain, difficulty=difficulty)
    return chain.with_config(run_name="rag-answer")


@lru_cache(maxsize=3)
def _cached_rag_resources(domain, difficulty):
    """서비스의 세 난이도별 체인과 검색 자원 보관 및 준비한 객체의 재사용."""
    # 호출한 난이도만 생성하며 난이도 전환 시 기존 워밍업 자원의 캐시 퇴출 방지
    k = SERVICE_TOP_K.get(difficulty)
    if SERVICE_IMPL == "retrieval_b":
        pool = RetrievalBPool(domain, difficulty, k=k)
    else:
        pool = RetrieverPool(domain, difficulty, top_k=k)
    chain = create_chain(
        domain=domain, difficulty=difficulty, impl=SERVICE_IMPL,
        include_context=True, trace=False, retriever_pool=pool,
    )
    return chain, pool


def get_rag_chain(domain, difficulty):
    """동시 최초 요청에서도 동일한 RAG 체인을 한 번만 생성하도록 캐시 조회 보호."""
    # lru_cache의 최초 계산 중 중복 실행을 막기 위해 캐시 조회 바깥에서 잠금 확보
    with RETRIEVAL_LOCK:
        return _cached_rag_resources(domain, difficulty)[0]


def warmup_service(domain, difficulty, modes):
    """실제 서비스 체인이 재사용할 검색 자원 준비 및 답변·검사 모델 호출 제외."""
    with RETRIEVAL_LOCK:
        _, pool = _cached_rag_resources(domain, difficulty)
        pool.warmup(modes)


def create_service_chain():
    """프롬프트와 모델 및 결과 처리 부품을 조립하는 전체 질문 처리 체인."""
    guard_model = create_model("guard")

    # 검사 모델의 실행 실패는 None으로 전달하고 guard.py에서 실패 상태와 후속 정책 처리
    query_check = (
        RunnableLambda(prepare_query_input) | query_guard_prompt
        | guard_model.with_structured_output(QueryDecision)
    ).with_fallbacks([RunnableLambda(lambda _: None)])
    # 사후 검사: 기본은 규칙 검사만 (LLM 호출 2번). RAG_ANSWER_GUARD=llm이면 예전처럼 LLM 사후 검사 (3번)
    #   보안 규칙은 answer_prompt(답변 단계)에 있어서 기본 경로에서도 답변 생성 시 함께 적용된다
    if (os.getenv("RAG_ANSWER_GUARD") or "rule").strip().lower() == "llm":
        answer_check = (
            RunnableLambda(lambda state: {
                "question": state["request"].question, "answer": state["result"]["answer"],
            })
            | answer_guard_prompt | guard_model.with_structured_output(AnswerCheck)
        ).with_fallbacks([RunnableLambda(lambda _: None)])
        check_step = RunnablePassthrough.assign(check=answer_check) | RunnableLambda(apply_answer_check)
    else:
        check_step = RunnableLambda(apply_answer_rules)

    # 직접 답변도 같은 위치에서 입력 준비와 모델 및 문자열 출력 처리를 연결
    direct_answer = (
        RunnableLambda(prepare_direct_input) | create_model() | StrOutputParser()
        | RunnableLambda(lambda text: {"answer": text, "status": "answered"})
    )

    def select_rag_chain(state):
        """요청한 난이도의 RAG 체인 선택 및 그 체인에 전달할 검색 입력 연결."""
        request, decision = state["request"], state["decision"]
        inputs = {
            "question": request.question, "search_question": decision.search_question,
            "sub_questions": decision.sub_questions, "retrieval_mode": request.retrieval_mode,
            "page_context": request.page_context,
        }
        # RunnableLambda가 반환한 체인은 LangChain이 이어서 실행하므로 별도 invoke 호출 불필요
        return (
            RunnableLambda(lambda _: inputs) | get_rag_chain(request.domain, request.difficulty)
            | RunnableLambda(lambda output: {
                **output["result"].model_dump(), "sources": output["sources"],
                "invalid_source_ids": output["invalid_source_ids"], "contexts": output["contexts"],
            })
        )

    def finish_response(state):
        """요청과 검사 중간값을 제외하고 화면에 필요한 결과만 반환."""
        decision, result = state["decision"], state["result"]
        return {
            **result, "route": state["route"], "guard_errors": state["guard_errors"],
            "search_question": decision.search_question if decision.action == "rag" else "",
            "sub_questions": decision.sub_questions if decision.action == "rag" else [],
            "contexts": result.get("contexts", []) if state["request"].include_context else [],
        }

    # 첫 조건 일치 시 해당 경로만 실행하며 마지막 항목은 기본 RAG 경로
    answer_branch = RunnableBranch(
        (lambda state: state["route"] == "block", RunnablePassthrough.assign(result=RunnableLambda(blocked_question_result))),
        (lambda state: state["route"] == "direct", RunnablePassthrough.assign(result=direct_answer)),
        RunnablePassthrough.assign(result=RunnableLambda(select_rag_chain)),
    )
    # 입력 차단 안내는 그대로 통과하고 실제 생성한 답변에만 사후 검사 적용
    check_branch = RunnableBranch(
        (lambda state: state["route"] == "block", RunnablePassthrough()),
        check_step,
    )
    # 요청별 상태 생성 후 사전 판단 → 답변 경로 선택 → 사후 검사 → 최종 응답 순서의 조립
    chain = (
        RunnableLambda(lambda request: {"request": request, "guard_errors": []})
        | RunnablePassthrough.assign(decision=query_check)
        | RunnableLambda(apply_query_decision)
        | answer_branch
        | check_branch
        | RunnableLambda(finish_response)
    )
    return with_langfuse(chain, run_name="question-workflow")
