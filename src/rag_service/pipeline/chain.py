from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from rag_service.contracts.schemas import RagAnswer
from rag_service.generation.prompt import answer_prompt
from rag_service.observability import with_langfuse
from rag_service.generation.sources import add_sources
from rag_service.storage.qdrant_store import get_pages

from rag_service.retrieval_a import build_hybrid_retriever, build_naive_retriever, build_rerank_retriever
from rag_service.retrieval_a.naive import collection_name, load_config


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[S{i}] {doc.metadata['book_id']}.pdf "
        f"/ page {doc.metadata['page']}\n{doc.page_content}"
        for i, doc in enumerate(docs, start=1)
    )


MAX_DOCS = 8  # 나눠 찾기(sub_questions)로 합친 검색 결과를 LLM에 넘기는 최대 개수


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


def create_chain(domain, difficulty):
    model = ChatOpenAI(model="gpt-6-luna")
    
    retrievers = {
        "naive": build_naive_retriever(domain=domain, difficulty=difficulty),
        "hybrid": build_hybrid_retriever(domain=domain, difficulty=difficulty),
        "hybrid_rerank": build_rerank_retriever(domain=domain, difficulty=difficulty),
        }
        
    collection = collection_name(load_config(), domain=domain, difficulty=difficulty)  # page 확장에서 쪽을 꺼낼 컬렉션

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