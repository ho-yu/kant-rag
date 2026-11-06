from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from rag_service.contracts.schemas import RagAnswer
from rag_service.generation.prompt import answer_prompt

from rag_service.retrieval_a import build_hybrid_retriever, build_naive_retriever, build_rerank_retriever


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[S{i}] {doc.metadata['book_id']}.pdf "
        f"/ {doc.metadata['page']}쪽\n{doc.page_content}"
        for i, doc in enumerate(docs, start=1)
    )


def create_chain(domain, difficulty):
    model = ChatOpenAI(model="gpt-6-luna")
    
    retrievers = {
        "naive": build_naive_retriever(domain=domain, difficulty=difficulty),
        "hybrid": build_hybrid_retriever(domain=domain, difficulty=difficulty),
        "hybrid_rerank": build_rerank_retriever(domain=domain, difficulty=difficulty),
        }
        
    def retrieve_context(request):
        search_question = request.get("question")
        mode = request.get("retrieval_mode", "naive")
        docs = retrievers[mode].invoke(search_question)
        return format_docs(docs)
    
    chain = (
            RunnablePassthrough.assign(
                context=RunnableLambda(retrieve_context)
            )
            | answer_prompt
            | model.with_structured_output(RagAnswer)
        )

    return chain