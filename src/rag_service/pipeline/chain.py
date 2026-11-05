from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from rag_service.contracts.schemas import RagAnswer

# TODO: retriever 함수 import
# build_naive_retriever
# build_hybrid_retriever
# build_rerank_retriever


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[S{i}] {doc.metadata['book_id']}.pdf "
        f"/ {doc.metadata['page']}쪽\n{doc.page_content}"
        for i, doc in enumerate(docs, start=1)
    )


def create_chain():
    retrievers = {
        "naive": build_naive_retriever(), #TODO: 실제 함수명으로 변경
        "hybrid": build_hybrid_retriever(), #TODO: 실제 함수명으로 변경
        "hybrid_rerank": build_rerank_retriever(), #TODO: 실제 함수명으로 변경
        }

    def retrieve_context(request):
        question = request.get("question")
        mode = request.get("retrieval_mode", "naive")
        docs = retrievers[mode].invoke(question)
        return format_docs(docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "제공된 검색 자료에 근거해 해운 업무 질문에 답하세요. "),
        ("human", "질문:\n{question}\n\n검색 자료:\n{context}")
    ])

    model = ChatOpenAI(model="gpt-6-luna")

    chain = (
        RunnablePassthrough.assign(
            context=RunnableLambda(retrieve_context)
        )
        | prompt
        | model.with_structured_output(RagAnswer)
    )

    return chain