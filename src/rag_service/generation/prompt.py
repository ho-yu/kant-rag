from langchain_core.prompts import ChatPromptTemplate

# langchain으로 모델에 제공할 prompt
answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "제공된 검색 자료에 근거해 업무 질문에 답하세요. "
        "필요한 조건이 부족하면 구체적으로 되물으세요. "
        "검색 자료로 확인할 수 없는 내용은 추측하지 마세요."
    ),
    ("human", "질문:\n{question}\n\n검색 자료:\n{context}"),
])