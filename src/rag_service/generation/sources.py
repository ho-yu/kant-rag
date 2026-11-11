"""RAG 답변 끝에 [출처]를 붙인다 (pipeline/chain.py의 create_chain 마지막 단계에서 사용).

LLM은 답변과 인용 번호(source_ids: S1, S2 …)까지만 만든다.
파일명·쪽 번호·원문은 LLM 호출이 끝난 뒤, 실제 검색 결과(docs)의 metadata로 코드가 붙인다.
"""
import re

from langchain_core.documents import Document

from rag_service.contracts.schemas import RagAnswer


def format_sources(source_ids: list[str], docs: list[Document], *, domain: str, difficulty: str) -> str:
    """LLM이 인용한 번호(S1, S2 …)의 자료를 고객이 보기 쉬운 글로 만든다.
    검색 결과에 없는 번호는 버리고, 같은 번호는 한 번만. 인용이 없으면 빈 문자열.
    """
    blocks, seen = [], set()
    for sid in source_ids:
        match = re.fullmatch(r"\[?S(\d+)\]?", sid.strip(), flags=re.IGNORECASE)
        if not match or not 1 <= int(match.group(1)) <= len(docs) or match.group(1) in seen:
            continue
        seen.add(match.group(1))
        doc = docs[int(match.group(1)) - 1]
        m = doc.metadata
        blocks.append(
            f"[S{match.group(1)}]\n"
            f"file_name : {m.get('file_name') or m['book_id'] + '.pdf'}\n"
            f"page : {m['page']}\n"
            f"book_id : {m['book_id']}\n"
            f"chunk_id : {m.get('chunk_id', '')}\n"
            f"domain : {m.get('domain') or domain}\n"
            f"difficulty : {m.get('difficulty') or difficulty}\n"
            f"page_content : {doc.page_content}"
        )
    return "\n\n".join(blocks)


def add_sources(request: dict, *, domain: str, difficulty: str) -> RagAnswer:
    """LLM 호출이 끝난 뒤, 답변 끝에 [출처]를 붙인다 (인용이 없으면 그대로).

    request: 체인 중간 결과 {"question", "retrieval_mode", "docs", "context", "answer"}
    """
    answer: RagAnswer = request["answer"]
    sources = format_sources(answer.source_ids, request["docs"], domain=domain, difficulty=difficulty)
    if not sources:
        return answer
    return answer.model_copy(update={"answer": f"{answer.answer}\n\n\n[출처]\n{sources}"})


def build_answer_result(request: dict) -> dict:
    """인용 검증과 구조화된 출처 및 실제 생성 문맥의 공통 결과 구성."""
    # contexts는 전체 모델 입력 자료이며 sources는 그중 답변에서 인용한 자료
    contexts = [
        {"id": f"S{i}", "content": doc.page_content, "metadata": doc.metadata}
        for i, doc in enumerate(request["docs"], start=1)
    ]
    sources, invalid_ids, seen = [], [], set()
    # 선언한 인용 목록과 답변 본문의 인용을 함께 확인하여 누락된 출처 검증
    references = request["answer"].source_ids + re.findall(
        r"\[S\d+\]", request["answer"].answer, flags=re.IGNORECASE
    )
    for sid in references:
        match = re.fullmatch(r"\[?S(\d+)\]?", sid.strip(), flags=re.IGNORECASE)
        index = int(match.group(1)) - 1 if match else -1
        if not 0 <= index < len(contexts):
            invalid_sid = f"S{index + 1}" if match else sid.strip()
            if invalid_sid not in invalid_ids:
                invalid_ids.append(invalid_sid)
        elif index not in seen:
            sources.append(contexts[index])
            seen.add(index)
    return {
        "result": request["answer"],
        "sources": sources,
        "invalid_source_ids": invalid_ids,
        "contexts": contexts,
    }
