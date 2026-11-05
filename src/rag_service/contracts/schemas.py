from typing import Literal
from pydantic import BaseModel, Field

class RagAnswer(BaseModel):
    """검색 자료에 근거한 RAG 답변의 공통 출력 형식."""

    status: Literal[
        "answered",
        "needs_clarification",
        "abstained",
    ] = Field(
        description=(
            "질문에 대한 응답 상태. "
            "answered: 필요한 조건과 근거가 충분하여 답변할 수 있음. "
            "needs_clarification: 답을 정하는 데 필요한 사용자 조건이 빠져 추가 질문이 필요함. "
            "abstained: 필요한 자료가 없거나 자료 간 충돌을 해결할 수 없어 답변을 보류함."
        )
    )

    answer: str = Field(
        description=(
            "사용자에게 보여줄 한국어 최종 응답. "
            "answered이면 질문에 대한 답과 필요한 적용 조건을 작성한다. "
            "needs_clarification이면 필요한 정보를 구체적으로 질문한다. "
            "abstained이면 확인할 수 없는 내용, 보류 이유와 필요한 자료를 설명한다. "
            "일부만 확인되면 확인된 내용과 미확인 내용을 구분한다. "
            "문서에 근거한 주요 주장에는 [S1]처럼 자료 ID를 표시한다."
        )
    )

    reason: str = Field(
        description=(
            "응답 상태를 선택한 근거를 1~2문장으로 요약한 설명. "
            "확인된 적용 조건이나 부족한 정보·근거를 구체적으로 적는다. "
            "사용자에게 필요한 추가 질문과 보류 이유를 이 필드에만 적지 않고 "
            "answer에도 포함한다."
        )
    )

    source_ids: list[str] = Field(
        description=(
            "응답에서 실제로 인용한 검색 자료 ID 목록. 예: ['S1', 'S2']. "
            "현재 요청의 검색 문맥에 부여된 ID만 중복 없이 사용한다. "
            "추가 확인이나 보류 응답도 문서의 사실을 설명했다면 해당 ID를 포함한다. "
            "인용할 근거가 없으면 빈 목록을 반환한다."
        )
    )