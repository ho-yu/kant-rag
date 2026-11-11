"""웹 화면이 보내는 질문과 서버가 돌려주는 답변의 데이터 형식 정의.

QueryRequest는 요청 값 검증용이며 QueryResponse는 최종 응답 검증용.
모델이 생성하는 답변 형식은 기존 contracts/schemas.py의 RagAnswer 사용.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HistoryMessage(BaseModel):
    """이전 대화 중 메시지 한 건의 발화자와 본문."""
    # 정의하지 않은 항목을 거절하여 필드 이름 오타와 잘못된 입력의 조기 확인
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]  # 사용자 질문 또는 이전 모델 답변 구분
    content: str = Field(max_length=20000)  # 출처 원문 목록을 제외한 대화 본문


class QueryRequest(BaseModel):
    """POST /query로 전달하는 질문 한 건과 검색 옵션."""
    model_config = ConfigDict(extra="forbid")

    # 네트워크 요청 크기 제한이며 검색 문서의 선택이나 축약 정책과 별개
    question: str = Field(min_length=1, max_length=10000)
    # 질문과 답변을 각각 한 건으로 계산하여 최근 다섯 턴까지 허용
    history: list[HistoryMessage] = Field(default_factory=list, max_length=10)
    domain: Literal["shipping"] = "shipping"  # 현재 서비스의 검색 대상 분야
    # advanced 선택 허용과 실제 모델 설정 및 컬렉션 준비 여부의 구분
    difficulty: Literal["basic", "intermediate", "advanced"] = "basic"
    retrieval_mode: Literal["naive", "hybrid", "hybrid_rerank"] = "naive"
    page_context: bool = False  # 검색 청크를 같은 페이지의 전체 원문으로 확장하는 옵션
    include_context: bool = False  # 답변 생성에 사용한 모든 검색 자료를 응답에도 포함하는 옵션

    @field_validator("question")
    @classmethod
    def nonblank_question(cls, value: str) -> str:
        """공백만 있는 질문 거절 및 질문 양끝의 불필요한 공백 제거."""
        if not value.strip():
            raise ValueError("질문을 입력해주세요")
        return value.strip()


class ContextDocument(BaseModel):
    """검색 자료 한 건의 인용 번호와 원문 및 문서 위치 정보."""
    id: str  # 현재 질문의 검색 문맥에서 부여한 S1 또는 S2 형태의 번호
    content: str  # 모델에 실제로 전달한 청크 또는 확장된 페이지 원문
    metadata: dict[str, Any]  # book_id와 page 및 chunk_id 등 검색 결과의 기존 정보


class QueryResponse(BaseModel):
    """웹 화면과 Python 호출 코드가 공통으로 받는 최종 처리 결과."""
    answer: str  # 별도 출처 목록 문자열이 붙지 않은 사용자용 답변 본문
    # 답변 완료 또는 추가 질문 필요 또는 근거 부족 보류 또는 검사에 의한 차단
    status: Literal["answered", "needs_clarification", "abstained", "blocked"]
    reason: str = ""  # 응답 상태나 차단 결정을 설명하는 짧은 사유
    source_ids: list[str] = Field(default_factory=list)  # 모델이 반환한 인용 번호 목록
    sources: list[ContextDocument] = Field(default_factory=list)  # 실제 자료와 대조하여 찾은 인용 출처
    invalid_source_ids: list[str] = Field(default_factory=list)  # 대응하는 검색 자료가 없는 인용 번호
    # 실제 실행 경로이며 blocked_answer는 생성 후 사후 검사에서 차단된 경우
    route: Literal["rag", "direct", "block", "blocked_answer"]
    search_question: str = ""  # 대화의 대상과 조건을 풀어 쓴 검색용 질문
    sub_questions: list[str] = Field(default_factory=list)  # 복합 질문을 나눈 추가 검색 질문
    contexts: list[ContextDocument] = Field(default_factory=list)  # 인용 여부와 무관한 전체 생성 자료
    guard_errors: list[str] = Field(default_factory=list)  # query_guard 또는 answer_guard 호출 실패 기록
    request_id: str  # 서버 로그에서 이 요청을 찾기 위한 API 요청 식별자
    trace_id: str | None = None  # Langfuse 기록 식별자이며 추적 미설정 또는 시작 실패 시 None
    elapsed: float  # 서버 처리 시간 단위 초이며 브라우저와의 네트워크 왕복 시간 제외
