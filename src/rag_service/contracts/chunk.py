"""청크 데이터 형식 (jsonl 한 줄 = Chunk 하나).

jsonl은 필드를 평평하게 저장하고, Qdrant에는 payload()의 LangChain 형식으로 올린다.

구조 (부모-자식)
    page  (쪽 전체)          parent_id = None
    ├─ para  (문단)          parent_id = 쪽 chunk_id
    ├─ table (표 전체)       parent_id = 쪽 chunk_id
    │   └─ row (표 한 행)    parent_id = 표 chunk_id
    └─ sent  (문장)          추후 사용 예정

chunk_id 형식
    쪽:   {book_id}-p{page}            예) SB01-p3
    나머지: {book_id}-p{page}-c{n}      예) SB01-p3-c2   (발제문 형식)
"""
from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field

Level = Literal["page", "para", "table", "row", "sent"]
ChunkType = Literal["page", "text", "table", "table_row", "sentence"]

LEVEL_TO_TYPE: dict[str, str] = {
    "page": "page",
    "para": "text",
    "table": "table",
    "row": "table_row",
    "sent": "sentence",
}


class Chunk(BaseModel):
    chunk_id: str = Field(description="SB01-p3-c2 형식. 쪽은 SB01-p3")
    parent_id: str | None = Field(default=None, description="부모 chunk_id (쪽은 None)")
    level: Level
    chunk_type: ChunkType
    book_id: str = Field(description="PDF 파일명에서 .pdf를 뺀 값")
    file_name: str
    page: int = Field(ge=1, description="PDF 뷰어 기준 쪽 번호 (첫 쪽 = 1)")
    doc_title: str = ""
    page_header: str = Field(default="", description="쪽 위 머리글 (문서번호·담당팀 등)")
    section_title: str = ""
    table_title: str | None = None
    row_index: int | None = Field(default=None, description="row 레벨에서 0부터 시작하는 행 번호")
    difficulty: str
    domain: str
    text: str = Field(description="원문. 답변 생성에는 이 값이나 부모 쪽만 쓴다")
    context: str = Field(default="", description="LLM이 쓴 맥락 한 줄. 검색(임베딩·BM25)에만 쓴다")

    @property
    def point_id(self) -> str:
        """Qdrant 포인트 ID. chunk_id 문자열을 그대로 쓸 수 없어서 uuid5로 바꾼다."""
        return chunk_point_id(self.chunk_id)

    def embedding_text(self) -> str:
        """임베딩·BM25에 넣는 문자열 = 접두 줄 + context + text."""
        head = f"[{self.doc_title} | {self.section_title} | p{self.page}]"
        parts = [head]
        if self.context:
            parts.append(self.context)
        parts.append(self.text)
        return "\n".join(parts)

    def payload(self) -> dict:
        """Qdrant payload. LangChain QdrantVectorStore가 그대로 읽는 형식으로 저장한다.

        {"page_content": 원문 text, "metadata": {chunk_id, book_id, page, level, context, ...}}
        → 검색 결과 Document.page_content = 원문, Document.metadata["book_id"], ["page"] …
        """
        meta = self.model_dump(exclude_none=True, exclude={"text"})
        return {"page_content": self.text, "metadata": meta}


def chunk_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))
