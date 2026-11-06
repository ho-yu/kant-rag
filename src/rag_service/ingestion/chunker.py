"""PdfDocument → 청크 목록 (쪽 → 문단 / 표 → row).

- 쪽 경계를 넘는 청크는 만들지 않는다. 오버랩도 없다.
- 문단이 max_chunk_tokens를 넘으면 문장 경계로만 나눈다 (문장 중간에서 자르지 않음).
- 표는 표 전체(table) 청크 하나 + 행마다 row 청크. 표와 row는 절대 자르지 않는다.
- row 문장 형식: "열 이름: 값 | 열 이름: 값 ..." (값 안의 '/'와 헷갈리지 않게 '|'로 구분)
- 같은 쪽 안의 청크 번호(c1, c2 …)는 읽는 순서대로 붙인다.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache

from rag_service.contracts.chunk import LEVEL_TO_TYPE, Chunk
from rag_service.ingestion.pdf_reader import PageContent, PdfDocument, Table

log = logging.getLogger(__name__)


# ───────────────────────── 토큰 수 ─────────────────────────
@lru_cache(maxsize=4)
def _tokenizer(name: str):
    try:
        from tokenizers import Tokenizer

        return Tokenizer.from_pretrained(name)
    except Exception as e:  # noqa: BLE001 — 네트워크·모델 이름 오류 등 어떤 실패든 어림값으로 대체
        log.warning("토크나이저 %s를 불러오지 못해 글자 수/2로 어림합니다: %s", name, e)
        return None


def count_tokens(text: str, tokenizer_name: str) -> int:
    tok = _tokenizer(tokenizer_name)
    if tok is None:
        return max(1, len(text) // 2)
    return len(tok.encode(text).ids)


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENT_SPLIT.split(text.strip()) if s]


def split_long_paragraph(text: str, max_tokens: int, tokenizer_name: str) -> list[str]:
    """max_tokens 이하로 문장을 묶는다. 문장 하나가 한도를 넘으면 그 문장은 통째로 둔다."""
    if count_tokens(text, tokenizer_name) <= max_tokens:
        return [text]
    parts, cur = [], ""
    for sent in split_sentences(text):
        cand = f"{cur} {sent}".strip()
        if cur and count_tokens(cand, tokenizer_name) > max_tokens:
            parts.append(cur)
            cur = sent
        else:
            cur = cand
    if cur:
        parts.append(cur)
    return parts


# ───────────────────────── 표 → 문자열 ─────────────────────────
def table_markdown(t: Table) -> str:
    lines = []
    if t.title:
        lines.append(f"[표] {t.title}")
    lines.append("| " + " | ".join(t.header) + " |")
    lines.append("|" + "---|" * len(t.header))
    for r in t.rows:
        cells = (r + [""] * len(t.header))[: len(t.header)]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [f"주: {n}" for n in t.notes]
    return "\n".join(lines)


def row_sentence(t: Table, row: list[str]) -> str:
    pairs = [f"{h}: {v}" for h, v in zip(t.header, row) if v]
    lines = []
    if t.title:
        lines.append(f"[표] {t.title}")
    lines.append(" | ".join(pairs))
    lines += [f"주: {n}" for n in t.notes]
    return "\n".join(lines)


def page_text(pc: PageContent) -> str:
    parts = []
    for b in pc.blocks:
        if b.kind == "table" and b.table is not None:
            parts.append(table_markdown(b.table))
        else:
            parts.append(b.text)
    return "\n\n".join(p for p in parts if p)


# ───────────────────────── 청크 만들기 ─────────────────────────
@dataclass
class ChunkSettings:
    max_chunk_tokens: int = 300
    jsonl_levels: tuple[str, ...] = ("page", "para", "table", "row")
    tokenizer: str = "BAAI/bge-m3"
    max_table_tokens: int = 8000


def chunk_document(doc: PdfDocument, difficulty: str, domain: str, settings: ChunkSettings) -> list[Chunk]:
    levels = set(settings.jsonl_levels)
    out: list[Chunk] = []
    for pc in doc.pages:
        base = {
            "book_id": doc.book_id,
            "file_name": doc.file_name,
            "page": pc.page,
            "doc_title": doc.title,
            "page_header": " | ".join(h for h in pc.header_lines if h != doc.title),
            "difficulty": difficulty,
            "domain": domain,
        }
        page_id = f"{doc.book_id}-p{pc.page}"
        first_heading = next((b.text for b in pc.blocks if b.kind == "heading"), "")
        if "page" in levels:
            out.append(Chunk(chunk_id=page_id, level="page", chunk_type="page",
                             section_title=first_heading, text=page_text(pc), **base))
        n = 0
        section = first_heading
        for b in pc.blocks:
            if b.kind == "heading":
                section = b.text
                continue
            if b.kind == "paragraph" and "para" in levels:
                for piece in split_long_paragraph(b.text, settings.max_chunk_tokens, settings.tokenizer):
                    n += 1
                    out.append(Chunk(chunk_id=f"{page_id}-c{n}", parent_id=page_id, level="para",
                                     chunk_type=LEVEL_TO_TYPE["para"], section_title=section,
                                     text=piece, **base))
            elif b.kind == "table" and b.table is not None:
                t = b.table
                table_id = None
                if "table" in levels:
                    n += 1
                    table_id = f"{page_id}-c{n}"
                    text = table_markdown(t)
                    if count_tokens(text, settings.tokenizer) > settings.max_table_tokens:
                        log.warning("%s 표가 %d토큰을 넘습니다 (자르지 않고 저장)", table_id, settings.max_table_tokens)
                    out.append(Chunk(chunk_id=table_id, parent_id=page_id, level="table",
                                     chunk_type="table", section_title=section, table_title=t.title or None,
                                     text=text, **base))
                if "row" in levels:
                    for i, row in enumerate(t.rows):
                        n += 1
                        out.append(Chunk(chunk_id=f"{page_id}-c{n}", parent_id=table_id or page_id,
                                         level="row", chunk_type="table_row", section_title=section,
                                         table_title=t.title or None, row_index=i,
                                         text=row_sentence(t, row), **base))
    return out
