"""PDF 한 쪽 → 머리글 / 제목 / 문단 / 표(제목·각주 포함) 블록으로 나눈다.

- 글자 크기와 위치는 pymupdf, 표는 pdfplumber로 읽는다.
- 본문 글자 크기(body)를 쪽마다 구해서 상대 크기로 판단한다.
    제목(heading)   : body × 1.3 이상
    표 제목(caption) : 굵은 본문 크기 글자이고 바로 아래에 표가 있음
    소제목           : 굵은 본문 크기 글자이고 아래에 표가 없음 → 다음 문단의 첫 줄로 붙임
    표 각주(note)    : 본문보다 작은 글자이고 표 바로 아래(40pt 이내)에 있음 → 표에 붙임
- 쪽 위쪽(첫 제목보다 위)은 머리글, 쪽 아래 60pt는 쪽 번호로 보고 본문에서 뺀다.
- 줄바꿈으로 끊긴 줄은 같은 문서의 표기 빈도와 Kiwi 띄어쓰기 교정으로 띄어쓰기 여부를 정해서 잇는다.
"""
from __future__ import annotations

import contextvars
import itertools
import re
import statistics
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pdfplumber
import pymupdf

HEADING_RATIO = 1.3      # body × 이 값 이상이면 제목
SMALL_RATIO = 0.95       # body × 이 값 미만이면 작은 글자 (각주 후보)
FOOTER_MARGIN = 60       # 쪽 아래 이 높이 안은 쪽 번호로 보고 버림
NOTE_GAP = 40            # 표 아래 이 거리 안의 작은 글자는 표 각주
CAPTION_GAP = 40         # 표 위 이 거리 안의 굵은 글자는 표 제목
PARA_GAP_RATIO = 1.35    # 줄 간격이 보통 간격 × 이 값보다 크면 새 문단


@dataclass
class Line:
    text: str            # 원래 줄 (앞뒤 공백 유지)
    y0: float
    y1: float
    x0: float
    size: float
    bold: bool


@dataclass
class Table:
    title: str
    header: list[str]
    rows: list[list[str]]
    notes: list[str] = field(default_factory=list)
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    @property
    def y0(self) -> float:
        return self.bbox[1]

    @property
    def y1(self) -> float:
        return self.bbox[3]


@dataclass
class Block:
    kind: str            # "heading" | "paragraph" | "table"
    y0: float
    text: str = ""
    table: Table | None = None


@dataclass
class PageContent:
    page: int            # 1부터
    header_lines: list[str]
    blocks: list[Block]


@dataclass
class PdfDocument:
    book_id: str
    file_name: str
    title: str
    pages: list[PageContent]


# ───────────────────────── 줄 잇기 ─────────────────────────
@lru_cache(maxsize=1)
def _kiwi():
    from kiwipiepy import Kiwi

    return Kiwi()


# 줄잇기에 참고할 문서 전체 원문 (read_pdf가 문서마다 설정)
_DOC_TEXT: contextvars.ContextVar[str] = contextvars.ContextVar("doc_text", default="")

_NO_JOIN_TAIL = re.compile(r"[.,:;!?)\]」』>]$")
_OPEN_HEAD = re.compile(r"^[‘“(\[「『<]")
_ASCII_TAIL = re.compile(r"[A-Za-z0-9]$")
_ASCII_HEAD = re.compile(r"^[A-Za-z0-9]")


def _vote(a: str, b: str) -> bool | None:
    """같은 문서 안에서 'A B'와 'AB' 중 어느 쪽이 더 많이 쓰였는지로 판단. 모르면 None."""
    doc = _DOC_TEXT.get()
    if not doc:
        return None
    w1 = re.split(r"\s", a)[-1]
    w2 = re.split(r"\s", b)[0]
    if not w1 or not w2:
        return None
    for k in (3, 2):
        x = w2[:k]
        joined, spaced = doc.count(w1 + x), doc.count(w1 + " " + x)
        if joined != spaced:
            return spaced > joined
    return None


@lru_cache(maxsize=50000)
def _kiwi_space(left: str, right: str) -> bool:
    """Kiwi 띄어쓰기 교정으로 경계에 공백이 들어가는지 본다."""
    out = _kiwi().space(left + right, reset_whitespace=False)
    n = len(re.sub(r"\s+", "", left))
    cnt = 0
    for i, ch in enumerate(out):
        if not ch.isspace():
            cnt += 1
            if cnt == n:
                return i + 1 < len(out) and out[i + 1] == " "
    return True


def needs_space(a: str, b: str) -> bool:
    if _NO_JOIN_TAIL.search(a) or _OPEN_HEAD.search(b):   # 문장부호 뒤, 여는 따옴표 앞은 띄운다
        return True
    if _ASCII_TAIL.search(a) and _ASCII_HEAD.search(b):    # 영문·숫자 사이 (예: 표 칸의 ID 두 줄)
        return True
    voted = _vote(a, b)
    if voted is not None:
        return voted
    return _kiwi_space(a[-15:], b[:15])


def join_two(prev: str, nxt: str) -> str:
    """앞 줄과 다음 줄을 잇는다.

    순서: 줄 끝/앞에 공백이 있으면 띄움 → 문장부호 규칙 → 같은 문서에서 더 많이 쓰인 형태
          → Kiwi 띄어쓰기 교정. (Basic·Intermediate 근거 인용 기준 줄 경계 589개 중 574개 일치)
    """
    if not prev:
        return nxt.strip()
    if prev.endswith((" ", "\u00a0")) or nxt.startswith((" ", "\u00a0")):
        return prev.rstrip() + " " + nxt.strip()
    a, b = prev.rstrip(), nxt.strip()
    if not b:
        return a
    return a + (" " if needs_space(a, b) else "") + b


def join_lines(lines: list[str]) -> str:
    out = ""
    for ln in lines:
        out = join_two(out, ln) if out else ln.strip()
    return out


# ───────────────────────── 표 ─────────────────────────
def _clean_cell(cell: str | None) -> str:
    if not cell:
        return ""
    parts = [p for p in cell.split("\n")]
    return join_lines(parts).replace("|", "/").strip()


def _read_tables(pl_page) -> list[Table]:
    tables = []
    for t in pl_page.find_tables():
        data = [[_clean_cell(c) for c in row] for row in t.extract()]
        data = [r for r in data if any(r)]
        if len(data) < 2:
            continue
        header = [h or f"열{i + 1}" for i, h in enumerate(data[0])]
        tables.append(Table(title="", header=header, rows=data[1:], bbox=tuple(t.bbox)))
    return tables


def _in_bbox(line: Line, bbox, pad: float = 2.0) -> bool:
    x0, top, x1, bottom = bbox
    cy = (line.y0 + line.y1) / 2
    return x0 - pad <= line.x0 <= x1 + pad and top - pad <= cy <= bottom + pad


# ───────────────────────── 쪽 하나 ─────────────────────────
def _read_lines(page) -> list[Line]:
    lines = []
    for b in page.get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            text = "".join(s["text"] for s in ln["spans"])
            if not text.strip():
                continue
            s0 = max(ln["spans"], key=lambda s: len(s["text"].strip()))
            x0, y0, _, y1 = ln["bbox"]
            lines.append(Line(text, y0, y1, x0, s0["size"], bool(s0["flags"] & 16)))
    lines.sort(key=lambda ln: (round(ln.y0, 1), ln.x0))
    return lines


def read_page(page, pl_page, page_no: int, raw_lines: list[Line] | None = None) -> PageContent:
    height = page.rect.height
    tables = _read_tables(pl_page)
    raw_lines = raw_lines if raw_lines is not None else _read_lines(page)
    all_lines = [ln for ln in raw_lines if ln.y0 < height - FOOTER_MARGIN]
    lines = [ln for ln in all_lines if not any(_in_bbox(ln, t.bbox) for t in tables)]

    sizes = [round(ln.size, 1) for ln in lines for _ in ln.text.strip()]
    body = statistics.mode(sizes) if sizes else 10.0

    first_heading = next((ln for ln in lines if ln.size >= body * HEADING_RATIO), None)
    if first_heading is not None:
        header = [ln for ln in lines if ln.y0 < first_heading.y0 - 1]
    else:
        header = [ln for ln in lines if ln.y0 < 100 and ln.size < body * SMALL_RATIO]
    lines = [ln for ln in lines if ln not in header]

    gaps = [b.y0 - a.y0 for a, b in itertools.pairwise(lines) if 0 < b.y0 - a.y0 < 40]
    pitch = statistics.median(gaps) if gaps else body * 1.6

    blocks: list[Block] = []
    para: list[Line] = []
    pending_sub: list[str] = []   # 소제목 (다음 문단 첫 줄로 붙임)

    def flush():
        nonlocal para, pending_sub
        if para:
            text = join_lines([ln.text for ln in para])
            if pending_sub:
                text = "\n".join(pending_sub + [text])
                pending_sub = []
            blocks.append(Block("paragraph", para[0].y0, text=text))
            para = []

    prev: Line | None = None
    for ln in lines:
        # 제목
        if ln.size >= body * HEADING_RATIO:
            flush()
            if blocks and blocks[-1].kind == "heading" and prev is not None and prev.size >= body * HEADING_RATIO:
                blocks[-1].text = join_two(blocks[-1].text, ln.text)
            else:
                blocks.append(Block("heading", ln.y0, text=ln.text.strip()))
            prev = ln
            continue
        # 표 각주: 작은 글자 + 표 바로 아래
        note_of = next((t for t in tables if t.y1 - 1 <= ln.y0 <= t.y1 + NOTE_GAP + 15 * len(t.notes)), None)
        if note_of is not None and ln.size < body * SMALL_RATIO and not ln.bold:
            flush()
            if note_of.notes and prev is not None and prev.size < body * SMALL_RATIO and ln.y0 - prev.y0 < pitch * PARA_GAP_RATIO:
                note_of.notes[-1] = join_two(note_of.notes[-1], ln.text)
            else:
                note_of.notes.append(ln.text.strip())
            prev = ln
            continue
        # 굵은 본문 크기: 표 제목 또는 소제목
        if ln.bold and abs(ln.size - body) < 0.6:
            flush()
            cap_of = next((t for t in tables if 0 <= t.y0 - ln.y1 <= CAPTION_GAP and not t.title), None)
            if cap_of is not None:
                cap_of.title = ln.text.strip()
            else:
                pending_sub.append(ln.text.strip())
            prev = ln
            continue
        # 본문 줄
        if para and prev is not None and (ln.y0 - prev.y0 > pitch * PARA_GAP_RATIO or prev not in para):
            flush()
        para.append(ln)
        prev = ln
    flush()
    if pending_sub:   # 쪽 끝에 남은 소제목
        blocks.append(Block("paragraph", lines[-1].y0 if lines else 0, text="\n".join(pending_sub)))

    for t in tables:
        blocks.append(Block("table", t.y0, table=t))
    blocks.sort(key=lambda b: b.y0)
    return PageContent(page=page_no, header_lines=[join_lines([h.text]) for h in header], blocks=blocks)


def read_pdf(path: str | Path) -> PdfDocument:
    path = Path(path)
    doc = pymupdf.open(path)
    try:
        title = (doc.metadata or {}).get("title") or ""
        raw = [_read_lines(page) for page in doc]
        token = _DOC_TEXT.set("\n".join(ln.text for lines in raw for ln in lines))
        try:
            pages = []
            with pdfplumber.open(path) as pl:
                for i, page in enumerate(doc):
                    pages.append(read_page(page, pl.pages[i], i + 1, raw[i]))
        finally:
            _DOC_TEXT.reset(token)
    finally:
        doc.close()
    if not title:
        title = next((h for p in pages[:1] for h in p.header_lines), path.stem)
    return PdfDocument(book_id=path.stem, file_name=path.name, title=title, pages=pages)
