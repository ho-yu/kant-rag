"""Contextual Retrieval: 청크마다 '이 청크가 문서에서 무엇에 관한 내용인지' 맥락 한 줄을 LLM으로 만든다.

- 쪽 하나당 LLM 호출 1번: 그 쪽의 원문을 주고, 쪽 안 청크들의 맥락을 JSON으로 한꺼번에 받는다.
- 입력은 해당 PDF 쪽 원문과 문서 제목뿐이다. 질문·정답·채점 기준은 절대 넣지 않는다.
- 맥락에 쪽 원문에 없는 숫자·날짜·ID가 있으면 버린다 (환각 방지).
- row 청크는 LLM을 부르지 않고, 속한 표(table)의 맥락을 그대로 쓴다.
- 캐시: (프롬프트 버전, 모델, 쪽 원문, 청크 원문)이 같으면 LLM을 다시 부르지 않는다.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from rag_service.contracts.chunk import Chunk

log = logging.getLogger(__name__)

SYSTEM_PROMPT = "너는 검색 색인에 붙일 짧은 맥락 문장을 쓰는 도우미다. 주어진 쪽 원문에 있는 내용만 사용한다."

USER_PROMPT = """<문서 제목>{doc_title}</문서 제목>
<머리글>{page_header}</머리글>
<쪽 원문 p{page}>
{page_text}
</쪽 원문>

아래 각 청크가 이 문서와 쪽에서 무엇에 관한 내용인지, 검색이 잘 되도록 짧은 맥락을 한국어 1~2문장(100자 안팎)으로 써라.
규칙:
- 쪽 원문에 없는 날짜, 숫자, 금액, ID, 이름, 조건을 새로 만들지 않는다.
- 청크 내용을 그대로 반복하지 말고, 대상과 주제, 적용 범위(어떤 문서의 어떤 절인지)를 밝힌다.
- 결론이나 판단(가능·불가, 적용·비적용)을 덧붙이지 않는다.
출력은 JSON 하나만: {{"contexts": [{{"chunk_id": "...", "context": "..."}}]}}

<청크 목록>
{chunk_list}
</청크 목록>"""

LlmCall = Callable[[list[dict]], tuple[str, dict]]   # messages → (응답 문자열, 토큰 사용량)


@dataclass
class ContextSettings:
    enabled: bool = True
    model: str = "gpt-6-luna"
    prompt_version: str = "v1"
    max_workers: int = 4
    max_chars: int = 300
    levels: tuple[str, ...] = ("para", "table")


@dataclass
class ContextStats:
    pages_called: int = 0
    pages_cached: int = 0
    chunks_filled: int = 0
    rejected: int = 0          # 숫자·길이 검사에서 버린 맥락
    missing: int = 0           # 실패 등으로 비어 있는 맥락
    prompt_tokens: int = 0
    completion_tokens: int = 0
    errors: list[str] = field(default_factory=list)
    rejected_samples: list[dict] = field(default_factory=list)   # 버린 맥락 예시 (원인 확인용)


# ───────────────────────── 검사 ─────────────────────────
_NUM = re.compile(r"\d+(?:[.,:/~\-]\d+)*")
_ID = re.compile(r"[A-Z]{2,}[A-Z0-9]*(?:-[A-Z0-9]+)+")


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def context_problems(ctx: str, allowed_text: str, max_chars: int) -> list[str]:
    """맥락의 문제 목록. 비어 있으면 통과.

    allowed_text = 쪽 원문 + 문서 제목 + 머리글 + 문서 ID + 쪽 번호.
    (LLM이 "SB01 문서 p17의 …"처럼 문서 ID나 쪽 번호를 쓰는 건 지어낸 숫자가 아니므로 허용)
    """
    if not ctx:
        return ["빈 값"]
    if len(ctx) > max_chars:
        return [f"너무 김({len(ctx)}자)"]
    allowed = _norm(allowed_text)
    return [f"원문에 없음: {tok}" for tok in _NUM.findall(ctx) + _ID.findall(ctx) if _norm(tok) not in allowed]


def validate_context(ctx: str, allowed_text: str, max_chars: int) -> bool:
    """맥락의 숫자·날짜·ID가 모두 원문(쪽 원문 + 문서 제목·머리글·ID·쪽 번호)에 있고 길이가 한도 안이면 True."""
    return not context_problems(ctx, allowed_text, max_chars)


# ───────────────────────── 캐시 ─────────────────────────
class ContextCache:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._data: dict[str, str] = {}
        if path.exists():
            with path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        self._data[rec["key"]] = rec["context"]

    @staticmethod
    def key(prompt_version: str, model: str, page_text: str, chunk_text: str) -> str:
        h = hashlib.sha256()
        for part in (prompt_version, model, page_text, chunk_text):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def put_many(self, items: dict[str, str]) -> None:
        if not items:
            return
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                for k, v in items.items():
                    self._data[k] = v
                    f.write(json.dumps({"key": k, "context": v}, ensure_ascii=False) + "\n")


# ───────────────────────── LLM 호출 ─────────────────────────
def openai_llm(model: str) -> LlmCall:
    from openai import OpenAI
    from tenacity import retry, stop_after_attempt, wait_exponential

    client = None   # 실제로 호출할 때 처음 만든다 → 캐시에 다 있으면 API 키 없이도 동작

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=2, max=30), reraise=True)
    def call(messages: list[dict]) -> tuple[str, dict]:
        nonlocal client
        if client is None:
            client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        usage = getattr(resp, "usage", None)
        return resp.choices[0].message.content or "", {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        }

    return call


def resolve_model(settings: ContextSettings) -> str:
    return os.getenv("OPENAI_CONTEXT_MODEL") or settings.model


# ───────────────────────── 실행 ─────────────────────────
class ContextGenerator:
    def __init__(self, settings: ContextSettings, cache: ContextCache, llm: LlmCall | None = None):
        self.settings = settings
        self.cache = cache
        self.model = resolve_model(settings)
        self.llm = llm or openai_llm(self.model)

    def _key(self, page: Chunk, c: Chunk) -> str:
        return ContextCache.key(self.settings.prompt_version, self.model, page.text, c.text)

    def _one_page(self, page: Chunk, targets: list[Chunk], stats: ContextStats, lock: threading.Lock) -> None:
        todo = [c for c in targets if self.cache.get(self._key(page, c)) is None]
        if not todo:
            with lock:
                stats.pages_cached += 1
            return
        chunk_list = "\n".join(f"- {c.chunk_id}: {c.text}" for c in todo)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(
                doc_title=page.doc_title, page_header=page.page_header, page=page.page,
                page_text=page.text, chunk_list=chunk_list)},
        ]
        try:
            raw, usage = self.llm(messages)
            items = json.loads(raw).get("contexts", [])
        except Exception as e:  # noqa: BLE001 — API·JSON 오류는 기록만 하고 다음 쪽으로 넘어간다
            with lock:
                stats.errors.append(f"{page.chunk_id}: {type(e).__name__}: {e}"[:300])
            log.warning("맥락 생성 실패 %s: %s", page.chunk_id, e)
            return
        got = {it.get("chunk_id"): (it.get("context") or "").strip() for it in items if isinstance(it, dict)}
        allowed = f"{page.text} {page.doc_title} {page.page_header} {page.book_id} {page.page}"
        new, rejected, samples = {}, 0, []
        for c in todo:
            ctx = got.get(c.chunk_id, "")
            problems = context_problems(ctx, allowed, self.settings.max_chars)
            if problems:
                if ctx:
                    rejected += 1
                    samples.append({"chunk_id": c.chunk_id, "context": ctx[:200], "problems": problems[:3]})
                continue
            new[self._key(page, c)] = ctx
        self.cache.put_many(new)
        with lock:
            stats.pages_called += 1
            stats.rejected += rejected
            stats.rejected_samples.extend(samples[: max(0, 10 - len(stats.rejected_samples))])
            stats.prompt_tokens += usage.get("prompt_tokens", 0)
            stats.completion_tokens += usage.get("completion_tokens", 0)

    def fill(self, chunks: list[Chunk]) -> ContextStats:
        """chunks의 context 필드를 채운다 (제자리 수정). 문서 하나 또는 여러 문서를 넣어도 된다."""
        stats = ContextStats()
        pages = {c.chunk_id: c for c in chunks if c.level == "page"}
        targets: dict[str, list[Chunk]] = defaultdict(list)
        for c in chunks:
            if c.level in self.settings.levels:
                targets[f"{c.book_id}-p{c.page}"].append(c)
        lock = threading.Lock()
        jobs = [(pages[pid], cs) for pid, cs in targets.items() if pid in pages]
        with ThreadPoolExecutor(max_workers=max(1, self.settings.max_workers)) as ex:
            list(ex.map(lambda job: self._one_page(job[0], job[1], stats, lock), jobs))

        by_id: dict[str, str] = {}
        for pid, cs in targets.items():
            page = pages.get(pid)
            for c in cs:
                ctx = self.cache.get(self._key(page, c)) if page else None
                c.context = ctx or ""
                by_id[c.chunk_id] = c.context
        for c in chunks:   # row는 속한 표의 맥락을 그대로
            if c.level == "row" and c.parent_id in by_id:
                c.context = by_id[c.parent_id]
        for c in chunks:
            if c.level in self.settings.levels or c.level == "row":
                if c.context:
                    stats.chunks_filled += 1
                else:
                    stats.missing += 1
        return stats
