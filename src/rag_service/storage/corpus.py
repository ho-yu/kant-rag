"""청킹 결과(최신 jsonl) 읽기 + 설정 읽기. 색인·BM25·답변 생성에서 공통으로 쓴다.

    from rag_service.storage.corpus import load_chunks, PageStore

    chunks = list(load_chunks("basic", levels=["para", "table"]))   # 최신 청크
    pages = PageStore("basic")
    parent_page = pages.page_of(chunks[0])                           # 부모 쪽 원문

history/ 아래 구버전과 context_cache.jsonl은 읽지 않는다.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import yaml

from rag_service.contracts.chunk import Chunk

DEFAULT_CONFIG = "config/chunking.yaml"
_NOT_CHUNKS = {"context_cache.jsonl"}


def repo_root(start: Path | None = None) -> Path:
    """pyproject.toml이 있는 폴더를 찾는다 (어느 폴더에서 실행해도 같은 경로를 쓰도록)."""
    p = (start or Path.cwd()).resolve()
    for d in [p, *p.parents]:
        if (d / "pyproject.toml").exists():
            return d
    return p


def load_config(path: str | Path | None = None) -> dict:
    root = repo_root()
    cfg_path = Path(path) if path else root / DEFAULT_CONFIG
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def corpus_dir(difficulty: str, domain: str = "shipping", config: dict | None = None) -> Path:
    cfg = config or load_config()
    return repo_root() / cfg["paths"]["corpus_dir"] / difficulty / domain


def chunk_files(base: Path) -> list[Path]:
    """최신 청크 파일 목록 (문서 이름순)."""
    return sorted(f for f in base.glob("*.jsonl") if f.name not in _NOT_CHUNKS)


def iter_jsonl(files: Iterable[Path]) -> Iterator[dict]:
    for f in files:
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def load_chunks(difficulty: str, domain: str = "shipping", levels: Iterable[str] | None = None,
                config: dict | None = None) -> Iterator[Chunk]:
    base = corpus_dir(difficulty, domain, config)
    files = chunk_files(base)
    if not files:
        raise FileNotFoundError(f"{base}에 청크 파일이 없습니다. 먼저 build_corpus를 실행하세요.")
    want = set(levels) if levels else None
    for rec in iter_jsonl(files):
        if want is None or rec["level"] in want:
            yield Chunk(**rec)


class PageStore:
    """부모 쪽 원문 조회용 (쪽 chunk_id → 쪽 Chunk). 쪽 단위는 jsonl에 항상 있다."""

    def __init__(self, difficulty: str, domain: str = "shipping", config: dict | None = None):
        self._pages = {c.chunk_id: c for c in load_chunks(difficulty, domain, ["page"], config)}

    def get(self, chunk_id: str | None) -> Chunk | None:
        return self._pages.get(chunk_id) if chunk_id else None

    def page_of(self, chunk: Chunk) -> Chunk | None:
        """청크가 속한 쪽 (row처럼 부모가 표인 경우도 쪽까지 올라간다)."""
        return self._pages.get(f"{chunk.book_id}-p{chunk.page}")

    def __len__(self) -> int:
        return len(self._pages)
