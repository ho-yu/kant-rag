"""청킹 결과를 문서별 jsonl로 저장하고, 바뀐 문서의 이전 파일은 history로 옮긴다 (모두 로컬 전용).

artifacts/corpus/{난이도}/{도메인}/
├─ SB01.jsonl … SB15.jsonl      ← 최신만. 문서 하나 = 파일 하나
├─ manifest.json                 ← 문서별 버전·지문·청크 수·이력
├─ context_cache.jsonl           ← 맥락 캐시 (이것만 깃에 커밋해서 팀원과 공유)
└─ history/SB02/v1.jsonl         ← 바뀌거나 지워진 문서의 이전 버전

- 문서 파일이 max_shard_mb를 넘으면 쪽 경계에서 SB01_001.jsonl, SB01_002.jsonl … 로 나눈다.
  한 쪽(쪽 레코드 + 그 자식들)은 절대 두 파일로 나누지 않는다.
- 줄 순서: 쪽 → (쪽 레코드, 그 쪽의 자식들) 순서로 항상 같게 쓴다.
"""
from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from rag_service.contracts.chunk import Chunk

MANIFEST = "manifest.json"
CACHE_FILE = "context_cache.jsonl"
HISTORY_DIR = "history"


def now() -> str:
    return datetime.now(UTC).astimezone().isoformat(timespec="seconds")


def serialize(chunks: list[Chunk]) -> list[str]:
    return [json.dumps(c.model_dump(exclude_none=True), ensure_ascii=False) for c in chunks]


def split_by_size(lines: list[str], max_shard_mb: float) -> list[list[str]]:
    """쪽 경계에서만 나눠 파일당 max_shard_mb 이하로 만든다 (쪽 하나가 한도를 넘으면 그 쪽만 한 파일)."""
    limit = int(max_shard_mb * 1024 * 1024)
    pages: list[list[str]] = []
    for line in lines:
        if json.loads(line)["level"] == "page" or not pages:
            pages.append([])
        pages[-1].append(line)
    parts: list[list[str]] = [[]]
    size = 0
    for page in pages:
        nbytes = sum(len(s.encode("utf-8")) + 1 for s in page)
        if parts[-1] and size + nbytes > limit:
            parts.append([])
            size = 0
        parts[-1].extend(page)
        size += nbytes
    return parts


def doc_file_names(book_id: str, n_parts: int) -> list[str]:
    if n_parts == 1:
        return [f"{book_id}.jsonl"]
    return [f"{book_id}_{i:03d}.jsonl" for i in range(1, n_parts + 1)]


def write_doc(base: Path, book_id: str, lines: list[str], max_shard_mb: float) -> list[str]:
    """문서 하나를 최신 파일로 쓴다. 반환: 파일 이름 목록."""
    base.mkdir(parents=True, exist_ok=True)
    parts = split_by_size(lines, max_shard_mb)
    names = doc_file_names(book_id, len(parts))
    for name, part in zip(names, parts, strict=True):
        with (base / name).open("w", encoding="utf-8", newline="\n") as f:
            for line in part:
                f.write(line + "\n")
    return names


def archive_doc(base: Path, book_id: str, entry: dict) -> list[str]:
    """최신 파일을 history/{book_id}/v{n}.jsonl 로 옮긴다. 반환: base 기준 상대 경로 목록."""
    files = entry.get("files", [])
    version = entry.get("version", 1)
    dest_dir = base / HISTORY_DIR / book_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    moved = []
    for i, name in enumerate(files, 1):
        src = base / name
        if not src.exists():
            continue
        suffix = "" if len(files) == 1 else f"_{i:03d}"
        dest = dest_dir / f"v{version}{suffix}.jsonl"
        shutil.move(str(src), dest)
        moved.append(dest.relative_to(base).as_posix())
    return moved


def read_lines(base: Path, files: list[str]) -> list[str]:
    out = []
    for name in files:
        with (base / name).open(encoding="utf-8") as f:
            out.extend(line.rstrip("\n") for line in f if line.strip())
    return out


def read_manifest(base: Path) -> dict:
    f = base / MANIFEST
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    return {"documents": {}, "removed": {}, "runs": []}


def write_manifest(base: Path, manifest: dict) -> None:
    base.mkdir(parents=True, exist_ok=True)
    manifest["updated_at"] = now()
    manifest["runs"] = manifest.get("runs", [])[-50:]
    (base / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
