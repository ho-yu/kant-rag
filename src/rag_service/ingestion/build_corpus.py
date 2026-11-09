"""PDF → 청크 → (맥락) → 문서별 jsonl → Qdrant 동기화. 한 번에 실행하는 진입점.

실행 예
    uv run python -m rag_service.ingestion.build_corpus --difficulty basic                 # 청킹 + 맥락 + 색인
    uv run python -m rag_service.ingestion.build_corpus --difficulty basic --no-context    # 맥락 없이 (API 키 불필요)
    uv run python -m rag_service.ingestion.build_corpus --difficulty basic --no-index      # jsonl까지만
    uv run python -m rag_service.ingestion.build_corpus --difficulty basic --index-only    # 청킹 없이 Qdrant만 맞춤
    uv run python -m rag_service.ingestion.build_corpus --difficulty basic --restore SB02@v1   # 문서 하나 되돌리기

바뀐 것만 다시 처리
    문서마다 지문(PDF 해시, 설정 해시, 청커 코드 해시)을 manifest.json에 남긴다.
    - 지문이 같은 문서: 건너뜀
    - 바뀐 문서: 다시 청킹. 맥락은 캐시에 없는 쪽만 LLM 호출. 결과가 이전과 다르면
      이전 파일을 history/{문서}/v{n}.jsonl로 옮기고 새 파일을 쓴다 (버전 +1)
    - 새 문서: 추가 (v1) / PDF가 지워진 문서: 파일을 history로 옮기고 manifest의 removed로
    Qdrant는 마지막에 포인트 단위로 비교해서 바뀐 청크만 지우고 다시 올린다 (index_sync.sync).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAIError

from rag_service.ingestion import chunker, context, pdf_reader
from rag_service.ingestion.chunker import ChunkSettings, chunk_document
from rag_service.ingestion.context import (
    ContextCache,
    ContextGenerator,
    ContextSettings,
    resolve_model,
)
from rag_service.ingestion.writer import (
    CACHE_FILE,
    archive_doc,
    now,
    read_lines,
    read_manifest,
    serialize,
    write_doc,
    write_manifest,
)
from rag_service.storage.corpus import corpus_dir, load_config, repo_root

CHUNKER_VERSION = "1.1.0"   # 청킹 규칙을 바꾸면 올린다 (코드 해시와 함께 지문에 들어감)

log = logging.getLogger("build_corpus")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def code_hash() -> str:
    """청킹 결과에 영향을 주는 코드 파일의 해시 (윈도우 줄바꿈 차이는 무시)."""
    from rag_service.contracts import chunk as chunk_mod

    h = hashlib.sha256(CHUNKER_VERSION.encode())
    for mod in (pdf_reader, chunker, context, chunk_mod):
        h.update(Path(mod.__file__).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=repo_root(), timeout=5, check=False).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def build_settings(cfg: dict, use_context: bool) -> tuple[ChunkSettings, ContextSettings]:
    c = cfg["chunking"]
    cs = ChunkSettings(
        max_chunk_tokens=int(c["max_chunk_tokens"]),
        jsonl_levels=tuple(c["jsonl_levels"]),
        tokenizer=c.get("tokenizer", "BAAI/bge-m3"),
        max_table_tokens=int(c.get("max_table_tokens", 8000)),
    )
    if int(c.get("overlap_tokens", 0)) != 0:
        log.warning("overlap_tokens는 현재 지원하지 않아 0으로 처리합니다.")
    x = cfg.get("context", {})
    xs = ContextSettings(
        enabled=bool(x.get("enabled", True)) and use_context,
        model=x.get("model", "gpt-6-luna"),
        prompt_version=str(x.get("prompt_version", "v1")),
        max_workers=int(x.get("max_workers", 4)),
        max_chars=int(x.get("max_chars", 300)),
        levels=tuple(x.get("levels", ["para", "table"])),
    )
    return cs, xs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PDF를 청킹해서 문서별 jsonl로 저장하고 Qdrant에 동기화")
    ap.add_argument("--difficulty", required=True, choices=["basic", "intermediate", "advanced"])
    ap.add_argument("--domain", default="shipping")
    ap.add_argument("--config", default=None, help="기본: config/chunking.yaml")
    ap.add_argument("--force", action="store_true", help="지문이 같아도 모든 문서를 다시 청킹")
    ap.add_argument("--no-context", action="store_true", help="LLM 맥락 생성을 건너뜀 (context는 빈 값)")
    ap.add_argument("--books", nargs="*", default=None, help="일부 문서만 처리 (예: SB01 SB02). 나머지는 그대로 둠")
    ap.add_argument("--no-index", action="store_true", help="Qdrant 동기화를 건너뜀 (jsonl까지만)")
    ap.add_argument("--index-only", action="store_true", help="청킹 없이 지금 jsonl에 맞춰 Qdrant만 동기화")
    ap.add_argument("--reindex", action="store_true", help="Qdrant 컬렉션을 지우고 모든 청크를 다시 임베딩")
    ap.add_argument("--restore", default=None, metavar="BOOK@vN",
                    help="문서 하나를 history의 이전 버전으로 되돌림 (예: SB02@v1)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("httpx", "httpcore", "huggingface_hub", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    load_dotenv(repo_root() / ".env")
    cfg = load_config(args.config)
    if args.restore:
        return restore(args, cfg)
    if args.index_only:
        return run_index(args, cfg)

    cs, xs = build_settings(cfg, use_context=not args.no_context)
    pdf_dir = repo_root() / cfg["paths"]["data_dir"] / args.difficulty / args.domain / "documents"
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if args.books:
        pdfs = [p for p in pdfs if p.stem in set(args.books)]
    if not pdfs:
        log.error("PDF가 없습니다: %s", pdf_dir)
        return 1

    base = corpus_dir(args.difficulty, args.domain, cfg)
    manifest = read_manifest(base)
    docs: dict = manifest.setdefault("documents", {})
    removed: dict = manifest.setdefault("removed", {})

    settings_snapshot, cfg_hash, c_hash = fingerprint_basis(cfg, xs)
    max_mb = float(cfg["jsonl"]["max_shard_mb"])

    generator = None
    if xs.enabled:
        try:
            generator = ContextGenerator(xs, ContextCache(base / CACHE_FILE))
        except OpenAIError as e:  # 예: OPENAI_API_KEY 없음
            log.error("맥락 생성 준비 실패 (%s). .env의 OPENAI_API_KEY를 확인하거나 --no-context로 실행하세요.", e)
            return 1

    result = {"added": [], "changed": [], "same_output": [], "skipped": [], "removed": [], "failed": []}
    token_usage: Counter = Counter()
    token_by_doc: dict[str, dict] = {}   # 문서별 맥락 생성 토큰 (Langfuse 기록용)
    started = time.time()
    for i, pdf in enumerate(pdfs, 1):
        book_id = pdf.stem
        fp = {"pdf_sha256": sha256_file(pdf), "config_hash": cfg_hash, "code_hash": c_hash}
        old = docs.get(book_id)
        files_ok = old is not None and all((base / f).exists() for f in old.get("files", []))
        if is_up_to_date(old, fp, base, xs.enabled) and not args.force:
            result["skipped"].append(book_id)
            continue

        t0 = time.time()
        try:
            doc = pdf_reader.read_pdf(pdf)
            chunks = chunk_document(doc, args.difficulty, args.domain, cs)
        except Exception as e:  # noqa: BLE001 — PDF 하나가 깨져도 나머지 문서는 계속 처리한다
            # 이전 결과(jsonl·manifest 항목)는 그대로 둔다 → 다음 실행 때 지문이 달라 다시 시도함
            result["failed"].append({"book_id": book_id, "error": f"{type(e).__name__}: {e}"[:300]})
            log.error("[%d/%d] %s 읽기·청킹 실패: %s: %s", i, len(pdfs), book_id, type(e).__name__, e)
            continue
        ctx_info: dict = {"enabled": xs.enabled}
        if generator is not None:
            st = generator.fill(chunks)
            ctx_info.update(pages_called=st.pages_called, pages_cached=st.pages_cached,
                            filled=st.chunks_filled, missing=st.missing, rejected=st.rejected,
                            prompt_tokens=st.prompt_tokens, completion_tokens=st.completion_tokens,
                            errors=st.errors[:5], rejected_samples=st.rejected_samples[:5])
            token_usage.update(prompt=st.prompt_tokens, completion=st.completion_tokens)
            if st.pages_called:
                token_by_doc[book_id] = {"prompt_tokens": st.prompt_tokens, "completion_tokens": st.completion_tokens,
                                         "pages_called": st.pages_called, "pages_cached": st.pages_cached,
                                         "errors": len(st.errors)}
        lines = serialize(chunks)
        info = {"file_name": pdf.name, "title": doc.title, "pages": len(doc.pages), "fingerprint": fp,
                "chunks": dict(Counter(c.level for c in chunks)), "context": ctx_info}

        if files_ok and read_lines(base, old["files"]) == lines:
            # 지문은 바뀌었지만(예: 코드 수정) 결과가 똑같음 → 파일·버전 그대로, 지문만 갱신
            old.update(info)
            result["same_output"].append(book_id)
            continue

        history = list(old.get("history", [])) if old else []
        version = 1
        if old is not None:
            moved = archive_doc(base, book_id, old) if files_ok else []
            history.append(_history_entry(old, moved, "changed"))
            version = old.get("version", 1) + 1
            result["changed"].append(book_id)
        else:
            if book_id in removed:            # 지웠다가 다시 들어온 문서
                gone = removed.pop(book_id)
                history = gone.get("history", [])
                version = gone.get("version", 1) + 1
            result["added"].append(book_id)
        files = write_doc(base, book_id, lines, max_mb)
        docs[book_id] = {**info, "version": version, "files": files, "updated_at": now(), "history": history}
        log.info("[%d/%d] %s v%d: %d쪽, 청크 %s (%.1f초)", i, len(pdfs), book_id, version, len(doc.pages),
                 info["chunks"], time.time() - t0)

    if not args.books:                        # PDF가 지워진 문서
        for book_id in sorted(set(docs) - {p.stem for p in pdfs}):
            old = docs.pop(book_id)
            moved = archive_doc(base, book_id, old)
            old.setdefault("history", []).append(_history_entry(old, moved, "removed"))
            removed[book_id] = {**old, "files": [], "removed_at": now()}
            result["removed"].append(book_id)
            log.info("%s: PDF가 없어져 history로 옮겼습니다.", book_id)

    manifest.update({
        "difficulty": args.difficulty, "domain": args.domain, "chunker_version": CHUNKER_VERSION,
        "code_hash": c_hash, "config_hash": cfg_hash, "git_commit": git_commit(), "settings": settings_snapshot,
    })
    touched = result["added"] + result["changed"] + result["removed"]
    if touched or result["same_output"] or result["failed"]:
        manifest.setdefault("runs", []).append({
            "at": now(), **{k: v for k, v in result.items() if v and k != "skipped"},
            "context_tokens": dict(token_usage),
        })
    write_manifest(base, manifest)

    total = sum(sum(d["chunks"].values()) for d in docs.values())
    log.info("완료 (%.0f초): 추가 %d, 변경 %d, 결과 같음 %d, 건너뜀 %d, 삭제 %d, 실패 %d / 전체 문서 %d, 청크 %d → %s",
             time.time() - started, len(result["added"]), len(result["changed"]), len(result["same_output"]),
             len(result["skipped"]), len(result["removed"]), len(result["failed"]), len(docs), total, base)
    if result["failed"]:
        log.warning("읽기·청킹에 실패한 문서: %s (manifest.json의 runs에 오류 기록)",
                    ", ".join(f["book_id"] for f in result["failed"]))
    if token_usage:
        log.info("맥락 생성 토큰: 입력 %d, 출력 %d", token_usage["prompt"], token_usage["completion"])
    report_to_langfuse(args, generator.model if generator else None, token_by_doc, result)
    missing = sum(d.get("context", {}).get("missing", 0) for d in docs.values())
    if xs.enabled and missing:
        log.warning("맥락이 비어 있는 청크 %d개. 다시 실행하면 그 문서만 다시 시도합니다.", missing)
    return run_index(args, cfg)


def report_to_langfuse(args, model: str | None, token_by_doc: dict[str, dict], result: dict) -> None:
    """이번 실행의 맥락 생성 토큰을 Langfuse에 기록 하나(ingestion-run)로 남긴다.
    문서마다 generation 하나(해당 문서의 LLM 호출 토큰 합계). LLM을 부르지 않았으면 기록하지 않는다.
    Langfuse 오류는 경고만 하고 청킹 결과에는 영향을 주지 않는다."""
    if not token_by_doc and not result["failed"]:
        return
    try:
        from langfuse import get_client as get_langfuse_client

        langfuse = get_langfuse_client()
        total_in = sum(d["prompt_tokens"] for d in token_by_doc.values())
        total_out = sum(d["completion_tokens"] for d in token_by_doc.values())
        with langfuse.start_as_current_observation(
            as_type="span", name="ingestion-run",
            input={"difficulty": args.difficulty, "domain": args.domain, "model": model},
            metadata={"kind": "ingestion"},
        ) as run:
            for book_id, d in token_by_doc.items():
                with langfuse.start_as_current_observation(
                    as_type="generation", name=f"context-{book_id}", model=model,
                    usage_details={"input": d["prompt_tokens"], "output": d["completion_tokens"]},
                    metadata=d,
                ):
                    pass
            run.update(output={
                "input_tokens": total_in, "output_tokens": total_out, "docs_with_llm_calls": len(token_by_doc),
                **{k: len(v) for k, v in result.items()}, "failed_docs": result["failed"],
            })
        langfuse.flush()   # 스크립트가 바로 끝나므로 전송을 기다린다
        log.info("Langfuse에 ingestion-run 기록을 남겼습니다.")
    except Exception as e:  # noqa: BLE001 — 기록 실패로 청킹이 실패하면 안 된다
        log.warning("Langfuse 기록 실패 (청킹 결과는 정상): %s: %s", type(e).__name__, e)


def _history_entry(entry: dict, moved: list[str], reason: str) -> dict:
    return {"version": entry.get("version", 1), "files": moved, "fingerprint": entry.get("fingerprint"),
            "pages": entry.get("pages"), "chunks": entry.get("chunks"), "updated_at": entry.get("updated_at"),
            "archived_at": now(), "reason": reason}


def fingerprint_basis(cfg: dict, xs: ContextSettings) -> tuple[dict, str, str]:
    """지문에 들어가는 설정 스냅샷, 설정 해시, 코드 해시."""
    snapshot = {
        "chunking": cfg["chunking"],
        "context": {"enabled": xs.enabled, "model": resolve_model(xs) if xs.enabled else None,
                    "prompt_version": xs.prompt_version, "levels": list(xs.levels), "max_chars": xs.max_chars},
    }
    return snapshot, sha256_json(snapshot)[:16], code_hash()


def is_up_to_date(old: dict | None, fp: dict, base: Path, context_enabled: bool) -> bool:
    if old is None or old.get("fingerprint") != fp:
        return False
    if not all((base / f).exists() for f in old.get("files", [])):
        return False
    return not context_enabled or old.get("context", {}).get("missing", 0) == 0


def scan_changes(difficulty: str, domain: str = "shipping", use_context: bool = True, force: bool = False,
                 config: dict | None = None) -> dict:
    """청킹하지 않고 무엇이 바뀌었는지만 본다 (메뉴의 미리보기용)."""
    cfg = config or load_config()
    _, xs = build_settings(cfg, use_context)
    _, cfg_hash, c_hash = fingerprint_basis(cfg, xs)
    pdf_dir = repo_root() / cfg["paths"]["data_dir"] / difficulty / domain / "documents"
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    base = corpus_dir(difficulty, domain, cfg)
    docs = read_manifest(base).get("documents", {})
    out = {"added": [], "changed": [], "removed": [], "same": []}
    for pdf in pdfs:
        old = docs.get(pdf.stem)
        fp = {"pdf_sha256": sha256_file(pdf), "config_hash": cfg_hash, "code_hash": c_hash}
        if old is None:
            out["added"].append(pdf.stem)
        elif force or not is_up_to_date(old, fp, base, xs.enabled):
            out["changed"].append(pdf.stem)
        else:
            out["same"].append(pdf.stem)
    out["removed"] = sorted(set(docs) - {p.stem for p in pdfs})
    return out


def run_index(args, cfg: dict) -> int:
    """jsonl에 맞춰 Qdrant를 동기화. 실패해도 jsonl은 그대로 남는다."""
    if args.no_index:
        log.info("--no-index: Qdrant 동기화를 건너뜁니다.")
        return 0
    from rag_service.ingestion import index_sync

    try:
        summary = index_sync.sync(args.difficulty, args.domain, recreate=args.reindex, config=cfg)
    except Exception as e:  # noqa: BLE001 — 연결·모델 오류 등. jsonl은 이미 저장됐으니 안내만 한다
        log.error("Qdrant 동기화 실패: %s: %s", type(e).__name__, e)
        log.error("jsonl은 저장됐습니다. .env의 QDRANT_URL·QDRANT_API_KEY를 확인한 뒤 같은 명령을 다시 실행하면 "
                  "청킹은 건너뛰고 Qdrant만 이어서 맞춥니다.")
        return 2
    log.info("Qdrant %s: 포인트 %d개 (추가·변경 %d, 삭제 %d, 새로 임베딩 %d)", summary["collection"],
             summary["count"], summary["upserted"], summary["deleted"], summary["embedded"])
    return 0


def restore(args, cfg: dict) -> int:
    """문서 하나를 history의 이전 버전 파일로 되돌린다. 지금 파일은 history로 옮긴다."""
    try:
        book_id, ver = args.restore.split("@")
        want = int(ver.lstrip("v"))
    except ValueError:
        log.error("--restore 형식: 문서@버전 (예: SB02@v1)")
        return 1
    base = corpus_dir(args.difficulty, args.domain, cfg)
    manifest = read_manifest(base)
    entry = manifest.get("documents", {}).get(book_id) or manifest.get("removed", {}).get(book_id)
    if entry is None:
        log.error("%s 문서 기록이 없습니다.", book_id)
        return 1
    hist = next((h for h in entry.get("history", []) if h["version"] == want and h.get("files")), None)
    if hist is None:
        have = [h["version"] for h in entry.get("history", []) if h.get("files")]
        log.error("%s의 v%d가 history에 없습니다. 있는 버전: %s", book_id, want, have)
        return 1
    lines = read_lines(base, hist["files"])
    history = list(entry.get("history", []))
    if entry.get("files"):
        history.append(_history_entry(entry, archive_doc(base, book_id, entry), "restored-over"))
    files = write_doc(base, book_id, lines, float(cfg["jsonl"]["max_shard_mb"]))
    version = max([entry.get("version", 1)] + [h["version"] for h in history]) + 1
    manifest.get("removed", {}).pop(book_id, None)
    manifest.setdefault("documents", {})[book_id] = {
        **{k: v for k, v in entry.items() if k != "removed_at"},
        "version": version, "files": files, "fingerprint": hist["fingerprint"],
        "chunks": hist.get("chunks"), "updated_at": now(), "restored_from": f"v{want}", "history": history,
    }
    manifest.setdefault("runs", []).append({"at": now(), "restored": [f"{book_id}@v{want} → v{version}"]})
    write_manifest(base, manifest)
    log.info("%s를 v%d 내용으로 되돌렸습니다 (새 버전 v%d).", book_id, want, version)
    log.warning("주의: PDF가 그대로면 다음 일반 실행 때 PDF에 맞춰 다시 청킹됩니다 (지문이 다르므로).")
    return run_index(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
