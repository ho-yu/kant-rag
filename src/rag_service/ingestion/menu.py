"""청킹·되돌리기 텍스트 메뉴. 번호를 입력하고 엔터. 0 = 뒤로, q = 종료.

실행: uv run python -m rag_service.ingestion.menu

    1. 청킹 실행      주제 → 난이도 → 바뀐 문서 확인·옵션 → 실행
    2. 되돌리기       주제 → 난이도 → 문서 → 버전 → 바뀐 쪽 미리보기 → 실행

실제 작업은 build_corpus.main()을 그대로 부른다 (명령줄 실행과 결과가 같다).
"""
from __future__ import annotations

import difflib
import json
import os
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from rag_service.ingestion import build_corpus
from rag_service.ingestion.writer import read_manifest
from rag_service.storage.corpus import corpus_dir, load_config, repo_root

DOMAINS = [("shipping", "해운", True), ("finance", "금융", False), ("manufacturing", "제조", False)]
DIFFICULTIES = ["basic", "intermediate", "advanced"]
REASONS = {"changed": "내용 변경", "removed": "PDF 삭제", "restored-over": "되돌리기 전 보관"}
LINE = "=" * 60
THIN = "-" * 60


class Quit(Exception):
    """q 입력 → 메뉴 종료"""


def ask(prompt: str, choices: set[str]) -> str:
    while True:
        try:
            v = input(f"{prompt} > ").strip().lower()
        except EOFError as e:
            raise Quit from e
        if v == "q":
            raise Quit
        if v in choices:
            return v
        print(f"  다시 입력하세요 ({', '.join(sorted(choices))} 또는 q)")


def pause() -> None:
    try:
        input("\n엔터를 누르면 메뉴로 돌아갑니다 ")
    except EOFError as e:
        raise Quit from e


def pad(text: str, width: int) -> str:
    """한글은 두 칸으로 계산해서 열을 맞춘다."""
    w = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in str(text))
    return f"{text}{' ' * max(0, width - w)}"


def short(text: str, n: int = 50) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


# ───────────────────────── 공통: 주제·난이도 ─────────────────────────
def choose_domain(title: str) -> str | None:
    print(f"\n[{title}] 주제 선택")
    choices = {"0"}
    for i, (name, ko, ok) in enumerate(DOMAINS, 1):
        if ok:
            print(f" {i}. {name} ({ko})")
            choices.add(str(i))
        else:
            print(f" -  {name:<14} (사용 안 함)")
    print(" 0. 뒤로")
    v = ask("선택", choices)
    return None if v == "0" else DOMAINS[int(v) - 1][0]


def choose_difficulty(title: str, domain: str, cfg: dict) -> str | None:
    print(f"\n[{title}] 난이도 선택  주제: {domain}")
    for i, d in enumerate(DIFFICULTIES, 1):
        pdfs = len(list((repo_root() / cfg["paths"]["data_dir"] / d / domain / "documents").glob("*.pdf")))
        m = read_manifest(corpus_dir(d, domain, cfg))
        done = len(m.get("documents", {}))
        state = f"청킹됨 {done}개 | 마지막 실행 {m['updated_at'][:16].replace('T', ' ')}" if done else "청킹 안 됨"
        print(f" {i}. {d:<13} PDF {pdfs:>3}개 | {state}")
    print(" 0. 뒤로")
    v = ask("선택", {"0", "1", "2", "3"})
    return None if v == "0" else DIFFICULTIES[int(v) - 1]


# ───────────────────────── 1. 청킹 실행 ─────────────────────────
def chunk_flow(cfg: dict) -> None:
    domain = choose_domain("청킹 실행")
    if domain is None:
        return
    difficulty = choose_difficulty("청킹 실행", domain, cfg)
    if difficulty is None:
        return
    opts = {"context": bool(cfg.get("context", {}).get("enabled", True)), "index": True, "force": False}
    labels = {"1": ("context", "맥락 생성 (OpenAI)"), "2": ("index", "Qdrant 동기화"),
              "3": ("force", "전부 다시 청킹 (force)")}
    while True:
        scan = build_corpus.scan_changes(difficulty, domain, opts["context"], opts["force"], cfg)
        todo = len(scan["added"]) + len(scan["changed"]) + len(scan["removed"])
        print(f"\n[청킹 실행] {domain} / {difficulty}")
        print(THIN)
        print(f" 변경 확인: 새 문서 {len(scan['added'])}{_names(scan['added'])} | "
              f"바뀐 문서 {len(scan['changed'])}{_names(scan['changed'])} | "
              f"삭제된 PDF {len(scan['removed'])}{_names(scan['removed'])} | 그대로 {len(scan['same'])}")
        last_ctx = read_manifest(corpus_dir(difficulty, domain, cfg)).get("settings", {}).get("context", {})
        if last_ctx and bool(last_ctx.get("enabled")) != opts["context"] and scan["changed"]:
            state = "켜고" if last_ctx.get("enabled") else "끄고"
            print(f" * 지난 실행은 맥락 생성을 {state} 했습니다. 설정이 달라서 기존 문서가 모두 '바뀐 문서'로 잡힙니다.")
        print(THIN)
        for key, (opt, label) in labels.items():
            print(f" {key}. {pad(label, 26)} [{'켜짐' if opts[opt] else '꺼짐'}]")
        if opts["context"] and not os.getenv("OPENAI_API_KEY"):
            print("    ! .env에 OPENAI_API_KEY가 없습니다. 맥락 생성을 끄거나 키를 넣으세요.")
        if opts["index"] and not os.getenv("QDRANT_URL"):
            print("    ! QDRANT_URL이 없어 Qdrant는 로컬 폴더(artifacts/qdrant_local)에 저장됩니다.")
        print(" (번호를 입력하면 켜짐/꺼짐이 바뀝니다)\n")
        print(" s. 실행   0. 뒤로")
        v = ask("선택", {"0", "1", "2", "3", "s"})
        if v == "0":
            return
        if v in labels:
            opt = labels[v][0]
            opts[opt] = not opts[opt]
            continue
        if todo == 0:
            msg = "바뀐 문서가 없습니다." + (" Qdrant만 확인할까요?" if opts["index"] else " 실행할 것이 없습니다.")
            if not opts["index"]:
                print(msg)
                pause()
                return
            q = f"{msg} (y/n)"
        else:
            q = f"실행할까요? 문서 {todo}개를 처리합니다. (y/n)"
        if ask(q, {"y", "n"}) == "n":
            continue
        argv = ["--difficulty", difficulty, "--domain", domain]
        argv += [] if opts["context"] else ["--no-context"]
        argv += [] if opts["index"] else ["--no-index"]
        argv += ["--force"] if opts["force"] else []
        _run(argv)
        pause()
        return


def _names(books: list[str], n: int = 3) -> str:
    if not books:
        return ""
    more = f" 외 {len(books) - n}" if len(books) > n else ""
    return f" ({', '.join(books[:n])}{more})"


def _run(argv: list[str]) -> int:
    print(f"\n실행: build_corpus {' '.join(argv)}\n")
    try:
        code = build_corpus.main(argv)
    except Exception as e:  # noqa: BLE001 — 메뉴는 계속 쓸 수 있게 오류만 보여준다
        print(f"\n오류: {type(e).__name__}: {e}")
        return 1
    print("\n" + ("완료했습니다." if code == 0 else f"실패했습니다 (코드 {code}). 위 메시지를 확인하세요."))
    return code


# ───────────────────────── 2. 되돌리기 ─────────────────────────
def rollback_flow(cfg: dict) -> None:
    domain = choose_domain("되돌리기")
    if domain is None:
        return
    difficulty = choose_difficulty("되돌리기", domain, cfg)
    if difficulty is None:
        return
    base = corpus_dir(difficulty, domain, cfg)
    while True:
        m = read_manifest(base)
        rows = []
        for book, e in sorted(m.get("documents", {}).items()):
            if any(h.get("files") for h in e.get("history", [])):
                rows.append((book, e, f"v{e['version']}"))
        for book, e in sorted(m.get("removed", {}).items()):
            if any(h.get("files") for h in e.get("history", [])):
                rows.append((book, e, "삭제됨"))
        hidden = len(m.get("documents", {})) - sum(1 for r in rows if r[2] != "삭제됨")
        print(f"\n[되돌리기] {domain} / {difficulty} — 이전 버전이 있는 문서")
        if not rows:
            print(" 이전 버전이 있는 문서가 없습니다.")
            pause()
            return
        print(f" 번호  {pad('문서', 7)} {pad('현재', 7)} {pad('이전 버전', 20)} 마지막 변경")
        for i, (book, e, cur) in enumerate(rows, 1):
            vers = ", ".join(f"v{h['version']}" for h in e["history"] if h.get("files"))
            when = (e.get("removed_at") or e.get("updated_at") or "")[:16].replace("T", " ")
            print(f" {i:>3}.  {pad(book, 7)} {pad(cur, 7)} {pad(vers, 20)} {when}")
        if hidden:
            print(f" (이전 버전이 없는 문서 {hidden}개는 숨김)")
        print(" 0. 뒤로")
        v = ask("선택", {"0"} | {str(i) for i in range(1, len(rows) + 1)})
        if v == "0":
            return
        book, entry, cur = rows[int(v) - 1]
        if version_flow(difficulty, domain, base, book, entry, cur):
            return


def version_flow(difficulty: str, domain: str, base: Path, book: str, entry: dict, cur: str) -> bool:
    hist = sorted((h for h in entry.get("history", []) if h.get("files")), key=lambda h: -h["version"])
    while True:
        print(f"\n[{book} 버전 목록] 현재 {cur}")
        print(f" 번호  {pad('버전', 5)} {pad('보관 시각', 18)} {pad('보관 이유', 18)}  쪽   청크")
        for i, h in enumerate(hist, 1):
            chunks = " / ".join(f"{k} {n}" for k, n in (h.get("chunks") or {}).items())
            when = (h.get("archived_at") or "")[:16].replace("T", " ")
            pages = h.get("pages") or (h.get("chunks") or {}).get("page") or "-"
            reason = REASONS.get(h.get("reason"), h.get("reason", ""))
            print(f" {i:>3}.  {pad('v' + str(h['version']), 5)} {pad(when, 18)} {pad(reason, 18)} {pages:>3}   {chunks}")
        print(" 0. 뒤로")
        v = ask("선택", {"0"} | {str(i) for i in range(1, len(hist) + 1)})
        if v == "0":
            return False
        h = hist[int(v) - 1]
        print(f"\n[v{h['version']} ↔ 현재 {cur} 차이]")
        for line in diff_summary(base, h["files"], entry.get("files", [])):
            print(line)
        print(f"\n{book}를 v{h['version']} 내용으로 되돌립니다.")
        if entry.get("files"):
            print(f" - 현재 {cur}는 history에 보관됩니다")
        print(" - 되돌린 내용은 새 버전 번호로 기록됩니다")
        print(" - Qdrant도 맞춥니다 (바뀐 청크만)")
        print(" - 주의: PDF가 그대로면 다음 '청킹 실행' 때 다시 최신으로 바뀝니다")
        if ask("실행할까요? (y/n)", {"y", "n"}) == "n":
            continue
        _run(["--difficulty", difficulty, "--domain", domain, "--restore", f"{book}@v{h['version']}"])
        pause()
        return True


def _by_page(base: Path, files: list[str]) -> dict[int, dict]:
    pages: dict[int, dict] = defaultdict(lambda: {"page": "", "items": []})
    for name in files:
        with (base / name).open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r["level"] == "page":
                    pages[r["page"]]["page"] = r["text"]
                elif r["level"] in ("para", "table"):
                    pages[r["page"]]["items"].append(r["text"])
    return pages


def diff_summary(base: Path, old_files: list[str], new_files: list[str], max_pages: int = 8) -> list[str]:
    """두 버전을 쪽 단위로 비교한다 (ID가 아니라 원문으로 비교해서 번호가 밀려도 정확함)."""
    if not new_files:
        return [" 현재 파일이 없습니다 (PDF 삭제됨). 이 버전으로 되살립니다."]
    old, new = _by_page(base, old_files), _by_page(base, new_files)
    out, same = [], 0
    for p in sorted(set(old) | set(new)):
        if p not in new:
            out.append(f" {p}쪽: 이전 버전에만 있음")
            continue
        if p not in old:
            out.append(f" {p}쪽: 현재 버전에만 있음")
            continue
        if old[p]["page"] == new[p]["page"]:
            same += 1
            continue
        sm = difflib.SequenceMatcher(a=old[p]["items"], b=new[p]["items"], autojunk=False)
        counts = defaultdict(int)
        sample = None
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == "replace":
                counts["다름"] += max(i2 - i1, j2 - j1)
                sample = sample or (old[p]["items"][i1], new[p]["items"][j1])
            elif op == "insert":
                counts["추가"] += j2 - j1
                sample = sample or ("(없음)", new[p]["items"][j1])
            elif op == "delete":
                counts["삭제"] += i2 - i1
                sample = sample or (old[p]["items"][i1], "(없음)")
        desc = ", ".join(f"문단 {n}개 {k}" for k, n in counts.items()) or "제목·띄어쓰기 등 작은 차이"
        out.append(f" {p}쪽: {desc}")
        if sample:
            out.append(f"   - 선택한 버전: {short(sample[0])}")
            out.append(f"   + 현재 버전  : {short(sample[1])}")
    if len(out) > max_pages * 3:
        out = out[: max_pages * 3] + [" … (더 있음)"]
    out.append(f" 나머지 {same}쪽: 같음")
    return out


# ───────────────────────── 시작 ─────────────────────────
def main() -> int:
    load_dotenv(repo_root() / ".env")
    cfg = load_config()
    try:
        while True:
            print(f"\n{LINE}\n KANT RAG 청킹 관리\n{LINE}")
            print(" 1. 청킹 실행 (PDF → jsonl → Qdrant)")
            print(" 2. 이전 버전으로 되돌리기")
            print(" q. 종료")
            v = ask("선택", {"1", "2"})
            if v == "1":
                chunk_flow(cfg)
            else:
                rollback_flow(cfg)
    except (Quit, KeyboardInterrupt):
        print("\n종료합니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
