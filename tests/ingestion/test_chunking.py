"""청킹 테스트. 레포에 들어 있는 실제 PDF(data/)를 사용한다.

실행: uv run pytest tests/ingestion -q
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from rag_service.contracts.chunk import Chunk, chunk_point_id
from rag_service.ingestion import build_corpus
from rag_service.ingestion.chunker import (
    ChunkSettings,
    chunk_document,
    split_long_paragraph,
)
from rag_service.ingestion.context import (
    ContextCache,
    ContextGenerator,
    ContextSettings,
    validate_context,
)
from rag_service.ingestion.pdf_reader import join_two, read_pdf
from rag_service.ingestion.writer import serialize, split_by_size

ROOT = Path(__file__).resolve().parents[2]
BASIC = ROOT / "data/basic/shipping/documents"
INTER = ROOT / "data/intermediate/shipping/documents"
ID_RE = re.compile(r"^[A-Z]+\d+-p\d+(-c\d+)?$")
SETTINGS = ChunkSettings()

needs_pdf = pytest.mark.skipif(not (BASIC / "SB01.pdf").exists(), reason="data PDF 없음")


@pytest.fixture(scope="module")
def sb01() -> list[Chunk]:
    return chunk_document(read_pdf(BASIC / "SB01.pdf"), "basic", "shipping", SETTINGS)


@pytest.fixture(scope="module")
def is01() -> list[Chunk]:
    return chunk_document(read_pdf(INTER / "IS01.pdf"), "intermediate", "shipping", SETTINGS)


# ───────────────────────── 줄 잇기 ─────────────────────────
def test_join_keeps_existing_space():
    assert join_two("바꾸어 사용하지", " 않는다.") == "바꾸어 사용하지 않는다."


def test_join_after_punctuation_adds_space():
    assert join_two("확인하며,", "확인서에는") == "확인하며, 확인서에는"


# ───────────────────────── 문단 청크 ─────────────────────────
@needs_pdf
def test_ids_and_parents(sb01):
    ids = {c.chunk_id for c in sb01}
    assert len(ids) == len(sb01), "chunk_id 중복"
    for c in sb01:
        assert ID_RE.match(c.chunk_id), c.chunk_id
        assert c.book_id == "SB01" and c.file_name == "SB01.pdf"
        assert c.text.strip()
        if c.level == "page":
            assert c.parent_id is None and c.chunk_id == f"SB01-p{c.page}"
        else:
            assert c.parent_id in ids
            assert c.chunk_id.startswith(f"SB01-p{c.page}-c")


@needs_pdf
def test_one_page_record_per_page(sb01):
    pages = [c.page for c in sb01 if c.level == "page"]
    assert pages == list(range(1, 25))


@needs_pdf
def test_paragraphs_never_cross_pages(sb01):
    page_text = {c.page: re.sub(r"\s+", "", c.text) for c in sb01 if c.level == "page"}
    for c in sb01:
        if c.level == "para":
            assert re.sub(r"\s+", "", c.text) in page_text[c.page]


@needs_pdf
def test_header_and_page_number_removed(sb01):
    p3 = next(c for c in sb01 if c.chunk_id == "SB01-p3")
    assert "3 / 24" not in p3.text
    assert p3.page_header == "SB01 | 예약지원팀"
    assert p3.section_title == "접수와 예약 확정의 구별"


def test_long_paragraph_split_on_sentence_boundary():
    text = " ".join(f"문장 {i}번은 길이를 늘리기 위한 예시 문장이다." for i in range(60))
    parts = split_long_paragraph(text, 100, "BAAI/bge-m3")
    assert len(parts) > 1
    assert all(p.endswith(".") for p in parts)
    assert re.sub(r"\s+", "", "".join(parts)) == re.sub(r"\s+", "", text)


# ───────────────────────── 표 ─────────────────────────
@needs_pdf
def test_table_and_rows(is01):
    tables = [c for c in is01 if c.level == "table" and c.page == 6]
    assert len(tables) == 1
    t = tables[0]
    assert t.table_title == "서비스 범위와 제외 대상 — 대상별 처리 원문"
    rows = [c for c in is01 if c.level == "row" and c.parent_id == t.chunk_id]
    assert len(rows) == 8
    headers = ["접수번호", "문의처·품명", "문의 항목", "범위", "소관 판단"]
    for r in rows:
        for h in headers:
            assert f"{h}:" in r.text, (h, r.text)
    assert "IS01-006-03" in rows[2].text and "안내 범위" in rows[2].text


@needs_pdf
def test_table_footnote_kept():
    doc = read_pdf(INTER / "IS11.pdf")
    chunks = chunk_document(doc, "intermediate", "shipping", SETTINGS)
    note = "행의 배열은 날짜순을 의미하지 않는다"
    t = next(c for c in chunks if c.level == "table" and c.page == 1)
    rows = [c for c in chunks if c.level == "row" and c.parent_id == t.chunk_id]
    assert note in t.text
    assert all(note in r.text for r in rows)
    assert not any(note in c.text for c in chunks if c.level == "para" and c.page == 1)


# ───────────────────────── jsonl 분할 ─────────────────────────
@needs_pdf
def test_big_document_split_on_page_boundary(sb01):
    lines = serialize(sb01)
    parts = split_by_size(lines, max_shard_mb=0.05)          # 약 50KB씩 → 여러 파일
    assert len(parts) > 1
    assert [ln for part in parts for ln in part] == lines     # 순서·내용 그대로
    for part in parts:
        assert json.loads(part[0])["level"] == "page"         # 각 파일은 쪽 레코드로 시작 (쪽이 안 쪼개짐)


def test_point_id_is_stable_uuid():
    assert chunk_point_id("SB01-p3-c2") == chunk_point_id("SB01-p3-c2")
    assert chunk_point_id("SB01-p3-c2") != chunk_point_id("SB01-p3-c3")


# ───────────────────────── 맥락 (가짜 LLM) ─────────────────────────
def test_validate_context_rejects_new_numbers():
    page = "보관료 무료기간은 3달력일이다. 예약 HA28-C007."
    assert validate_context("HA28-C007 예약의 보관료 무료기간 설명", page, 300)
    assert not validate_context("무료기간 5일에 대한 설명", page, 300)       # 5는 원문에 없음
    assert not validate_context("HA28-C999 예약 설명", page, 300)          # 없는 ID
    assert not validate_context("가" * 301, page, 300)


@needs_pdf
def test_context_generator_with_fake_llm(tmp_path, is01):
    chunks = [c.model_copy() for c in is01 if c.page == 6]
    calls = []

    def fake_llm(messages):
        calls.append(messages)
        ids = re.findall(r"^- (\S+):", messages[-1]["content"], flags=re.MULTILINE)
        return json.dumps({"contexts": [{"chunk_id": i, "context": "서비스 범위와 제외 대상 절의 내용"} for i in ids]}), \
            {"prompt_tokens": 100, "completion_tokens": 10}

    gen = ContextGenerator(ContextSettings(max_workers=1), ContextCache(tmp_path / "cache.jsonl"), llm=fake_llm)
    st = gen.fill(chunks)
    assert st.pages_called == 1 and st.missing == 0
    table = next(c for c in chunks if c.level == "table")
    assert all(c.context == table.context for c in chunks if c.level == "row")   # row는 표 맥락
    # 두 번째 실행은 캐시에서 → LLM 호출 없음
    again = [c.model_copy(update={"context": ""}) for c in chunks]
    st2 = ContextGenerator(ContextSettings(max_workers=1), ContextCache(tmp_path / "cache.jsonl"), llm=fake_llm).fill(again)
    assert len(calls) == 1 and st2.pages_cached == 1 and st2.missing == 0


# ───────────────────────── 증분 실행·버전 ─────────────────────────
@pytest.fixture()
def mini_repo(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='t'\n", encoding="utf-8")
    (tmp_path / "config").mkdir(parents=True)
    shutil.copy(ROOT / "config/chunking.yaml", tmp_path / "config/chunking.yaml")
    docs = tmp_path / "data/basic/shipping/documents"
    docs.mkdir(parents=True)
    for b in ("SB01", "SB02"):
        shutil.copy(BASIC / f"{b}.pdf", docs / f"{b}.pdf")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def corpus(repo: Path) -> Path:
    return repo / "artifacts/corpus/basic/shipping"


def manifest(repo: Path) -> dict:
    return json.loads((corpus(repo) / "manifest.json").read_text(encoding="utf-8"))


def run(*extra) -> int:
    return build_corpus.main(["--difficulty", "basic", "--no-context", "--no-index", *extra])


@needs_pdf
def test_only_changed_documents_are_rewritten(mini_repo):
    base = corpus(mini_repo)
    assert run() == 0
    assert sorted(f.name for f in base.glob("*.jsonl")) == ["SB01.jsonl", "SB02.jsonl"]
    sb01_before = (base / "SB01.jsonl").read_bytes()
    m = manifest(mini_repo)
    assert m["documents"]["SB01"]["version"] == 1 and m["documents"]["SB02"]["version"] == 1

    # 바뀐 게 없으면 아무 파일도 안 바뀜
    runs = len(m["runs"])
    assert run() == 0
    assert len(manifest(mini_repo)["runs"]) == runs

    # SB02 PDF가 바뀜 → SB02만 v2, 이전 파일은 history로. SB01 파일은 바이트까지 그대로
    shutil.copy(BASIC / "SB03.pdf", mini_repo / "data/basic/shipping/documents/SB02.pdf")
    assert run() == 0
    m = manifest(mini_repo)
    assert m["runs"][-1]["changed"] == ["SB02"]
    assert m["documents"]["SB02"]["version"] == 2 and m["documents"]["SB01"]["version"] == 1
    assert (base / "SB01.jsonl").read_bytes() == sb01_before
    assert (base / "history/SB02/v1.jsonl").exists()
    assert m["documents"]["SB02"]["history"][0]["files"] == ["history/SB02/v1.jsonl"]


@needs_pdf
def test_added_removed_and_restore(mini_repo):
    base = corpus(mini_repo)
    docs = mini_repo / "data/basic/shipping/documents"
    assert run() == 0
    v1_text = (base / "SB02.jsonl").read_text(encoding="utf-8")

    # 새 문서 추가 → SB03만 처리
    shutil.copy(BASIC / "SB03.pdf", docs / "SB03.pdf")
    assert run() == 0
    last = manifest(mini_repo)["runs"][-1]
    assert last["added"] == ["SB03"] and "changed" not in last

    # PDF 삭제 → 최신 파일은 history로, manifest의 removed로
    (docs / "SB03.pdf").unlink()
    assert run() == 0
    m = manifest(mini_repo)
    assert not (base / "SB03.jsonl").exists() and (base / "history/SB03/v1.jsonl").exists()
    assert "SB03" in m["removed"] and "SB03" not in m["documents"]

    # SB02를 바꿨다가 v1으로 되돌리기
    shutil.copy(BASIC / "SB04.pdf", docs / "SB02.pdf")
    assert run() == 0
    assert (base / "SB02.jsonl").read_text(encoding="utf-8") != v1_text
    assert run("--restore", "SB02@v1") == 0
    m = manifest(mini_repo)
    assert (base / "SB02.jsonl").read_text(encoding="utf-8") == v1_text
    assert m["documents"]["SB02"]["restored_from"] == "v1" and m["documents"]["SB02"]["version"] == 3
    assert (base / "history/SB02/v2.jsonl").exists()                    # 되돌리기 전 최신본도 보관
    assert run("--restore", "SB02@v9") == 1                              # 없는 버전


@needs_pdf
def test_loader_reads_latest_only(mini_repo):
    from rag_service.storage.corpus import PageStore, load_chunks

    assert run() == 0
    shutil.copy(BASIC / "SB03.pdf", mini_repo / "data/basic/shipping/documents/SB02.pdf")
    assert run() == 0
    chunks = list(load_chunks("basic"))
    assert {c.book_id for c in chunks} == {"SB01", "SB02"}
    pages = PageStore("basic")
    assert len(pages) == len([c for c in chunks if c.level == "page"])
    para = next(c for c in chunks if c.level == "para" and c.book_id == "SB02")
    assert pages.page_of(para).chunk_id == para.parent_id
    # history의 구버전은 읽지 않는다 (SB02 쪽 수 = 현재 PDF 쪽 수)
    assert len([c for c in chunks if c.level == "page" and c.book_id == "SB02"]) == \
        manifest(mini_repo)["documents"]["SB02"]["pages"]


@needs_pdf
def test_cached_context_needs_no_api_key(tmp_path, is01, monkeypatch):
    """팀원이 context_cache.jsonl만 받아도 OpenAI 키 없이 같은 맥락을 채운다."""
    from rag_service.ingestion import context as ctx_mod

    chunks = [c.model_copy() for c in is01 if c.page == 6]

    def fake_llm(messages):
        ids = re.findall(r"^- (\S+):", messages[-1]["content"], flags=re.MULTILINE)
        return json.dumps({"contexts": [{"chunk_id": i, "context": "서비스 범위 절의 내용"} for i in ids]}), {}

    ContextGenerator(ContextSettings(max_workers=1), ContextCache(tmp_path / "c.jsonl"), llm=fake_llm).fill(chunks)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    again = [c.model_copy(update={"context": ""}) for c in chunks]
    gen = ContextGenerator(ContextSettings(max_workers=1), ContextCache(tmp_path / "c.jsonl"),
                           llm=ctx_mod.openai_llm("gpt-6-luna"))           # 진짜 OpenAI 호출 함수 (키 없음)
    st = gen.fill(again)
    assert st.missing == 0 and st.pages_called == 0 and [c.context for c in again] == [c.context for c in chunks]


def test_context_may_mention_doc_id_and_page_number():
    """LLM이 쓴 문서 ID·쪽 번호는 지어낸 숫자가 아니다 (실제 실행에서 이것 때문에 절반이 버려졌던 문제)."""
    from rag_service.ingestion.context import context_problems

    page_text = "예약 확정은 상태가 확정인 예약 확인서로 확인한다."
    allowed = f"{page_text} 가온루트해운 예약 안내 SB01 | 예약지원팀 SB01 17"
    assert context_problems("SB01 문서 p17의 예약 확정 절", allowed, 300) == []
    assert context_problems("예약 확정 절, 출항 3일 전", allowed, 300) == ["원문에 없음: 3"]
