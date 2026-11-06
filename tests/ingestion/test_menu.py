"""텍스트 메뉴 테스트: 입력을 순서대로 흘려 넣고 화면 출력과 결과 파일을 확인한다."""
from __future__ import annotations

import builtins
import json
import shutil
from pathlib import Path

import pytest

from rag_service.ingestion import build_corpus, menu

ROOT = Path(__file__).resolve().parents[2]
BASIC = ROOT / "data/basic/shipping/documents"
needs_pdf = pytest.mark.skipif(not (BASIC / "SB01.pdf").exists(), reason="data PDF 없음")


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='t'\n", encoding="utf-8")
    (tmp_path / "config").mkdir(parents=True)
    shutil.copy(ROOT / "config/chunking.yaml", tmp_path / "config/chunking.yaml")
    docs = tmp_path / "data/basic/shipping/documents"
    docs.mkdir(parents=True)
    for b in ("SB01", "SB02"):
        shutil.copy(BASIC / f"{b}.pdf", docs / f"{b}.pdf")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("QDRANT_URL", raising=False)
    return tmp_path


def feed(monkeypatch, keys: list[str]) -> None:
    it = iter(keys)

    def fake_input(prompt=""):
        print(prompt, end="")
        try:
            return next(it)
        except StopIteration as e:
            raise EOFError from e

    monkeypatch.setattr(builtins, "input", fake_input)


@needs_pdf
def test_chunk_menu_toggles_and_runs(repo, monkeypatch, capsys):
    # 청킹 → shipping → basic → 맥락 끔 → Qdrant 끔 → 실행 → y → 엔터 → 종료
    feed(monkeypatch, ["1", "1", "1", "1", "2", "s", "y", "", "q"])
    assert menu.main() == 0
    out = capsys.readouterr().out
    assert "새 문서 2 (SB01, SB02)" in out
    assert "OPENAI_API_KEY가 없습니다" in out                      # 맥락 켜짐 상태의 경고
    assert "--no-context --no-index" in out and "완료했습니다." in out
    assert (repo / "artifacts/corpus/basic/shipping/SB01.jsonl").exists()

    # 다시 들어가면 바뀐 게 없음 → 실행할 것 없음
    feed(monkeypatch, ["1", "1", "1", "1", "2", "s", "", "q"])
    menu.main()
    out = capsys.readouterr().out
    assert "그대로 2" in out and "실행할 것이 없습니다" in out


@needs_pdf
def test_rollback_menu_shows_diff_and_restores(repo, monkeypatch, capsys):
    base = repo / "artifacts/corpus/basic/shipping"
    assert build_corpus.main(["--difficulty", "basic", "--no-context", "--no-index"]) == 0
    v1 = (base / "SB02.jsonl").read_text(encoding="utf-8")
    shutil.copy(BASIC / "SB03.pdf", repo / "data/basic/shipping/documents/SB02.pdf")
    assert build_corpus.main(["--difficulty", "basic", "--no-context", "--no-index"]) == 0
    monkeypatch.setattr("rag_service.ingestion.build_corpus.run_index", lambda args, cfg: 0)

    # 되돌리기 → shipping → basic → 문서 1(SB02) → 버전 1(v1) → y → 엔터 → 종료
    feed(monkeypatch, ["2", "1", "1", "1", "1", "y", "", "q"])
    assert menu.main() == 0
    out = capsys.readouterr().out
    assert "SB02" in out and "v1" in out and "(이전 버전이 없는 문서 1개는 숨김)" in out
    assert "[v1 ↔ 현재 v2 차이]" in out and "쪽:" in out
    assert (base / "SB02.jsonl").read_text(encoding="utf-8") == v1
    m = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    assert m["documents"]["SB02"]["restored_from"] == "v1"


def test_back_and_invalid_input(repo, monkeypatch, capsys):
    feed(monkeypatch, ["9", "1", "0", "q"])                     # 잘못된 입력 → 다시, 뒤로, 종료
    assert menu.main() == 0
    out = capsys.readouterr().out
    assert "다시 입력하세요" in out and "종료합니다." in out


def test_diff_ignores_shifted_ids(tmp_path):
    """번호가 밀려도 원문으로 비교해서 '1개 추가'로 보인다."""
    def rec(cid, page, level, text):
        return json.dumps({"chunk_id": cid, "page": page, "level": level, "text": text}, ensure_ascii=False)
    old = [rec("X-p1", 1, "page", "A B C"), rec("X-p1-c1", 1, "para", "A"), rec("X-p1-c2", 1, "para", "B"),
           rec("X-p1-c3", 1, "para", "C"), rec("X-p2", 2, "page", "D"), rec("X-p2-c1", 2, "para", "D")]
    new = [rec("X-p1", 1, "page", "A N B C"), rec("X-p1-c1", 1, "para", "A"), rec("X-p1-c2", 1, "para", "N"),
           rec("X-p1-c3", 1, "para", "B"), rec("X-p1-c4", 1, "para", "C"), rec("X-p2", 2, "page", "D"),
           rec("X-p2-c1", 2, "para", "D")]
    (tmp_path / "old.jsonl").write_text("\n".join(old), encoding="utf-8")
    (tmp_path / "new.jsonl").write_text("\n".join(new), encoding="utf-8")
    lines = menu.diff_summary(tmp_path, ["old.jsonl"], ["new.jsonl"])
    assert lines[0] == " 1쪽: 문단 1개 추가" and lines[-1] == " 나머지 1쪽: 같음"
