"""실험 비교판(evaluation/reporting/experiment_board.py) 테스트. API·Qdrant·모델 없이 돈다.

가짜 결과 파일을 tmp 폴더에 만들고, 비교판이 읽은 값과 만든 HTML·CSV를 확인한다.

    uv run pytest tests/evaluation/test_experiment_board.py -q
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

BOARD_PATH = (
    Path(__file__).resolve().parents[2]
    / "evaluation"
    / "reporting"
    / "experiment_board.py"
)


@pytest.fixture(scope="module")
def board():
    spec = importlib.util.spec_from_file_location("experiment_board", BOARD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _jsonl(path: Path, rows, extra: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    path.write_text(text + "\n" + extra, encoding="utf-8")


def _summary(page_hit: float, mrr: float = 0.3) -> dict:
    stats = {
        "doc_hit": 0.9,
        "doc_recall": 0.8,
        "doc_group_recall": 0.8,
        "page_hit": page_hit,
        "page_recall": 0.3,
        "page_group_recall": 0.5,
        "doc_mrr": 0.7,
        "page_mrr": mrr,
        "n": 28,
    }
    return {"1": {**stats, "page_hit": page_hit / 2}, "5": stats}


def _search_result(page_hit: float) -> dict:
    return {
        "ks": [1, 5],
        "n_questions": 30,
        "n_errors": 0,
        "short_results": 0,
        "latency_ms": {"mean": 40.0, "median": 38.5, "max": 60.0},
        "summary": _summary(page_hit),
        "build_seconds": 3.2,
        "items": [],
    }


def _answer(qid: str, status: str, latency: float | None = 4000.0, tokens=None) -> dict:
    return {
        "id": qid,
        "mode": "x",
        "question": f"{qid} 질문",
        "status": status,
        "answer": "" if status == "error" else "답",
        "reason": "",
        "source_ids": [],
        "contexts": [],
        "retrieval_ms": 100.0 if status != "error" else None,
        "latency_ms": latency if status != "error" else None,
        "tokens": tokens if tokens is not None else {"input": 1000, "output": 200},
        "error": "TimeoutError: x" if status == "error" else None,
    }


def _faith(qid: str, state: str, value: float | None, status: str = "answered") -> dict:
    return {
        "id": qid,
        "mode": "x",
        "status": status,
        "faithfulness": value,
        "state": state,
        "cause": None,
        "reason": "",
        "eval_ms": 10.0,
    }


def _config(k: int = 5, rerank_n: int = 50) -> dict:
    return {
        "created_at": "2026-11-10T02:29:53+00:00",
        "split": "development",
        "domain": "shipping",
        "difficulty": "basic",
        "generation": {"model": "gpt-6-luna"},
        "retrieval": {
            "impl": "retrieval_b",
            "k": k,
            "levels": ["para", "table"],
            "candidate_k": 30,
            "rrf_k": 60,
            "rerank_n": rerank_n,
            "rerank_max_length": 1024,
        },
        "pipeline": "검색기 → format_docs → answer_prompt → 모델",
    }


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """레포 모양의 tmp 폴더: 골든셋 + 검색 비교 2개 + 진단 1개 + 답변 평가 폴더 2개 + 깨진 파일."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    _write(
        tmp_path / "data/basic/shipping/evaluation/development.json",
        [
            {"id": "Q1", "expected_action": "answer"},
            {"id": "Q2", "expected_action": "answer"},
            {"id": "Q3", "expected_action": "clarify"},
            {"id": "Q4", "expected_action": "abstain"},
        ],
    )
    search = tmp_path / "results/comparison/retrieval_b"
    # 예전 형식: settings.retrieval_b가 없다 (튜닝 값 기록 없음)
    _write(
        search / "basic_development_para-table.json",
        {
            "created_at": "2026-11-10T02:02:00+00:00",
            "git_commit": "f0ae1b3",
            "settings": {
                "domain": "shipping",
                "difficulty": "basic",
                "split": "development",
                "levels": ["para", "table"],
                "ks": [1, 5],
                "limit": None,
                "modes": ["naive", "rerank", "a_rerank"],
            },
            "results": {
                "naive": _search_result(0.607),
                "rerank": _search_result(0.571),
                "a_rerank": {"error": "RuntimeError: 모델 없음"},
            },
        },
    )
    _write(
        search / "basic_development_para-table_n20.json",
        {
            "created_at": "2026-11-10T03:30:00+00:00",
            "git_commit": "f0ae1b3",
            "settings": {
                "domain": "shipping",
                "difficulty": "basic",
                "split": "development",
                "levels": ["para", "table"],
                "ks": [1, 5],
                "limit": None,
                "modes": ["rerank"],
                "retrieval_b": {
                    "candidate_k": 30,
                    "rrf_k": 60,
                    "rerank_n": 20,
                    "rerank_max_length": 1024,
                    "rerank_batch_size": 16,
                },
            },
            "results": {"rerank": _search_result(0.643)},
        },
    )
    (search / "broken.json").write_text("{깨진", encoding="utf-8")
    _write(
        tmp_path / "results/diagnose/retrieval_b/rerank_basic_development_k5_n20.json",
        {
            "created_at": "2026-11-10T03:40:00+00:00",
            "settings": {
                "domain": "shipping",
                "difficulty": "basic",
                "levels": ["para", "table"],
                "split": "development",
                "k": 5,
                "rerank_n": 20,
                "probe_n": 100,
                "hybrid_k": 10,
                "limit": None,
            },
            "summary": {
                "n": 28,
                "labels": {
                    "둘 다 맞힘": 15,
                    "밀림": 3,
                    "이김": 3,
                    "후보 없음": 2,
                    "둘 다 놓침": 5,
                    "정답 없음": 2,
                },
                "same_book": 9,
                "hybrid_page_hit": 0.6429,
                "rerank_page_hit": 0.6429,
                "candidate_page_hit": 0.9286,
            },
            "items": [],
        },
    )
    ans = tmp_path / "results/answer/retrieval_b"
    k5 = ans / "basic_development_k5"
    _write(k5 / "config.json", _config())
    _jsonl(
        k5 / "answers_naive.jsonl",
        [
            _answer("Q1", "answered", 3000.0),
            _answer("Q2", "abstained", 5000.0),
            _answer("Q3", "needs_clarification", 4000.0),
            _answer("Q4", "error"),
        ],
        extra="{깨진 줄\n",  # 끊긴 실행: 깨진 줄은 건너뛴다
    )
    _jsonl(
        k5 / "answers_hybrid.jsonl",
        [
            _answer("Q1", "answered", 3500.0),
            _answer("Q2", "answered", 4500.0, tokens={"input": 900, "output": 100}),
            _answer("Q3", "needs_clarification", 4000.0),
            _answer("Q4", "abstained", 4200.0),
        ],
    )
    _jsonl(
        k5 / "faithfulness_naive.jsonl",
        [
            _faith("Q1", "scored", 1.0),
            _faith("Q2", "scored", 0.0, "abstained"),  # 0점도 평균에 넣는다
            _faith("Q3", "na", None, "needs_clarification"),
            _faith("Q4", "unavailable", None, "error"),
        ],
    )
    _jsonl(
        k5 / "faithfulness_hybrid.jsonl",
        [
            _faith("Q1", "scored", 0.8),
            _faith("Q2", "scored", 0.6),
            _faith("Q3", "scored", 1.0, "needs_clarification"),
            _faith("Q4", "scored", 0.5, "abstained"),
        ],
    )
    _jsonl(
        k5 / "grading_usage.jsonl",
        [
            {
                "step": "faithfulness",
                "mode": "naive",
                "calls": 2,
                "input": 300,
                "output": 50,
            },
            {
                "step": "faithfulness",
                "mode": "naive",
                "calls": 2,
                "input": 200,
                "output": 50,
            },
            {
                "step": "faithfulness",
                "mode": "hybrid",
                "calls": 2,
                "input": 100,
                "output": 10,
            },
        ],
    )
    # config.json이 없는 폴더: 이름으로 난이도·split을 짐작하고, 설정은 "기록 없음"
    k3 = ans / "basic_development_k3"
    _jsonl(k3 / "answers_naive.jsonl", [_answer("Q1", "answered")])
    (ans / "empty_folder").mkdir(parents=True)
    return tmp_path


# ---------- 기본값 ----------


def test_defaults_match_settings(board):
    settings = pytest.importorskip("rag_service.retrieval_b.settings")
    s = settings.Settings()
    assert board.DEFAULTS["k"] == s.k
    assert board.DEFAULTS["levels"] == "+".join(s.levels)
    for key in board.TUNING_KEYS:
        assert board.DEFAULTS[key] == getattr(s, key), key


def test_changed_settings(board):
    assert board.changed_settings({"rerank_n": 50}, "para+table") == {}
    assert board.changed_settings({"rerank_n": 20, "rrf_k": 60}, "page", 3) == {
        "k": 3,
        "levels": "page",
        "rerank_n": 20,
    }
    assert board.changed_settings(None, "para+table") is None  # 기록 없음
    assert board.changed_settings(None, "page") == {
        "levels": "page",
        "_unrecorded": True,
    }


# ---------- 검색 비교 ----------


def test_collect_search(board, root):
    rows, skipped = board.collect_search(root)
    by = {(r["run"], r["mode"]): r for r in rows}
    old = by[("basic_development_para-table", "rerank")]
    assert old["changed"] is None  # 예전 파일: 튜닝 값 기록 없음
    assert old["by_k"]["5"]["page_hit"] == 0.571
    assert old["latency_median"] == 38.5
    new = by[("basic_development_para-table_n20", "rerank")]
    assert new["changed"] == {"rerank_n": 20}
    # a 방식은 retrieval_b 튜닝 값을 안 쓰므로 기본값, 실패한 방식은 오류만
    a = by[("basic_development_para-table", "a_rerank")]
    assert a["changed"] == {} and "모델 없음" in a["error"] and "by_k" not in a
    assert any("broken.json" in s for s in skipped)


def test_search_csv_uses_k(board, root):
    rows, _ = board.collect_search(root)
    out = board.search_csv_rows(rows, 5)
    first = next(r for r in out if r["실행"].endswith("_n20"))
    assert first["쪽 Hit@5"] == 0.643
    assert first["쪽 Hit@1"] == pytest.approx(0.3215)
    assert first["바꾼 설정"] == "rerank_n=20"


# ---------- 리랭크 진단 ----------


def test_collect_diagnose(board, root):
    rows, skipped = board.collect_diagnose(root)
    assert skipped == []
    (r,) = rows
    assert r["changed"] == {"rerank_n": 20}
    assert r["labels"]["밀림"] == 3
    assert r["rerank_page_hit"] == 0.6429
    assert r["mode"] == "rerank"  # 기준 실행과 짝지을 때 쓴다


# ---------- 답변 평가 ----------


def test_collect_answers_counts_and_faithfulness(board, root):
    rows, _ = board.collect_answers(root)
    by = {(r["run"], r["mode"]): r for r in rows}
    naive = by[("basic_development_k5", "naive")]
    assert naive["n"] == 4  # 깨진 줄은 빼고 센다
    assert (
        naive["n_answered"],
        naive["n_abstained"],
        naive["n_needs_clarification"],
        naive["n_error"],
    ) == (1, 1, 1, 1)
    assert naive["error_rate"] == 0.25
    # Q1 answered=answer, Q3 clarify 일치 / Q2 abstained≠answer, Q4 error≠abstain
    assert (naive["status_match"], naive["status_total"]) == (2, 4)
    # 0점은 넣고 해당 없음·평가 불가는 뺀다: (1.0 + 0.0) / 2
    assert naive["faith_mean"] == 0.5
    assert (
        naive["faith_n_scored"],
        naive["faith_n_na"],
        naive["faith_n_unavailable"],
    ) == (2, 1, 1)
    assert naive["faith_answered"] == 1.0
    # 오류 줄은 시간에서 뺀다
    assert naive["latency_median"] == 4000.0
    assert naive["tokens_in"] == 4000 and naive["tokens_known"] == 4
    assert (naive["grading_in"], naive["grading_out"]) == (500, 100)
    assert naive["changed"] == {}


def test_collect_answers_common_answered(board, root):
    rows, _ = board.collect_answers(root)
    by = {(r["run"], r["mode"]): r for r in rows}
    naive, hybrid = (
        by[("basic_development_k5", "naive")],
        by[("basic_development_k5", "hybrid")],
    )
    # 두 방식 모두 answered인 문항은 Q1뿐
    assert naive["common_n"] == hybrid["common_n"] == 1
    assert naive["faith_common"] == 1.0 and hybrid["faith_common"] == 0.8
    assert hybrid["faith_answered"] == 0.7  # Q1 0.8, Q2 0.6
    assert hybrid["status_match"] == 4  # 네 문항 모두 기대 동작과 같다
    assert hybrid["tokens_in"] == 3900


def test_collect_answers_without_config(board, root):
    rows, skipped = board.collect_answers(root)
    k3 = next(r for r in rows if r["run"] == "basic_development_k3")
    assert k3["changed"] is None  # config.json 없음 → 기록 없음
    # 폴더 이름으로 난이도·split을 짐작한다
    assert (k3["difficulty"], k3["split"]) == ("basic", "development")
    assert k3["status_match"] == 1
    assert k3["faith_mean"] is None and k3["faith_n_rows"] == 0  # 아직 채점 전
    assert any("empty_folder" in s for s in skipped)


# ---------- 파일 쓰기 ----------


def test_write_board_html_and_csv(board, root):
    out = root / "results" / "experiment_board.html"
    data = board.write_board(root, out)
    html = out.read_text(encoding="utf-8")
    assert "<title>실험 비교판</title>" in html
    start = html.index('id="board-data">') + len('id="board-data">')
    payload = json.loads(
        html[start : html.index("</script>", start)].replace("<\\/", "</")
    )
    assert (
        payload["counts"] == data["counts"] == {"search": 4, "diagnose": 1, "answer": 3}
    )
    for name in ("search", "diagnose", "answer"):
        csv_path = root / "results" / f"experiment_board_{name}.csv"
        assert csv_path.read_bytes().startswith(b"\xef\xbb\xbf")  # 엑셀용 BOM
    assert "rerank_n=20" in (root / "results/experiment_board_search.csv").read_text(
        encoding="utf-8-sig"
    )


def test_payload_cannot_close_script_tag(board):
    data = board.collect(Path("/nonexistent-root"))
    data["skipped"] = ["</script><b>x</b>"]
    html = board.build_html(data)
    body = html[html.index('id="board-data">') :]
    assert "</script><b>" not in body.split("</script>", 1)[0]


def test_main_prints_counts(board, root, capsys):
    out = board.main(["--root", str(root), "--out", str(root / "board.html")])
    assert out.exists()
    text = capsys.readouterr().out
    assert "검색 4줄, 리랭크 진단 1줄, 답변 평가 3줄" in text
    assert "건너뜀" in text


def test_empty_results(board, tmp_path):
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    data = board.write_board(tmp_path, tmp_path / "results" / "b.html")
    assert data["counts"] == {"search": 0, "diagnose": 0, "answer": 0}
    # 행이 없으면 CSV를 안 만든다
    assert not (tmp_path / "results" / "b_search.csv").exists()


# ---------- k 그래프 선 묶기 ----------


def _row(board_mod, run, mode, created, by_k, changed=None, levels="para+table"):
    return {
        "run": run,
        "mode": mode,
        "created_at": created,
        "split": "development",
        "difficulty": "basic",
        "levels": levels,
        "changed": {} if changed is None else changed,
        "error": None,
        "by_k": by_k,
    }


def test_search_lines_join_runs_that_differ_only_in_k(board):
    """k만 다른 실행은 한 선, 설정이 다르면 다른 선. 같은 k는 가장 최근 파일 값을 쓴다."""
    rows = [
        _row(
            board,
            "base",
            "naive",
            "2026-11-10T03:00:00+00:00",
            {"1": {"page_hit": 0.2}, "3": {"page_hit": 0.5}, "5": {"page_hit": 0.6}},
        ),
        _row(
            board,
            "base_k3",
            "naive",
            "2026-11-10T06:00:00+00:00",
            {"3": {"page_hit": 0.57}},
        ),
        _row(
            board,
            "base_k7",
            "naive",
            "2026-11-10T06:05:00+00:00",
            {"7": {"page_hit": 0.61}},
        ),
        _row(
            board, "base", "bm25", "2026-11-10T03:00:00+00:00", {"5": {"page_hit": 0.7}}
        ),
        _row(
            board,
            "n20",
            "rerank",
            "2026-11-10T04:00:00+00:00",
            {"5": {"page_hit": 0.64}},
            changed={"rerank_n": 20},
        ),
        _row(
            board,
            "page",
            "naive",
            "2026-11-10T04:10:00+00:00",
            {"5": {"page_hit": 0.64}},
            changed={"levels": "page"},
            levels="page",
        ),
        {
            **_row(board, "broken", "naive", "2026-11-10T07:00:00+00:00", {}),
            "error": "x",
        },
    ]
    lines = {line["label"]: line for line in board.search_lines(rows)}
    assert set(lines) == {
        "naive · 기본값",
        "bm25 · 기본값",
        "rerank · rerank_n=20",
        "naive · levels=page",
    }
    naive = lines["naive · 기본값"]
    assert sorted(naive["by_k"], key=int) == ["1", "3", "5", "7"]
    assert naive["by_k"]["3"]["page_hit"] == 0.57  # 최근 파일(base_k3)이 이긴다
    assert naive["source"] == {"1": "base", "3": "base_k3", "5": "base", "7": "base_k7"}


def test_search_lines_label_for_unrecorded_and_other_split(board):
    row = _row(
        board, "old", "hybrid", "2026-11-09T00:00:00+00:00", {"5": {"page_hit": 0.6}}
    )
    row["changed"] = None
    row["split"] = "additional"
    assert board.line_label(row) == "hybrid · 기록 없음 · additional"


def test_board_payload_has_search_lines(board, root):
    data = board.collect(root)
    assert "search_lines" in data
    assert all(
        {"label", "mode", "by_k", "source"} <= set(line)
        for line in data["search_lines"]
    )


def test_html_search_defaults_to_k_graph_and_hides_rows_without_k(board, root):
    """검색 섹션은 처음에 k별 그래프로 열리고, 표에서는 고른 k 값이 없는 줄을 숨기는 안내가 있다 (JS 설정 확인)."""
    html = board.build_html(board.collect(root))
    assert "defaultView: 'ktable', hideMissingK: true" in html
    assert "id: 'ktable', label: 'k별 표'" in html
    assert "toggle-missing" in html and "board-view2-" in html
    assert "tableLabel: '파일별 표'" in html and "data-kmetric" in html


# ---------- 자동 갱신 ----------


def test_refresh_writes_board_and_prints_uri(board, root, capsys):
    out = board.refresh(root)
    assert out == root / board.OUT_HTML and out.exists()
    assert "비교판 갱신: file:" in capsys.readouterr().out


def test_refresh_failure_does_not_raise(board, root, monkeypatch, capsys):
    def boom(*a, **k):
        raise RuntimeError("깨짐")

    monkeypatch.setattr(board, "write_board", boom)
    assert board.refresh(root) is None
    assert "비교판 갱신 실패" in capsys.readouterr().out


def test_board_writes_version_file_and_polls_it(board, root):
    out = root / board.OUT_HTML
    data = board.write_board(root, out)
    version = root / "results" / "experiment_board_version.js"
    expected = f'window.BOARD_VERSION = "{data["version"]}";\n'
    assert version.read_text(encoding="utf-8") == expected
    html = out.read_text(encoding="utf-8")
    assert f'"version": "{data["version"]}"' in html  # 페이지에 자기 버전이 들어 있다
    assert "experiment_board_version.js?t=" in html and 'id="autorefresh"' in html
    again = board.write_board(root, out)
    assert again["version"] != data["version"]  # 다시 만들면 버전이 바뀐다


# ---------- 실험 선택 ----------


def test_experiment_id_merges_k_files_only(board):
    assert (
        board.experiment_id("basic_development_para-table_k3")
        == "basic_development_para-table_k*"
    )
    assert (
        board.experiment_id("basic_development_para-table_k15")
        == "basic_development_para-table_k*"
    )
    assert (
        board.experiment_id("basic_development_para-table")
        == "basic_development_para-table"
    )
    assert (
        board.experiment_id("basic_development_para-table_n20")
        == "basic_development_para-table_n20"
    )
    assert (
        board.experiment_id("basic_development_para-table_n20_k7-15")
        == "basic_development_para-table_n20_k7-15"
    )


def _exp_row(run, created, ks):
    return {
        "run": run,
        "experiment": None,
        "created_at": created,
        "by_k": {str(k): {"page_hit": 0.5} for k in ks},
    }


def test_search_experiments_labels_ks_and_newest_first(board):
    rows = []
    for k, t in ((3, "06:16"), (5, "06:17"), (7, "06:19")):
        for mode in ("naive", "bm25"):
            rows.append(
                {
                    **_exp_row(f"base_k{k}", f"2026-11-10T{t}:00+00:00", [k]),
                    "mode": mode,
                }
            )
    rows.append(
        {
            **_exp_row("base", "2026-11-10T03:51:00+00:00", [1, 3, 5, 10]),
            "mode": "naive",
        }
    )
    rows.append(
        {**_exp_row("base_n20", "2026-11-10T03:58:00+00:00", [1, 5]), "mode": "rerank"}
    )
    for r in rows:
        r["experiment"] = board.experiment_id(r["run"])
    exps = board.search_experiments(rows)
    assert [e["id"] for e in exps] == ["base_k*", "base_n20", "base"]  # 최근 것부터
    k = exps[0]
    assert k["label"] == "base_k* (k3~k7, 3개 파일)"
    assert k["ks"] == [3, 5, 7] and k["files"] == ["base_k3", "base_k5", "base_k7"]
    assert exps[2]["label"] == "base" and exps[2]["ks"] == [1, 3, 5, 10]


def test_payload_has_experiments_and_lines_per_experiment(board, root):
    data = board.collect(root)
    assert "search_experiments" in data and "search_lines_by_experiment" in data
    ids = {e["id"] for e in data["search_experiments"]}
    assert ids == set(data["search_lines_by_experiment"])
    assert all("experiment" in r for r in data["search"])


def test_empty_results_have_no_experiments(board, tmp_path):
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    data = board.collect(tmp_path)
    assert data["search_experiments"] == [] and data["search_lines_by_experiment"] == {}


def test_search_lines_keep_latency_per_k(board):
    rows = [
        {
            **_row(
                board,
                "a_k3",
                "naive",
                "2026-11-10T06:00:00+00:00",
                {"3": {"page_hit": 0.5}},
            ),
            "latency_median": 40.0,
        },
        {
            **_row(
                board,
                "a_k5",
                "naive",
                "2026-11-10T06:01:00+00:00",
                {"5": {"page_hit": 0.6}},
            ),
            "latency_median": 35.5,
        },
    ]
    (line,) = board.search_lines(rows)
    assert line["latency"] == {"3": 40.0, "5": 35.5}


# ---------- k별 표 ----------


def _kline(mode, by_k, latency, label=None):
    return {
        "label": label or f"{mode} · 기본값",
        "mode": mode,
        "by_k": {k: {"page_hit": v} for k, v in by_k.items()},
        "latency": latency,
        "source": {k: f"x_k{k}" for k in by_k},
    }


def test_by_k_table_rows_best_ties_and_delta(board):
    lines = [
        _kline("hybrid", {"3": 0.5, "5": 0.643}, {"3": 40.0, "5": 45.0}),
        _kline("naive", {"3": 0.571, "5": 0.607}, {"3": 30.0, "5": 30.0}),
        _kline("bm25", {"3": 0.536, "5": 0.643}, {"3": 3.0, "5": 4.0}),
    ]
    t = board.by_k_table(lines)
    assert [line["mode"] for line in t["lines"]] == [
        "naive",
        "bm25",
        "hybrid",
    ]  # 방식 순서 고정
    assert t["ks"] == [3, 5]
    hit3 = t["cells"]["page_hit"]["3"]
    assert [c["best"] for c in hit3] == [True, False, False]
    hit5 = t["cells"]["page_hit"]["5"]
    assert [c["best"] for c in hit5] == [False, True, True]  # 동점은 모두
    assert [c["delta"] for c in hit3] == [None, None, None]  # 첫 k는 차이 없음
    assert [c["delta"] for c in hit5] == [0.036, 0.107, 0.143]
    lat5 = t["cells"]["latency"]["5"]
    assert [c["best"] for c in lat5] == [False, True, False]  # ms는 낮을수록 좋음
    assert lat5[0]["delta"] == 0.0 and hit5[1]["src"] == "x_k5"
    assert t["cells"]["page_mrr"]["3"][0]["v"] is None  # 없는 값은 None


def test_search_by_k_csv(board, tmp_path):
    data = {
        "search_experiments": [{"id": "e_k*", "label": "e_k* (k3~k5, 2개 파일)"}],
        "search_ktables": {
            "e_k*": board.by_k_table(
                [_kline("naive", {"3": 0.5, "5": 0.6}, {"3": 30.0, "5": 31.0})]
            )
        },
    }
    rows = board.search_by_k_csv_rows(data)
    assert [r["k"] for r in rows] == [3, 5]
    assert rows[0]["실험"] == "e_k* (k3~k5, 2개 파일)"
    assert rows[1]["쪽 Hit naive · 기본값"] == 0.6
    assert rows[1]["검색 ms 중앙값 naive · 기본값"] == 31.0
    assert rows[1]["파일"] == "x_k5"
    path = tmp_path / "by_k.csv"
    board.write_csv(path, rows)
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")  # 다른 CSV와 같은 BOM


def test_write_board_writes_search_by_k_csv(board, root):
    out = root / board.OUT_HTML
    board.write_board(root, out)
    assert (root / "results" / "experiment_board_search_by_k.csv").exists()
    data = board.collect(root)
    assert "" in data["search_ktables"]
