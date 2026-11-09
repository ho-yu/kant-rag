"""report.py: 검색 비교 결과 → HTML 페이지."""

from __future__ import annotations

import importlib.util
import json

from rag_service.retrieval_b import report as report_mod
from rag_service.retrieval_b.report import page_link, render_html, write_html
from rag_service.storage.corpus import repo_root


def _score(hit, group, missing, ids):
    return {
        "doc_hit": hit,
        "doc_recall": hit,
        "doc_group_recall": hit,
        "page_hit": hit,
        "page_recall": group,
        "page_group_recall": group,
        "found_chunk_ids": ids,
        "missing_pages": missing,
    }


def _mode(summary_hit, q1, q2, latency=10.0):
    summary = dict.fromkeys(
        [
            "doc_hit",
            "doc_recall",
            "doc_group_recall",
            "page_hit",
            "page_recall",
            "page_group_recall",
        ],
        summary_hit,
    )
    summary["n"] = 1
    return {
        "latency_ms": {"mean": latency, "median": latency, "max": latency},
        "build_seconds": 1.0,
        "n_errors": 0,
        "short_results": 0,
        "summary": {"5": summary},
        "items": [
            {
                "id": "Q1",
                "expected_action": "answer",
                "n_groups": 2,
                "error": None,
                "scores": {"5": q1},
            },
            {
                "id": "Q2",
                "expected_action": "clarify",
                "n_groups": 0,
                "error": None,
                "scores": {"5": q2},
            },
        ],
    }


NO_GOLD = {**_score(None, None, [], ["X-p1-c1"]), "doc_group_recall": None}


def make_report():
    return {
        "created_at": "2026-11-09T05:47:02+00:00",
        "git_commit": "abc1234",
        "settings": {
            "difficulty": "basic",
            "split": "development",
            "levels": ["para", "table"],
            "ks": [5],
        },
        "results": {
            "naive": _mode(
                0.5, _score(0, 0.0, [["SB01", 1], ["SB02", 3]], ["SB09-p1-c1"]), NO_GOLD
            ),
            "bm25": _mode(
                1.0, _score(1, 0.5, [["SB02", 3]], ["SB01-p1-c1"]), NO_GOLD, latency=2.0
            ),
            "a_hybrid": {"error": "RuntimeError: 연결 <실패>"},
        },
    }


ITEMS = [
    {"id": "Q1", "question": "부산–LA <b>예약</b> 정보는?", "evidence_groups": []},
    {"id": "Q2", "question": "마감이 몇 시죠?", "evidence_groups": []},
]


def test_page_is_self_contained():
    page = render_html(make_report(), ITEMS)
    assert page.startswith("<!doctype html>")
    assert "<meta charset='utf-8'>" in page
    assert "<script src" not in page and "<link" not in page


def test_header_and_failed_mode_escaped():
    page = render_html(make_report(), ITEMS)
    assert "검색 비교 결과: basic / development / para+table" in page
    assert "커밋 abc1234" in page
    assert "<b>a_hybrid 실패:</b> RuntimeError: 연결 &lt;실패&gt;" in page


def test_best_value_highlighted_only_for_winner():
    page = render_html(make_report(), ITEMS)
    bm25 = next(r for r in page.split("<tr") if r.startswith("><td>bm25"))
    naive = next(r for r in page.split("<tr") if r.startswith("><td>naive"))
    assert 'class="num best">1.000' in bm25
    assert "best" not in naive


def test_question_rows_marks_groups_and_escape():
    page = render_html(make_report(), ITEMS)
    assert (
        '<tr data-group="split"><td class="nw">Q1</td><td class=\'nw\'>답변</td>'
        in page
    )
    assert "부산–LA &lt;b&gt;예약&lt;/b&gt; 정보는?" in page
    assert '<td class="bad">× (0/2)</td><td class="ok">○ (1/2)</td>' in page
    assert '<tr data-group="no_gold"><td class="nw">Q2</td>' in page
    assert "놓친 정답 쪽: SB01 p1, SB02 p3" in page
    assert "<code>SB01-p1-c1</code>" in page


def test_filter_counts():
    page = render_html(make_report(), ITEMS)
    assert ">전체 2</button>" in page
    assert ">방식마다 갈림 1</button>" in page
    assert ">모든 방식이 놓침 0</button>" in page
    assert ">채점 제외 1</button>" in page


def test_all_missed_group():
    report = make_report()
    report["results"]["bm25"]["items"][0]["scores"]["5"] = _score(
        0, 0.0, [["SB01", 1]], ["SB07-p2-c1"]
    )
    page = render_html(report, ITEMS)
    assert '<tr data-group="all_missed"><td class="nw">Q1</td>' in page


def test_without_items_and_detail_k_fallback():
    page = render_html(make_report(), None, detail_k=10)
    assert "한눈에 보기 (k=5)" in page
    assert "<td class='q'>-</td>" in page


def test_only_failed_modes():
    report = make_report()
    report["results"] = {"naive": {"error": "boom"}}
    page = render_html(report)
    assert "<b>naive 실패:</b> boom" in page
    assert "한눈에 보기" not in page


def test_write_html_utf8(tmp_path):
    out = write_html(make_report(), tmp_path / "sub" / "r.html", ITEMS)
    assert "부산–LA" in out.read_text(encoding="utf-8")


def test_cli_converts_json(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "retrieval_report",
        repo_root() / "evaluation" / "reporting" / "retrieval_report.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "load_golden", lambda path: ITEMS)
    src = tmp_path / "basic_development_para-table.json"
    src.write_text(json.dumps(make_report(), ensure_ascii=False), encoding="utf-8")
    opened = []
    monkeypatch.setattr(report_mod.webbrowser, "open", opened.append)
    out = mod.main([str(src), "--open"])
    assert out == src.with_suffix(".html")
    assert "부산–LA" in out.read_text(encoding="utf-8")
    assert opened == [out.resolve().as_uri()]


def test_page_link_returns_file_uri_and_opens_only_when_asked(tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(report_mod.webbrowser, "open", opened.append)
    path = tmp_path / "r.html"
    path.write_text("x", encoding="utf-8")
    uri = page_link(path)
    assert uri.startswith("file:") and uri.endswith("/r.html")
    assert opened == []
    assert page_link(path, open_browser=True) == uri
    assert opened == [uri]
