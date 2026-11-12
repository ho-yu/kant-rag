"""서버 Judge 점수 모으기 (evaluation/answer/server_judge_summary.py) 테스트.

Langfuse 없이 가짜 점수와 가짜 SDK 객체로 해당 없음·평가 불가 표시, 서비스 오류 규칙, 중복, 집계를 확인한다.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
from types import SimpleNamespace

import pytest

from rag_service.storage.corpus import repo_root


def _load():
    path = repo_root() / "evaluation" / "answer" / "server_judge_summary.py"
    spec = importlib.util.spec_from_file_location("server_judge_summary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sj = _load()


def _hit(value, comment="판단: 맞음\n근거: S1", ts="2026-11-12T01:00:00Z"):
    return {"value": value, "comment": comment, "timestamp": ts}


def _by_name(scores):
    return {s["name"]: s for s in scores}


def test_plain_scores_are_scored_with_reason():
    row = sj.from_server("judge_facts", [_hit(1)], "answered")
    assert (row["state"], row["value"]) == ("scored", 1)
    assert row["reason"].startswith("판단:")
    assert sj.from_server("judge_facts", [_hit(0)], "answered")["value"] == 0


def test_na_mark_becomes_na_but_not_for_decision():
    na = _hit(0, "[해당 없음] 판단: 되묻기만 함\n근거: 문서 내용 없음")
    assert (
        sj.from_server("judge_evidence", [na], "needs_clarification")["state"] == "na"
    )
    decision = sj.from_server("judge_decision", [na], "needs_clarification")
    assert (decision["state"], decision["cause"]) == ("unavailable", "judge_na")


@pytest.mark.parametrize(
    ("value", "state", "expected"),
    [
        ("pass", "scored", 1),
        ("fail", "scored", 0),
        ("na", "na", None),
        ("maybe", "unavailable", None),
    ],
)
def test_old_categorical_values_are_read(value, state, expected):
    row = sj.from_server("judge_conditions", [_hit(value)], "answered")
    assert (row["state"], row["value"]) == (state, expected)
    old_na_decision = sj.from_server("judge_decision", [_hit("na")], "answered")
    assert (old_na_decision["state"], old_na_decision["cause"]) == (
        "unavailable",
        "judge_na",
    )


def test_unavailable_mark_becomes_unavailable():
    row = sj.from_server("judge_facts", [_hit(0, "[평가 불가] 출력 없음")], "answered")
    assert (row["state"], row["cause"]) == ("unavailable", "judge_unavailable")


@pytest.mark.parametrize(
    ("comment", "mark"),
    [
        ("[해당 없음] 판단: 되묻기만 함", "[해당 없음]"),
        ("  판단: [해당 없음] 되묻기만 함\n근거: 없음", "[해당 없음]"),
        ("판단:[평가 불가] 출력 없음", "[평가 불가]"),
        ("판단: 0 — 필수 사실을 빠뜨림\n근거: [해당 없음]이 아님", None),
        ("", None),
        (None, None),
    ],
)
def test_mark_is_read_at_the_start_or_right_after_judgement(comment, mark):
    assert sj.mark_of(comment) == mark


def test_service_error_rule_overrides_server_scores():
    by = _by_name(sj.item_scores("error", {name: [_hit(1)] for name in sj.JUDGE_NAMES}))
    for name in ("judge_facts", "judge_conditions", "judge_evidence"):
        assert (by[name]["state"], by[name]["cause"]) == (
            "unavailable",
            "service_error",
        )
    assert (by["judge_decision"]["state"], by["judge_decision"]["value"]) == (
        "scored",
        0,
    )


def test_missing_and_bad_values_are_unavailable():
    by = _by_name(sj.item_scores("answered", {"judge_facts": [_hit(0.5)]}))
    assert by["judge_facts"]["cause"] == "bad_value"
    assert by["judge_conditions"]["cause"] == "missing"


def test_latest_duplicate_wins_and_is_counted():
    hits = [_hit(0, ts="2026-11-12T01:00:00Z"), _hit(1, ts="2026-11-12T01:05:00Z")]
    assert sj.from_server("judge_facts", hits, "answered")["value"] == 1
    items = [{"status": "answered", "scores": {"judge_facts": hits}}]
    assert sj.duplicates(items) == {"judge_facts": 1}


def test_summary_excludes_na_and_unavailable_but_keeps_zero():
    items = [
        {"status": "answered", "scores": {n: [_hit(1)] for n in sj.JUDGE_NAMES}},
        {"status": "answered", "scores": {n: [_hit(0)] for n in sj.JUDGE_NAMES}},
        {
            "status": "needs_clarification",
            "scores": {
                **{n: [_hit(0, "[해당 없음] 되묻기만 함")] for n in sj.JUDGE_NAMES},
                "judge_decision": [_hit(1)],
            },
        },
        {"status": "error", "scores": {}},
    ]
    s = sj.summarize_run(items)
    facts = s["judge_facts"]
    assert (
        facts["mean"],
        facts["n_scored"],
        facts["n_na"],
        facts["n_unavailable"],
    ) == (
        0.5,
        2,
        1,
        1,
    )
    decision = s["judge_decision"]
    # 1, 0, 1(되묻기), 0(서비스 오류) → 평균 0.5, 해당 없음 없음
    assert (decision["mean"], decision["n_scored"], decision["n_na"]) == (0.5, 4, 0)


def _page(data, cursor=None):
    return SimpleNamespace(data=data, meta=SimpleNamespace(cursor=cursor))


def _item(trace_id, output):
    return SimpleNamespace(trace_id=trace_id, output=output)


def _score(
    name,
    value,
    trace_id,
    comment="판단: 맞음",
    ts="2026-11-12T01:00:00Z",
    kind="observation",
):
    subject = (
        SimpleNamespace(kind="trace", id=trace_id)
        if kind == "trace"
        else SimpleNamespace(
            kind="observation", id="obs-" + trace_id, trace_id=trace_id
        )
    )
    return SimpleNamespace(
        name=name, value=value, comment=comment, timestamp=ts, subject=subject
    )


def test_load_items_reads_status_across_pages():
    pages = [
        _page([_item("t1", {"status": "answered"})], "next"),
        _page([_item("t2", json.dumps({"status": "error"})), _item("t3", None)]),
    ]
    calls = []

    def list_items(**kw):
        calls.append(kw)
        return pages[len(calls) - 1]

    api = SimpleNamespace(experiments=SimpleNamespace(list_items=list_items))
    since = dt.datetime(2026, 11, 5, tzinfo=dt.UTC)
    items = sj.load_items(api, "naive_20261112-1039", since)
    assert [c["cursor"] for c in calls] == [None, "next"]
    assert calls[0]["experiment_name"] == "naive_20261112-1039"
    assert calls[0]["from_start_time"] == since
    assert "io" in calls[0]["fields"]
    assert [(i["trace_id"], i["status"]) for i in items] == [
        ("t1", "answered"),
        ("t2", "error"),
        ("t3", "unknown"),
    ]


def test_load_judge_scores_asks_eval_scores_by_trace_batches():
    trace_ids = [f"t{i}" for i in range(12)]
    calls = []

    def get_many(**kw):
        calls.append(kw)
        if kw["trace_id"].startswith("t0") and kw["cursor"] is None:
            return _page(
                [
                    _score("judge_facts", 1.0, "t0", comment="[해당 없음] 되묻기만 함"),
                    _score("hit_at_5", 1.0, "t0"),  # Judge가 아닌 점수는 버린다
                    _score(
                        "judge_facts", 0.0, "other-run"
                    ),  # 다른 실행의 trace는 버린다
                ],
                "next",
            )
        if kw["cursor"] == "next":
            return _page([_score("judge_decision", 1.0, "t1", kind="trace")])
        return _page([])

    api = SimpleNamespace(scores=SimpleNamespace(get_many=get_many))
    out = sj.load_judge_scores(api, trace_ids)
    assert [c["trace_id"].count(",") + 1 for c in calls if c["cursor"] is None] == [
        10,
        2,
    ]
    assert all(c["source"] == "EVAL" for c in calls)
    assert all("details" in c["fields"] and "subject" in c["fields"] for c in calls)
    assert set(calls[0]["name"].split(",")) == set(sj.JUDGE_NAMES)
    assert out["t0"]["judge_facts"][0]["comment"].startswith("[해당 없음]")
    assert list(out["t0"]) == ["judge_facts"]
    assert out["t1"]["judge_decision"][0]["value"] == 1.0
    assert "other-run" not in out


def test_load_run_joins_status_and_scores_and_flags_absent_names():
    pages = [
        _page(
            [
                _item("t1", {"status": "answered"}),
                _item("t2", {"status": "needs_clarification"}),
            ]
        )
    ]

    def list_items(**kw):
        return pages[0]

    def get_many(**kw):
        return _page(
            [
                _score("judge_facts", 1.0, "t1"),
                _score(
                    "judge_evidence",
                    0.0,
                    "t2",
                    comment="[해당 없음] 근거가 필요한 주장 없음",
                ),
            ]
        )

    api = SimpleNamespace(
        experiments=SimpleNamespace(list_items=list_items),
        scores=SimpleNamespace(get_many=get_many),
    )
    items = sj.load_run(api, "run", dt.datetime(2026, 11, 5, tzinfo=dt.UTC))
    assert items[0]["scores"] == {
        "judge_facts": [
            {"value": 1.0, "comment": "판단: 맞음", "timestamp": "2026-11-12T01:00:00Z"}
        ]
    }
    assert list(items[1]["scores"]) == ["judge_evidence"]
    assert sj.absent_names(items) == ["judge_conditions", "judge_decision"]
    s = sj.summarize_run(items)
    assert s["judge_evidence"]["n_na"] == 1
    assert s["judge_conditions"]["n_unavailable"] == 2


def test_local_runs_reads_run_names_and_skips_judge_files(monkeypatch, tmp_path):
    monkeypatch.setattr(sj, "LOCAL", tmp_path)
    (tmp_path / "naive_x.json").write_text(
        json.dumps({"dataset": "ds", "run_name": "naive_x"}), encoding="utf-8"
    )
    (tmp_path / "naive_x_judge.json").write_text("{}", encoding="utf-8")
    assert sj.local_runs([], 3) == ["naive_x"]
    assert sj.local_runs(["hybrid_y"], 3) == ["hybrid_y"]
    monkeypatch.setattr(sj, "LOCAL", tmp_path / "empty")
    with pytest.raises(SystemExit):
        sj.local_runs([], 3)
