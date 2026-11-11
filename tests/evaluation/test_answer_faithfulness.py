"""evaluation/answer/faithfulness.py 테스트.

API 없이 가짜 채점기(scorer)와 가짜 OpenAI 응답으로 상태 구분, 이어 하기, 요약, 채점 토큰 세기를 확인한다.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import math
from types import SimpleNamespace

import httpx2
import openai
import pytest

from rag_service.storage.corpus import repo_root


def _load(name: str):
    path = repo_root() / "evaluation" / "answer" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"answer_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fa = _load("faithfulness")


def _rec(qid="Q1", **kw):
    rec = {
        "id": qid,
        "mode": "naive",
        "question": "일반 건화물 컨테이너 종류는?",
        "status": "answered",
        "answer": "20GP와 40GP입니다 [S1]",
        "contexts": [
            {"sid": "S1", "text": "일반 건화물은 20GP와 40GP로 예약한다."},
            {"sid": "S2", "text": "반입 마감은 출항 2일 전이다."},
        ],
        "error": None,
    }
    rec.update(kw)
    return rec


class FakeScorer:
    """ascore가 values에서 하나씩 꺼내 돌려준다. Exception이면 던진다."""

    def __init__(self, *values):
        self.values = list(values)
        self.calls = []

    async def ascore(self, **kwargs):
        self.calls.append(kwargs)
        value = self.values.pop(0)
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(value=value)


def _score(rec, *values):
    scorer = FakeScorer(*values)
    return asyncio.run(fa.score_one(rec, scorer)), scorer


# ---------- 문항 하나 ----------


def test_scored_passes_question_answer_and_context_texts():
    row, scorer = _score(_rec(), 0.5)
    assert row["state"] == "scored" and row["faithfulness"] == 0.5
    assert scorer.calls == [
        {
            "user_input": "일반 건화물 컨테이너 종류는?",
            "response": "20GP와 40GP입니다 [S1]",
            "retrieved_contexts": [
                "일반 건화물은 20GP와 40GP로 예약한다.",
                "반입 마감은 출항 2일 전이다.",
            ],
        }
    ]


def test_zero_is_a_score_not_na():
    row, _ = _score(_rec(), 0.0)
    assert row["state"] == "scored" and row["faithfulness"] == 0.0


def test_nan_means_no_claims_na():
    row, _ = _score(
        _rec(status="needs_clarification", answer="장비 종류를 알려 주세요."), math.nan
    )
    assert row["state"] == "na" and row["cause"] == "no_claims"
    assert row["faithfulness"] is None
    assert row["status"] == "needs_clarification"


def test_grading_failure_is_unavailable_with_reason():
    row, _ = _score(_rec(), TimeoutError("응답 없음"))
    assert row["state"] == "unavailable" and row["cause"] == "grading_failed"
    assert "TimeoutError" in row["reason"]
    assert row["faithfulness"] is None


@pytest.mark.parametrize(
    ("rec", "cause"),
    [
        (
            _rec(status="error", answer="", error="RuntimeError: Qdrant"),
            "service_error",
        ),
        (_rec(answer="  "), "empty_answer"),
        (_rec(contexts=[]), "empty_context"),
        (_rec(contexts=[{"sid": "S1", "text": ""}]), "empty_context"),
    ],
)
def test_precheck_cases_do_not_call_scorer(rec, cause):
    row, scorer = _score(rec)  # 값을 하나도 안 줬으니 부르면 IndexError
    assert row["state"] == "unavailable" and row["cause"] == cause
    assert scorer.calls == []


# ---------- 한 방식 전체 ----------


def test_score_records_writes_and_resumes(tmp_path):
    path = tmp_path / "faithfulness_naive.jsonl"
    records = [_rec("Q1"), _rec("Q2"), _rec("Q3")]
    logs: list[str] = []
    new = asyncio.run(
        fa.score_records(records[:2], FakeScorer(1.0, 0.5), path, log=logs.append)
    )
    assert {r["id"] for r in new} == {"Q1", "Q2"}
    assert logs[0] == "저장됨 0, 새로 채점 2"

    new = asyncio.run(fa.score_records(records, FakeScorer(0.0), path, log=logs.append))
    assert [r["id"] for r in new] == ["Q3"]  # 이미 채점한 Q1, Q2는 건너뜀
    assert logs[-2] == "저장됨 2, 새로 채점 1"
    assert {r["id"] for r in fa.read_jsonl(path)} == {"Q1", "Q2", "Q3"}


# ---------- 요약 ----------


def _row(qid, value, state, cause=None, status="answered"):
    return {
        "id": qid,
        "faithfulness": value,
        "state": state,
        "cause": cause,
        "status": status,
    }


def test_summarize_excludes_na_and_unavailable_but_keeps_zero():
    rows = [
        _row("Q1", 1.0, "scored"),
        _row("Q2", 0.0, "scored"),
        _row("Q3", 0.5, "scored", status="abstained"),
        _row("Q4", None, "na", "no_claims", status="needs_clarification"),
        _row("Q5", None, "unavailable", "service_error", status="error"),
        _row("Q6", None, "unavailable", "grading_failed"),
    ]
    s = fa.summarize(rows, n_answers=7)
    assert s["mean"] == 0.5  # (1.0 + 0.0 + 0.5) / 3
    assert (s["n_scored"], s["n_zero"], s["n_na"], s["n_unavailable"]) == (3, 1, 1, 2)
    assert (s["n_service_error"], s["n_grading_failed"], s["n_missing"]) == (1, 1, 1)
    assert s["by_status"]["answered"] == {"mean": 0.5, "n": 2}
    assert s["by_status"]["abstained"] == {"mean": 0.5, "n": 1}
    assert s["by_status"]["needs_clarification"] == {"mean": None, "n": 0}


def test_summarize_empty():
    s = fa.summarize([], n_answers=0)
    assert s["mean"] is None and s["n_scored"] == 0


def test_write_summary_and_report(tmp_path):
    answers = [_rec("Q1"), _rec("Q2")]
    (tmp_path / "answers_naive.jsonl").write_text(
        "\n".join(json.dumps(a, ensure_ascii=False) for a in answers), encoding="utf-8"
    )
    (tmp_path / "faithfulness_naive.jsonl").write_text(
        json.dumps(_row("Q1", 0.8, "scored")), encoding="utf-8"
    )
    (tmp_path / "grading_usage.jsonl").write_text(
        json.dumps(
            {
                "step": "faithfulness",
                "mode": "naive",
                "calls": 2,
                "input": 300,
                "output": 40,
            }
        )
        + "\n"
        + json.dumps(
            {"step": "judge", "mode": "naive", "calls": 9, "input": 999, "output": 99}
        ),
        encoding="utf-8",
    )
    summary = fa.write_summary(tmp_path, ["naive", "hybrid"], model="gpt-6-luna")
    m = summary["modes"]["naive"]
    assert "hybrid" not in summary["modes"]  # 답변 파일이 없는 방식은 뺀다
    assert m["mean"] == 0.8 and m["n_missing"] == 1
    assert m["grading_usage"] == {"calls": 2, "input": 300, "output": 40, "unknown": 0}

    csv_text = (tmp_path / "faithfulness_per_question.csv").read_text(
        encoding="utf-8-sig"
    )
    lines = csv_text.strip().splitlines()
    assert lines[0].startswith("id,mode,status,faithfulness,state")
    assert len(lines) == 3 and "missing" in lines[2]  # 아직 채점 안 한 Q2

    table = fa.report(summary)
    assert "| naive | 0.800 | 1 | 0 | 0 (0 / 0) | 300 / 40 |" in table


# ---------- 채점 호출 토큰 ----------


def test_count_usage_wraps_create():
    class Completions:
        async def create(self, **kwargs):
            usage = kwargs.get("usage")
            return SimpleNamespace(usage=usage)

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    counter = fa.UsageCounter()
    fa.count_usage(client, counter)
    asyncio.run(
        client.chat.completions.create(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=3)
        )
    )
    asyncio.run(client.chat.completions.create(usage=None))
    assert counter.as_dict() == {"calls": 2, "input": 10, "output": 3, "unknown": 1}


def _fake_openai(statements, verdicts, seen):
    """RAGAS가 보내는 두 호출(주장 나누기 → 주장별 판정)에 정해진 JSON으로 답하는 가짜 OpenAI."""

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        text = json.dumps(body, ensure_ascii=False)
        if "verdict" in text:
            content = {"statements": verdicts}
        else:
            content = {"statements": statements}
        return httpx2.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(content, ensure_ascii=False),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                },
            },
        )

    return openai.AsyncOpenAI(
        api_key="test",
        http_client=openai.DefaultAsyncHttpxClient(
            transport=httpx2.MockTransport(handler)
        ),
    )


def test_real_ragas_faithfulness_with_fake_openai():
    """실제 ragas Faithfulness가 우리 입력으로 돌아가는지 (API 대신 가짜 응답)."""
    seen: list[dict] = []
    verdicts = [
        {"statement": "20GP 예약 가능", "reason": "문맥에 있음", "verdict": 1},
        {"statement": "45GP 예약 가능", "reason": "문맥에 없음", "verdict": 0},
    ]
    client = _fake_openai(["20GP 예약 가능", "45GP 예약 가능"], verdicts, seen)
    counter = fa.UsageCounter()
    scorer = fa.make_scorer("gpt-6-luna", counter, client=client)
    row = asyncio.run(
        fa.score_one(_rec(answer="20GP와 45GP로 예약합니다 [S1]"), scorer)
    )
    assert row["state"] == "scored" and row["faithfulness"] == 0.5
    assert counter.as_dict() == {"calls": 2, "input": 200, "output": 40, "unknown": 0}
    # gpt-6-luna: RAGAS가 temperature 1.0, max_completion_tokens로 바꿔 보낸다
    assert seen[0]["temperature"] == 1.0
    assert seen[0]["max_completion_tokens"] == fa.MAX_TOKENS
    assert "20GP와 40GP로 예약한다" in json.dumps(
        seen[1], ensure_ascii=False
    )  # 두 번째 호출에 문맥


def test_real_ragas_no_statements_is_na():
    client = _fake_openai([], [], [])
    scorer = fa.make_scorer("gpt-6-luna", fa.UsageCounter(), client=client)
    row = asyncio.run(fa.score_one(_rec(answer="어떤 장비인지 알려 주세요."), scorer))
    assert row["state"] == "na" and row["cause"] == "no_claims"


def test_out_dir_matches_run_answers_and_run_overrides():
    from rag_service.retrieval_b.settings import Settings

    s = Settings(k=3)
    assert fa.run_name("basic", "development") == "basic_development"
    assert fa.run_name("basic", "development", settings=s) == "basic_development_k3"
    assert (
        fa.run_name("basic", "development", tag="r2", settings=s)
        == "basic_development_k3_r2"
    )
    assert fa.out_dir_for("basic", "development", run="mine", settings=s).name == "mine"


def test_main_finds_folder_from_cli_tuning(monkeypatch, tmp_path, capsys):
    """run_answers.py --rerank-n 20 으로 만든 폴더를 같은 옵션으로 찾는다."""
    monkeypatch.setattr(fa, "repo_root", lambda: tmp_path)
    folder = tmp_path / fa.OUT_DIR / "basic_development_n20"
    folder.mkdir(parents=True)
    (folder / "answers_naive.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(fa, "refresh_board", lambda *a: None)
    fa.main(["--rerank-n", "20", "--summary-only"])
    assert "[답변 폴더] results" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="basic_development_n20"):
        fa.main(["--summary-only"])  # 기본 설정 폴더는 없다 → 있는 폴더 목록에 n20


def test_available_runs_lists_only_answer_folders(monkeypatch, tmp_path):
    monkeypatch.setattr(fa, "repo_root", lambda: tmp_path)
    base = tmp_path / fa.OUT_DIR
    for name, has_answers in [
        ("basic_development", True),
        ("basic_development_k3", True),
        ("basic_development_empty", False),
        ("basic_additional", True),
    ]:
        (base / name).mkdir(parents=True)
        if has_answers:
            (base / name / "answers_naive.jsonl").write_text("", encoding="utf-8")
    assert fa.available_runs("basic", "development") == [
        "basic_development",
        "basic_development_k3",
    ]


def test_main_stops_when_answer_folder_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(fa, "repo_root", lambda: tmp_path)
    (tmp_path / fa.OUT_DIR / "basic_development_k3").mkdir(parents=True)
    (tmp_path / fa.OUT_DIR / "basic_development_k3" / "answers_naive.jsonl").write_text(
        "", encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="basic_development_k3"):
        fa.main(["--run", "nope"])


# ---------- 실험 비교판 자동 갱신 ----------


def _answers_folder(tmp_path):
    folder = tmp_path / fa.OUT_DIR / "basic_development_x"
    folder.mkdir(parents=True)
    (folder / "answers_naive.jsonl").write_text(
        json.dumps({"id": "Q1", "mode": "naive", "status": "answered"}) + "\n",
        encoding="utf-8",
    )
    return folder


def test_main_refreshes_board_unless_no_board(monkeypatch, tmp_path):
    monkeypatch.setattr(fa, "repo_root", lambda: tmp_path)
    _answers_folder(tmp_path)
    calls = []
    monkeypatch.setattr(fa, "refresh_board", lambda *a: calls.append(1))
    fa.main(["--run", "basic_development_x", "--modes", "naive", "--summary-only"])
    assert calls == [1]
    fa.main(
        [
            "--run",
            "basic_development_x",
            "--modes",
            "naive",
            "--summary-only",
            "--no-board",
        ]
    )
    assert calls == [1]


def test_refresh_board_is_silent_without_board_file(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(fa, "repo_root", lambda: tmp_path)  # 비교판 스크립트가 없는 곳
    assert fa.refresh_board() is None
    assert capsys.readouterr().out == ""


def test_out_dir_and_available_runs_with_exp(monkeypatch, tmp_path):
    monkeypatch.setattr(fa, "repo_root", lambda: tmp_path)
    exp_answers = (
        tmp_path / "results" / "experiments" / "page_k" / "answer" / "retrieval_b"
    )
    (exp_answers / "basic_development_k9").mkdir(parents=True)
    (exp_answers / "basic_development_k9" / "answers_naive.jsonl").write_text(
        "", encoding="utf-8"
    )
    assert (
        fa.out_dir_for("basic", "development", run="basic_development_k9", exp="page_k")
        == exp_answers / "basic_development_k9"
    )
    assert fa.available_runs("basic", "development", "page_k") == [
        "basic_development_k9"
    ]
    assert fa.available_runs("basic", "development") == []  # 기본 폴더에는 없음
