"""evaluation/answer/judge.py 테스트.

API 없이 가짜 평가 LLM(FakeJudge)과 가짜 OpenAI 응답으로 입력 만들기, 상태 구분, 이어 하기, 요약을 확인한다.
"""

from __future__ import annotations

import importlib.util
import json

import httpx2
import openai
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from rag_service.storage.corpus import repo_root


def _load(name: str):
    path = repo_root() / "evaluation" / "answer" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"answer_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


jd = _load("judge")

GOLDEN_ITEM = {
    "id": "Q1",
    "question": "일반 건화물 컨테이너 종류는?",
    "reference_answer": "20GP와 40GP입니다.",
    "required_facts": ["일반 건화물 20GP", "일반 건화물 40GP"],
    "forbidden_claims": ["냉동 장비도 승인 가능"],
    "expected_action": "answer",
    "evidence_groups": [[{"book_id": "SB01", "page": 1, "quote": "정답 근거 원문"}]],
    "grading_notes": "같은 의미의 표현을 허용한다.",
}
GOLDEN = {
    "Q1": GOLDEN_ITEM,
    "Q2": {**GOLDEN_ITEM, "id": "Q2"},
    "Q3": {**GOLDEN_ITEM, "id": "Q3"},
}


def _rec(qid="Q1", **kw):
    rec = {
        "id": qid,
        "mode": "naive",
        "question": "일반 건화물 컨테이너 종류는?",
        "status": "answered",
        "answer": "20GP와 40GP입니다 [S1]",
        "reason": "문맥에 장비 종류가 있음",
        "source_ids": ["S1"],
        "contexts": [
            {
                "sid": "S1",
                "book_id": "SB01",
                "page": 1,
                "text": "대상 장비는 20GP와 40GP다.",
            },
            {
                "sid": "S2",
                "book_id": "SB02",
                "page": 3,
                "text": "반입 마감은 출항 2일 전이다.",
            },
        ],
        "error": None,
    }
    rec.update(kw)
    return rec


def _result(facts=1, conditions=1, evidence=1, decision=1):
    def s(v):
        return {"score": v, "reason": f"이유 {v}"}

    return jd.JudgeResult(
        facts=s(facts),
        conditions=s(conditions),
        evidence=s(evidence),
        decision=s(decision),
    )


class FakeJudge:
    """invoke가 results에서 하나씩 꺼내 돌려준다. Exception이면 던진다."""

    def __init__(self, *results, model=None):
        self.results, self.model, self.inputs = list(results), model, []

    def invoke(self, inputs):
        self.inputs.append(inputs)
        if self.model is not None:  # 토큰 사용량을 남기는 가짜 모델 호출
            self.model.invoke("hi")
        value = self.results.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def _fake_model(input_tokens=500, output_tokens=80):
    msg = AIMessage(
        content="x",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        response_metadata={"model_name": "fake"},  # 콜백은 모델 이름이 있어야 센다
    )
    return GenericFakeChatModel(messages=iter([msg]))


# ---------- ① 서비스 오류 (precheck) ----------


def test_service_error_items_decision_zero_others_unavailable():
    items = jd.service_error_items(
        _rec(status="error", answer="", error="RuntimeError: Qdrant")
    )
    assert items["decision"] == {
        "score": 0,
        "state": "scored",
        "reason": "서비스 오류로 답변을 만들지 못함",
    }
    for key in ("facts", "conditions", "evidence"):
        assert items[key]["score"] is None and items[key]["state"] == "unavailable"
        assert "RuntimeError: Qdrant" in items[key]["reason"]


def test_precheck_cases():
    assert jd.precheck(_rec(), GOLDEN_ITEM) is None  # 채점할 문항
    items, cause = jd.precheck(_rec(status="error", error="x"), GOLDEN_ITEM)
    assert cause == "service_error" and items["decision"]["score"] == 0
    items, cause = jd.precheck(_rec(), None)
    assert cause == "not_in_golden"
    assert all(items[k]["state"] == "unavailable" for k in jd.ITEMS)


# ---------- ② 평가 입력 ----------


def test_golden_inputs_five_fields_without_evidence_groups():
    g = jd.golden_inputs(GOLDEN_ITEM)
    assert g == {
        "reference_answer": "20GP와 40GP입니다.",
        "required_facts": "- 일반 건화물 20GP\n- 일반 건화물 40GP",
        "forbidden_claims": "- 냉동 장비도 승인 가능",
        "expected_action": "answer (API 상태 answered)",
        "grading_notes": "같은 의미의 표현을 허용한다.",
    }
    assert "정답 근거 원문" not in json.dumps(g, ensure_ascii=False)


def test_golden_inputs_empty_values():
    g = jd.golden_inputs({"expected_action": "clarify", "forbidden_claims": []})
    assert g["reference_answer"] == "(없음)" and g["forbidden_claims"] == "(없음)"
    assert g["expected_action"] == "clarify (API 상태 needs_clarification)"


def test_build_inputs_fills_every_prompt_slot():
    inputs = jd.build_inputs(_rec(), GOLDEN_ITEM)
    assert set(inputs) == set(jd.judge_prompt.input_variables)  # 11칸
    assert inputs["context"].startswith(
        "[S1] SB01.pdf / page 1\n대상 장비는 20GP와 40GP다."
    )
    assert inputs["citations"] == "S1: SB01 1쪽"
    assert inputs["status"] == "answered"


def test_format_citations_marks_unknown_sid():
    text = jd.format_citations(_rec(source_ids=["S2", "S9"]))
    assert text == "S2: SB02 3쪽\nS9: 실제 최종 문맥에 없는 번호"
    assert jd.format_citations(_rec(source_ids=[])) == "(인용 없음)"


# ---------- ③ 평가 LLM 호출 ----------


def test_call_judge_returns_result_and_tokens():
    judge = FakeJudge(_result(), model=_fake_model(500, 80))
    result, tokens = jd.call_judge(judge, {"question": "q"})
    assert result.facts.score == 1
    assert tokens == {"input": 500, "output": 80}
    assert judge.inputs == [{"question": "q"}]


def test_call_judge_tokens_none_when_unknown():
    _, tokens = jd.call_judge(FakeJudge(_result()), {"question": "q"})
    assert tokens is None  # 0이 아니라 확인 불가


# ---------- ④ 점수 → 상태 ----------


@pytest.mark.parametrize(
    ("key", "score", "state"),
    [
        ("facts", 1, "scored"),
        ("facts", 0, "scored"),  # 0은 해당 없음이 아니다
        ("facts", None, "na"),
        ("evidence", None, "na"),
        ("decision", 0, "scored"),
        ("decision", None, "unavailable"),  # 판단은 null을 받아들이지 않는다
    ],
)
def test_item_state(key, score, state):
    assert jd.item_state(key, score) == state


def test_to_items():
    items = jd.to_items(_result(facts=0, evidence=None, decision=None))
    assert items["facts"] == {"score": 0, "state": "scored", "reason": "이유 0"}
    assert items["conditions"]["score"] == 1
    assert items["evidence"] == {"score": None, "state": "na", "reason": "이유 None"}
    assert (
        items["decision"]["state"] == "unavailable"
        and items["decision"]["score"] is None
    )


# ---------- 문항 하나 ----------


def test_judge_one_scored_row():
    judge = FakeJudge(_result(conditions=0), model=_fake_model())
    row = jd.judge_one(_rec(), GOLDEN, judge)
    assert row["expected_action"] == "answer" and row["status"] == "answered"
    assert row["conditions"]["score"] == 0 and row["facts"]["score"] == 1
    assert row["cause"] is None and row["tokens"] == {"input": 500, "output": 80}
    assert row["eval_ms"] >= 0
    # 정답 쪽 칸은 평가 LLM 입력에만 들어간다
    assert judge.inputs[0]["reference_answer"] == "20GP와 40GP입니다."


def test_judge_one_grading_failure_is_unavailable():
    row = jd.judge_one(_rec(), GOLDEN, FakeJudge(TimeoutError("응답 없음")))
    assert row["cause"] == "grading_failed"
    for key in jd.ITEMS:
        assert row[key]["state"] == "unavailable" and row[key]["score"] is None
        assert "TimeoutError" in row[key]["reason"]


def test_judge_one_service_error_does_not_call():
    judge = FakeJudge()  # 결과를 하나도 안 줬으니 부르면 IndexError
    row = jd.judge_one(_rec(status="error", answer="", error="x"), GOLDEN, judge)
    assert row["cause"] == "service_error" and judge.inputs == []
    assert row["decision"]["score"] == 0


# ---------- ⑤ 이어 하기 ----------


def test_todo_records_skips_saved_ids(tmp_path):
    path = tmp_path / "judge_naive.jsonl"
    path.write_text(json.dumps({"id": "Q1"}) + "\n", encoding="utf-8")
    todo = jd.todo_records([_rec("Q1"), _rec("Q2")], path)
    assert [r["id"] for r in todo] == ["Q2"]
    assert len(jd.todo_records([_rec("Q1")], tmp_path / "none.jsonl")) == 1


def test_judge_records_writes_and_resumes(tmp_path):
    path = tmp_path / "judge_naive.jsonl"
    records = [_rec("Q1"), _rec("Q2"), _rec("Q3")]
    logs: list[str] = []
    new = jd.judge_records(
        records[:2], GOLDEN, FakeJudge(_result(), _result()), path, log=logs.append
    )
    assert [r["id"] for r in new] == ["Q1", "Q2"]
    assert logs[0] == "저장됨 0, 새로 채점 2"
    assert logs[1] == "  1/2 Q1 사실=1 조건=1 근거=1 판단=1"

    new = jd.judge_records(
        records, GOLDEN, FakeJudge(_result(evidence=None)), path, log=logs.append
    )
    assert [r["id"] for r in new] == ["Q3"]  # 이미 채점한 Q1, Q2는 건너뜀
    assert logs[-2] == "저장됨 2, 새로 채점 1"
    assert logs[-1] == "  1/1 Q3 사실=1 조건=1 근거=해당없음 판단=1"
    assert [r["id"] for r in jd.read_jsonl(path)] == ["Q1", "Q2", "Q3"]


def test_usage_of_skips_rows_without_call():
    rows = [
        {"cause": None, "tokens": {"input": 10, "output": 2}},
        {"cause": None, "tokens": None},
        {"cause": "grading_failed", "tokens": None},  # 호출은 했으니 센다
        {"cause": "service_error", "tokens": None},  # 호출 안 함
    ]
    assert jd.usage_of(rows) == {"calls": 3, "input": 10, "output": 2, "unknown": 2}


# ---------- ⑥ 요약 ----------


def _row(qid, expected="answer", **scores):
    row = {"id": qid, "expected_action": expected, "status": "answered", "cause": None}
    for key in jd.ITEMS:
        v = scores.get(key, 1)
        state = {"na": "na", "un": "unavailable"}.get(v, "scored")
        row[key] = {
            "score": v if state == "scored" else None,
            "state": state,
            "reason": "",
        }
    return row


def test_item_rate_excludes_na_and_unavailable_keeps_zero():
    rows = [
        _row("Q1", facts=1),
        _row("Q2", facts=0),
        _row("Q3", facts=1),
        _row("Q4", facts="na"),
        _row("Q5", facts="un"),
    ]
    assert jd.item_rate(rows, "facts") == {
        "rate": 0.6667,  # 2 / 3
        "n_scored": 3,
        "n_one": 2,
        "n_na": 1,
        "n_unavailable": 1,
    }
    assert jd.item_rate([], "facts")["rate"] is None


def test_summarize_counts_and_decision_by_expected_action():
    rows = [
        _row("Q1", decision=1),
        _row("Q2", decision=0),
        _row("Q3", "clarify", facts="na", evidence="na", decision=1),
        {
            **_row("Q4", decision=0, facts="un", conditions="un", evidence="un"),
            "cause": "service_error",
        },
    ]
    s = jd.summarize(rows, n_answers=5)
    assert s["items"]["decision"]["rate"] == 0.5  # (1 + 0 + 1 + 0) / 4
    assert (
        s["items"]["facts"]["n_na"] == 1 and s["items"]["facts"]["n_unavailable"] == 1
    )
    assert (
        s["n_rows"],
        s["n_service_error"],
        s["n_grading_failed"],
        s["n_missing"],
    ) == (4, 1, 0, 1)
    assert s["by_expected_action"]["answer"] == {"n": 3, "decision": 0.3333}
    assert s["by_expected_action"]["clarify"] == {"n": 1, "decision": 1.0}
    assert s["by_expected_action"]["abstain"] == {"n": 0, "decision": None}


def test_write_summary_and_report(tmp_path):
    answers = [_rec("Q1"), _rec("Q2")]
    (tmp_path / "answers_naive.jsonl").write_text(
        "\n".join(json.dumps(a, ensure_ascii=False) for a in answers), encoding="utf-8"
    )
    (tmp_path / "judge_naive.jsonl").write_text(
        json.dumps(_row("Q1", facts=0, evidence="na"), ensure_ascii=False),
        encoding="utf-8",
    )
    usage = [
        {"step": "judge", "mode": "naive", "calls": 1, "input": 900, "output": 120},
        {
            "step": "faithfulness",
            "mode": "naive",
            "calls": 2,
            "input": 300,
            "output": 40,
        },
    ]
    (tmp_path / "grading_usage.jsonl").write_text(
        "\n".join(json.dumps(u) for u in usage), encoding="utf-8"
    )
    summary = jd.write_summary(tmp_path, ["naive", "hybrid"], model="gpt-6-luna")
    m = summary["modes"]["naive"]
    assert "hybrid" not in summary["modes"]  # 답변 파일이 없는 방식은 뺀다
    assert m["n_missing"] == 1
    assert m["grading_usage"] == {"calls": 1, "input": 900, "output": 120, "unknown": 0}

    lines = (
        (tmp_path / "judge_per_question.csv")
        .read_text(encoding="utf-8-sig")
        .strip()
        .splitlines()
    )
    assert lines[0].startswith(
        "id,mode,expected_action,status,facts,facts_reason,conditions"
    )
    assert lines[1].startswith("Q1,naive,answer,answered,0,,1,,해당없음,,1,")
    assert len(lines) == 3 and "missing" in lines[2]  # 아직 채점 안 한 Q2

    table = jd.report(summary)
    assert (
        "| naive | 0.000 (0/1) | 1.000 (1/1) | - | 1.000 (1/1) | 0/0/1 | 0 (0 / 0) | 900 / 120 |"
        in table
    )
    assert "| naive | 1.000 (n=1) | - (n=0) | - (n=0) |" in table


# ---------- 답변 폴더 (run_answers.py, faithfulness.py와 같은 규칙) ----------

SETTINGS_VARS = (
    "K",
    "LEVELS",
    "CANDIDATE_K",
    "RRF_K",
    "RERANK_N",
    "RERANK_MAX_LENGTH",
    "RERANK_BATCH_SIZE",
)


@pytest.fixture
def default_settings(monkeypatch):
    """.env 값이 결과를 바꾸지 않게 RETRIEVAL_B_*를 비우고, 끝나면 설정 캐시를 지운다."""
    from rag_service.retrieval_b.settings import get_settings

    for name in SETTINGS_VARS:
        monkeypatch.setenv(f"RETRIEVAL_B_{name}", "")
    get_settings.cache_clear()
    yield monkeypatch
    get_settings.cache_clear()


def _answer_dir(root, name):
    d = root / "results" / "answer" / "retrieval_b" / name
    d.mkdir(parents=True)
    (d / "answers_naive.jsonl").write_text("{}", encoding="utf-8")
    return d


def test_out_dir_follows_settings_and_run_overrides(default_settings):
    from rag_service.retrieval_b.settings import get_settings

    assert jd.out_dir_for("basic", "development").name == "basic_development"
    assert jd.out_dir_for("basic", "development", tag="r2").name == (
        "basic_development_r2"
    )
    default_settings.setenv("RETRIEVAL_B_K", "3")
    get_settings.cache_clear()
    assert jd.out_dir_for("basic", "development").name == "basic_development_k3"
    # --run은 폴더 이름 전체라서 설정 태그와 --tag를 무시한다
    assert jd.out_dir_for("basic", "development", run="mine", tag="r2").name == "mine"


def test_available_runs_lists_only_answer_folders(tmp_path, monkeypatch):
    monkeypatch.setattr(jd, "repo_root", lambda: tmp_path)
    assert jd.available_runs("basic", "development") == []
    _answer_dir(tmp_path, "basic_development")
    _answer_dir(tmp_path, "basic_development_k3")
    _answer_dir(tmp_path, "basic_test")
    (
        tmp_path / "results" / "answer" / "retrieval_b" / "basic_development_empty"
    ).mkdir()
    assert jd.available_runs("basic", "development") == [
        "basic_development",
        "basic_development_k3",
    ]


def test_main_stops_when_answer_folder_missing(tmp_path, monkeypatch, default_settings):
    monkeypatch.setattr(jd, "repo_root", lambda: tmp_path)
    _answer_dir(tmp_path, "basic_development_k3")
    with pytest.raises(SystemExit) as stop:
        jd.main(["--summary-only"])
    message = str(stop.value)
    assert "basic_development" in message and "--run" in message
    assert "있는 폴더: basic_development_k3" in message
    assert not list(tmp_path.rglob("judge_summary.json"))  # 아무것도 쓰지 않는다


# ---------- 실제 LangChain 체인 (API 대신 가짜 응답) ----------


def test_real_chain_with_fake_openai(monkeypatch):
    """make_judge가 만든 진짜 체인(프롬프트 → ChatOpenAI → JudgeResult)이 우리 입력으로 돌아가는지."""
    monkeypatch.setenv("OPENAI_API_KEY", "test")  # 가짜 응답이라 실제 키는 쓰지 않는다
    seen: list[dict] = []
    answer = {
        "facts": {"score": 1, "reason": "20GP, 40GP를 모두 제시"},
        "conditions": {"score": 1, "reason": "일반 건화물 범위"},
        "evidence": {"score": 1, "reason": "S1이 뒷받침"},
        "decision": {"score": 1, "reason": "답변이 적절"},
    }

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
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
                            "content": json.dumps(answer, ensure_ascii=False),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 700,
                    "completion_tokens": 90,
                    "total_tokens": 790,
                },
            },
        )

    client = openai.DefaultHttpxClient(transport=httpx2.MockTransport(handler))
    judge = jd.make_judge("gpt-6-luna", http_client=client)
    row = jd.judge_one(_rec(), GOLDEN, judge)
    assert row["cause"] is None
    assert [row[k]["score"] for k in jd.ITEMS] == [1, 1, 1, 1]
    assert row["tokens"] == {"input": 700, "output": 90}
    sent = json.dumps(seen[0], ensure_ascii=False)
    assert seen[0]["model"] == "gpt-6-luna"
    assert "대상 장비는 20GP와 40GP다." in sent  # 실제 최종 문맥
    assert "일반 건화물 20GP" in sent  # 필수 사실
    assert "정답 근거 원문" not in sent  # 근거 그룹은 보내지 않는다
