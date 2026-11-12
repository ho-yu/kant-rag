"""evaluation/answer/quality.py 테스트 (RAGAS 4개 + Judge 4항목을 한 번에).

API 없이 가짜 지표·가짜 Judge로 상태 규칙과 요약을 보고,
실제 ragas 지표 객체는 가짜 OpenAI 응답과 가짜 임베딩으로 한 번씩 돌려 본다.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json

import httpx2
import openai
import pytest

from rag_service.storage.corpus import repo_root


def _load():
    path = repo_root() / "evaluation" / "answer" / "quality.py"
    spec = importlib.util.spec_from_file_location("answer_quality", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


q = _load()

GOLDEN = {
    "reference_answer": "20GP와 40GP입니다.",
    "required_facts": ["일반 건화물 20GP", "일반 건화물 40GP"],
    "forbidden_claims": ["냉동 장비도 승인 가능"],
    "expected_action": "answer",
    "evidence_groups": [[{"book_id": "SB01", "page": 1, "quote": "정답 근거 원문"}]],
    "grading_notes": "같은 의미의 표현을 허용한다.",
}


def _rec(**kw):
    rec = {
        "id": "Q1",
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


class FakeMetric:
    """ascore가 value(또는 Exception)를 돌려준다. 받은 입력을 남긴다."""

    def __init__(self, value):
        self.value, self.calls = value, []

    async def ascore(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.value, Exception):
            raise self.value
        return type("R", (), {"value": self.value})()


class FakeJudge:
    def __init__(self, result):
        self.result, self.inputs = result, []

    def invoke(self, inputs):
        self.inputs.append(inputs)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _judge_result(facts=1, conditions=1, evidence=1, decision=1):
    def s(v):
        return {"score": v, "reason": f"이유 {v}"}

    return q.JudgeResult(
        facts=s(facts),
        conditions=s(conditions),
        evidence=s(evidence),
        decision=s(decision),
    )


def _scorer(values=None, judge=None):
    values = values or {}
    metrics = {name: FakeMetric(values.get(name, 0.8)) for name in q.RAGAS_METRICS}
    return q.QualityScorer(metrics=metrics, judge=judge or FakeJudge(_judge_result()))


def _by_name(scores):
    return {s["name"]: s for s in scores}


# ---------- 한 문항 ----------


def test_score_returns_eight_named_scores_in_order():
    scores = asyncio.run(_scorer().score(_rec(), GOLDEN))
    assert [s["name"] for s in scores] == list(q.ALL_NAMES)
    assert all(s["state"] == "scored" for s in scores)


def test_without_judge_only_ragas_scores_are_returned():
    metrics = {name: FakeMetric(0.8) for name in q.RAGAS_METRICS}
    scorer = q.QualityScorer(metrics=metrics, judge=None)
    scores = asyncio.run(scorer.score(_rec(), GOLDEN))
    assert [s["name"] for s in scores] == list(q.RAGAS_METRICS)
    assert scorer.grading_usage()["judge"]["calls"] == 0


def test_create_without_judge_does_not_build_one():
    emb = FakeEmbeddings()
    client = _fake_openai(lambda _: {}, [])
    scorer = q.QualityScorer.create(
        "gpt-6-luna",
        embeddings=emb,
        client=client,
        judge=FakeJudge(_judge_result()),
        with_judge=False,
    )
    assert scorer.judge is None
    assert set(scorer.metrics) == set(q.RAGAS_METRICS)


def test_ragas_inputs_match_ragas_043_signatures():
    scorer = _scorer()
    asyncio.run(scorer.score(_rec(), GOLDEN))
    texts = ["대상 장비는 20GP와 40GP다.", "반입 마감은 출항 2일 전이다."]
    question, answer = "일반 건화물 컨테이너 종류는?", "20GP와 40GP입니다 [S1]"
    assert scorer.metrics["faithfulness"].calls == [
        {"user_input": question, "response": answer, "retrieved_contexts": texts}
    ]
    assert scorer.metrics["answer_relevancy"].calls == [
        {"user_input": question, "response": answer}
    ]
    for name in ("context_precision", "context_recall"):
        assert scorer.metrics[name].calls == [
            {
                "user_input": question,
                "reference": "20GP와 40GP입니다.",
                "retrieved_contexts": texts,
            }
        ]


def test_judge_gets_golden_fields_but_not_evidence_groups():
    judge = FakeJudge(_judge_result())
    asyncio.run(_scorer(judge=judge).score(_rec(), GOLDEN))
    sent = json.dumps(judge.inputs[0], ensure_ascii=False)
    assert "일반 건화물 20GP" in sent and "냉동 장비도 승인 가능" in sent
    assert "정답 근거 원문" not in sent  # evidence_groups의 quote


def test_service_error_all_ragas_unavailable_and_judge_decision_zero():
    scorer = _scorer()
    rec = _rec(status="error", answer="", contexts=[], error="RuntimeError: Qdrant")
    by = _by_name(asyncio.run(scorer.score(rec, GOLDEN)))
    for name in q.RAGAS_METRICS:
        assert (
            by[name]["state"] == "unavailable" and by[name]["cause"] == "service_error"
        )
        assert "RuntimeError: Qdrant" in by[name]["reason"]
    assert (
        by["judge_decision"]["state"] == "scored" and by["judge_decision"]["value"] == 0
    )
    assert by["judge_facts"]["state"] == "unavailable"
    assert all(not m.calls for m in scorer.metrics.values())  # 채점 호출 없음


def test_faithfulness_nan_is_na_but_recall_failure_is_unavailable():
    values = {"faithfulness": float("nan"), "context_recall": RuntimeError("timeout")}
    by = _by_name(asyncio.run(_scorer(values).score(_rec(), GOLDEN)))
    assert (
        by["faithfulness"]["state"] == "na"
        and by["faithfulness"]["cause"] == "no_claims"
    )
    assert (
        by["context_recall"]["state"] == "unavailable"
        and by["context_recall"]["cause"] == "grading_failed"
    )
    assert "timeout" in by["context_recall"]["reason"]
    assert by["context_precision"]["state"] == "scored"


def test_empty_context_and_missing_reference():
    by = _by_name(asyncio.run(_scorer().score(_rec(contexts=[]), GOLDEN)))
    assert by["faithfulness"]["cause"] == "empty_context"
    assert by["context_precision"]["cause"] == "empty_context"
    assert by["answer_relevancy"]["state"] == "scored"  # 문맥 없이도 채점

    by = _by_name(
        asyncio.run(_scorer().score(_rec(), {**GOLDEN, "reference_answer": ""}))
    )
    assert (
        by["context_precision"]["state"] == "na"
        and by["context_precision"]["cause"] == "no_reference"
    )
    assert by["context_recall"]["state"] == "na"


def test_no_golden_context_metrics_and_judge_unavailable():
    by = _by_name(asyncio.run(_scorer().score(_rec(), None)))
    assert by["faithfulness"]["state"] == "scored"  # 정답이 필요 없는 지표
    assert by["answer_relevancy"]["state"] == "scored"
    assert by["context_recall"]["cause"] == "no_golden"
    assert all(by[n]["state"] == "unavailable" for n in q.JUDGE_NAMES.values())


def test_clarify_answer_relevancy_zero_still_scored():
    rec = _rec(
        status="needs_clarification", answer="어떤 장비인지 알려 주세요.", source_ids=[]
    )
    by = _by_name(asyncio.run(_scorer({"answer_relevancy": 0.0}).score(rec, GOLDEN)))
    assert by["answer_relevancy"] == {
        **by["answer_relevancy"],
        "state": "scored",
        "value": 0.0,
    }


def test_judge_null_items():
    judge = FakeJudge(_judge_result(facts=None, evidence=None, decision=None))
    by = _by_name(asyncio.run(_scorer(judge=judge).score(_rec(), GOLDEN)))
    assert (
        by["judge_facts"]["state"] == "na" and by["judge_facts"]["cause"] == "judge_na"
    )
    assert (
        by["judge_decision"]["state"] == "unavailable"
        and by["judge_decision"]["cause"] == "judge_null"
    )
    assert by["judge_conditions"]["value"] == 1


def test_judge_failure_is_unavailable_and_ragas_still_scored():
    by = _by_name(
        asyncio.run(
            _scorer(judge=FakeJudge(ValueError("bad json"))).score(_rec(), GOLDEN)
        )
    )
    for name in q.JUDGE_NAMES.values():
        assert (
            by[name]["state"] == "unavailable" and by[name]["cause"] == "grading_failed"
        )
    assert by["faithfulness"]["state"] == "scored"


def test_missing_metric_object_is_rejected():
    with pytest.raises(ValueError, match="answer_relevancy"):
        q.QualityScorer(
            metrics={"faithfulness": FakeMetric(1.0)}, judge=FakeJudge(None)
        )


# ---------- 요약 ----------


def test_summarize_excludes_na_and_unavailable_and_splits_by_status():
    rows = [
        (
            "answered",
            [
                q.score("faithfulness", 1.0, "scored"),
                q.score("answer_relevancy", 0.9, "scored"),
            ],
        ),
        (
            "answered",
            [
                q.score("faithfulness", 0.0, "scored"),
                q.score("answer_relevancy", 0.7, "scored"),
            ],
        ),
        (
            "needs_clarification",
            [
                q.score("faithfulness", None, "na", "no_claims"),
                q.score("answer_relevancy", 0.0, "scored"),
            ],
        ),
        (
            "error",
            [
                q.score("faithfulness", None, "unavailable", "service_error"),
                q.score("answer_relevancy", None, "unavailable", "service_error"),
            ],
        ),
    ]
    s = q.summarize(rows)
    f = s["faithfulness"]
    assert f["mean"] == 0.5 and f["n_scored"] == 2 and f["n_zero"] == 1
    assert f["n_na"] == 1 and f["n_unavailable"] == 1
    assert f["causes"] == {"no_claims": 1, "service_error": 1}
    ar = s["answer_relevancy"]["by_status"]
    assert ar["answered"] == {"mean": 0.8, "n_scored": 2, "n": 2}
    assert ar["needs_clarification"]["mean"] == 0.0
    assert ar["error"] == {"mean": None, "n_scored": 0, "n": 1}
    table = q.report(s)
    assert "| faithfulness | 0.500 | 2 | 1 | 1 |" in table


# ---------- 실제 ragas 지표 객체 (가짜 OpenAI + 가짜 임베딩) ----------


class FakeEmbeddings:
    """LangChain 임베딩 흉내. 같은 글이면 같은 벡터."""

    def __init__(self):
        self.calls = []

    def embed_query(self, text):
        self.calls.append(text)
        return [1.0, 0.0] if "컨테이너" in text else [0.0, 1.0]


def _fake_openai(content_for, seen):
    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        content = content_for(json.dumps(body, ensure_ascii=False))
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
                    "prompt_tokens": 50,
                    "completion_tokens": 10,
                    "total_tokens": 60,
                },
            },
        )

    return openai.AsyncOpenAI(
        api_key="test",
        http_client=openai.DefaultAsyncHttpxClient(
            transport=httpx2.MockTransport(handler)
        ),
    )


def test_real_answer_relevancy_uses_query_embeddings():
    seen = []
    client = _fake_openai(
        lambda _: {"question": "일반 건화물 컨테이너 종류는?", "noncommittal": 0}, seen
    )
    emb = FakeEmbeddings()
    scorer = q.QualityScorer.create(
        "gpt-6-luna", embeddings=emb, client=client, judge=FakeJudge(_judge_result())
    )
    row = asyncio.run(
        q.ragas_score(
            "answer_relevancy", scorer.metrics["answer_relevancy"], _rec(), GOLDEN
        )
    )
    assert row["state"] == "scored" and row["value"] == pytest.approx(1.0)
    assert len(seen) == 3  # strictness 3: 답변에서 질문 세 번 만들기
    assert emb.calls[0] == "일반 건화물 컨테이너 종류는?"
    assert scorer.grading_usage()["ragas"] == {
        "calls": 3,
        "input": 150,
        "output": 30,
        "unknown": 0,
    }


def test_real_answer_relevancy_noncommittal_is_zero():
    client = _fake_openai(
        lambda _: {"question": "무엇을 원하나요?", "noncommittal": 1}, []
    )
    scorer = q.QualityScorer.create(
        "gpt-6-luna", embeddings=FakeEmbeddings(), client=client, judge=FakeJudge(None)
    )
    row = asyncio.run(
        q.ragas_score(
            "answer_relevancy", scorer.metrics["answer_relevancy"], _rec(), GOLDEN
        )
    )
    assert row == {**row, "state": "scored", "value": 0.0}


def test_real_context_precision_and_recall():
    def content_for(text):
        if "classifications" in text:  # context_recall: 정답 문장이 문맥에 있는지
            return {
                "classifications": [
                    {"statement": "20GP", "reason": "있음", "attributed": 1},
                    {"statement": "40GP", "reason": "없음", "attributed": 0},
                ]
            }
        verdict = (
            1 if "20GP와 40GP다" in text else 0
        )  # context_precision: 문맥마다 쓸모 판정
        return {"reason": "판정", "verdict": verdict}

    client = _fake_openai(content_for, [])
    scorer = q.QualityScorer.create(
        "gpt-6-luna", embeddings=FakeEmbeddings(), client=client, judge=FakeJudge(None)
    )
    cp = asyncio.run(
        q.ragas_score(
            "context_precision", scorer.metrics["context_precision"], _rec(), GOLDEN
        )
    )
    cr = asyncio.run(
        q.ragas_score(
            "context_recall", scorer.metrics["context_recall"], _rec(), GOLDEN
        )
    )
    assert cp["state"] == "scored" and cp["value"] == pytest.approx(
        1.0
    )  # 쓸모 있는 문맥이 1등
    assert cr["state"] == "scored" and cr["value"] == 0.5
