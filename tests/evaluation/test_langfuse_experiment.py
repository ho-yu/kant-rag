"""evaluation/langfuse_experiment.py 테스트.

Langfuse 서버·OpenAI·Qdrant 없이: chain 출력 → 기록, 점수 ↔ Evaluation 변환, 평가기, 실행 요약을 본다.
마지막 테스트는 실제 Langfuse SDK의 run_experiment를 서버 없이(로컬 문항) 돌려 평가기 연결을 확인한다.
"""

from __future__ import annotations

import asyncio
import importlib.util
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from rag_service.contracts.schemas import RagAnswer
from rag_service.pipeline import chain as chain_mod
from rag_service.storage.corpus import repo_root


def _load():
    path = repo_root() / "evaluation" / "langfuse_experiment.py"
    spec = importlib.util.spec_from_file_location("langfuse_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lx = _load()

DOCS = [
    Document(
        page_content="대상 장비는 20GP와 40GP다.",
        metadata={"book_id": "SB01", "page": 1, "chunk_id": "SB01-p1-c0"},
    ),
    Document(
        page_content="반입 마감은 출항 2일 전이다.",
        metadata={"book_id": "SB02", "page": 3, "chunk_id": "SB02-p3-c1"},
    ),
]
ANSWER = RagAnswer(
    status="answered",
    answer="20GP와 40GP입니다 [S1]\n\n\n[출처]\n[S1]\nfile_name : SB01.pdf",
    reason="문맥에 있음",
    source_ids=["S1"],
)
GOLDEN = {
    "reference_answer": "20GP와 40GP입니다.",
    "required_facts": ["20GP"],
    "expected_action": "answer",
    "evidence_groups": [[{"book_id": "SB01", "page": 1, "quote": "q"}]],
    "grading_notes": "",
}


class FakeChain:
    """chain.py와 같은 모양의 작은 LangChain chain: retrieve_docs 단계 → 답(RagAnswer)만 돌려준다.

    result가 Exception이면 검색 단계에서 그 오류를 낸다. step_name을 바꾸면 검색 단계 이름이 달라진다.
    """

    def __init__(self, result, docs=DOCS, step_name="retrieve_docs"):
        self.requests = []

        def retrieve_docs(request):
            self.requests.append(
                {k: request[k] for k in ("question", "retrieval_mode", "page_context")}
            )
            if isinstance(result, Exception):
                raise result
            return list(docs)

        retrieve_docs.__name__ = step_name
        self.runnable = RunnablePassthrough.assign(
            docs=RunnableLambda(retrieve_docs)
        ) | RunnableLambda(lambda request: result)

    async def ainvoke(self, request, config=None):
        return await self.runnable.ainvoke(request, config=config)


class FakeScorer:
    def __init__(self, scores):
        self.scores, self.calls = scores, []

    async def score(self, rec, golden):
        self.calls.append((rec, golden))
        return self.scores

    def grading_usage(self):
        return {"total": {"calls": 2, "input": 300, "output": 40, "unknown": 0}}


def test_dataset_name_matches_uploader():
    assert (
        lx.dataset_name("shipping", "basic", "expansion_validation")
        == "shipping-basic-expansion-validation-v1"
    )


def test_record_from_chain_strips_sources_and_keeps_final_context_order():
    rec = lx.record_from_chain("질문", {"answer": ANSWER, "docs": DOCS})
    assert rec["answer"] == "20GP와 40GP입니다 [S1]"
    assert [c["sid"] for c in rec["contexts"]] == ["S1", "S2"]
    assert rec["contexts"][1] == {
        "sid": "S2",
        "book_id": "SB02",
        "page": 3,
        "chunk_id": "SB02-p3-c1",
        "text": "반입 마감은 출항 2일 전이다.",
    }
    assert rec["status"] == "answered" and rec["source_ids"] == ["S1"]


def test_task_sends_only_question_and_records_service_error():
    chain = FakeChain(ANSWER)
    task = lx.make_task(chain, mode="hybrid_rerank", page_context=True)
    item = SimpleNamespace(
        input={"question": "일반 건화물 컨테이너 종류는?"}, expected_output=GOLDEN
    )
    rec = asyncio.run(task(item=item))
    assert chain.requests == [
        {
            "question": "일반 건화물 컨테이너 종류는?",
            "retrieval_mode": "hybrid_rerank",
            "page_context": True,
        }
    ]  # 정답 칸은 chain에 가지 않는다
    assert rec["status"] == "answered" and "latency_ms" in rec

    rec = asyncio.run(
        lx.make_task(FakeChain(RuntimeError("Qdrant 끊김")), mode="naive")(
            item={"input": {"question": "q"}}
        )
    )
    assert (
        rec["status"] == "error"
        and rec["error"] == "RuntimeError: Qdrant 끊김"
        and rec["contexts"] == []
    )


def test_task_captures_final_context_without_changing_chain():
    rec = asyncio.run(
        lx.make_task(FakeChain(ANSWER), mode="naive")(item={"input": {"question": "q"}})
    )
    assert rec["status"] == "answered"
    assert [c["chunk_id"] for c in rec["contexts"]] == ["SB01-p1-c0", "SB02-p3-c1"]


def test_missing_retrieve_step_is_error_not_empty_context():
    chain = FakeChain(ANSWER, step_name="search")  # chain.py에서 함수 이름이 바뀐 경우
    rec = asyncio.run(
        lx.make_task(chain, mode="naive")(item={"input": {"question": "q"}})
    )
    assert rec["status"] == "error" and "retrieve_docs" in rec["error"]


class CountingHandler(BaseCallbackHandler):
    def __init__(self):
        self.starts = 0

    def on_chain_start(self, *args, **kwargs):
        self.starts += 1


def test_real_create_chain_docs_captured_and_langfuse_callback_kept(monkeypatch):
    """실제 pipeline/chain.py create_chain (검색기·LLM만 가짜)에서 최종 문맥을 꺼낸다."""

    class FakeModel:
        def __init__(self, **kwargs):
            pass

        def with_structured_output(self, schema):
            return RunnableLambda(lambda prompt_value: ANSWER)

    retriever = RunnableLambda(lambda question: list(DOCS))
    for name in (
        "build_naive_retriever",
        "build_hybrid_retriever",
        "build_rerank_retriever",
    ):
        monkeypatch.setattr(chain_mod, name, lambda **kw: retriever)
    # retrieval_a는 RetrieverPool로 검색기를 준비하므로 풀도 가짜로 바꾼다
    monkeypatch.setattr(chain_mod, "RetrieverPool", lambda *a, **kw: SimpleNamespace(get=lambda mode: retriever))
    monkeypatch.setattr(chain_mod, "load_config", dict)
    monkeypatch.setattr(chain_mod, "collection_name", lambda *a, **kw: "test")
    monkeypatch.setattr(chain_mod, "create_model", lambda role="answer": FakeModel())
    service_handler = CountingHandler()  # with_langfuse가 붙이는 콜백 대신
    monkeypatch.setattr(
        chain_mod,
        "with_langfuse",
        lambda chain, **kw: chain.with_config(callbacks=[service_handler]),
    )

    chain = chain_mod.create_chain("shipping", "basic")
    out = asyncio.run(
        lx.invoke_with_docs(chain, {"question": "q", "retrieval_mode": "hybrid"})
    )
    assert out["answer"] == chain.invoke({"question": "q", "retrieval_mode": "hybrid"})
    assert [d.metadata["chunk_id"] for d in out["docs"]] == ["SB01-p1-c0", "SB02-p3-c1"]
    assert service_handler.starts > 0  # 서비스 쪽 콜백(Langfuse trace)도 그대로 불린다


def test_to_evaluations_numeric_for_scored_categorical_state_otherwise():
    scores = [
        lx.quality.score("faithfulness", 0.75, "scored"),
        lx.quality.score("context_recall", None, "na", "no_reference", "정답 없음"),
        lx.quality.score(
            "judge_facts", None, "unavailable", "grading_failed", "채점 실패: X"
        ),
    ]
    ev = lx.to_evaluations(scores, status="answered")
    assert (ev[0].name, ev[0].value, ev[0].data_type) == (
        "faithfulness",
        0.75,
        "NUMERIC",
    )
    assert (ev[1].name, ev[1].value, ev[1].data_type) == (
        "context_recall_state",
        "na",
        "CATEGORICAL",
    )
    assert ev[1].comment == "no_reference: 정답 없음"
    assert ev[2].metadata == {
        "state": "unavailable",
        "cause": "grading_failed",
        "status": "answered",
    }
    back = lx.from_evaluations(ev)
    assert [(b["name"], b["state"], b["value"]) for b in back] == [
        ("faithfulness", "scored", 0.75),
        ("context_recall", "na", None),
        ("judge_facts", "unavailable", None),
    ]


def test_answer_quality_evaluator_passes_record_and_golden():
    scorer = FakeScorer([lx.quality.score("faithfulness", 1.0, "scored")])
    evaluators = lx.make_evaluators(scorer)
    assert [e.__name__ for e in evaluators] == ["answer_quality", "retrieval_judgement"]
    out = asyncio.run(
        evaluators[0](
            input={"question": "q"},
            output={"status": "answered", "answer": "a"},
            expected_output=GOLDEN,
            metadata=None,
        )
    )
    assert out[0].name == "faithfulness"
    rec, golden = scorer.calls[0]
    assert rec["question"] == "q" and golden is GOLDEN


def test_retrieval_evaluator_uploads_doc_scores_and_can_be_turned_off():
    evaluators = lx.make_evaluators(FakeScorer([]))
    output = {"status": "answered", "contexts": [{"sid": "S1", "book_id": "SB01", "page": 1, "text": "t"}]}
    out = evaluators[1](input={}, output=output, expected_output=GOLDEN, metadata=None)
    assert [(e.name, e.value) for e in out] == [("hit_at_5", 1.0), ("recall_at_5", 1.0), ("mrr_at_5", 1.0)]
    names = [e.__name__ for e in lx.make_evaluators(FakeScorer([]), retrieval_k=None)]
    assert names == ["answer_quality"]


def test_without_code_scorer_only_retrieval_is_uploaded():
    assert [e.__name__ for e in lx.make_evaluators(None)] == ["retrieval_judgement"]
    assert lx.make_evaluators(None, retrieval_k=None) == []


def test_retrieval_scores_are_converted_when_returned(monkeypatch):
    monkeypatch.setattr(
        lx.retrieval_scores,
        "score_retrieval",
        lambda rec, golden, k: [lx.quality.score(f"doc_hit_at_{k}", 1.0, "scored")],
    )
    evaluators = lx.make_evaluators(FakeScorer([]), retrieval_k=3)
    out = evaluators[1](
        input={}, output={"status": "answered"}, expected_output=GOLDEN, metadata=None
    )
    assert (out[0].name, out[0].value) == ("doc_hit_at_3", 1.0)


def test_run_summary_means_counts_and_grading_tokens():
    def item(status, scores):
        return SimpleNamespace(
            output={"status": status},
            evaluations=lx.to_evaluations(scores, status=status),
        )

    q = lx.quality
    items = [
        item(
            "answered",
            [
                q.score("faithfulness", 1.0, "scored"),
                q.score("context_recall", None, "na", "no_reference"),
            ],
        ),
        item(
            "answered",
            [
                q.score("faithfulness", 0.5, "scored"),
                q.score("context_recall", None, "na", "no_reference"),
            ],
        ),
        item(
            "error",
            [
                q.score("faithfulness", None, "unavailable", "service_error"),
                q.score("context_recall", None, "unavailable", "service_error"),
            ],
        ),
    ]
    out = {e.name: e for e in lx.make_run_evaluator(FakeScorer([]))(item_results=items)}
    assert out["faithfulness_mean"].value == 0.75
    assert out["faithfulness_mean"].comment.startswith(
        "평가 대상 2 · 해당 없음 0 · 평가 불가 1"
    )
    assert "context_recall_mean" not in out  # 평가 대상 0건이면 평균을 만들지 않는다
    assert (
        out["grading_tokens_input"].value == 300
        and out["grading_tokens_output"].value == 40
    )


def test_run_summary_without_code_scorer_has_no_grading_tokens():
    item = SimpleNamespace(
        output={"status": "answered"},
        evaluations=lx.to_evaluations(
            [lx.quality.score("hit_at_5", 1.0, "scored")], status="answered"
        ),
    )
    out = {e.name: e for e in lx.make_run_evaluator(None)(item_results=[item])}
    assert out["hit_at_5_mean"].value == 1.0
    assert "grading_tokens_input" not in out


def test_run_name_for():
    now = datetime(2026, 11, 11, 13, 5, tzinfo=UTC)
    assert (
        lx.run_name_for("hybrid_rerank", page_context=False, now=now)
        == "hybrid_rerank_20261111-1305"
    )
    assert (
        lx.run_name_for("naive", page_context=True, tag="r2", now=now)
        == "naive_page_20261111-1305_r2"
    )
    assert (
        lx.run_name_for("hybrid", page_context=False, now=now, impl="retrieval_b")
        == "b_hybrid_20261111-1305"
    )


def test_impl_lists_match_chain():
    assert lx.IMPLS == chain_mod.IMPLS


def test_chain_picks_retriever_impl(monkeypatch):
    from rag_service import retrieval_b

    assert chain_mod.retriever_builders() == (
        chain_mod.build_naive_retriever,
        chain_mod.build_hybrid_retriever,
        chain_mod.build_rerank_retriever,
    )
    assert chain_mod.retriever_builders("retrieval_b") == (
        retrieval_b.build_naive_retriever,
        retrieval_b.build_hybrid_retriever,
        retrieval_b.build_rerank_retriever,
    )
    with pytest.raises(ValueError):
        chain_mod.retriever_builders("retrieval_c")

    seen = []
    for name in ("build_naive_retriever", "build_hybrid_retriever", "build_rerank_retriever"):
        monkeypatch.setattr(retrieval_b, name, lambda _n=name, **kw: seen.append((_n, kw)) or RunnableLambda(lambda q: []))
        monkeypatch.setattr(chain_mod, name, lambda **kw: pytest.fail("retrieval_a 검색기를 부르면 안 됨"))
    monkeypatch.setattr(chain_mod, "load_config", dict)
    monkeypatch.setattr(chain_mod, "collection_name", lambda *a, **kw: "test")
    monkeypatch.setattr(chain_mod, "create_model", lambda role="answer": SimpleNamespace(with_structured_output=lambda s: RunnableLambda(lambda p: ANSWER)))
    chain = chain_mod.create_chain("shipping", "advanced", impl="retrieval_b")
    # 검색기는 처음 요청한 검색 방식만 그때 만든다 (만들 때는 아무것도 준비하지 않음)
    assert seen == []
    for mode in ("naive", "hybrid", "hybrid_rerank"):
        chain.invoke({"question": "q", "retrieval_mode": mode})
    assert [n for n, _ in seen] == ["build_naive_retriever", "build_hybrid_retriever", "build_rerank_retriever"]
    assert all(kw == {"domain": "shipping", "difficulty": "advanced", "profile": None} for _, kw in seen)


def test_run_mode_passes_impl_to_chain(monkeypatch, tmp_path):
    import rag_service.pipeline as pipeline

    seen = {}

    def fake_create_chain(**kw):
        seen.update(kw)
        return FakeChain(ANSWER)

    class FakeLangfuse:
        def run_experiment(self, **kw):
            seen["run_name"], seen["metadata"] = kw["run_name"], kw["metadata"]
            return SimpleNamespace(item_results=[], dataset_run_url=None)

    monkeypatch.setattr(pipeline, "create_chain", fake_create_chain)
    monkeypatch.setattr(lx.quality.QualityScorer, "create", classmethod(lambda cls, model: FakeScorer([])))
    monkeypatch.setattr(lx, "ROOT", tmp_path)
    args = SimpleNamespace(
        domain="shipping", difficulty="basic", impl="retrieval_b", page_context=False, tag="",
        no_retrieval=False, code_answer_scores=True, k=5, model="m", dataset="d", concurrency=1,
    )
    payload = lx.run_mode(FakeLangfuse(), [], args, "naive")
    assert seen["impl"] == "retrieval_b" and seen["difficulty"] == "basic"
    assert seen["run_name"].startswith("b_naive_") and seen["metadata"]["impl"] == "retrieval_b"
    assert payload["impl"] == "retrieval_b"


@pytest.mark.parametrize("code_answer_scores", [False, True])
def test_answer_scores_come_from_langfuse_unless_code_flag(monkeypatch, tmp_path, code_answer_scores):
    from rag_service import pipeline

    seen = {"created": 0}

    class FakeLangfuse:
        def run_experiment(self, **kw):
            seen["evaluators"] = [e.__name__ for e in kw["evaluators"]]
            seen["metadata"], seen["description"] = kw["metadata"], kw["description"]
            return SimpleNamespace(item_results=[], dataset_run_url=None)

    def fake_create(cls, model):
        seen["created"] += 1
        return FakeScorer([])

    monkeypatch.setattr(pipeline, "create_chain", lambda **kw: FakeChain(ANSWER))
    monkeypatch.setattr(lx.quality.QualityScorer, "create", classmethod(fake_create))
    monkeypatch.setattr(lx.score_configs, "config_ids", lambda langfuse: {})
    monkeypatch.setattr(lx, "ROOT", tmp_path)
    args = SimpleNamespace(
        domain="shipping", difficulty="basic", impl="retrieval_a", page_context=False, tag="",
        no_retrieval=False, code_answer_scores=code_answer_scores, k=5, model="m", dataset="d",
        concurrency=1,
    )
    payload = lx.run_mode(FakeLangfuse(), [], args, "naive")
    assert seen["created"] == int(code_answer_scores)
    expected = ["answer_quality", "retrieval_judgement"] if code_answer_scores else ["retrieval_judgement"]
    assert seen["evaluators"] == expected
    assert seen["metadata"]["code_answer_scores"] == str(code_answer_scores)
    assert payload["code_answer_scores"] is code_answer_scores
    assert (payload["grading_usage"] is None) is not code_answer_scores
    assert ("Langfuse evaluator" in seen["description"]) is not code_answer_scores


def test_real_run_experiment_wires_task_and_evaluators(monkeypatch):
    """실제 Langfuse SDK(키 없음, 서버 없음)로 로컬 문항 두 개를 돌린다."""
    for key in (
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST",
        "LANGFUSE_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)
    from langfuse import Langfuse

    langfuse = Langfuse(
        public_key="pk-test",
        secret_key="sk-test",
        host="http://127.0.0.1:9",
        tracing_enabled=False,
    )
    scorer = FakeScorer(
        [
            lx.quality.score("faithfulness", 0.9, "scored"),
            lx.quality.score("judge_facts", None, "na", "judge_na", "되묻기"),
        ]
    )
    data = [
        {"input": {"question": "q1"}, "expected_output": GOLDEN},
        {"input": {"question": "q2"}, "expected_output": GOLDEN},
    ]
    result = langfuse.run_experiment(
        name="test",
        run_name="local",
        data=data,
        task=lx.make_task(FakeChain(ANSWER), mode="naive"),
        evaluators=lx.make_evaluators(scorer, retrieval_k=None),
        run_evaluators=[lx.make_run_evaluator(scorer)],
        max_concurrency=2,
    )
    assert len(result.item_results) == 2
    names = {e.name for r in result.item_results for e in r.evaluations}
    assert names == {"faithfulness", "judge_facts_state"}
    run = {e.name: e.value for e in result.run_evaluations}
    assert run["faithfulness_mean"] == pytest.approx(0.9)
    assert lx.summarize_items(result.item_results)["judge_facts"]["n_na"] == 2
