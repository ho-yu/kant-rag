"""evaluation/answer/run_answers.py 테스트.

API·Qdrant 없이 가짜 검색기와 가짜 답변 체인으로 저장 형식, 오류 기록, 이어 하기를 확인한다.
"""

from __future__ import annotations

import importlib.util
import json

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from rag_service.contracts.schemas import RagAnswer
from rag_service.storage.corpus import repo_root


def _load(name: str):
    path = repo_root() / "evaluation" / "answer" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"answer_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ra = _load("run_answers")

DOCS = [
    Document(
        page_content="일반 건화물은 20GP와 40GP로 예약한다.",
        metadata={
            "chunk_id": "SB01-p1-c1",
            "book_id": "SB01",
            "page": 1,
            "level": "para",
        },
    ),
    Document(
        page_content="| 장비 | 마감 |",
        metadata={
            "chunk_id": "SB02-p3-c2",
            "book_id": "SB02",
            "page": 3,
            "level": "table",
        },
    ),
]
ANSWER = RagAnswer(
    status="answered",
    answer="20GP와 40GP로 예약할 수 있습니다 [S1]",
    reason="문맥에 장비 종류가 있음",
    source_ids=["S1"],
)
QUESTIONS = [{"id": f"Q{i}", "question": f"질문 {i}"} for i in range(1, 4)]


class FakeRetriever:
    def __init__(self, docs=DOCS, fail: bool = False):
        self.docs, self.fail, self.calls = docs, fail, []

    def invoke(self, question):
        self.calls.append(question)
        if self.fail:
            raise RuntimeError("Qdrant 연결 실패")
        return list(self.docs)


class FakeChain:
    def __init__(self, answer: RagAnswer = ANSWER, fail: bool = False, model=None):
        self.answer, self.fail, self.model, self.inputs = answer, fail, model, []

    def invoke(self, inputs):
        self.inputs.append(inputs)
        if self.model is not None:  # 토큰 사용량을 남기는 가짜 모델 호출
            self.model.invoke("hi")
        if self.fail:
            raise TimeoutError("응답 없음")
        return self.answer


def fake_format(docs):
    return "\n\n".join(
        f"[S{i}] {d.metadata['book_id']}.pdf / page {d.metadata['page']}\n{d.page_content}"
        for i, d in enumerate(docs, start=1)
    )


def _run_one(retriever=None, chain=None):
    return ra.run_one(
        "Q1",
        "컨테이너 종류는?",
        mode="naive",
        retriever=retriever or FakeRetriever(),
        answer_chain=chain or FakeChain(),
        format_context=fake_format,
    )


# ---------- 질문 읽기 ----------


def test_load_questions_keeps_only_id_and_question(tmp_path):
    path = tmp_path / "q.json"
    rows = [{"id": "Q1", "question": "질문", "reference_answer": "정답"}]
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    assert ra.load_questions(path) == [{"id": "Q1", "question": "질문"}]


@pytest.mark.parametrize(
    "rows",
    [
        [{"id": "Q1", "question": "a"}, {"id": "Q1", "question": "b"}],
        [{"id": "Q1"}],
        [{"id": "Q1", "question": "  "}],
    ],
)
def test_load_questions_rejects_bad_rows(tmp_path, rows):
    path = tmp_path / "q.json"
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError):
        ra.load_questions(path)


def test_development_questions_file_has_30():
    path = ra.questions_path("development", domain="shipping", difficulty="basic")
    assert path.name == "development_questions.json"
    questions = ra.load_questions(path)
    assert len(questions) == 30
    assert questions[0]["id"] == "S-DEV-01"


def test_questions_path_rejects_unknown_split():
    with pytest.raises(ValueError):
        ra.questions_path("test", domain="shipping", difficulty="basic")


# ---------- 기록 한 줄 ----------


def test_context_records_numbers_like_format_docs():
    recs = ra.context_records(DOCS + [Document(page_content="메타 없음")])
    assert [r["sid"] for r in recs] == ["S1", "S2", "S3"]
    assert recs[1] == {
        "sid": "S2",
        "chunk_id": "SB02-p3-c2",
        "book_id": "SB02",
        "page": 3,
        "level": "table",
        "text": "| 장비 | 마감 |",
    }
    assert recs[2]["book_id"] is None


def test_context_labels_match_chain_format_docs():
    """chain.py format_docs가 붙이는 [S번호]와 context_records의 sid가 같은 순서인지."""
    text = ra.default_format_context(DOCS)
    for rec in ra.context_records(DOCS):
        assert f"[{rec['sid']}] {rec['book_id']}.pdf" in text


def test_usage_totals():
    assert ra.usage_totals({}) is None
    assert ra.usage_totals(None) is None
    usage = {
        "m1": {"input_tokens": 10, "output_tokens": 2},
        "m2": {"input_tokens": 5, "output_tokens": 1},
    }
    assert ra.usage_totals(usage) == {"input": 15, "output": 3}


def test_run_one_success_record():
    chain = FakeChain()
    rec = _run_one(chain=chain)
    assert rec["status"] == "answered"
    assert rec["answer"] == ANSWER.answer
    assert rec["source_ids"] == ["S1"]
    assert [c["sid"] for c in rec["contexts"]] == ["S1", "S2"]
    assert rec["error"] is None and rec["error_stage"] is None
    assert (
        rec["tokens"] is None
    )  # 가짜 체인은 사용량을 남기지 않음 → 0이 아니라 확인 불가
    assert rec["latency_ms"] >= rec["retrieval_ms"] >= 0
    # 모델 입력: 질문과, format_docs 모양의 문맥
    assert chain.inputs == [
        {"question": "컨테이너 종류는?", "context": fake_format(DOCS)}
    ]


def test_run_one_counts_tokens_from_model_calls():
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="x",
                    usage_metadata={
                        "input_tokens": 120,
                        "output_tokens": 30,
                        "total_tokens": 150,
                    },
                    response_metadata={
                        "model_name": "fake"
                    },  # 콜백은 모델 이름이 있어야 센다
                )
            ]
        )
    )
    rec = _run_one(chain=FakeChain(model=model))
    assert rec["tokens"] == {"input": 120, "output": 30}


def test_run_one_retrieval_error_is_recorded():
    rec = _run_one(retriever=FakeRetriever(fail=True))
    assert rec["status"] == "error"
    assert rec["error_stage"] == "retrieval"
    assert rec["error"].startswith("RuntimeError")
    assert rec["contexts"] == [] and rec["retrieval_ms"] is None
    assert rec["latency_ms"] is not None  # 실패한 요청의 시간도 남긴다


def test_run_one_generation_error_keeps_contexts():
    rec = _run_one(chain=FakeChain(fail=True))
    assert rec["status"] == "error"
    assert rec["error_stage"] == "generation"
    assert rec["error"].startswith("TimeoutError")
    assert len(rec["contexts"]) == 2  # 무엇을 찾았는지는 남긴다


# ---------- 저장과 이어 하기 ----------


def test_run_mode_writes_one_line_per_question_and_resumes(tmp_path):
    path = tmp_path / "answers_naive.jsonl"
    retriever = FakeRetriever()
    made = []

    def factory():
        made.append(1)
        return retriever

    logs: list[str] = []
    new = ra.run_mode(
        "naive",
        QUESTIONS,
        path,
        retriever_factory=factory,
        answer_chain=FakeChain(),
        format_context=fake_format,
        log=logs.append,
    )
    rows = ra.read_jsonl(path)
    assert [r["id"] for r in rows] == ["Q1", "Q2", "Q3"] == [r["id"] for r in new]
    assert retriever.calls[0] == ra.WARMUP_QUESTION  # 워밍업이 먼저
    assert ra.WARMUP_QUESTION not in [
        r["question"] for r in rows
    ]  # 워밍업은 저장 안 함
    assert logs[0] == "[naive] 저장됨 0, 새로 실행 3"

    again = ra.run_mode(
        "naive",
        QUESTIONS,
        path,
        retriever_factory=factory,
        answer_chain=FakeChain(),
        format_context=fake_format,
        log=logs.append,
    )
    assert again == [] and len(made) == 1  # 할 일이 없으면 검색기를 만들지도 않는다
    assert logs[-1] == "[naive] 저장됨 3, 새로 실행 0"


def test_error_rows_count_as_done(tmp_path):
    """오류 줄도 완료로 본다. 다시 돌려도 덮어쓰지 않는다 (오류율에 그대로 남게)."""
    path = tmp_path / "answers_naive.jsonl"
    kwargs = {
        "retriever_factory": FakeRetriever,
        "format_context": fake_format,
        "log": lambda _: None,
    }
    ra.run_mode(
        "naive", QUESTIONS[:1], path, answer_chain=FakeChain(fail=True), **kwargs
    )
    ra.run_mode("naive", QUESTIONS[:1], path, answer_chain=FakeChain(), **kwargs)
    rows = ra.read_jsonl(path)
    assert len(rows) == 1 and rows[0]["status"] == "error"


def test_read_jsonl_skips_broken_line(tmp_path):
    path = tmp_path / "a.jsonl"
    path.write_text('{"id": "Q1"}\n{"id": "Q2"\n\n{"id": "Q3"}\n', encoding="utf-8")
    assert [r["id"] for r in ra.read_jsonl(path)] == ["Q1", "Q3"]
    assert ra.read_jsonl(tmp_path / "none.jsonl") == []


# ---------- 실행 설정 ----------


def test_config_is_written_once_and_checked(tmp_path):
    config = {"split": "development", "retrieval": {"k": 5}}
    ra.check_or_write_config(tmp_path, config)
    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert saved["retrieval"] == {"k": 5} and "created_at" in saved
    ra.check_or_write_config(tmp_path, config)  # 같으면 통과
    with pytest.raises(SystemExit):
        ra.check_or_write_config(tmp_path, {**config, "retrieval": {"k": 10}})


def test_run_config_records_same_settings_for_all_modes(monkeypatch):
    monkeypatch.setenv(
        "EMBEDDING_MODEL", "my-embedding"
    )  # .env 값보다 먼저 (load_dotenv는 덮어쓰지 않음)
    cfg = ra.run_config(
        split="development", domain="shipping", difficulty="basic", model="m"
    )
    assert cfg["generation"]["model"] == "m"
    assert cfg["retrieval"]["impl"] == "retrieval_b"
    assert cfg["retrieval"]["embedding_model"] == "my-embedding"
    assert cfg["retrieval"]["k"] >= 1
    assert "사전 판단" in cfg["pipeline"]


def test_out_dir_name_follows_changed_settings(monkeypatch):
    from rag_service.retrieval_b import settings as settings_mod

    for var in ("K", "LEVELS", "CANDIDATE_K", "RRF_K", "RERANK_N", "RERANK_MAX_LENGTH"):
        monkeypatch.setenv(f"RETRIEVAL_B_{var}", "")  # 빈 값 = 기본값 (.env보다 먼저)
    settings_mod.get_settings.cache_clear()
    try:
        assert ra.run_name("basic", "development") == "basic_development"
        assert ra.run_name("basic", "development", tag="r2") == "basic_development_r2"
        assert (
            ra.run_name("basic", "development", model="other")
            == "basic_development_other"
        )
        monkeypatch.setenv("RETRIEVAL_B_K", "3")
        monkeypatch.setenv("RETRIEVAL_B_RERANK_N", "100")
        settings_mod.get_settings.cache_clear()
        assert ra.run_name("basic", "development") == "basic_development_k3_n100"
        assert (
            ra.out_dir_for("basic", "development").name == "basic_development_k3_n100"
        )
    finally:
        settings_mod.get_settings.cache_clear()


# ---------- 실험 비교판 자동 갱신 ----------


def test_main_refreshes_board_unless_no_board(monkeypatch, tmp_path):
    monkeypatch.setattr(ra, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(ra, "load_questions", lambda path: [])
    monkeypatch.setattr(ra, "out_dir_for", lambda *a, **k: tmp_path / "out")
    monkeypatch.setattr(ra, "check_or_write_config", lambda *a, **k: None)
    monkeypatch.setattr(ra, "run_config", lambda **k: {})
    monkeypatch.setattr(ra, "make_answer_chain", lambda model: None)
    monkeypatch.setattr(ra, "run_mode", lambda *a, **k: None)
    calls = []
    monkeypatch.setattr(ra, "refresh_board", lambda *a: calls.append(1))
    ra.main(["--modes", "naive"])
    assert calls == [1]
    ra.main(["--modes", "naive", "--no-board"])
    assert calls == [1]


def test_refresh_board_is_silent_without_board_file(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(ra, "repo_root", lambda: tmp_path)  # 비교판 스크립트가 없는 곳
    assert ra.refresh_board() is None
    assert capsys.readouterr().out == ""


def test_out_dir_with_exp(monkeypatch, tmp_path):
    monkeypatch.setattr(ra, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(ra, "run_name", lambda *a, **k: "basic_development_k9")
    out = ra.out_dir_for("basic", "development", exp="page_k")
    assert (
        out
        == tmp_path
        / "results"
        / "experiments"
        / "page_k"
        / "answer"
        / "retrieval_b"
        / "basic_development_k9"
    )
    assert (
        ra.out_dir_for("basic", "development")
        == tmp_path / "results" / "answer" / "retrieval_b" / "basic_development_k9"
    )


def test_run_config_records_exp(monkeypatch):
    cfg = ra.run_config(
        split="development",
        domain="shipping",
        difficulty="basic",
        model="m",
        exp="page_k",
    )
    assert cfg["exp"] == "page_k"
