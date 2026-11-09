"""metrics.py와 비교 실행기(evaluation/retrieval/retrieval_b_compare.py) 테스트.

기본 테스트: 손계산 값과 가짜 검색기만 쓴다. 팀 Qdrant·모델 없이 돈다.
integration 테스트: 팀 Qdrant(읽기 전용)와 bge-m3로 development 전체를 채점한다.

실행 (레포 맨 위에서)
    uv run pytest tests/retrieval_b/test_metrics.py -q
    uv run pytest tests/retrieval_b/test_metrics.py -q -m "not integration"
"""

from __future__ import annotations

import importlib.util
import json
import os

import pytest
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from rag_service.retrieval_b import metrics
from rag_service.retrieval_b.metrics import (
    evaluate,
    gold_docs,
    gold_pages,
    golden_path,
    group_recall,
    hit_recall,
    load_golden,
    page_groups,
    score_question,
)
from rag_service.storage.corpus import repo_root

load_dotenv(repo_root() / ".env")


def doc(book_id: str, page: int, n: int = 1) -> Document:
    return Document(
        page_content=f"{book_id} {page}쪽",
        metadata={
            "chunk_id": f"{book_id}-p{page}-c{n}",
            "book_id": book_id,
            "page": page,
        },
    )


def item(groups, qid: str = "Q1", action: str = "answer") -> dict:
    """groups = [[(book_id, page), ...], ...] → 골든셋 항목 하나"""
    return {
        "id": qid,
        "question": f"{qid} 질문",
        "expected_action": action,
        "evidence_groups": [
            [{"book_id": b, "page": p, "quote": "..."} for b, p in g] for g in groups
        ],
    }


@pytest.fixture(scope="module")
def dev_items():
    return load_golden(golden_path("development"))


# ---------- hit_recall / group_recall: 개선판 8장 2강 손계산 ----------


def test_hit_recall_hand_calculation():
    # 8장 2강 §3: 정답 {meal, travel} 중 meal만 찾음 → Hit 1, Recall 0.5
    assert hit_recall({"meal"}, {"meal", "travel"}) == (1, 0.5)


def test_hit_recall_all_found():
    assert hit_recall({"A", "B", "X"}, {"A", "B"}) == (1, 1.0)


def test_hit_recall_no_gold_is_none():
    # 8장 2강 §4: 정답이 빈 질문은 평균에서 뺀다
    assert hit_recall(set(), set()) is None
    assert hit_recall({"A"}, set()) is None


def test_hit_recall_empty_result_is_zero():
    # 검색 결과가 비면 빼지 않고 0점
    assert hit_recall(set(), {"A"}) == (0, 0.0)


def test_group_recall_or_candidates():
    groups = [{"A", "B"}, {"C"}]  # 묶음 1은 A 또는 B면 충분
    assert group_recall({"A"}, groups) == 0.5
    assert group_recall({"B", "C"}, groups) == 1.0
    assert group_recall({"X"}, groups) == 0.0
    assert group_recall({"A"}, []) is None


# ---------- score_question ----------


def test_cut_k_before_making_set():
    """8장 2강 §6: 결과 A, A, B / 정답 {A, B} / k=2 → 앞 2개는 A, A라서 Recall 0.5"""
    q = item([[("A", 1)], [("B", 1)]])
    docs = [doc("A", 1, 1), doc("A", 1, 2), doc("B", 1)]
    s2 = score_question(q, docs, k=2)
    assert (s2["page_hit"], s2["page_recall"]) == (1, 0.5)
    s3 = score_question(q, docs, k=3)
    assert s3["page_recall"] == 1.0


def test_page_and_doc_level_differ():
    """같은 문서의 다른 쪽을 찾으면 문서 기준은 맞고 쪽 기준은 틀린다 (8장 2강 §7 질문 1)."""
    q = item([[("SB01", 2)]])
    s = score_question(q, [doc("SB01", 5)], k=1)
    assert (s["doc_hit"], s["doc_recall"]) == (1, 1.0)
    assert (s["page_hit"], s["page_recall"]) == (0, 0.0)
    assert s["missing_pages"] == [["SB01", 2]]


def test_union_and_group_recall_on_or_group():
    """묶음 하나에 후보 두 개(OR). 하나만 찾으면 합집합 0.5, 묶음 방식 1.0."""
    q = item([[("SB01", 1), ("SB14", 3)]])
    s = score_question(q, [doc("SB01", 1)], k=5)
    assert s["page_recall"] == 0.5
    assert s["page_group_recall"] == 1.0


def test_no_gold_question_scores_none():
    s = score_question(item([]), [doc("SB01", 1)], k=5)
    assert all(s[m] is None for m in metrics.METRICS)


def test_score_question_keeps_found_chunk_ids_in_order():
    docs = [doc("SB02", 1), doc("SB01", 3)]
    s = score_question(item([[("SB01", 3)]]), docs, k=5)
    assert s["found_chunk_ids"] == ["SB02-p1-c1", "SB01-p3-c1"]


def test_score_question_rejects_bad_k():
    with pytest.raises(ValueError):
        score_question(item([[("A", 1)]]), [], k=0)


def test_page_as_string_is_normalized():
    d = Document(
        page_content="x",
        metadata={"chunk_id": "SB01-p1-c1", "book_id": "SB01", "page": "1"},
    )
    assert score_question(item([[("SB01", 1)]]), [d], k=1)["page_hit"] == 1


# ---------- 골든셋 읽기 (11/9 main 데이터 기준) ----------


def test_s_dev_01_gold(dev_items):
    first = dev_items[0]
    assert first["id"] == "S-DEV-01"
    assert gold_docs(first) == {"SB01", "SB14"}
    assert gold_pages(first) == {("SB01", 1), ("SB14", 3)}
    assert page_groups(first) == [{("SB01", 1), ("SB14", 3)}]  # 묶음 하나, 후보 둘


def test_development_set_shape(dev_items):
    assert len(dev_items) == 30
    empty = [x["id"] for x in dev_items if not gold_pages(x)]
    assert empty == ["S-DEV-12", "S-DEV-27"]
    actions = [x["expected_action"] for x in dev_items]
    assert (
        actions.count("answer"),
        actions.count("clarify"),
        actions.count("abstain"),
    ) == (24, 3, 3)


@pytest.mark.parametrize(
    ("split", "count"),
    [("development", 30), ("additional", 30), ("expansion_validation", 12)],
)
def test_all_splits_load(split, count):
    assert len(load_golden(golden_path(split))) == count


def test_golden_path_rejects_unknown_split():
    with pytest.raises(ValueError):
        golden_path("security_tasks")


def test_golden_path_layout():
    p = golden_path("additional", difficulty="intermediate")
    assert p.parts[-5:] == (
        "data",
        "intermediate",
        "shipping",
        "evaluation",
        "additional.json",
    )


def test_load_golden_rejects_bad_files(tmp_path):
    not_list = tmp_path / "a.json"
    not_list.write_text('{"id": "x"}', encoding="utf-8")
    with pytest.raises(TypeError, match="list"):
        load_golden(not_list)

    bad_page = tmp_path / "b.json"
    bad = [
        {
            "id": "Q",
            "question": "q",
            "evidence_groups": [[{"book_id": "SB01", "page": "1"}]],
        }
    ]
    bad_page.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="Q"):
        load_golden(bad_page)

    dup = tmp_path / "c.json"
    dup.write_text(json.dumps([item([]), item([])]), encoding="utf-8")
    with pytest.raises(ValueError, match="겹칩"):
        load_golden(dup)


# ---------- evaluate: 가짜 검색기 ----------


def fake_retriever(answers: dict[str, list[Document]], calls: list[str]):
    def _invoke(question: str) -> list[Document]:
        calls.append(question)
        return answers[question]

    return RunnableLambda(_invoke)


def test_evaluate_calls_retriever_once_per_question():
    q1 = item([[("SB01", 1)]], "Q1")
    q2 = item([[("SB02", 2)], [("SB03", 3)]], "Q2")
    calls: list[str] = []
    r = fake_retriever(
        {"Q1 질문": [doc("SB09", 9), doc("SB01", 1)], "Q2 질문": [doc("SB02", 2)]},
        calls,
    )
    res = evaluate(r, [q1, q2], ks=(1, 3))
    assert calls == ["Q1 질문", "Q2 질문"]
    assert res["summary"]["1"]["page_hit"] == 0.5  # Q1 0, Q2 1
    assert res["summary"]["3"]["page_hit"] == 1.0
    assert res["summary"]["3"]["page_recall"] == 0.75  # Q1 1.0, Q2 0.5
    assert res["summary"]["3"]["n"] == 2
    assert res["items"][0]["scores"]["1"]["found_chunk_ids"] == ["SB09-p9-c1"]


def test_evaluate_skips_no_gold_in_average_but_keeps_row():
    calls: list[str] = []
    qs = [item([[("SB01", 1)]], "Q1"), item([], "Q2", action="clarify")]
    r = fake_retriever(
        {"Q1 질문": [doc("SB01", 1)], "Q2 질문": [doc("SB05", 1)]}, calls
    )
    res = evaluate(r, qs, ks=(5,))
    assert res["n_questions"] == 2
    assert res["summary"]["5"]["n"] == 1
    assert res["summary"]["5"]["page_hit"] == 1.0
    assert res["summary_answer_only"]["5"]["n"] == 1


def test_evaluate_records_error_as_zero_score():
    def boom(question: str):
        raise RuntimeError("Qdrant 연결 실패")

    res = evaluate(RunnableLambda(boom), [item([[("SB01", 1)]])], ks=(1,))
    assert res["n_errors"] == 1
    assert "Qdrant 연결 실패" in res["items"][0]["error"]
    assert res["summary"]["1"]["page_hit"] == 0.0
    assert res["latency_ms"]["mean"] is None  # 실패한 질문은 검색 시간에서 뺀다


def test_evaluate_counts_short_results_and_sorts_ks():
    calls: list[str] = []
    r = fake_retriever({"Q1 질문": [doc("SB01", 1)]}, calls)
    res = evaluate(r, [item([[("SB01", 1)]])], ks=(10, 1, 10))
    assert res["ks"] == [1, 10]
    assert res["short_results"] == 1  # 10개를 기대했는데 1개만 옴


def test_evaluate_rejects_bad_ks():
    with pytest.raises(ValueError):
        evaluate(RunnableLambda(lambda q: []), [], ks=(0, 5))


def test_evaluate_output_is_json_serializable():
    calls: list[str] = []
    r = fake_retriever({"Q1 질문": [doc("SB01", 1)]}, calls)
    json.dumps(evaluate(r, [item([[("SB01", 1)]])], ks=(1,)), ensure_ascii=False)


# ---------- 비교 실행기 ----------


@pytest.fixture
def compare_module():
    path = repo_root() / "evaluation" / "retrieval" / "retrieval_b_compare.py"
    spec = importlib.util.spec_from_file_location("retrieval_b_compare", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compare_writes_report(
    compare_module, monkeypatch, tmp_path, dev_items, capsys
):
    """가짜 검색기로 development 앞 3문항을 돌려 결과 파일 모양을 확인한다."""
    built = []

    def fake_builder(domain, difficulty, k, levels):
        built.append((domain, difficulty, k, levels))
        answers = {x["question"]: [doc(*min(gold_pages(x)))] for x in dev_items[:3]}
        answers[compare_module.WARMUP_QUESTION] = []
        return RunnableLambda(lambda q: answers[q])

    def broken_builder(*args):
        raise RuntimeError("모델 없음")

    monkeypatch.setitem(compare_module.MODES, "naive", fake_builder)
    monkeypatch.setitem(compare_module.MODES, "a_naive", broken_builder)
    out = compare_module.main(
        [
            "--modes",
            "naive",
            "a_naive",
            "--limit",
            "3",
            "--ks",
            "1",
            "5",
            "--out",
            str(tmp_path / "r.json"),
        ]
    )
    report = json.loads(out.read_text(encoding="utf-8"))
    assert built == [("shipping", "basic", 5, ["para", "table"])]  # k는 ks 중 최댓값
    assert report["settings"]["split"] == "development"
    assert report["results"]["naive"]["summary"]["1"]["page_hit"] == 1.0
    assert report["results"]["naive"]["n_questions"] == 3
    # 한 방식이 실패해도 나머지는 저장
    assert "모델 없음" in report["results"]["a_naive"]["error"]
    assert "naive" in capsys.readouterr().out


def test_compare_default_output_name(compare_module, monkeypatch, tmp_path):
    """--out을 안 주면 results/comparison/retrieval_b/{난이도}_{split}_{levels}.json (진짜 레포에는 안 씀)."""
    monkeypatch.setitem(
        compare_module.MODES, "naive", lambda *a: RunnableLambda(lambda q: [])
    )
    monkeypatch.setattr(compare_module, "repo_root", lambda: tmp_path)
    out = compare_module.main(["--limit", "1", "--levels", "page"])
    expected = tmp_path / "results" / "comparison" / "retrieval_b"
    assert out == expected / "basic_development_page.json"
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["git_commit"] is None  # tmp_path는 git 폴더가 아님


# ---------- integration: 팀 Qdrant (읽기 전용) + bge-m3 ----------


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("QDRANT_URL"), reason=".env에 QDRANT_URL 없음")
def test_team_naive_baseline_on_development(dev_items):
    from rag_service.retrieval_b import build_naive_retriever

    retriever = build_naive_retriever(domain="shipping", difficulty="basic", k=10)
    res = evaluate(retriever, dev_items, ks=(1, 5, 10))
    assert res["n_errors"] == 0
    assert res["summary"]["10"]["n"] == 28
    s = res["summary"]
    # k가 크면 줄지 않는다
    assert s["1"]["page_hit"] <= s["5"]["page_hit"] <= s["10"]["page_hit"]
    assert s["10"]["doc_hit"] >= s["10"]["page_hit"]  # 문서 기준이 더 느슨하다
