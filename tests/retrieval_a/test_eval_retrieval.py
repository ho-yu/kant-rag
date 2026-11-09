import json

from rag_service.retrieval_a.eval_retrieval import group_rank, load_golden, parse_cli_output, quote_in_chunk

QUOTE = "대상 장비는 20GP와 40GP다. 두 장비 모두 일반 건화물용이며, 크기가 다르다는 이유로 예약 확인과 서류 제출 절차가 생략되지는 않는다."


def test_quote_match_ignores_spacing_and_punctuation():
    chunk = "서비스 범위\n대상 장비는 20GP와  40GP다 — 두 장비 모두 일반 건화물용이며 크기가 다르다는 이유로 예약 확인과 서류 제출 절차가 생략되지는 않는다"
    assert quote_in_chunk(chunk, QUOTE)
    assert not quote_in_chunk("운임 청구서는 출항 후 발행하며 수정 요청은 별도로 접수한다.", QUOTE)


def test_quote_spanning_chunks_matches_by_chunk_side():
    part = "크기가 다르다는 이유로 예약 확인과 서류 제출 절차가 생략되지는 않는다."
    assert quote_in_chunk(part, QUOTE)
    assert not quote_in_chunk("두 장비 모두 일반", QUOTE)


def test_other_page_does_not_match():
    groups = [[{"book_id": "SB01", "page": 1, "quote": QUOTE}]]
    texts = {"SB01-p2-c1": QUOTE}
    hits = [{"rank": 1, "book_id": "SB01", "page": 2, "chunk_id": "SB01-p2-c1"}]
    assert group_rank(hits, groups, texts) is None
    assert group_rank(hits, groups, texts, tolerance=1) == 1


def test_group_rules_or_inside_and_across_latest_rank():
    a = {"book_id": "SB01", "page": 1, "quote": "가나다라마바사아자차카타파하 첫째 근거 문장입니다"}
    a2 = {"book_id": "SB14", "page": 3, "quote": "대체 근거 문장으로 같은 사실을 뒷받침합니다 하나둘셋"}
    b = {"book_id": "sb11", "page": 2, "quote": "둘째 그룹 근거 문장은 이메일 경로를 유지한다고 적는다"}
    texts = {
        "X-p9-c1": "무관한 내용",
        "SB14-p3-c1": a2["quote"],
        "SB11-p2-c4": b["quote"],
    }
    hits = [
        {"rank": 1, "book_id": "X", "page": 9, "chunk_id": "X-p9-c1"},
        {"rank": 2, "book_id": "SB14", "page": 3, "chunk_id": "SB14-p3-c1"},
        {"rank": 4, "book_id": "SB11", "page": 2, "chunk_id": "SB11-p2-c4"},
    ]
    assert group_rank(hits, [[a, a2]], texts) == 2
    assert group_rank(hits, [[a, a2], [b]], texts) == 4
    assert group_rank(hits[:2], [[a, a2], [b]], texts) is None
    assert group_rank(hits, [[a, a2], [b]], texts, use_quote=False) == 4


def test_parse_cli_output():
    out = """UserWarning: something
=== bge-m3 · rerank ===
[S1] SB01.pdf 1쪽 (SB01-p1-c2) rerank=0.999 hybrid#2 dense#2 sparse#2
    대상 장비는 20GP와 40GP다. ...
Loading weights: 100%
[S2] SB01.pdf 1쪽 (para, SB01-p1-c1) rerank=0.973 hybrid#1 dense#1 sparse#1
"""
    assert parse_cli_output(out) == [
        {"rank": 1, "book_id": "SB01", "page": 1, "chunk_id": "SB01-p1-c2"},
        {"rank": 2, "book_id": "SB01", "page": 1, "chunk_id": "SB01-p1-c1"},
    ]


def test_load_golden_skips_empty_groups_and_reads_jsonl(tmp_path):
    rows = [
        {"id": "A", "question": "q1", "evidence_groups": [[{"book_id": "SB01", "page": 1, "quote": "x"}]]},
        {"id": "B", "question": "q2", "evidence_groups": []},
    ]
    arr, jl = tmp_path / "g.json", tmp_path / "g.jsonl"
    arr.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    jl.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    for path in (arr, jl):
        items, skipped = load_golden([str(path)])
        assert [q["id"] for q in items] == ["A"] and skipped == ["B"]
