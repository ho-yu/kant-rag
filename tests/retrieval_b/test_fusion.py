"""fusion.py: RRF로 순위 합치기."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from rag_service.retrieval_b.fusion import RRF_K, rrf_fuse


def doc(chunk_id: str, score: float | None = None) -> Document:
    meta = {"chunk_id": chunk_id, "book_id": chunk_id[:4], "page": 1}
    if score is not None:
        meta["score"] = score
    return Document(page_content=f"text {chunk_id}", metadata=meta)


def ids(docs) -> list[str]:
    return [d.metadata["chunk_id"] for d in docs]


DENSE = [doc("SB01-p1-c1", 0.9), doc("SB02-p1-c1", 0.8), doc("SB03-p1-c1", 0.7)]
BM25 = [doc("SB02-p1-c1", 12.0), doc("SB04-p1-c1", 9.0), doc("SB01-p1-c1", 3.0)]


def test_scores_follow_the_formula():
    out = rrf_fuse({"dense": DENSE, "bm25": BM25}, k=4)
    scores = {d.metadata["chunk_id"]: d.metadata["score"] for d in out}
    assert scores["SB02-p1-c1"] == pytest.approx(1 / (RRF_K + 2) + 1 / (RRF_K + 1))
    assert scores["SB01-p1-c1"] == pytest.approx(1 / (RRF_K + 1) + 1 / (RRF_K + 3))
    assert scores["SB03-p1-c1"] == pytest.approx(1 / (RRF_K + 3))
    assert scores["SB04-p1-c1"] == pytest.approx(1 / (RRF_K + 2))


def test_both_lists_beat_one_list():
    """두 검색기 모두에 나온 청크가 한쪽에만 1위로 나온 청크보다 앞선다."""
    out = rrf_fuse({"dense": DENSE, "bm25": BM25}, k=4)
    assert ids(out) == ["SB02-p1-c1", "SB01-p1-c1", "SB04-p1-c1", "SB03-p1-c1"]


def test_k_cuts_the_list():
    assert ids(rrf_fuse({"dense": DENSE, "bm25": BM25}, k=1)) == ["SB02-p1-c1"]


def test_ranks_and_original_scores_are_kept():
    out = {
        d.metadata["chunk_id"]: d.metadata
        for d in rrf_fuse({"dense": DENSE, "bm25": BM25}, k=4)
    }
    assert (out["SB02-p1-c1"]["dense_rank"], out["SB02-p1-c1"]["bm25_rank"]) == (2, 1)
    assert (out["SB02-p1-c1"]["dense_score"], out["SB02-p1-c1"]["bm25_score"]) == (
        0.8,
        12.0,
    )
    assert out["SB03-p1-c1"]["bm25_rank"] is None
    assert out["SB03-p1-c1"]["bm25_score"] is None
    assert out["SB03-p1-c1"]["book_id"] == "SB03"


def test_ties_are_broken_by_chunk_id():
    out = rrf_fuse({"dense": [doc("SB09-p1-c1")], "bm25": [doc("SB01-p1-c1")]}, k=2)
    assert ids(out) == ["SB01-p1-c1", "SB09-p1-c1"]


def test_duplicate_inside_one_list_counts_once():
    out = rrf_fuse(
        {"dense": [doc("SB01-p1-c1"), doc("SB01-p1-c1"), doc("SB02-p1-c1")]}, k=2
    )
    assert out[0].metadata["score"] == pytest.approx(1 / (RRF_K + 1))
    assert out[1].metadata["dense_rank"] == 2  # 중복을 건너뛰고 2위


def test_inputs_are_not_changed():
    before = [dict(d.metadata) for d in DENSE]
    rrf_fuse({"dense": DENSE, "bm25": BM25}, k=4)
    assert [d.metadata for d in DENSE] == before


def test_empty_lists_give_empty_result():
    assert rrf_fuse({"dense": [], "bm25": []}, k=5) == []


def test_rrf_k_changes_the_gap():
    small = rrf_fuse({"dense": DENSE}, k=2, rrf_k=0)
    assert [d.metadata["score"] for d in small] == pytest.approx([1.0, 0.5])


@pytest.mark.parametrize("k, rrf_k", [(0, 60), (5, -1)])
def test_bad_arguments(k, rrf_k):
    with pytest.raises(ValueError):
        rrf_fuse({"dense": DENSE}, k=k, rrf_k=rrf_k)
