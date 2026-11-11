"""여러 검색 결과를 RRF(Reciprocal Rank Fusion)로 하나의 순위로 합친다.

RRF 점수 = Σ 1 / (rrf_k + 순위)   (순위는 1부터, 그 검색기에 안 나왔으면 0점)

- 점수 크기가 다른 검색기(Dense 코사인 0~1, BM25 수십)도 순위만 쓰므로 바로 합칠 수 있다.
- rrf_k(기본 60)가 크면 순위 차이가 덜 중요해지고, 작으면 1위가 크게 앞선다.
- 같은 청크는 chunk_id로 알아본다.

결과 Document metadata
- score: RRF 점수 (검색기 공통 키)
- {이름}_rank, {이름}_score: 각 검색기에서의 순위와 원래 점수 (안 나왔으면 None)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from langchain_core.documents import Document

RRF_K = 60


def rrf_fuse(
    rankings: Mapping[str, Sequence[Document]],
    k: int,
    rrf_k: int = RRF_K,
) -> list[Document]:
    """{검색기 이름: 순위대로 정렬된 Document 목록}을 RRF로 합쳐 상위 k개를 돌려준다.

    동점이면 chunk_id 순서로 정렬해서 실행할 때마다 결과가 같다.
    넘긴 Document는 바꾸지 않고 metadata를 복사한 새 Document를 만든다.
    """
    if k < 1:
        raise ValueError(f"k는 1 이상이어야 합니다: {k}")
    if rrf_k < 0:
        raise ValueError(f"rrf_k는 0 이상이어야 합니다: {rrf_k}")

    scores: dict[str, float] = {}
    first: dict[str, Document] = {}
    ranks: dict[str, dict[str, tuple[int, float | None]]] = {}
    for name, docs in rankings.items():
        rank = 0
        for doc in docs:
            cid = doc.metadata["chunk_id"]
            if name in ranks.get(cid, {}):
                continue  # 한 검색기 안의 중복은 처음 순위만 센다
            rank += 1
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            first.setdefault(cid, doc)
            ranks.setdefault(cid, {})[name] = (rank, doc.metadata.get("score"))

    top = sorted(scores, key=lambda cid: (-scores[cid], cid))[:k]
    out = []
    for cid in top:
        meta = {**first[cid].metadata, "score": scores[cid]}
        for name in rankings:
            rank, score = ranks[cid].get(name, (None, None))
            meta[f"{name}_rank"] = rank
            meta[f"{name}_score"] = score
        out.append(Document(page_content=first[cid].page_content, metadata=meta))
    return out
