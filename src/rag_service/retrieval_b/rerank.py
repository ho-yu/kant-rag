"""Hybrid 후보를 Cross-Encoder(리랭커)로 다시 점수 매겨 순서를 바꾼다.

- 모델: BAAI/bge-reranker-v2-m3 (팀 결정 11/6). RERANKER_MODEL 환경변수로 바꿀 수 있다.
- 설정은 retrieval_a와 같게 맞춘다: 후보 50개 → 상위 5개, max_length 1024,
  GPU면 fp16, 점수는 sigmoid(0~1). 같은 조건이라 두 구현을 바로 비교할 수 있다.
- 리랭커 입력: (질문, "[제목 | 섹션 | p쪽]" + context + 원문). BM25·Dense 색인 문자열과 같은 모양.
- 리랭커는 후보 안에서 순서만 바꾼다. 후보에 없는 청크는 찾아오지 못한다 (7장 3강 §2.1).

결과 Document metadata
- score, rerank_score: 리랭커 점수 (score는 검색기 공통 키)
- prev_rank: 리랭크 전 Hybrid 순위 (1부터), hybrid_score: 리랭크 전 RRF 점수
- dense_rank, bm25_rank 등 Hybrid가 붙인 값은 그대로 남는다
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any

from langchain_core.documents import Document

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_N = 50  # 리랭커에 넘길 Hybrid 후보 수
MAX_LENGTH = 1024
BATCH_SIZE = 16

# (질문, 본문 목록) → 본문마다 점수. 실제로는 Cross-Encoder, 테스트에서는 가짜 함수.
ScoreFn = Callable[[str, list[str]], list[float]]


def rerank_text(doc: Document) -> str:
    """리랭커에 넣을 본문. Chunk.embedding_text()와 같은 모양으로 metadata에서 만든다."""
    m = doc.metadata
    head = " | ".join(
        str(x)
        for x in (m.get("doc_title"), m.get("section_title"), f"p{m.get('page')}")
        if x
    )
    parts = [f"[{head}]", m.get("context") or "", doc.page_content]
    return "\n".join(p for p in parts if p)


def model_name_from_env(model_name: str | None = None) -> str:
    return model_name or os.getenv("RERANKER_MODEL") or RERANKER_MODEL


@lru_cache(maxsize=2)
def load_cross_encoder(model_name: str, max_length: int = MAX_LENGTH) -> Any:
    """Cross-Encoder를 한 번만 불러온다. GPU가 있으면 cuda + fp16, 없으면 cpu."""
    import torch
    from sentence_transformers import CrossEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {"dtype": torch.float16} if device == "cuda" else {}
    return CrossEncoder(
        model_name,
        device=device,
        max_length=max_length,
        model_kwargs=model_kwargs,
        activation_fn=torch.nn.Sigmoid(),
    )


def cross_encoder_scorer(
    model_name: str | None = None,
    max_length: int = MAX_LENGTH,
    batch_size: int = BATCH_SIZE,
) -> ScoreFn:
    """실제 리랭커 점수 함수. 모델은 처음 부를 때 불러온다 (만들 때는 안 불러옴)."""
    name = model_name_from_env(model_name)

    def score(question: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        model = load_cross_encoder(name, max_length)
        scores = model.predict(
            [(question, t) for t in texts],
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [float(s) for s in scores]

    return score


def rerank(
    question: str,
    docs: Sequence[Document],
    k: int,
    score_fn: ScoreFn,
) -> list[Document]:
    """후보 docs를 리랭커 점수 순으로 다시 세워 상위 k개를 돌려준다.

    점수가 아니라 후보의 행 번호를 정렬해서 원래 Document와 연결한다 (7장 3강 §3.3).
    동점이면 chunk_id 순. 넘긴 Document는 바꾸지 않는다.
    """
    if k < 1:
        raise ValueError(f"k는 1 이상이어야 합니다: {k}")
    if not docs:
        return []
    scores = [float(s) for s in score_fn(question, [rerank_text(d) for d in docs])]
    if len(scores) != len(docs):
        raise ValueError(f"후보 {len(docs)}개인데 점수가 {len(scores)}개입니다.")

    order = sorted(
        range(len(docs)), key=lambda i: (-scores[i], docs[i].metadata["chunk_id"])
    )
    out = []
    for i in order[:k]:
        doc = docs[i]
        meta = {
            **doc.metadata,
            "score": scores[i],
            "rerank_score": scores[i],
            "prev_rank": i + 1,
            "hybrid_score": doc.metadata.get("score"),
        }
        out.append(Document(page_content=doc.page_content, metadata=meta))
    return out
