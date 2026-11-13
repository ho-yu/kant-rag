"""BM25: 팀 청크를 Kiwi로 토큰화해 키워드 점수로 검색한다.

- 입력 (source로 고른다. 기본값은 settings.Settings.bm25_source)
  - "jsonl": storage.corpus.load_chunks (artifacts/corpus/{난이도}/{도메인}/*.jsonl)
  - "qdrant": Dense와 같은 팀 Qdrant 컬렉션 (kant_{도메인}_{난이도}_bge-m3)의 payload. 벡터는 받지 않는다.
  두 곳 모두 청크를 jsonl 파일과 같은 순서로 맞춘다. 내용이 같으면 색인과 점수도 똑같다.
- 색인 문자열: Chunk.embedding_text() (Dense에 넣은 것과 같은 "[제목 | 섹션 | pN]" + context + text)
- 결과: Dense와 같은 모양의 Document (page_content = 원문 text, metadata에 chunk_id·book_id·page·level·score)

토큰화(Kiwi)와 점수 계산(BM25Okapi)은 어느 입력이든 같다. 바뀌는 것은 청크를 읽는 곳뿐이다.

[2026-11-11 기능 추가] 이유: BM25만 로컬 jsonl을 읽어 Dense(팀 Qdrant)와 다른 곳의 청크를 볼 수 있었음 → 효과: BM25도 기본으로 팀 Qdrant에서 청크를 읽음 (chain의 hybrid·hybrid_rerank 포함). --bm25-source jsonl로 로컬 파일도 고를 수 있음 (검색 계산은 그대로).
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable, Iterator
from functools import lru_cache

from kiwipiepy import Kiwi
from langchain_core.documents import Document
from qdrant_client.http.exceptions import ResponseHandlingException
from rank_bm25 import BM25Okapi

from rag_service.contracts.chunk import Chunk
from rag_service.retrieval_b.dense import get_store
from rag_service.retrieval_b.settings import BM25_SOURCES
from rag_service.storage.corpus import load_chunks
from rag_service.storage.qdrant_store import level_filter

# 장비 코드가 "20" + "GP"로 쪼개지지 않게 한 단어로 등록한다 (basic·intermediate jsonl에 나오는 코드)
USER_WORDS = ["20GP", "40GP", "40HC", "40RF", "B/L"]

# 뜻이 있는 품사만 남긴다: 명사, 수사, 외국어·숫자, 동사·형용사 어간, 어근, 일반 부사
KEEP_TAGS = frozenset(
    {"NNG", "NNP", "NNB", "NR", "NP", "SL", "SN", "SH", "VV", "VA", "XR", "MAG"}
)

Tokenizer = Callable[[str], list[str]]


@lru_cache(maxsize=1)
def get_kiwi() -> Kiwi:
    kiwi = Kiwi()
    for word in USER_WORDS:
        kiwi.add_user_word(word, "NNP")
    return kiwi


def tokenize(text: str) -> list[str]:
    """조사·어미·문장부호를 빼고 남은 형태소. 영문은 소문자로 맞춘다 (si = SI)."""
    return [t.form.lower() for t in get_kiwi().tokenize(text) if t.tag in KEEP_TAGS]


class BM25Index:
    """문서 목록과 색인 문자열로 BM25 색인을 만든다. 만들 때 한 번만 토큰화한다."""

    def __init__(
        self,
        docs: list[Document],
        texts: list[str] | None = None,
        tokenizer: Tokenizer = tokenize,
    ) -> None:
        if not docs:
            raise ValueError("BM25 색인에 넣을 문서가 없습니다.")
        texts = texts if texts is not None else [d.page_content for d in docs]
        if len(texts) != len(docs):
            raise ValueError(
                f"문서 {len(docs)}개와 색인 문자열 {len(texts)}개의 수가 다릅니다."
            )
        self.docs = docs
        self.tokenizer = tokenizer
        self.bm25 = BM25Okapi([tokenizer(t) for t in texts])

    def search(self, question: str, k: int) -> list[Document]:
        """점수가 0보다 큰 문서 중 상위 k개. 정렬은 (-점수, chunk_id)로 고정한다."""
        if not question.strip():
            raise ValueError("질문이 비어 있습니다.")
        if k < 1:
            raise ValueError(f"k는 1 이상이어야 합니다: {k}")
        scores = self.bm25.get_scores(self.tokenizer(question))
        ranked = sorted(
            (
                (float(score), doc)
                for score, doc in zip(scores, self.docs, strict=True)
                if score > 0
            ),
            key=lambda pair: (-pair[0], pair[1].metadata["chunk_id"]),
        )
        # 색인 안의 Document는 여러 질문이 같이 쓰므로 복사본에 점수를 붙인다
        return [
            Document(
                page_content=doc.page_content, metadata={**doc.metadata, "score": score}
            )
            for score, doc in ranked[:k]
        ]


# ---------- 청크 읽기: jsonl 또는 Qdrant ----------

SCROLL_LIMIT = 1000  # Qdrant에서 한 번에 받는 청크 수
SCROLL_RETRIES = 4  # 연결이 끊기면 1·2·4초 쉬고 다시 받는다 (모두 4번)


def chunk_from_payload(payload: dict) -> Chunk:
    """Qdrant payload {"page_content": 원문, "metadata": 나머지 칸} → jsonl 한 줄과 같은 Chunk."""
    return Chunk.model_validate(
        {**payload["metadata"], "text": payload["page_content"]}
    )


def file_order(chunk: Chunk) -> tuple[str, int, int]:
    """jsonl 파일 안 순서: 문서(book_id) → 쪽 → 쪽 안 번호(-cN, 쪽 청크는 0).

    Qdrant는 포인트 ID 순으로 돌려주므로 이 순서로 다시 세운다.
    BM25Okapi는 문서 순서에 따라 idf 평균의 소수점 끝자리가 달라질 수 있어서, 순서까지 jsonl과 맞춘다.
    """
    found = re.search(r"-c(\d+)$", chunk.chunk_id)
    return chunk.book_id, chunk.page, int(found.group(1)) if found else 0


def _scroll_page(client, collection: str, levels: tuple[str, ...], offset):
    """한 묶음(SCROLL_LIMIT개)을 받는다. 연결이 끊기면 조금 쉬고 다시 받는다."""
    for attempt in range(SCROLL_RETRIES):
        try:
            return client.scroll(
                collection_name=collection,
                scroll_filter=level_filter(list(levels)),
                limit=SCROLL_LIMIT,
                offset=offset,
                with_payload=["page_content", "metadata"],
                with_vectors=False,
            )
        except ResponseHandlingException as error:
            if attempt == SCROLL_RETRIES - 1:
                raise RuntimeError(
                    "Qdrant에서 BM25 청크를 받다가 연결이 계속 끊겼습니다. 네트워크를 "
                    "확인하거나, 로컬 파일로 돌리려면 --bm25-source jsonl을 주세요."
                ) from error
            time.sleep(2**attempt)
    raise AssertionError("도달하지 않음")


def scroll_payloads(client, collection: str, levels: tuple[str, ...]) -> Iterator[dict]:
    """컬렉션에서 levels에 해당하는 청크의 payload를 끝까지 받는다 (벡터는 받지 않는다)."""
    offset = None
    while True:
        points, offset = _scroll_page(client, collection, levels, offset)
        for point in points:
            yield point.payload or {}
        if offset is None:
            return


def load_qdrant_chunks(
    domain: str, difficulty: str, levels: tuple[str, ...]
) -> list[Chunk]:
    """Dense와 같은 팀 Qdrant 컬렉션에서 청크를 읽는다. 난이도마다 그 난이도의 컬렉션을 본다."""
    # 연결·컬렉션 확인과 오류 안내는 Dense와 같다 (이때 Dense용 bge-m3도 불러온다)
    store = get_store(domain, difficulty)
    chunks = [
        chunk_from_payload(payload)
        for payload in scroll_payloads(store.client, store.collection_name, levels)
    ]
    if not chunks:
        raise RuntimeError(
            f"Qdrant 컬렉션 {store.collection_name!r}에 {list(levels)} 청크가 없습니다."
        )
    return sorted(chunks, key=file_order)


def load_team_chunks(
    domain: str, difficulty: str, levels: tuple[str, ...], source: str
) -> list[Chunk]:
    """source("jsonl" 또는 "qdrant")에서 levels의 청크를 jsonl 파일 순서로 읽는다."""
    if source == "jsonl":
        return list(load_chunks(difficulty, domain=domain, levels=levels))
    if source == "qdrant":
        return load_qdrant_chunks(domain, difficulty, levels)
    raise ValueError(f"source는 {BM25_SOURCES} 중 하나여야 합니다: {source!r}")


@lru_cache(maxsize=9)
def get_bm25_index(
    domain: str, difficulty: str, levels: tuple[str, ...], source: str
) -> BM25Index:
    """팀 청크로 색인을 만든다. (domain, difficulty, levels, source)마다 한 번만 만든다."""
    chunks = load_team_chunks(domain, difficulty, levels, source)
    docs = [Document(**chunk.payload()) for chunk in chunks]
    return BM25Index(docs, [chunk.embedding_text() for chunk in chunks], tokenizer=bm25_tokens)


def as_levels(levels: Iterable[str]) -> tuple[str, ...]:
    """lru_cache 키로 쓰려고 순서와 중복을 정리한 tuple로 바꾼다."""
    return tuple(sorted(set(levels)))
