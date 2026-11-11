"""BM25: 팀 jsonl 청크를 Kiwi로 토큰화해 키워드 점수로 검색한다.

- 입력: storage.corpus.load_chunks (artifacts/corpus/{난이도}/{도메인}/*.jsonl)
- 색인 문자열: Chunk.embedding_text() (Dense에 넣은 것과 같은 "[제목 | 섹션 | pN]" + context + text)
- 결과: Dense와 같은 모양의 Document (page_content = 원문 text, metadata에 chunk_id·book_id·page·level·score)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import lru_cache

from kiwipiepy import Kiwi
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from rag_service.storage.corpus import load_chunks

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


@lru_cache(maxsize=9)
def get_bm25_index(domain: str, difficulty: str, levels: tuple[str, ...]) -> BM25Index:
    """팀 jsonl로 색인을 만든다. (domain, difficulty, levels)마다 한 번만 만든다."""
    chunks = list(load_chunks(difficulty, domain=domain, levels=levels))
    docs = [Document(**chunk.payload()) for chunk in chunks]
    return BM25Index(docs, [chunk.embedding_text() for chunk in chunks])


def as_levels(levels: Iterable[str]) -> tuple[str, ...]:
    """lru_cache 키로 쓰려고 순서와 중복을 정리한 tuple로 바꾼다."""
    return tuple(sorted(set(levels)))
