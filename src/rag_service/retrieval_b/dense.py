"""Naive RAG: 팀 Qdrant 컬렉션에서 bge-m3 Dense 검색을 한다."""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from rag_service.storage.corpus import load_config, repo_root
from rag_service.storage.qdrant_store import collection_name, get_client, level_filter

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """질문을 벡터로 바꾸는 임베딩 객체. 문서 벡터와 같은 bge-m3를 쓴다."""
    load_dotenv(repo_root() / ".env")
    model_name = (os.getenv("EMBEDDING_MODEL") or "").strip() or DEFAULT_EMBEDDING_MODEL
    return HuggingFaceEmbeddings(
        model_name=model_name, encode_kwargs={"normalize_embeddings": True}
    )


@lru_cache(maxsize=9)
def get_store(domain: str, difficulty: str) -> QdrantVectorStore:
    """팀 Qdrant 컬렉션에 읽기 전용으로 연결한다. 연결이 안 되면 고칠 방법을 알려 준다."""
    load_dotenv(repo_root() / ".env")
    if not (os.getenv("QDRANT_URL") or "").strip():
        raise RuntimeError(
            "QDRANT_URL이 비어 있습니다. .env.example을 복사해 .env를 만들고 "
            "QDRANT_URL, QDRANT_API_KEY를 채워 주세요."
        )
    cfg = load_config()
    name = collection_name(cfg, difficulty=difficulty, domain=domain)
    client = get_client(cfg)
    try:
        exists = client.collection_exists(name)
    except UnexpectedResponse as error:
        if error.status_code in (401, 403):
            raise RuntimeError(
                "Qdrant가 접속을 거절했습니다. .env의 QDRANT_API_KEY를 확인하고, "
                "맞는데도 안 되면 담당자에게 물어보세요."
            ) from error
        raise
    except ResponseHandlingException as error:
        raise RuntimeError(
            "Qdrant 서버에 연결하지 못했습니다. .env의 QDRANT_URL과 "
            "네트워크(같은 망·VPN)를 확인해 주세요."
        ) from error
    if not exists:
        raise RuntimeError(
            f"Qdrant에 컬렉션 {name!r}이 없습니다. 아직 적재되지 않은 주제·난이도일 수 "
            "있으니 담당자에게 확인해 주세요."
        )
    return QdrantVectorStore(
        client=client, collection_name=name, embedding=get_embeddings()
    )


def dense_search(
    store: QdrantVectorStore, question: str, k: int, levels: list[str]
) -> list[Document]:
    """질문과 가까운 청크 k개를 점수와 함께 돌려준다. levels로 검색할 레벨을 고른다."""
    if not question.strip():
        raise ValueError("질문이 비어 있습니다.")
    if k < 1:
        raise ValueError(f"k는 1 이상이어야 합니다: {k}")

    pairs = store.similarity_search_with_score(
        question, k=k, filter=level_filter(levels)
    )
    docs = []

    for doc, score in pairs:
        doc.metadata["score"] = float(score)
        docs.append(doc)

    return sorted(docs, key=lambda d: (-d.metadata["score"], d.metadata["chunk_id"]))


if __name__ == "__main__":
    store = get_store("shipping", "basic")
    question = "부산에서 LA로 컨테이너를 예약할 때 필요한 정보는?"
    for rank, doc in enumerate(
        dense_search(store, question, k=5, levels=["para", "table"]), 1
    ):
        m = doc.metadata
        print(
            rank,
            m["chunk_id"],
            m["book_id"],
            m["page"],
            round(m["score"], 4),
            doc.page_content[:40].replace("\n", " "),
        )
    for bad_question, bad_k in [("", 5), ("   ", 5), (question, 0)]:
        try:
            dense_search(store, bad_question, k=bad_k, levels=["para", "table"])
        except ValueError as error:
            print("ValueError:", error)
    docs = dense_search(store, question, k=10, levels=["para", "table"])
    scores = [d.metadata["score"] for d in docs]
    print(len(docs), scores == sorted(scores, reverse=True))
