"""검색 구현 A — Naive RAG.

질문을 bge-m3로 임베딩해 Qdrant에서 가장 가까운 청크 top_k개를 그대로 돌려준다.
키워드 검색(BM25)·리랭킹 없이 dense 벡터 유사도만 쓴다.

반환: LangChain 검색기. `retriever.invoke(question)` → list[Document]
  - page_content: 청크 원문
  - metadata["book_id"]: PDF 파일명(확장자 제외), metadata["page"]: 1부터 시작하는 쪽 번호

직접 실행:
  uv run python -m rag_service.retrieval_a.naive "VGM은 언제까지 내야 하나요?"
"""

from __future__ import annotations

import argparse
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from langchain_core.retrievers import BaseRetriever
from qdrant_client import models

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "config" / "retrieval_a" / "naive.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


DIFFICULTIES = ("basic", "intermediate", "advanced")
DOMAINS = ("shipping", "finance", "manufacturing")  # client.py 선택지와 같게


def collection_name(cfg: dict[str, Any], *, domain: str | None = None, difficulty: str | None = None) -> str:
    """chain.py가 넘기는 domain·difficulty로 컬렉션 이름을 만든다. 예) kant_shipping_basic_bge-m3"""
    c = cfg["collection"]
    domain = (domain or c["domain"]).strip().lower()
    difficulty = (difficulty or c["difficulty"]).strip().lower()
    if domain not in DOMAINS:
        raise ValueError(f"domain은 {DOMAINS} 중 하나여야 합니다: {domain!r}")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty는 {DIFFICULTIES} 중 하나여야 합니다: {difficulty!r}")
    return f"{c['prefix']}_{domain}_{difficulty}_{c['model_tag']}"


def level_filter(levels: list[str]) -> models.Filter:
    """metadata.level이 levels 중 하나인 포인트만 검색 (page와 문단이 중복으로 나오지 않게)."""
    return models.Filter(
        must=[models.FieldCondition(key="metadata.level", match=models.MatchAny(any=list(levels)))]
    )


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=4)
def _embeddings(model_name: str, device: str, normalize: bool):
    """임베딩 모델은 무거우니 한 번만 올린다."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": _resolve_device(device)},
        encode_kwargs={"normalize_embeddings": normalize},
    )


def _connect_client(name: str):
    """Qdrant에 연결하고 컬렉션이 있는지 확인한다 (임베딩 모델은 올리지 않음)."""
    url = os.getenv("QDRANT_URL")
    if not url:
        raise RuntimeError(
            "QDRANT_URL이 비어 있습니다. .env.example을 복사해 .env를 만들고 "
            "QDRANT_URL, QDRANT_API_KEY를 채워 주세요."
        )
    api_key = os.getenv("QDRANT_API_KEY") or None

    from qdrant_client import QdrantClient

    client = QdrantClient(url=url, api_key=api_key)
    if not client.collection_exists(name):
        raise RuntimeError(
            f"Qdrant에 컬렉션 {name!r}이 없습니다. 아직 적재되지 않은 주제·난이도일 수 있습니다 "
            "(지금은 shipping 위주로 적재됨)."
        )
    return client


def _connect_store(cfg: dict[str, Any], name: str):
    from langchain_qdrant import QdrantVectorStore

    emb_cfg = cfg["embedding"]
    embedding = _embeddings(emb_cfg["model_name"], emb_cfg["device"], emb_cfg["normalize"])
    client = _connect_client(name)
    return QdrantVectorStore(
        client=client,
        collection_name=name,
        embedding=embedding,
        vector_name=cfg["collection"].get("vector_name", ""),
    )


def build_naive_retriever(
    *,
    domain: str | None = None,
    difficulty: str | None = None,
    top_k: int | None = None,
    levels: list[str] | None = None,
    config_path: Path = CONFIG_PATH,
) -> BaseRetriever:
    """Naive RAG 검색기 (공통 체인용).

    chain.py의 create_chain(domain, difficulty)에서
    `build_naive_retriever(domain=domain, difficulty=difficulty)`로 한 번 만들고,
    질문마다 `retriever.invoke(question)`으로 list[Document]를 받는다.
    인자를 비우면 config/retrieval_a/naive.yaml 값을 쓴다.
    """
    cfg = load_config(config_path)
    search = cfg["search"]
    store = _connect_store(cfg, collection_name(cfg, domain=domain, difficulty=difficulty))
    return store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": top_k or search["top_k"],
            "filter": level_filter(levels or search["levels"]),
        },
    )


def _main() -> None:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(description="retrieval_a Naive 검색 결과 확인")
    parser.add_argument("question")
    parser.add_argument("--domain", default=None)
    parser.add_argument("--difficulty", default=None)
    parser.add_argument("--k", type=int, default=None)
    args = parser.parse_args()

    retriever = build_naive_retriever(domain=args.domain, difficulty=args.difficulty, top_k=args.k)
    for i, doc in enumerate(retriever.invoke(args.question), start=1):
        m = doc.metadata
        print(f"[S{i}] {m.get('book_id')}.pdf {m.get('page')}쪽 ({m.get('level')}, {m.get('chunk_id')})")
        print("    " + doc.page_content[:200].replace("\n", " "))


if __name__ == "__main__":
    _main()
