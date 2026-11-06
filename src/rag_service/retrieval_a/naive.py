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
DOMAINS = ("shipping", "finance", "manufacturing")


def collection_name(cfg: dict[str, Any], *, domain: str | None = None, difficulty: str | None = None) -> str:
    c = cfg["collection"]
    domain = (domain or c["domain"]).strip().lower()
    difficulty = (difficulty or c["difficulty"]).strip().lower()
    if domain not in DOMAINS:
        raise ValueError(f"domain은 {DOMAINS} 중 하나여야 합니다: {domain!r}")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty는 {DIFFICULTIES} 중 하나여야 합니다: {difficulty!r}")
    return f"{c['prefix']}_{domain}_{difficulty}_{c['model_tag']}"


def level_filter(levels: list[str]) -> models.Filter:
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
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": _resolve_device(device)},
        encode_kwargs={"normalize_embeddings": normalize},
    )


def _connect_client(name: str):
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
