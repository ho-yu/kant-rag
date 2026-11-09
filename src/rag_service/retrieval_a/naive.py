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
MODELS_PATH = REPO_ROOT / "config" / "retrieval_a" / "models.yaml"
LOCAL_QDRANT_PATH = REPO_ROOT / "artifacts" / "retrieval_a" / "qdrant_local"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


DIFFICULTIES = ("basic", "intermediate", "advanced")
DOMAINS = ("shipping", "finance", "manufacturing")
SPARSE_BACKENDS = ("bge-m3", "bm25")


def model_slug(model: str) -> str:
    return model.split("/")[-1].lower()


def resolve_scope(cfg: dict[str, Any], domain: str | None, difficulty: str | None) -> tuple[str, str]:
    c = cfg["collection"]
    domain = (domain or c["domain"]).strip().lower()
    difficulty = (difficulty or c["difficulty"]).strip().lower()
    if domain not in DOMAINS:
        raise ValueError(f"domain은 {DOMAINS} 중 하나여야 합니다: {domain!r}")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty는 {DIFFICULTIES} 중 하나여야 합니다: {difficulty!r}")
    return domain, difficulty


def model_profile(difficulty: str, path: Path = MODELS_PATH) -> dict[str, str]:
    profiles = load_config(path)
    if difficulty not in profiles:
        raise ValueError(f"{path.name}에 {difficulty!r} 설정이 없습니다. 있는 이름: {list(profiles)}")
    p = profiles[difficulty]
    profile = {
        "embedding": p["embedding"],
        "query_prefix": p.get("query_prefix") or "",
        "sparse": p.get("sparse", "bm25"),
        "reranker": p["reranker"],
    }
    if profile["sparse"] not in SPARSE_BACKENDS:
        raise ValueError(f"sparse는 {SPARSE_BACKENDS} 중 하나여야 합니다: {profile['sparse']!r}")
    if profile["sparse"] == "bge-m3" and model_slug(profile["embedding"]) != "bge-m3":
        raise ValueError(
            f"sparse: bge-m3는 임베딩이 BAAI/bge-m3일 때만 쓸 수 있습니다 "
            f"({difficulty}: {profile['embedding']}). sparse: bm25로 바꿔 주세요."
        )
    return profile


def collection_name(
    cfg: dict[str, Any], *, domain: str | None = None, difficulty: str | None = None, profile: str | None = None
) -> str:
    domain, difficulty = resolve_scope(cfg, domain, difficulty)
    tag = model_slug(model_profile(profile or difficulty)["embedding"])
    return f"{cfg['collection']['prefix']}_{domain}_{difficulty}_{tag}"


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
def _embeddings(model_name: str, device: str, normalize: bool, query_prefix: str = ""):
    from langchain_huggingface import HuggingFaceEmbeddings

    query_kwargs = {"normalize_embeddings": normalize}
    if query_prefix:
        query_kwargs["prompt"] = query_prefix
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": _resolve_device(device)},
        encode_kwargs={"normalize_embeddings": normalize},
        query_encode_kwargs=query_kwargs,
    )


@lru_cache(maxsize=1)
def _remote_client(url: str, api_key: str | None):
    from qdrant_client import QdrantClient

    return QdrantClient(url=url, api_key=api_key, timeout=60)


@lru_cache(maxsize=1)
def local_client(path: str = str(LOCAL_QDRANT_PATH)):
    from qdrant_client import QdrantClient

    Path(path).mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=path)


def _connect_client(name: str):
    url = (os.getenv("QDRANT_URL") or "").strip()
    if url:
        client = _remote_client(url, os.getenv("QDRANT_API_KEY") or None)
        if client.collection_exists(name):
            return client
    if LOCAL_QDRANT_PATH.exists():
        client = local_client()
        if client.collection_exists(name):
            return client
    where = "Qdrant 서버와 로컬 색인" if url else "로컬 색인 (QDRANT_URL이 비어 있음)"
    raise RuntimeError(
        f"{where}에 컬렉션 {name!r}이 없습니다. 아직 적재되지 않은 주제·난이도이거나 "
        "models.yaml의 모델 설정과 적재된 모델이 다를 수 있습니다."
    )


def _connect_store(cfg: dict[str, Any], name: str, profile: dict[str, str]):
    from langchain_qdrant import QdrantVectorStore

    emb_cfg = cfg["embedding"]
    embedding = _embeddings(profile["embedding"], emb_cfg["device"], emb_cfg["normalize"], profile["query_prefix"])
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
    profile: str | None = None,
    config_path: Path = CONFIG_PATH,
) -> BaseRetriever:
    cfg = load_config(config_path)
    search = cfg["search"]
    domain, difficulty = resolve_scope(cfg, domain, difficulty)
    name = collection_name(cfg, domain=domain, difficulty=difficulty, profile=profile)
    store = _connect_store(cfg, name, model_profile(profile or difficulty))
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
    parser.add_argument("--profile", default=None, help="models.yaml의 모델 설정 이름 (기본: 난이도 이름)")
    args = parser.parse_args()

    retriever = build_naive_retriever(
        domain=args.domain, difficulty=args.difficulty, top_k=args.k, profile=args.profile
    )
    for i, doc in enumerate(retriever.invoke(args.question), start=1):
        m = doc.metadata
        print(f"[S{i}] {m.get('book_id')}.pdf {m.get('page')}쪽 ({m.get('level')}, {m.get('chunk_id')})")
        print("    " + doc.page_content[:200].replace("\n", " "))


if __name__ == "__main__":
    _main()
