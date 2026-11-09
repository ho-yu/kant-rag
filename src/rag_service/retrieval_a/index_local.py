from __future__ import annotations

import argparse
import json

from rag_service.ingestion.index_sync import sync_points
from rag_service.retrieval_a.naive import (
    LOCAL_QDRANT_PATH,
    collection_name,
    load_config,
    local_client,
    model_profile,
    resolve_scope,
)
from rag_service.storage import qdrant_store
from rag_service.storage.corpus import load_chunks
from rag_service.storage.corpus import load_config as load_chunking_config


def build_local_index(
    profile: str,
    *,
    domain: str | None = None,
    difficulty: str | None = None,
    batch_size: int = 8,
    recreate: bool = False,
) -> dict:
    cfg = load_config()
    domain, difficulty = resolve_scope(cfg, domain, difficulty)
    model = model_profile(profile)["embedding"]
    name = collection_name(cfg, domain=domain, difficulty=difficulty, profile=profile)
    chunking = load_chunking_config()
    chunks = list(load_chunks(difficulty, domain, chunking["index"]["levels"], chunking))

    def make_embedder():
        return qdrant_store.Embedder(model, batch_size, chunking["index"].get("device", "cuda"))

    summary = sync_points(local_client(), name, chunks, make_embedder, recreate)
    summary["model"] = model
    summary["path"] = str(LOCAL_QDRANT_PATH)
    return summary


def _main() -> None:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="정원님 청크(jsonl)를 다른 임베딩 모델로 로컬 Qdrant에 색인")
    parser.add_argument("--profile", required=True, help="models.yaml의 모델 설정 이름 (예: qwen3)")
    parser.add_argument("--domain", default=None)
    parser.add_argument("--difficulty", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()
    summary = build_local_index(
        args.profile,
        domain=args.domain,
        difficulty=args.difficulty,
        batch_size=args.batch_size,
        recreate=args.recreate,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
