"""Qdrant 공통 기능 (읽기 쪽). 검색·생성 코드와 청킹(index_sync)이 같이 쓴다.

- collection_name(cfg, 난이도, 주제) → 예) kant_shipping_basic_bge-m3
- get_client(cfg): .env의 QDRANT_URL·QDRANT_API_KEY로 접속 (비어 있으면 로컬 폴더 index.local_path)
- Embedder: bge-m3 dense 임베딩 (정규화 → cosine, 기본 GPU fp16, 없으면 CPU)
- level_filter(["para", "table"]): 검색할 레벨만 고르는 필터
- get_pages(컬렉션, ["SB11-p5", ...]): page 청크(쪽 전체)를 chunk_id로 꺼낸다 (page 확장용)

Qdrant에 올리기(동기화)는 청킹 담당 전용이라 rag_service/ingestion/index_sync.py에 있다.

payload는 LangChain 형식: {"page_content": 원문, "metadata": {book_id, page, level, ...}, "content_hash"}
LangChain에서 바로 쓰기:
    from langchain_qdrant import QdrantVectorStore
    from langchain_huggingface import HuggingFaceEmbeddings
    emb = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", encode_kwargs={"normalize_embeddings": True})
    store = QdrantVectorStore.from_existing_collection(collection_name="kant_shipping_basic_bge-m3",
                                                      embedding=emb, url=QDRANT_URL, api_key=QDRANT_API_KEY)
    store.similarity_search("질문", k=5, filter=level_filter(["para", "table"]))
검색 필터 키는 "metadata.level", "metadata.book_id", "metadata.page" (페이로드 인덱스 있음).
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache

from rag_service.storage.corpus import repo_root

log = logging.getLogger("qdrant_store")


def model_slug(model: str) -> str:
    return model.split("/")[-1].lower()


def collection_name(cfg: dict, difficulty: str, domain: str = "shipping") -> str:
    ix = cfg["index"]
    return f"{ix.get('collection_prefix', 'kant')}_{domain}_{difficulty}_{model_slug(ix['embedding_model'])}"


def get_client(cfg: dict):
    from qdrant_client import QdrantClient

    url = os.getenv("QDRANT_URL", "").strip()
    if url:
        return QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY") or None, timeout=60)
    path = repo_root() / cfg["index"].get("local_path", "artifacts/qdrant_local")
    log.info("QDRANT_URL이 비어 있어 로컬 폴더에 저장합니다: %s", path)
    return QdrantClient(path=str(path))


def resolve_device(want: str = "cuda") -> str:
    """config의 index.device → 실제 장치. GPU를 못 쓰면 CPU로 바꾸고 이유를 알려준다."""
    import torch

    want = (want or "cuda").lower()
    if want == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    reason = "CPU 전용 PyTorch가 설치됨" if torch.version.cuda is None else "GPU 또는 드라이버를 찾지 못함"
    log.warning("GPU를 쓸 수 없어 CPU로 임베딩합니다 (%s). 느릴 수 있습니다.", reason)
    return "cpu"


class Embedder:
    """sentence-transformers로 bge-m3 dense 벡터를 만든다 (정규화 → cosine). 기본은 GPU(fp16)."""

    def __init__(self, model: str, batch_size: int = 32, device: str = "cuda"):
        from sentence_transformers import SentenceTransformer

        self.device = resolve_device(device)
        self.model = SentenceTransformer(model, device=self.device)
        if self.device == "cuda":
            import torch

            self.model.half()   # fp16: 메모리 절반, 속도 향상 (bge-m3 결과 차이는 무시할 수준)
            log.info("임베딩 장치: GPU %s (fp16)", torch.cuda.get_device_name(0))
        else:
            log.info("임베딩 장치: CPU")
        self.batch_size = batch_size
        self.dim = len(self.encode(["dim check"])[0])

    def encode(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 show_progress_bar=len(texts) > self.batch_size)
        return [v.tolist() for v in vecs]


@lru_cache(maxsize=1)
def _read_client():
    """읽기 전용 접속 (.env의 QDRANT_URL·QDRANT_API_KEY). 처음 한 번만 접속한다."""
    from qdrant_client import QdrantClient

    return QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY") or None, timeout=60)


def get_pages(collection: str, page_ids: list[str]) -> dict[str, dict]:
    """page 청크를 chunk_id로 꺼낸다 → {chunk_id: payload}. Qdrant에 없는 쪽은 빠진다.

    page 청크 chunk_id = {book_id}-p{page} (예: SB11-p5), 포인트 ID = chunk_point_id(chunk_id).
    payload = {"page_content": 쪽 전체 원문, "metadata": {...}}
    """
    from rag_service.contracts.chunk import chunk_point_id

    if not page_ids:
        return {}
    points = _read_client().retrieve(
        collection, ids=[chunk_point_id(p) for p in page_ids], with_payload=True, with_vectors=False
    )
    pages = {}
    for point in points:
        payload = point.payload or {}
        pages[payload.get("metadata", {}).get("chunk_id")] = payload
    return pages


def level_filter(levels: list[str], negate: bool = False):
    """검색할 레벨만 고르는 Qdrant 필터. 예) level_filter(["para", "table"])"""
    from qdrant_client import models

    cond = models.FieldCondition(key="metadata.level", match=models.MatchAny(any=list(levels)))
    return models.Filter(must_not=[cond]) if negate else models.Filter(must=[cond])
