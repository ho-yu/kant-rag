"""최신 jsonl 청크 → bge-m3 임베딩 → Qdrant 동기화 (색인). 청킹 담당 전용.

보통은 build_corpus가 청킹 뒤에 sync()를 자동으로 부른다. 색인만 다시 맞출 때는
    uv run python -m rag_service.ingestion.build_corpus --difficulty basic --index-only

동작
- 컬렉션: 난이도별 하나. 이름 = {prefix}_{도메인}_{난이도}_{모델}  예) kant_shipping_basic_bge-m3
  항상 최신 jsonl(구버전 없음)과 같게 맞춘다.
- 올리는 레벨은 config의 index.levels (기본 page, para, table / row는 무료 플랜 용량 때문에 off).
- 포인트 ID는 uuid5(chunk_id). payload의 content_hash(임베딩 문자열의 해시)로 내용이 바뀌었는지 본다.
- 동기화: Qdrant에 있는 (ID, content_hash)와 최신 청크를 비교해서
    새로 생겼거나 내용이 바뀐 포인트만 올리고, 없어진 포인트는 지운다.
    벡터는 같은 content_hash가 이미 Qdrant에 있으면 복사하고(ID가 밀린 경우 포함), 없을 때만 임베딩한다.
    → 문장 하나가 바뀌면 그 쪽의 청크 몇 개만 임베딩·업로드된다.
- 접속·임베딩·컬렉션 이름은 공통 모듈 rag_service.storage.qdrant_store를 쓴다.
"""
from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable

from rag_service.contracts.chunk import Chunk
from rag_service.storage import qdrant_store
from rag_service.storage.corpus import load_chunks, load_config

log = logging.getLogger("index_sync")

BATCH = 256


def content_hash(c: Chunk) -> str:
    return hashlib.sha256(c.embedding_text().encode("utf-8")).hexdigest()[:16]


def ensure_collection(client, name: str, dim: int) -> None:
    from qdrant_client import models

    if client.collection_exists(name):
        return
    client.create_collection(name, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))
    for field, schema in (("metadata.level", models.PayloadSchemaType.KEYWORD),
                          ("metadata.book_id", models.PayloadSchemaType.KEYWORD),
                          ("metadata.page", models.PayloadSchemaType.INTEGER)):
        client.create_payload_index(name, field_name=field, field_schema=schema)


def existing_points(client, name: str) -> dict[str, str]:
    """Qdrant에 있는 {포인트 ID: content_hash} (벡터는 받지 않아서 가볍다)."""
    out: dict[str, str] = {}
    offset = None
    while True:
        pts, offset = client.scroll(name, limit=2000, offset=offset, with_payload=["content_hash"],
                                    with_vectors=False)
        for p in pts:
            out[str(p.id)] = (p.payload or {}).get("content_hash", "")
        if offset is None:
            return out


def sync_points(client, name: str, chunks: list[Chunk], make_embedder: Callable[[], object],
                recreate: bool = False) -> dict:
    """컬렉션 name을 chunks와 똑같이 맞춘다. 바뀐 것만 올리고 지운다."""
    from qdrant_client import models

    if recreate and client.collection_exists(name):
        client.delete_collection(name)
    want = {c.point_id: (c, content_hash(c)) for c in chunks}
    have = existing_points(client, name) if client.collection_exists(name) else {}

    to_upsert = [(pid, c, h) for pid, (c, h) in want.items() if have.get(pid) != h]
    to_delete = [pid for pid in have if pid not in want]   # 내용만 바뀐 것은 같은 ID로 덮어씀

    # 벡터 재사용: 같은 content_hash를 가진 기존 포인트(ID가 밀린 경우 포함)에서 복사
    need = {h for _, _, h in to_upsert}
    donor_ids = list({h: pid for pid, h in have.items() if h in need}.values())
    vectors: dict[str, list[float]] = {}
    for i in range(0, len(donor_ids), BATCH):
        for p in client.retrieve(name, ids=donor_ids[i:i + BATCH], with_payload=["content_hash"],
                                 with_vectors=True):
            if p.vector is not None:
                vectors[(p.payload or {}).get("content_hash", "")] = p.vector

    todo = [(c, h) for _, c, h in to_upsert if h not in vectors]
    exists = client.collection_exists(name)
    embedder = make_embedder() if (todo or not exists) else None
    if not exists:
        ensure_collection(client, name, embedder.dim)

    t0 = time.time()
    for i in range(0, len(todo), BATCH):
        part = todo[i:i + BATCH]
        for (_, h), v in zip(part, embedder.encode([c.embedding_text() for c, _ in part]), strict=True):
            vectors[h] = v
    for i in range(0, len(to_upsert), BATCH):
        part = to_upsert[i:i + BATCH]
        client.upsert(name, points=[
            models.PointStruct(id=pid, vector=vectors[h], payload={**c.payload(), "content_hash": h})
            for pid, c, h in part
        ])
    for i in range(0, len(to_delete), BATCH):   # 새 포인트를 올린 뒤에 지워서 검색이 비는 순간이 없게
        client.delete(name, points_selector=models.PointIdsList(points=to_delete[i:i + BATCH]))
    count = client.count(name, exact=True).count
    return {"collection": name, "points": len(chunks), "count": count, "upserted": len(to_upsert),
            "deleted": len(to_delete), "embedded": len({h for _, h in todo}),
            "reused": len(to_upsert) - len(todo), "seconds": round(time.time() - t0, 1)}


def sync(difficulty: str, domain: str = "shipping", levels: list[str] | None = None, recreate: bool = False,
         config: dict | None = None) -> dict:
    """최신 jsonl(index.levels)에 맞춰 난이도별 컬렉션을 동기화한다."""
    cfg = config or load_config()
    levels = levels or list(cfg["index"]["levels"])
    chunks = list(load_chunks(difficulty, domain, levels, cfg))
    name = qdrant_store.collection_name(cfg, difficulty, domain)
    log.info("Qdrant 동기화: %s ← 레벨 %s, 청크 %d개", name, levels, len(chunks))

    def make_embedder():
        ix = cfg["index"]
        return qdrant_store.Embedder(ix["embedding_model"], int(ix.get("embedding_batch_size", 32)), ix.get("device", "cuda"))

    client = qdrant_store.get_client(cfg)
    try:
        summary = sync_points(client, name, chunks, make_embedder, recreate)
    finally:
        client.close()
    summary["levels"] = levels
    return summary
