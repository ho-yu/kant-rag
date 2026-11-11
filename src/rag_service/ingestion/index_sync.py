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
- sparse 벡터 (config index.sparse: [m3, bm25]): 포인트마다 sparse 벡터도 같이 올린다.
    m3 = bge-m3 단어 가중치, bm25 = Kiwi 형태소 BM25 (둘 다 [제목 | 절 | p쪽] + 맥락 + 원문으로 계산)
    → hybrid 검색이 컬렉션 전체를 내려받지 않고 질문마다 Qdrant에 바로 물어볼 수 있다.
    설정한 sparse가 없는 기존 컬렉션은 처음 한 번 새 구성으로 다시 만든다 (dense와 이미 있던 sparse는
    재사용, 없는 것만 계산). 다시 만드는 동안은 컬렉션이 비어 있으니 팀원 검색이 잠깐 안 된다.
- 접속·임베딩·컬렉션 이름은 공통 모듈 rag_service.storage.qdrant_store를 쓴다.
"""
from __future__ import annotations

import hashlib
import logging
import time
from collections import Counter
from collections.abc import Callable

from rag_service.contracts.chunk import Chunk
from rag_service.storage import qdrant_store
from rag_service.storage.corpus import load_chunks, load_config

log = logging.getLogger("index_sync")

BATCH = 256
UPLOAD_GROUP = 2048   # 이만큼씩 (임베딩·sparse 계산 → 업로드) 반복. 중간에 끊겨도 올린 것까지는 남는다


def content_hash(c: Chunk) -> str:
    return hashlib.sha256(c.embedding_text().encode("utf-8")).hexdigest()[:16]


def ensure_collection(client, name: str, dim: int, kinds: list[str] | None = None) -> None:
    from qdrant_client import models

    if client.collection_exists(name):
        return
    client.create_collection(
        name, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        sparse_vectors_config={k: qdrant_store.sparse_params(k) for k in kinds} if kinds else None,
    )
    for field, schema in (("metadata.level", models.PayloadSchemaType.KEYWORD),
                          ("metadata.book_id", models.PayloadSchemaType.KEYWORD),
                          ("metadata.page", models.PayloadSchemaType.INTEGER)):
        client.create_payload_index(name, field_name=field, field_schema=schema)


def existing_points(client, name: str, kinds: list[str] | None = None) -> dict[str, str]:
    """Qdrant에 있는 {포인트 ID: content_hash} (벡터는 받지 않아서 가볍다).
    kinds가 있으면 그 sparse가 다 없는 포인트(payload sparse_kinds로 확인)는 hash를 ""로 줘서 다시 올리게 한다."""
    need = set(kinds or [])
    out: dict[str, str] = {}
    offset = None
    while True:
        pts, offset = client.scroll(name, limit=2000, offset=offset,
                                    with_payload=["content_hash", "sparse_kinds"], with_vectors=False)
        for p in pts:
            pl = p.payload or {}
            ok = need <= set(pl.get("sparse_kinds") or [])
            out[str(p.id)] = pl.get("content_hash", "") if ok else ""
        if offset is None:
            return out


def _split_vector(v) -> tuple[object | None, dict]:
    """Qdrant에서 받은 벡터 → (dense, {sparse 이름: SparseVector}).
    sparse 없는 컬렉션은 리스트, 있으면 {"": dense, "m3": ..., "bm25": ...}."""
    if v is None:
        return None, {}
    if isinstance(v, dict):
        return v.get(""), {k: x for k, x in v.items() if k in qdrant_store.SPARSE_KINDS}
    return v, {}


def _all_vectors(client, name: str) -> dict[str, tuple]:
    """컬렉션의 모든 포인트 {content_hash: (dense, {sparse 이름: 벡터})}. 컬렉션을 다시 만들 때 재사용용."""
    import numpy as np

    out: dict = {}
    offset = None
    while True:
        pts, offset = client.scroll(name, limit=512, offset=offset, with_payload=["content_hash"],
                                    with_vectors=True)
        for p in pts:
            dense, sparse = _split_vector(p.vector)
            if dense is not None:   # float32 배열로 보관 (파이썬 리스트보다 메모리 1/8)
                out[(p.payload or {}).get("content_hash", "")] = (np.asarray(dense, dtype=np.float32), sparse)
        if offset is None:
            return out


def _encode_sparse(kind: str, enc, chunks: list[Chunk]) -> list[dict]:
    texts = [qdrant_store.sparse_text(c.payload()) for c in chunks]
    if kind == "bm25":
        return enc.encode(texts, [c.level for c in chunks], progress=True)
    return enc.encode(texts, progress=True)


def sync_points(client, name: str, chunks: list[Chunk], make_embedder: Callable[[], object],
                recreate: bool = False, make_sparse: dict[str, Callable[[], object]] | None = None) -> dict:
    """컬렉션 name을 chunks와 똑같이 맞춘다. 바뀐 것만 올리고 지운다.
    make_sparse = {sparse 이름: 만드는 함수}. 있으면 그 sparse 벡터도 같이 올린다
    (설정한 sparse가 없는 기존 컬렉션은 새 구성으로 다시 만든다)."""
    from qdrant_client import models

    make_sparse = make_sparse or {}
    kinds = [k for k in qdrant_store.SPARSE_KINDS if k in make_sparse]
    if recreate and client.collection_exists(name):
        client.delete_collection(name)

    donors: dict = {}   # content_hash → (dense, {sparse}) (컬렉션을 다시 만들 때만 사용)
    rebuilt = False
    if kinds and client.collection_exists(name) and not qdrant_store.has_sparse(client, name, kinds):
        log.info("%s에 sparse %s가 없어 새 구성으로 다시 만듭니다 (있는 벡터는 재사용, 없는 것만 계산)", name, kinds)
        donors = _all_vectors(client, name)
        dim = len(next(iter(donors.values()))[0]) if donors else None
        client.delete_collection(name)
        if dim:
            ensure_collection(client, name, dim, kinds)
        rebuilt = True

    want = {c.point_id: (c, content_hash(c)) for c in chunks}
    have = existing_points(client, name, kinds) if client.collection_exists(name) else {}

    to_upsert = [(pid, c, h) for pid, (c, h) in want.items() if have.get(pid) != h]
    to_delete = [pid for pid in have if pid not in want]   # 내용만 바뀐 것은 같은 ID로 덮어씀

    # 벡터 재사용: 같은 content_hash를 가진 기존 포인트(ID가 밀린 경우 포함)에서 복사
    need = {h for _, _, h in to_upsert}
    dense_of: dict = {}
    sparse_of: dict[str, dict] = {k: {} for k in kinds}
    for h, (dense, sp) in donors.items():
        if h in need:
            dense_of[h] = dense
            for k in kinds:
                if k in sp:
                    sparse_of[k][h] = sp[k]
    if not rebuilt:   # 같은 내용이 다른 ID로 이미 있으면 그 벡터를 받아 온다
        all_have = existing_points(client, name) if client.collection_exists(name) else {}
        donor_ids = list({h: pid for pid, h in all_have.items() if h in need and h not in dense_of}.values())
        for i in range(0, len(donor_ids), BATCH):
            for p in client.retrieve(name, ids=donor_ids[i:i + BATCH], with_payload=["content_hash"],
                                     with_vectors=True):
                h = (p.payload or {}).get("content_hash", "")
                dense, sp = _split_vector(p.vector)
                if dense is not None:
                    dense_of[h] = dense
                for k in kinds:
                    if k in sp:
                        sparse_of[k][h] = sp[k]
    donors.clear()

    todo = {h for _, _, h in to_upsert if h not in dense_of}
    exists = client.collection_exists(name)
    embedder = make_embedder() if (todo or not exists) else None
    if not exists:
        ensure_collection(client, name, embedder.dim, kinds)
    sparse_todo = {k: {h for _, _, h in to_upsert if h not in sparse_of[k]} for k in kinds}
    encoders = {k: make_sparse[k]() for k in kinds if sparse_todo[k]}
    for k in kinds:
        if sparse_todo[k]:
            log.info("%s sparse 계산: %d개", k, len(sparse_todo[k]))
    if todo:
        log.info("dense 새로 임베딩: %d개", len(todo))

    t0 = time.time()
    done = 0
    uses = Counter(h for _, _, h in to_upsert)   # 같은 내용(hash)이 여러 포인트에 쓰일 수 있어 다 쓴 뒤에만 비운다
    for g in range(0, len(to_upsert), UPLOAD_GROUP):   # 묶음마다 계산 → 바로 업로드 (끊겨도 올린 것까지 남음)
        group = to_upsert[g:g + UPLOAD_GROUP]
        new_dense = list({h: c for _, c, h in group if h in todo and h not in dense_of}.items())
        for i in range(0, len(new_dense), BATCH):
            part = new_dense[i:i + BATCH]
            for (h, _), v in zip(part, embedder.encode([c.embedding_text() for _, c in part]), strict=True):
                dense_of[h] = v
        for k in kinds:
            new_sparse = list({h: c for _, c, h in group if h in sparse_todo[k] and h not in sparse_of[k]}.items())
            if new_sparse:
                weights = _encode_sparse(k, encoders[k], [c for _, c in new_sparse])
                for (h, _), w in zip(new_sparse, weights, strict=True):
                    sparse_of[k][h] = qdrant_store.to_sparse_vector(w)
        for i in range(0, len(group), BATCH):
            part = group[i:i + BATCH]
            points = []
            for pid, c, h in part:
                dense = dense_of[h]
                dense = dense.tolist() if hasattr(dense, "tolist") else dense
                payload = {**c.payload(), "content_hash": h}
                if kinds:
                    vector = {"": dense, **{k: sparse_of[k][h] for k in kinds}}
                    payload["sparse_kinds"] = kinds
                else:
                    vector = dense
                points.append(models.PointStruct(id=pid, vector=vector, payload=payload))
            client.upsert(name, points=points)
        done += len(group)
        if len(to_upsert) > UPLOAD_GROUP:
            log.info("업로드 %d/%d", done, len(to_upsert))
        for _, _, h in group:   # 더 쓸 데가 없는 벡터는 메모리에서 비운다
            uses[h] -= 1
            if uses[h] == 0:
                dense_of.pop(h, None)
                for k in kinds:
                    sparse_of[k].pop(h, None)
    for i in range(0, len(to_delete), BATCH):   # 새 포인트를 올린 뒤에 지워서 검색이 비는 순간이 없게
        client.delete(name, points_selector=models.PointIdsList(points=to_delete[i:i + BATCH]))
    count = client.count(name, exact=True).count
    return {"collection": name, "points": len(chunks), "count": count, "upserted": len(to_upsert),
            "deleted": len(to_delete), "embedded": len(todo),
            "sparse": {k: len(v) for k, v in sparse_todo.items()}, "sparse_kinds": kinds,
            "rebuilt_for_sparse": rebuilt, "reused": len(to_upsert) - len(todo),
            "seconds": round(time.time() - t0, 1)}


def sync(difficulty: str, domain: str = "shipping", levels: list[str] | None = None, recreate: bool = False,
         config: dict | None = None, sparse: list[str] | None = None) -> dict:
    """최신 jsonl(index.levels)에 맞춰 난이도별 컬렉션을 동기화한다.
    sparse: 올릴 sparse 종류 (None이면 config index.sparse)."""
    cfg = config or load_config()
    ix = cfg["index"]
    levels = levels or list(ix["levels"])
    kinds = qdrant_store.sparse_kinds({"index": {"sparse": sparse}}) if sparse is not None \
        else qdrant_store.sparse_kinds(cfg)
    chunks = list(load_chunks(difficulty, domain, levels, cfg))
    name = qdrant_store.collection_name(cfg, difficulty, domain)
    log.info("Qdrant 동기화: %s ← 레벨 %s, 청크 %d개, sparse %s", name, levels, len(chunks), kinds or "없음")

    def make_embedder():
        return qdrant_store.Embedder(ix["embedding_model"], int(ix.get("embedding_batch_size", 32)), ix.get("device", "cuda"))

    def make_m3():
        return qdrant_store.SparseEncoder(ix.get("sparse_model") or ix["embedding_model"],
                                          int(ix.get("sparse_batch_size", 16)),
                                          int(ix.get("sparse_max_length", 1024)), ix.get("device", "cuda"))

    def make_bm25():
        texts = [qdrant_store.sparse_text(c.payload()) for c in chunks]
        avgdl = qdrant_store.BM25Encoder.fit_avgdl(texts, [c.level for c in chunks])
        log.info("bm25 레벨별 평균 토큰 수: %s", {k: round(v, 1) for k, v in avgdl.items()})
        return qdrant_store.BM25Encoder(avgdl, float(ix.get("bm25_k1", 1.5)), float(ix.get("bm25_b", 0.75)))

    makers = {"m3": make_m3, "bm25": make_bm25}
    client = qdrant_store.get_client(cfg)
    try:
        summary = sync_points(client, name, chunks, make_embedder, recreate, {k: makers[k] for k in kinds})
    finally:
        client.close()
    summary["levels"] = levels
    return summary
