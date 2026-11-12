from __future__ import annotations

import argparse
import hashlib
import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from rag_service.retrieval_a.naive import (
    REPO_ROOT,
    _connect_client,
    _embeddings,
    _resolve_device,
    collection_name,
    level_filter,
    load_config,
    model_profile,
    resolve_scope,
)

CONFIG_PATH = REPO_ROOT / "config" / "retrieval_a" / "hybrid.yaml"

Sparse = dict[str, float]


class BGEM3Encoder:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "auto", max_length: int = 1024):
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = _resolve_device(device)
        self.max_length = max_length
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, dtype=dtype).to(self.device).eval()
        self.sparse_linear = torch.nn.Linear(self.model.config.hidden_size, 1)
        state = torch.load(hf_hub_download(model_name, "sparse_linear.pt"), map_location="cpu")
        self.sparse_linear.load_state_dict(state)
        self.sparse_linear = self.sparse_linear.to(self.device, dtype=dtype).eval()
        self.skip_ids = {
            self.tokenizer.cls_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
            self.tokenizer.unk_token_id,
        }

    def encode(self, texts: list[str], batch_size: int = 16) -> tuple[np.ndarray, list[Sparse]]:
        torch = self.torch
        dense_out, sparse_out = [], []
        with torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                batch = self.tokenizer(
                    texts[i : i + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                hidden = self.model(**batch).last_hidden_state
                cls = torch.nn.functional.normalize(hidden[:, 0].float(), dim=-1)
                dense_out.append(cls.cpu().numpy())
                weights = torch.relu(self.sparse_linear(hidden)).squeeze(-1).float().cpu().numpy()
                for ids, w in zip(batch["input_ids"].cpu().numpy(), weights, strict=True):
                    sparse_out.append(_token_weights(ids, w, self.skip_ids))
        return np.concatenate(dense_out), sparse_out


def _token_weights(ids, weights, skip_ids) -> Sparse:
    out: Sparse = {}
    for tid, w in zip(ids, weights, strict=True):
        tid = int(tid)
        if tid in skip_ids or w <= 0:
            continue
        key = str(tid)
        if w > out.get(key, 0.0):
            out[key] = float(w)
    return out


@lru_cache(maxsize=2)
def _encoder(model_name: str, device: str, max_length: int) -> BGEM3Encoder:
    return BGEM3Encoder(model_name, device, max_length)


_KEEP_TAGS = ("NN", "NR", "NP", "VV", "VA", "XR", "SL", "SN", "SH", "MAG")


@lru_cache(maxsize=1)
def _kiwi():
    from kiwipiepy import Kiwi

    return Kiwi()


def tokenize(text: str) -> list[str]:
    return [t.form.lower() for t in _kiwi().tokenize(text) if t.tag.startswith(_KEEP_TAGS)]


def build_bm25(docs: list[Document], *, use_context: bool):
    from rank_bm25 import BM25Okapi

    return BM25Okapi([tokenize(index_text(d, use_context)) for d in docs])


def sparse_score(q: Sparse, d: Sparse) -> float:
    if len(q) > len(d):
        q, d = d, q
    return sum(w * d[t] for t, w in q.items() if t in d)


def min_max(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)


def ranks_of(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=int)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def fuse(
    dense: np.ndarray,
    sparse: np.ndarray,
    *,
    method: str = "minmax",
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    rrf_k: int = 60,
) -> np.ndarray:
    if method == "minmax":
        return dense_weight * min_max(dense) + sparse_weight * min_max(sparse)
    if method == "rrf":
        return dense_weight / (rrf_k + ranks_of(dense)) + sparse_weight / (rrf_k + ranks_of(sparse))
    raise ValueError(f"fusion.method는 minmax 또는 rrf여야 합니다: {method!r}")


def index_text(doc: Document, use_context: bool = True) -> str:
    if not use_context:
        return doc.page_content
    m = doc.metadata
    head = " | ".join(str(x) for x in (m.get("doc_title"), m.get("section_title"), f"p{m.get('page')}") if x)
    return "\n".join(x for x in (f"[{head}]", m.get("context") or "", doc.page_content) if x)


_POINTS: dict[tuple[str, tuple[str, ...]], tuple[list[Document], np.ndarray]] = {}
_SCROLL_RETRIES = 4


def _with_retry(call):
    """Qdrant가 응답 없이 연결을 끊으면 1·2·4초 뒤 다시 부른다."""
    from qdrant_client.http.exceptions import ResponseHandlingException

    for attempt in range(_SCROLL_RETRIES):
        try:
            return call()
        except ResponseHandlingException as e:
            if attempt == _SCROLL_RETRIES - 1:
                raise
            wait = 2**attempt
            print(f"[hybrid] Qdrant 연결 끊김, {wait}초 뒤 재시도 ({attempt + 1}/{_SCROLL_RETRIES - 1}): {e}")
            time.sleep(wait)


def _scroll(client, **kwargs):
    return _with_retry(lambda: client.scroll(**kwargs))


def load_points(client, collection: str, levels: list[str]) -> tuple[list[Document], np.ndarray]:
    key = (collection, tuple(levels))
    if key in _POINTS:
        return _POINTS[key]
    docs, vecs = [], []
    offset = None
    while True:
        points, offset = _scroll(
            client,
            collection_name=collection,
            scroll_filter=level_filter(levels),
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        for p in points:
            payload = p.payload or {}
            vec = p.vector.get("") if isinstance(p.vector, dict) else p.vector  # 261111 이정원, 청킹 수정이후 에러나서 수정
            docs.append(Document(page_content=payload.get("page_content", ""), metadata=dict(payload.get("metadata", {}))))
            vecs.append(vec)
        if offset is None:
            break
    if not docs:
        raise RuntimeError(f"Qdrant 컬렉션 {collection!r}에서 {levels} 청크를 찾지 못했습니다.")
    mat = np.asarray(vecs, dtype=np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    _POINTS[key] = (docs, mat)
    return docs, mat


def _doc_key(doc: Document, text: str) -> str:
    cid = doc.metadata.get("chunk_id") or f"{doc.metadata.get('book_id')}-p{doc.metadata.get('page')}"
    return f"{cid}:{hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]}"


_SPARSE_FILES: dict[Path, dict[str, Sparse]] = {}


def doc_sparse_weights(
    docs: list[Document], encoder_fn, cache_path: Path, *, use_context: bool, batch_size: int
) -> list[Sparse]:
    cache = _SPARSE_FILES.get(cache_path)
    if cache is None:
        cache = {}
        if cache_path.exists():
            with open(cache_path, encoding="utf-8") as f:
                cache = json.load(f)
        _SPARSE_FILES[cache_path] = cache
    texts = [index_text(d, use_context) for d in docs]
    keys = [_doc_key(d, t) for d, t in zip(docs, texts, strict=True)]
    todo = [i for i, k in enumerate(keys) if k not in cache]
    if todo:
        print(f"[hybrid] sparse 가중치 계산: {len(todo)}개 청크 (캐시 {len(keys) - len(todo)}개)")
        _, weights = encoder_fn().encode([texts[i] for i in todo], batch_size=batch_size)
        for i, w in zip(todo, weights, strict=True):
            cache[keys[i]] = w
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        keep = set(keys)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in cache.items() if k in keep}, f, ensure_ascii=False)
    return [cache[k] for k in keys]


class HybridRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    docs: list[Document]
    dense_matrix: Any
    score_query: Any
    top_k: int = 5
    method: str = "minmax"
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    rrf_k: int = 60

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        q_dense, sparse = self.score_query(query)
        dense = self.dense_matrix @ np.asarray(q_dense, dtype=np.float32)
        sparse = np.asarray(sparse, dtype=np.float32)
        fused = fuse(
            dense,
            sparse,
            method=self.method,
            dense_weight=self.dense_weight,
            sparse_weight=self.sparse_weight,
            rrf_k=self.rrf_k,
        )
        d_rank, s_rank = ranks_of(dense), ranks_of(sparse)
        out = []
        for i in np.argsort(-fused, kind="stable")[: self.top_k]:
            doc = self.docs[i]
            meta = {
                **doc.metadata,
                "hybrid_score": round(float(fused[i]), 6),
                "dense_score": round(float(dense[i]), 6),
                "sparse_score": round(float(sparse[i]), 6),
                "dense_rank": int(d_rank[i]),
                "sparse_rank": int(s_rank[i]),
            }
            out.append(Document(page_content=doc.page_content, metadata=meta))
        return out


def _candidate_scores(hits: list[dict], key: str) -> np.ndarray:
    """한쪽 검색에만 나온 후보는 다른 쪽 점수를 꼴찌보다 조금 낮게 채운다 (min-max에서 0, 순위는 맨 뒤)."""
    vals = [h[key] for h in hits if h[key] is not None]
    if not vals:
        return np.zeros(len(hits), dtype=np.float32)
    floor = min(vals) - ((max(vals) - min(vals)) or 1.0) * 1e-3
    return np.asarray([floor if h[key] is None else h[key] for h in hits], dtype=np.float32)


class QdrantHybridRetriever(BaseRetriever):
    """dense·sparse 후보를 Qdrant에서 각각 받아, 후보 안에서만 점수를 합친다 (컬렉션 전체를 내려받지 않음)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: Any
    collection: str
    encode_query: Any
    sparse_kind: str = "m3"
    levels: list[str]
    candidate_k: int = 100
    top_k: int = 5
    method: str = "minmax"
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    rrf_k: int = 60

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        from rag_service.storage import qdrant_store

        q_dense, q_sparse = self.encode_query(query)
        hits = _with_retry(
            lambda: qdrant_store.search_dense_sparse(
                self.client,
                self.collection,
                [float(x) for x in q_dense],
                q_sparse,
                kind=self.sparse_kind,
                levels=self.levels,
                limit=max(self.candidate_k, self.top_k),
            )
        )
        if not hits:
            return []
        dense, sparse = _candidate_scores(hits, "dense_score"), _candidate_scores(hits, "sparse_score")
        fused = fuse(
            dense,
            sparse,
            method=self.method,
            dense_weight=self.dense_weight,
            sparse_weight=self.sparse_weight,
            rrf_k=self.rrf_k,
        )
        d_rank, s_rank = ranks_of(dense), ranks_of(sparse)
        out = []
        for i in np.argsort(-fused, kind="stable")[: self.top_k]:
            payload = hits[i]["payload"]
            meta = {
                **dict(payload.get("metadata", {})),
                "hybrid_score": round(float(fused[i]), 6),
                "dense_score": round(float(dense[i]), 6),
                "sparse_score": round(float(sparse[i]), 6),
                "dense_rank": int(d_rank[i]),
                "sparse_rank": int(s_rank[i]),
            }
            out.append(Document(page_content=payload.get("page_content", ""), metadata=meta))
        return out


def _sparse_in_qdrant(client, name: str, kind: str) -> bool:
    from rag_service.storage import qdrant_store

    try:
        return qdrant_store.has_sparse(client, name, [kind])
    except Exception:  # noqa: BLE001 — 확인 실패면 기존 방식으로
        return False


def _query_encoder(profile: dict[str, str], emb: dict[str, Any], kind: str):
    """질문 → (dense, sparse). 색인 때와 같은 모델·규칙 (m3: BGEM3Encoder, bm25: qdrant_store.bm25_query)."""
    if kind == "m3":

        def encode(q: str):
            dense, sparse = _encoder(profile["embedding"], emb["device"], emb.get("max_length", 1024)).encode([q])
            return dense[0], sparse[0]

        return encode

    from rag_service.storage.qdrant_store import bm25_query

    embedder = _embeddings(profile["embedding"], emb["device"], True, profile["query_prefix"])
    return lambda q: (embedder.embed_query(q), bm25_query(q))


def build_hybrid_retriever(
    *,
    domain: str | None = None,
    difficulty: str | None = None,
    top_k: int | None = None,
    levels: list[str] | None = None,
    profile: str | None = None,
    config_path: Path = CONFIG_PATH,
) -> BaseRetriever:
    cfg = load_config(config_path)
    levels = levels or cfg["search"]["levels"]
    domain, difficulty = resolve_scope(cfg, domain, difficulty)
    name = collection_name(cfg, domain=domain, difficulty=difficulty, profile=profile)
    profile = model_profile(profile or difficulty)
    emb = cfg["embedding"]
    sp = cfg["sparse"]
    fusion = cfg["fusion"]
    kind = "m3" if profile["sparse"] == "bge-m3" else "bm25"
    client = _connect_client(name)

    if sp.get("source", "qdrant") == "qdrant":
        if _sparse_in_qdrant(client, name, kind):
            return QdrantHybridRetriever(
                client=client,
                collection=name,
                encode_query=_query_encoder(profile, emb, kind),
                sparse_kind=kind,
                levels=list(levels),
                candidate_k=sp.get("candidate_k", 100),
                top_k=top_k or cfg["search"]["top_k"],
                method=fusion.get("method", "minmax"),
                dense_weight=fusion["dense_weight"],
                sparse_weight=fusion["sparse_weight"],
                rrf_k=fusion.get("rrf_k", 60),
            )
        print(f"[hybrid] {name}에 sparse 벡터 {kind!r}가 없어 청크를 모두 받아 직접 계산합니다 (sparse.source: local)")

    docs, dense_matrix = load_points(client, name, levels)
    use_context = sp.get("use_context", True)

    if profile["sparse"] == "bge-m3":

        def encoder() -> BGEM3Encoder:
            return _encoder(profile["embedding"], emb["device"], emb.get("max_length", 1024))

        doc_sparse = doc_sparse_weights(
            docs,
            encoder,
            REPO_ROOT / sp["cache_dir"] / f"{name}_{'-'.join(levels)}.json",
            use_context=use_context,
            batch_size=emb.get("batch_size", 16),
        )

        def score_query(q: str):
            dense, sparse = encoder().encode([q])
            return dense[0], [sparse_score(sparse[0], d) for d in doc_sparse]

    else:
        bm25 = build_bm25(docs, use_context=use_context)
        embedder = _embeddings(profile["embedding"], emb["device"], True, profile["query_prefix"])

        def score_query(q: str):
            return embedder.embed_query(q), bm25.get_scores(tokenize(q))

    return HybridRetriever(
        docs=docs,
        dense_matrix=dense_matrix,
        score_query=score_query,
        top_k=top_k or cfg["search"]["top_k"],
        method=fusion.get("method", "minmax"),
        dense_weight=fusion["dense_weight"],
        sparse_weight=fusion["sparse_weight"],
        rrf_k=fusion.get("rrf_k", 60),
    )


def _main() -> None:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(description="retrieval_a Hybrid 검색 결과 확인")
    parser.add_argument("question")
    parser.add_argument("--domain", default=None)
    parser.add_argument("--difficulty", default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--profile", default=None, help="models.yaml의 모델 설정 이름 (기본: 난이도 이름)")
    args = parser.parse_args()

    retriever = build_hybrid_retriever(
        domain=args.domain, difficulty=args.difficulty, top_k=args.k, profile=args.profile
    )
    for i, doc in enumerate(retriever.invoke(args.question), start=1):
        m = doc.metadata
        print(
            f"[S{i}] {m.get('book_id')}.pdf {m.get('page')}쪽 ({m.get('level')}, {m.get('chunk_id')}) "
            f"hybrid={m['hybrid_score']:.3f} dense={m['dense_score']:.3f}(#{m['dense_rank']}) "
            f"sparse={m['sparse_score']:.3f}(#{m['sparse_rank']})"
        )
        print("    " + doc.page_content[:200].replace("\n", " "))


if __name__ == "__main__":
    _main()
