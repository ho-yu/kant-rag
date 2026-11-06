from __future__ import annotations

import argparse
import hashlib
import json
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
    _resolve_device,
    collection_name,
    level_filter,
    load_config,
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


def load_points(client, collection: str, levels: list[str]) -> tuple[list[Document], np.ndarray]:
    docs, vecs = [], []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            scroll_filter=level_filter(levels),
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        for p in points:
            payload = p.payload or {}
            vec = p.vector if not isinstance(p.vector, dict) else next(iter(p.vector.values()))
            docs.append(Document(page_content=payload.get("page_content", ""), metadata=dict(payload.get("metadata", {}))))
            vecs.append(vec)
        if offset is None:
            break
    if not docs:
        raise RuntimeError(f"Qdrant 컬렉션 {collection!r}에서 {levels} 청크를 찾지 못했습니다.")
    mat = np.asarray(vecs, dtype=np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return docs, mat


def _doc_key(doc: Document, text: str) -> str:
    cid = doc.metadata.get("chunk_id") or f"{doc.metadata.get('book_id')}-p{doc.metadata.get('page')}"
    return f"{cid}:{hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]}"


def doc_sparse_weights(
    docs: list[Document], encoder_fn, cache_path: Path, *, use_context: bool, batch_size: int
) -> list[Sparse]:
    cache: dict[str, Sparse] = {}
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
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
    doc_sparse: list[Sparse]
    encode_query: Any
    top_k: int = 5
    method: str = "minmax"
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    rrf_k: int = 60

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        q_dense, q_sparse = self.encode_query(query)
        dense = self.dense_matrix @ np.asarray(q_dense, dtype=np.float32)
        sparse = np.array([sparse_score(q_sparse, d) for d in self.doc_sparse], dtype=np.float32)
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


def build_hybrid_retriever(
    *,
    domain: str | None = None,
    difficulty: str | None = None,
    top_k: int | None = None,
    levels: list[str] | None = None,
    config_path: Path = CONFIG_PATH,
) -> BaseRetriever:
    cfg = load_config(config_path)
    levels = levels or cfg["search"]["levels"]
    name = collection_name(cfg, domain=domain, difficulty=difficulty)
    docs, dense_matrix = load_points(_connect_client(name), name, levels)

    emb = cfg["embedding"]

    def encoder() -> BGEM3Encoder:
        return _encoder(emb["model_name"], emb["device"], emb.get("max_length", 1024))

    sp = cfg["sparse"]
    level_tag = "-".join(levels)
    doc_sparse = doc_sparse_weights(
        docs,
        encoder,
        REPO_ROOT / sp["cache_dir"] / f"{name}_{level_tag}.json",
        use_context=sp.get("use_context", True),
        batch_size=emb.get("batch_size", 16),
    )

    def encode_query(q: str):
        dense, sparse = encoder().encode([q])
        return dense[0], sparse[0]

    fusion = cfg["fusion"]
    return HybridRetriever(
        docs=docs,
        dense_matrix=dense_matrix,
        doc_sparse=doc_sparse,
        encode_query=encode_query,
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
    args = parser.parse_args()

    retriever = build_hybrid_retriever(domain=args.domain, difficulty=args.difficulty, top_k=args.k)
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
