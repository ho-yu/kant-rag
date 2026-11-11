"""Qdrant 공통 기능 (읽기 쪽). 검색·생성 코드와 청킹(index_sync)이 같이 쓴다.

- collection_name(cfg, 난이도, 주제) → 예) kant_shipping_basic_bge-m3
- get_client(cfg): .env의 QDRANT_URL·QDRANT_API_KEY로 접속 (비어 있으면 로컬 폴더 index.local_path)
- Embedder: bge-m3 dense 임베딩 (정규화 → cosine, 기본 GPU fp16, 없으면 CPU)
- level_filter(["para", "table"]): 검색할 레벨만 고르는 필터
- get_pages(컬렉션, ["SB11-p5", ...]): page 청크(쪽 전체)를 chunk_id로 꺼낸다 (page 확장용)
- SparseEncoder: bge-m3 sparse(단어 가중치) 벡터. 청크를 올릴 때와 질문을 검색할 때 같은 걸 쓴다
- bm25_tokens / bm25_query: BM25 토큰 (번호·코드는 통째로 한 단어 + Kiwi 형태소: 팀 hybrid.py 규칙 + W_SERIAL)
- search_dense_sparse(..., kind="m3"|"bm25"): Qdrant에 dense·sparse로 각각 물어서 후보와 점수를 받는다

컬렉션 벡터 구성
    dense : 이름 없는 기본 벡터 (1024차원, cosine)  ← 기존과 같음. naive 검색은 그대로 동작
    m3    : bge-m3 sparse (config index.sparse에 m3가 있을 때)
    bm25  : BM25 (config index.sparse에 bm25가 있을 때. Qdrant가 IDF를 계산: modifier=IDF)
    → hybrid 검색이 컬렉션 전체를 내려받지 않고 질문마다 Qdrant에 바로 물어볼 수 있다.

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
import re
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


# ───────────────────────── sparse 벡터 (hybrid 검색용) ─────────────────────────
SPARSE_KINDS = ("m3", "bm25")   # 컬렉션 안의 sparse 벡터 이름 (dense는 이름 없는 기본 벡터 "")
#   m3  : bge-m3 sparse(단어 가중치). 토큰 ID = bge-m3 토크나이저 번호
#   bm25: Kiwi 형태소 BM25. 토큰 ID = 형태소 해시(bm25_token_id). IDF는 Qdrant가 검색 때 계산(modifier=IDF)


def sparse_kinds(cfg: dict) -> list[str]:
    """config index.sparse → 올릴 sparse 종류 목록. 예) [m3, bm25] / [] (끔) / "m3,bm25" / true(=m3)"""
    v = cfg["index"].get("sparse") or []
    if v is True:
        v = ["m3"]
    if isinstance(v, str):
        v = [] if v.strip().lower() in ("", "none", "off") else [x.strip() for x in v.split(",") if x.strip()]
    bad = set(v) - set(SPARSE_KINDS)
    if bad:
        raise ValueError(f"index.sparse는 {list(SPARSE_KINDS)} 중에서 고르세요: {sorted(bad)}")
    return [k for k in SPARSE_KINDS if k in v]


def sparse_params(kind: str):
    """컬렉션을 만들 때 sparse 벡터 설정. bm25는 Qdrant가 IDF를 곱한다 (문서 쪽엔 TF 부분만 저장)."""
    from qdrant_client import models

    if kind == "bm25":
        return models.SparseVectorParams(modifier=models.Modifier.IDF)
    return models.SparseVectorParams()


def sparse_text(payload: dict) -> str:
    """sparse를 계산할 글 = [문서 제목 | 절 제목 | p쪽] + 맥락 + 원문. (m3·bm25 공통)
    retrieval_a/hybrid.py의 index_text(use_context=True)와 같은 규칙.
    advanced의 예약번호(HG2030-00362 등)는 절 제목에만 있어서 절 제목이 꼭 들어가야 한다."""
    m = payload.get("metadata", {})
    head = " | ".join(str(x) for x in (m.get("doc_title"), m.get("section_title"), f"p{m.get('page')}") if x)
    return "\n".join(x for x in (f"[{head}]", m.get("context") or "", payload.get("page_content", "")) if x)


class SparseEncoder:
    """bge-m3 sparse(단어 가중치) 벡터를 만든다. 청크(색인)와 질문(검색) 모두 이걸로 만들어야 점수가 맞는다.

    계산: 토큰마다 relu(sparse_linear(마지막 hidden)) → 같은 토큰은 큰 값, 특수 토큰은 뺀다.
    (retrieval_a/hybrid.py의 BGEM3Encoder와 같은 계산. 기본 GPU fp16)
    결과: [{토큰 ID: 가중치}, ...]  → to_sparse_vector()로 Qdrant 형식으로 바꾼다.
    검색 예) q = SparseEncoder().encode(["질문"])[0] → search_dense_sparse(..., q, kind="m3")
    """

    def __init__(self, model: str = "BAAI/bge-m3", batch_size: int = 16, max_length: int = 1024,
                 device: str = "cuda"):
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = resolve_device(device)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model, dtype=dtype).to(self.device).eval()
        self.head = torch.nn.Linear(self.model.config.hidden_size, 1)
        self.head.load_state_dict(torch.load(hf_hub_download(model, "sparse_linear.pt"), map_location="cpu"))
        self.head = self.head.to(self.device, dtype=dtype).eval()
        self.skip = {self.tokenizer.cls_token_id, self.tokenizer.eos_token_id,
                     self.tokenizer.pad_token_id, self.tokenizer.unk_token_id}
        self.batch_size, self.max_length = batch_size, max_length

    def encode(self, texts: list[str], progress: bool = False) -> list[dict[int, float]]:
        import time

        torch = self.torch
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))   # 길이순으로 묶어 패딩 낭비를 줄인다
        out: list[dict[int, float]] = [{}] * len(texts)
        t0, steps = time.time(), max(1, (len(texts) + self.batch_size - 1) // self.batch_size)
        with torch.inference_mode():
            for step, i in enumerate(range(0, len(order), self.batch_size), start=1):
                idx = order[i:i + self.batch_size]
                batch = self.tokenizer([texts[j] for j in idx], padding=True, truncation=True,
                                       max_length=self.max_length, return_tensors="pt").to(self.device)
                hidden = self.model(**batch).last_hidden_state
                weights = torch.relu(self.head(hidden)).squeeze(-1).float().cpu().numpy()
                for j, ids, w in zip(idx, batch["input_ids"].cpu().numpy(), weights, strict=True):
                    d: dict[int, float] = {}
                    for tid, v in zip(ids.tolist(), w.tolist(), strict=True):
                        if tid not in self.skip and v > 0 and v > d.get(tid, 0.0):
                            d[tid] = v
                    out[j] = d
                if progress and (step % 50 == 0 or step == steps):
                    sec = time.time() - t0
                    log.info("m3 sparse %d/%d (%.0f%%) 경과 %.0f초, 남은 시간 약 %.0f초",
                             min(i + self.batch_size, len(texts)), len(texts), 100 * step / steps,
                             sec, sec / step * (steps - step))
        return out


# BM25 토큰 규칙
#  1) 번호·코드는 통째로 한 단어: 영문과 숫자가 함께 있고 하이픈(-)으로 이어진 덩어리
#     HL-40GP-00362 → hl-40gp-00362, N1-20-F09118 → n1-20-f09118, 20GP → 20gp
#     슬래시(/)는 나눔 (S2/40GP → s2, 40gp). 쪽 번호(p116)·금액(USD12, 20USD)·무게(18100kg)는 묶지 않음
#  2) B/L은 한 단어
#  3) 나머지 문장은 Kiwi 형태소: retrieval_a/hybrid.py의 _KEEP_TAGS + W_SERIAL(일련번호·날짜. 예: 2029-09-11)
#  4) 모두 소문자
BM25_KEEP_TAGS = ("NN", "NR", "NP", "VV", "VA", "XR", "SL", "SN", "SH", "MAG", "W_SERIAL")
_CODE = re.compile(r"(?<![A-Za-z0-9-])(?=[A-Za-z0-9-]*[A-Za-z])(?=[A-Za-z0-9-]*\d)"
                   r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*(?![A-Za-z0-9])")
_NOT_CODE = re.compile(r"(?i)^(p\d+|usd\d+|\d+(kg|usd|cbm|ton|t))$")   # 쪽 번호·금액·무게
_WHOLE_WORDS = re.compile(r"(?i)(?<![A-Za-z0-9])b/l(?![A-Za-z0-9])")


def _split_codes(text: str) -> tuple[list[str], str]:
    """글 → (통째로 묶은 번호·코드 토큰들, 나머지 글). 묶은 자리는 공백으로 비운다."""
    codes: list[str] = []

    def take(m: re.Match) -> str:
        if _NOT_CODE.match(m.group()):
            return m.group()
        codes.append(m.group().lower())
        return " "

    rest = _CODE.sub(take, text)

    def take_word(m: re.Match) -> str:
        codes.append(m.group().lower())
        return " "

    return codes, _WHOLE_WORDS.sub(take_word, rest)


def _kiwi_tokens(toks) -> list[str]:
    return [t.form.lower() for t in toks if t.tag.startswith(BM25_KEEP_TAGS)]


@lru_cache(maxsize=1)
def _kiwi():
    from kiwipiepy import Kiwi

    return Kiwi(num_workers=max(1, (os.cpu_count() or 2) - 1))


def bm25_tokens(text: str) -> list[str]:
    """BM25 토큰 (위 규칙). 색인과 질문 모두 이 함수를 써야 점수가 맞는다."""
    codes, rest = _split_codes(text)
    return codes + _kiwi_tokens(_kiwi().tokenize(rest))


def bm25_tokens_many(texts: list[str]) -> list[list[str]]:
    """여러 글을 한 번에 (Kiwi 멀티스레드). 결과는 bm25_tokens()를 하나씩 부른 것과 같다."""
    split = [_split_codes(t) for t in texts]
    return [codes + _kiwi_tokens(toks)
            for (codes, _), toks in zip(split, _kiwi().tokenize([r for _, r in split]), strict=True)]


def bm25_token_id(token: str) -> int:
    """형태소 → Qdrant sparse 번호 (blake2b 32비트 해시). 단어장 파일 없이 색인·질문이 같은 번호를 쓴다."""
    import hashlib

    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest(), "little")


def bm25_query(text: str) -> dict[int, float]:
    """질문 → BM25 질의 벡터 {토큰 번호: 질문 속 등장 횟수}. Qdrant가 IDF를 곱해 점수를 낸다.
    검색 예) search_dense_sparse(client, 컬렉션, dense, bm25_query("질문"), kind="bm25")"""
    out: dict[int, float] = {}
    for t in bm25_tokens(text):
        tid = bm25_token_id(t)
        out[tid] = out.get(tid, 0.0) + 1.0
    return out


class BM25Encoder:
    """청크 쪽 BM25 값 = tf·(k1+1) / (tf + k1·(1 − b + b·문서길이/평균길이)). IDF는 Qdrant가 곱한다.
    평균 길이는 레벨(page·para·table)마다 따로 (page가 훨씬 길어서 섞으면 para 점수가 틀어진다)."""

    def __init__(self, avgdl: dict[str, float], k1: float = 1.5, b: float = 0.75):
        self.avgdl, self.k1, self.b = avgdl, k1, b

    @staticmethod
    def fit_avgdl(texts: list[str], levels: list[str], sample: int = 3000) -> dict[str, float]:
        """레벨별 평균 토큰 수. 레벨마다 최대 sample개만 고르게 뽑아 잰다 (전체를 다 토큰화하지 않게)."""
        by_level: dict[str, list[str]] = {}
        for t, lv in zip(texts, levels, strict=True):
            by_level.setdefault(lv, []).append(t)
        out = {}
        for lv, ts in by_level.items():
            step = max(1, len(ts) // sample)
            picked = ts[::step][:sample]
            out[lv] = max(1.0, sum(len(x) for x in bm25_tokens_many(picked)) / len(picked))
        return out

    def encode(self, texts: list[str], levels: list[str], progress: bool = False) -> list[dict[int, float]]:
        import time

        t0, out = time.time(), []
        fallback = sum(self.avgdl.values()) / max(1, len(self.avgdl))
        for n, (toks, lv) in enumerate(zip(bm25_tokens_many(texts), levels, strict=True), start=1):
            tf: dict[int, int] = {}
            for t in toks:
                tid = bm25_token_id(t)
                tf[tid] = tf.get(tid, 0) + 1
            norm = self.k1 * (1 - self.b + self.b * len(toks) / self.avgdl.get(lv, fallback))
            out.append({tid: c * (self.k1 + 1) / (c + norm) for tid, c in tf.items()})
            if progress and (n % 5000 == 0 or n == len(texts)):
                log.info("bm25 %d/%d 경과 %.0f초", n, len(texts), time.time() - t0)
        return out


def to_sparse_vector(weights: dict):
    """{토큰 번호: 가중치} → Qdrant SparseVector. (키가 문자열이어도 된다: 기존 sparse 캐시 형식)"""
    from qdrant_client import models

    if isinstance(weights, models.SparseVector):
        return weights
    items = sorted((int(k), float(v)) for k, v in weights.items())
    return models.SparseVector(indices=[k for k, _ in items], values=[v for _, v in items])


def collection_sparse_kinds(client, name: str) -> set[str]:
    """컬렉션에 설정된 sparse 벡터 이름들."""
    return set(client.get_collection(name).config.params.sparse_vectors or {})


def has_sparse(client, name: str, kinds=SPARSE_KINDS) -> bool:
    """컬렉션에 kinds sparse 벡터가 모두 있는지."""
    return set(kinds) <= collection_sparse_kinds(client, name)


def search_dense_sparse(client, collection: str, dense: list[float] | None, sparse: dict | None, *,
                        kind: str = "m3", levels: list[str] | None = None, limit: int = 50) -> list[dict]:
    """hybrid 검색용 후보: Qdrant에 dense로 limit개, sparse(kind)로 limit개를 각각 물어서 합친다.

    sparse: m3면 SparseEncoder().encode([질문])[0], bm25면 bm25_query(질문)
    반환: [{"payload": {...}, "dense_score": float | None, "sparse_score": float | None}, ...]
          한쪽 결과에만 있는 포인트는 다른 쪽 점수가 None (보통 0점·꼴찌로 보고 합친다).
    점수 합치는 방법(min-max 가중합, RRF 등)은 검색 코드가 정한다. 컬렉션 전체를 내려받지 않는다.
    """
    flt = level_filter(levels) if levels else None
    hits: dict = {}
    queries = []
    if dense is not None:
        queries.append(("dense_score", dense, None))
    if sparse:
        queries.append(("sparse_score", to_sparse_vector(sparse), kind))
    for key, query, using in queries:
        res = client.query_points(collection, query=query, using=using, query_filter=flt, limit=limit,
                                  with_payload=True, with_vectors=False)
        for p in res.points:
            h = hits.setdefault(p.id, {"payload": p.payload or {}, "dense_score": None, "sparse_score": None})
            h[key] = p.score
    return list(hits.values())


def level_filter(levels: list[str], negate: bool = False):
    """검색할 레벨만 고르는 Qdrant 필터. 예) level_filter(["para", "table"])"""
    from qdrant_client import models

    cond = models.FieldCondition(key="metadata.level", match=models.MatchAny(any=list(levels)))
    return models.Filter(must_not=[cond]) if negate else models.Filter(must=[cond])
