# 청킹·색인·Qdrant 사용 가이드 (팀 공유용)

> 청킹 담당: 이정원 / 자세한 규칙은 `docs/chunking_spec.md`
> 청킹 담당이 아니라면 **0번(처음 한 번)**, **1번(Qdrant 접속)**, **3번(검색 코드에서 쓰는 법)** 만 보면 된다.

## 0. 처음 받았을 때 (한 번만)

1. `git pull` 후 `uv sync`
2. `.env` 확인: `OPENAI_API_KEY`, `OPENAI_MODEL=gpt-6-luna`, `QDRANT_URL`, `QDRANT_API_KEY` (Qdrant 값은 1번 참고)
3. 청크 파일 만들기 — **청크 jsonl은 깃에 없어서(로컬 전용) 각자 한 번 만들어야 한다.**

```
uv run python -m rag_service.ingestion.menu
```

`1. 청킹 실행` → `1. shipping` → 난이도 선택 → **`2`를 눌러 Qdrant 동기화를 끈다** → `s` → `y`

- 깃에 커밋된 `context_cache.jsonl`이 있으면 OpenAI를 다시 부르지 않고 담당자와 **똑같은** 청크가 만들어진다 (키 없어도 됨).
- Qdrant는 팀이 같은 클러스터를 쓰므로 이미 올라가 있다. **팀원은 동기화를 끄고 실행한다** (4번 주의 참고).
- 결과 위치: `artifacts/corpus/{난이도}/shipping/SB01.jsonl …` (Basic 약 25초, Intermediate 약 3분, Advanced 약 10분)

## 1. Qdrant 접속 (팀원)

### 1-1. 키 받기

- Qdrant 클러스터는 팀이 **하나를 같이** 쓴다. 각자 클러스터를 만들 필요 없다.
- `QDRANT_URL`과 `QDRANT_API_KEY`는 담당자에게 **개인 메시지로** 받는다. (단톡방·깃·노션에 올리지 않는다)
- 키는 사람마다 따로 발급한다. 한 사람 키가 새면 그 키만 지우면 된다.
- 쓰기 권한이 있는 키를 받았더라도 **Qdrant 동기화(메뉴 `1`에서 동기화 켜기, `--index-only`, `--reindex`)는 돌리지 않는다.** 4번 참고.

### 1-2. `.env`에 넣기

```
QDRANT_URL=https://xxxxxxxx.xxx.cloud.qdrant.io:6333
QDRANT_API_KEY=받은 키
```

- `QDRANT_URL`을 비워 두면 Qdrant Cloud 대신 내 PC의 `artifacts/qdrant_local/`을 쓴다 (비어 있으므로 검색 결과가 0개).

### 1-3. 연결 확인

```python
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()
c = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
print([x.name for x in c.get_collections().collections])      # ['kant_shipping_basic_bge-m3', …]

name = "kant_shipping_basic_bge-m3"
print(c.count(name).count)                                     # Basic 전체 1960
for lv in ("page", "para", "table"):
    f = models.Filter(must=[models.FieldCondition(key="metadata.level", match=models.MatchValue(value=lv))])
    print(lv, c.count(name, count_filter=f).count)
```

- 브라우저로 보고 싶으면 `{QDRANT_URL}/dashboard`에 접속해 API 키를 넣는다 (컬렉션·포인트 내용 확인).
- `401/403` → 키가 틀렸거나 만료됨. `Not found: Collection` → 그 난이도는 아직 안 올라감. 둘 다 담당자에게 알린다.

### 1-4. 컬렉션에 무엇이 있나

| 항목 | 내용 |
|---|---|
| 컬렉션 이름 | `kant_shipping_{basic/intermediate/advanced}_bge-m3` (난이도마다 하나, 항상 최신 PDF 기준) |
| 들어 있는 레벨 | `page`(쪽 전체), `para`(문단), `table`(표 전체). `row`(표 한 줄)는 무료 용량 때문에 jsonl에만 있음 |
| 벡터 | bge-m3 dense 1024차원, cosine, 정규화됨 |
| payload | LangChain 형식: `page_content`(원문) + `metadata`(chunk_id, parent_id, level, book_id, page, doc_title, section_title, table_title, context …) + `content_hash` |
| 인덱스 걸린 필드 | `metadata.level`, `metadata.book_id`, `metadata.page` → 필터 검색 가능 |
| 포인트 ID | `uuid5(chunk_id)` — 같은 청크는 언제나 같은 ID |

- 현재 올라간 것: **Basic** (1,960개). Intermediate·Advanced는 담당자가 올리면 공지.

## 2. 언제 무엇을 쓰나

| 상황 | 할 일 | 누가 |
|---|---|---|
| 처음 받았거나 내 PC에 청크 jsonl이 없음 | 메뉴 `1` (Qdrant 동기화 **끔**) | 모두 |
| Qdrant 접속 확인 | 1-3 코드 | 모두 |
| PDF가 추가·변경·삭제됨 | 메뉴 `1` (동기화 켬) → `context_cache.jsonl` 커밋 | 청킹 담당 |
| 청킹 규칙·설정을 바꿈 | 코드·`config/chunking.yaml` 수정 → `uv run pytest` → 근거 보존 검사 → 메뉴 `1` | 청킹 담당 |
| 잘못 바뀐 문서를 예전으로 되돌림 | 메뉴 `2` (문서·버전 선택, 차이 미리보기 후 실행) | 청킹 담당 |
| Qdrant가 비었음 (무료 클러스터 삭제, 업로드 중 끊김) | `uv run python -m rag_service.ingestion.build_corpus --difficulty basic --index-only` | 청킹 담당 |
| 청크 크기 등을 바꾼 뒤 근거가 남는지 확인 | `uv run python evaluation/retrieval/check_chunk_coverage.py --difficulty basic` | 청킹 담당 |
| Dense 검색 (Naive RAG) | 3-1 코드 | 검색 담당 |
| BM25 (하이브리드) | 3-2 코드 | 검색 담당 |
| Hybrid (Qdrant에 dense + sparse(m3·bm25) 바로 질의) | 3-3 코드 | 검색 담당 |
| 답변 생성에 부모 쪽 원문 넣기, 출처 표기 | 3-4 코드 | 생성 담당 |
| 화면(클라이언트)에서 RAG로 답 받기 | `uv run streamlit run client/client.py` → 검색 방식 라디오 선택 | 모두 |
| 어떤 코드든 고친 뒤 기존 기능 확인 | `uv run pytest` (전부 통과해야 함) | 고친 사람 |

## 3. 검색·생성 코드에서 쓰는 법

### 3-1. Dense 검색 (Qdrant + LangChain)

```python
import os
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from rag_service.storage.qdrant_store import level_filter

load_dotenv()
emb = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", encode_kwargs={"normalize_embeddings": True})
store = QdrantVectorStore.from_existing_collection(
    collection_name="kant_shipping_basic_bge-m3",          # kant_shipping_{난이도}_bge-m3
    embedding=emb, url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

docs = store.similarity_search("질문", k=5, filter=level_filter(["para", "table"]))   # 실험 B
# 실험 A(쪽 단위)는 level_filter(["page"])
docs[0].page_content          # 원문
docs[0].metadata["book_id"], docs[0].metadata["page"], docs[0].metadata["parent_id"]
```

- 질문 임베딩도 bge-m3, `normalize_embeddings=True`, 접두어 없음.
- **반드시 `level_filter`를 넣는다.** 안 넣으면 쪽(page)과 문단(para)이 같이 나와 같은 내용이 중복된다.
- GPU가 있으면 `HuggingFaceEmbeddings(..., model_kwargs={"device": "cuda"})`.

### 3-2. BM25 (하이브리드용)

```python
from rag_service.storage.corpus import load_chunks

chunks = list(load_chunks("basic", levels=["para", "table"]))   # Qdrant와 같은 레벨
texts = [c.embedding_text() for c in chunks]                    # 검색용 문자열 (Qdrant 임베딩과 같은 입력)
# texts를 Kiwi로 토큰화해서 rank_bm25에 넣는다. 결과 i번 → chunks[i]
# LangChain Document로 바꿀 때: Document(page_content=c.text, metadata=c.payload()["metadata"])
```

- BM25는 Qdrant가 아니라 **내 PC의 jsonl**을 읽는다. 그래서 0번(청크 파일 만들기)이 먼저 필요하다.
- Qdrant의 청크와 jsonl의 청크는 `chunk_id`가 같다 → RRF로 합칠 때 `chunk_id`로 맞춘다.

### 3-3. Hybrid 검색 (Qdrant에 dense + sparse로 바로 질의)

컬렉션의 포인트(page·para·table 전부)에는 벡터가 세 개 들어 있다. 컬렉션 전체를 내려받거나 sparse 캐시를 만들 필요 없이,
질문마다 Qdrant에 상위 후보만 물어보면 된다.

| 벡터 이름 | 내용 | 질문 쪽 만드는 법 |
|---|---|---|
| `""` (기본) | bge-m3 dense 1024차원 | 기존과 같음 (naive 검색 그대로) |
| `"m3"` | bge-m3 sparse (토큰 번호 = bge-m3 토크나이저 번호) | `qs.SparseEncoder().encode([q])[0]` 또는 hybrid.py `BGEM3Encoder.encode([q])[1][0]` (같은 계산) |
| `"bm25"` | Kiwi BM25 (토큰 번호 = 형태소 해시, IDF는 Qdrant가 계산) | `qs.bm25_query(q)` |

```python
from rag_service.storage import qdrant_store as qs

name = "kant_shipping_advanced_bge-m3"
# dense + bm25
hits = qs.search_dense_sparse(client, name, dense_vec, qs.bm25_query(question), kind="bm25",
                              levels=["para", "table"], limit=50)
# dense + m3
hits = qs.search_dense_sparse(client, name, dense_vec, m3_weights, kind="m3",
                              levels=["para", "table"], limit=50)
# hits = [{"payload": {...}, "dense_score": 0.71 | None, "sparse_score": 3.2 | None}, ...]
# 점수 합치기(minmax 0.7/0.3, 가중 RRF 등)는 검색 코드에서. 한쪽에만 나온 후보는 다른 쪽을 0점·꼴찌로 본다.
```

- sparse 글은 `[문서 제목 | 절 제목 | p쪽] + 맥락 + 원문` (`qs.sparse_text(payload)`, 기존 hybrid의 `index_text`와 같은 규칙).
  advanced의 예약번호(`HG2030-00362`)는 **절 제목에만** 있어서 이 규칙이 중요하다.
- BM25 토큰 규칙 (`qs.bm25_tokens`). 질문도 꼭 `qs.bm25_tokens`/`qs.bm25_query`로 만든다
  (규칙이 다르면 같은 단어도 번호가 달라 점수가 안 맞는다). k1=1.5, b=0.75, 평균 길이는 레벨별.
  1. 번호·코드는 통째로 한 단어: 영문과 숫자가 함께 있고 하이픈으로 이어진 덩어리
     (`HL-40GP-00362` → `hl-40gp-00362`, `N1-20-F09118` → `n1-20-f09118`, `20GP` → `20gp`)
  2. 슬래시는 나눔 (`S2/40GP` → `s2`, `40gp`). 쪽 번호(`p116`)·금액(`USD12`)·무게(`18100kg`)는 묶지 않음. `B/L`은 한 단어
  3. 나머지는 Kiwi 형태소: hybrid.py `_KEEP_TAGS` + `W_SERIAL`(일련번호·날짜. 예: `2029-09-11`), 모두 소문자
- BM25 점수는 rank_bm25(BM25Okapi)와 IDF 식이 조금 달라 값은 다르지만 순위는 거의 같다.
- sparse 종류는 `config/chunking.yaml`의 `index.sparse: [m3, bm25]` (청킹 담당만). 메뉴 4·5번으로 켜고 끈다.

### 3-4. 부모 쪽 원문과 출처 (답변 생성)

```python
from rag_service.storage.corpus import PageStore

pages = PageStore("basic")
m = docs[0].metadata
parent = pages.get(f"{m['book_id']}-p{m['page']}")   # 그 청크가 속한 쪽 전체
parent.text                                          # LLM에 넘길 부모 쪽 원문
source = f"{m['book_id']}.pdf / {m['page']}쪽"        # 출처는 metadata로 붙인다 (LLM이 만들게 하지 않음)
```

- 답변 생성에는 `page_content`(원문)나 부모 쪽만 쓴다. `metadata["context"]`(LLM이 쓴 맥락)는 검색용이라 답변 근거로 쓰지 않는다.
- Hit@k·Recall@k는 같은 쪽 청크가 여러 개 나올 수 있으니 `(book_id, page)` 기준으로 중복을 지우고 센다.

## 4. 하면 안 되는 것

- **평가 JSON(정답·`quote`·채점 기준)을 색인·검색어·필터·프롬프트에 넣지 않는다.** (`check_chunk_coverage.py`만 예외: 청킹 결과 검사용)
- **팀원 여럿이 Qdrant 동기화를 켜고 돌리지 않는다.** 동기화는 Qdrant를 "내 PC의 jsonl"과 똑같이 맞춘다. 내 jsonl이 담당자와 다르면(예: 맥락 없이 청킹) 공유 클러스터를 덮어쓴다. 동기화는 청킹 담당만.
- **공유 컬렉션(`kant_shipping_*`)에 직접 쓰거나 지우지 않는다** (`upsert`, `delete`, `recreate_collection`, `from_documents` 등). 실험용으로 따로 올리고 싶으면 `내이름_실험명`처럼 **다른 이름의 컬렉션**을 만들고, 다 쓰면 지운다 (무료 용량 1GB를 같이 쓴다).
- Qdrant URL·API 키를 코드에 직접 쓰지 않는다. 항상 `.env`에서 읽는다.
- 청크 jsonl·`manifest.json`·`history/`는 커밋하지 않는다 (.gitignore에 있음). `context_cache.jsonl`은 커밋한다.
- `.env`(API 키)는 커밋하지 않는다.
- Qdrant 무료 클러스터는 1주 미사용 시 중지, 4주 후 삭제된다. 제출(11/13) 전까지 주기적으로 사용.

## 5. 파일 지도

| 경로 | 파일 | 역할 |
|---|---|---|
| `src/rag_service/ingestion/` **(청킹 전용)** | `menu.py` | 텍스트 메뉴 (청킹 실행, 되돌리기) |
| | `build_corpus.py` | 명령줄 실행 (메뉴가 내부에서 호출) |
| | `pdf_reader.py`, `chunker.py`, `context.py`, `writer.py` | PDF 읽기, 청크 만들기, 맥락 생성, jsonl 저장 |
| | `index_sync.py` | Qdrant 업로드 (바뀐 청크만 동기화) |
| `src/rag_service/storage/` **(공통)** | `corpus.py` | jsonl 읽기 `load_chunks`, 부모 쪽 `PageStore` ← **검색·생성 코드에서 사용** |
| | `qdrant_store.py` | 컬렉션 이름, 접속, 임베딩, `level_filter` ← **검색 코드에서 사용** |
| `src/rag_service/contracts/` | `chunk.py` | 청크 형식 (필드 정의) |
| `config/` | `chunking.yaml` | 청킹·색인·맥락 설정 |
| `client/` | `client.py` | Streamlit 화면 (검색 방식 라디오: naive / hybrid / hybrid+rerank) |
| `evaluation/retrieval/` | `check_chunk_coverage.py` | 근거 보존 검사 (평가 전용) |
| `tests/ingestion/` | `test_chunking.py`, `test_menu.py`, `test_index_sync.py` | 청킹 코드 확인 (README.md 참고) |
| `tests/storage/` | `test_qdrant_store.py` | 공통 기능 확인 (README.md 참고) |

## 6. 자주 묻는 것

- **다시 돌리면 전부 다시 해?** 아니. 바뀐 문서만 다시 청킹하고, 바뀐 청크만 임베딩한다. 바뀐 게 없으면 몇 초 만에 끝난다.
- **맥락 생성이 일부 실패했어.** 같은 명령을 다시 실행하면 실패한 문서만 다시 시도한다 (성공한 건 캐시에서).
- **Qdrant 동기화가 실패했어.** (담당자) jsonl은 이미 저장됐다. `.env`의 `QDRANT_URL`·`QDRANT_API_KEY`를 확인하고 같은 명령을 다시 실행하면 빠진 것만 이어서 올린다.
- **Qdrant 검색은 되는데 BM25가 0개야.** 내 PC에 청크 jsonl이 없는 것. 0번을 먼저 한다.
- **검색 결과가 0개야.** `.env`의 `QDRANT_URL`이 비어 있으면 로컬 빈 폴더를 본다. 1-3으로 연결과 개수를 확인한다.
- **임베딩이 너무 느려.** 윈도우 기본 PyTorch는 CPU 버전일 수 있다. Basic(약 2천 청크)은 CPU로도 괜찮지만 Advanced(약 10.7만)는 GPU가 필요하다.
- **리랭커 첫 실행이 오래 걸려.** `BAAI/bge-reranker-v2-m3`(약 2.3GB)를 처음 한 번 내려받는 시간이다.
