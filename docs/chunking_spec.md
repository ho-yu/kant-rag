# 청킹 사양 (확정안)

> 작성: 2026-11-05 / 대상: Basic 해운(shipping)부터, 이후 Intermediate·Advanced
> 제출: 2026-11-13 13:00 KST

## 1. 구조 (부모-자식)

| level | 단위 | 부모 | jsonl | Qdrant |
|---|---|---|---|---|
| `page` | 쪽 전체 | - | O | O |
| `para` | 문단 | 쪽 | O | O |
| `table` | 표 전체 | 쪽 | O | O |
| `row` | 표의 한 행 + 컬럼명 (절대 안 쪼갬) | 표 | O | **off** (Qdrant 무료 플랜 용량 제한) |
| `sent` | 문장 | 문단 | 추후 | 추후 |

- 오버랩 없음. 문단이 300토큰을 넘으면 문장 경계로만 분할 (안전장치. Basic·Intermediate에는 해당 문단 없음).
- 쪽 경계를 넘는 청크 없음 (`page` 보존).
- 임베딩: bge-m3 (dense). BM25: `rank-bm25`.
- 답변 생성 모델: gpt-6-luna.

## 2. PDF 읽기 규칙 (`pdf_reader.py`)

글자 크기는 쪽마다 본문 크기(body) 대비 상대값으로 판단한다.

| 구분 | 판단 기준 | 처리 |
|---|---|---|
| 머리글 | 쪽의 첫 제목보다 위에 있는 줄 (`SB01 \| 예약지원팀`, 문서 제목 등) | 본문에서 빼고 `page_header`로 저장 |
| 쪽 번호 | 쪽 아래 60pt 안 | 버림 |
| 제목 | body × 1.3 이상 | `section_title` |
| 표 제목 | 굵은 본문 크기 글자 + 바로 아래(40pt)에 표 | `table_title` |
| 소제목 | 굵은 본문 크기 글자 + 아래에 표 없음 | 다음 문단의 첫 줄로 붙임 |
| 표 각주 | 본문보다 작은 글자 + 표 바로 아래 | 표와 모든 row에 `주: …`로 붙임 |
| 문단 | 줄 간격이 평소 × 1.35보다 크면 새 문단 | `para` |

- 줄바꿈으로 끊긴 줄 잇기: 줄 끝/앞 공백 → 문장부호 규칙 → 같은 문서에서 더 많이 쓰인 표기(`AB` vs `A B`) → Kiwi 띄어쓰기 교정 순서로 판단.
  근거 인용 기준 줄 경계 589개 중 574개(97.5%) 일치. 틀려도 공백 차이뿐이라 내용은 그대로다.

## 3. 표 처리

- 표 전체를 청크 하나(`table`, 마크다운 표)로 저장하고, 행마다 `row` 청크를 따로 만든다.
- row 형식: `열 이름: 값 | 열 이름: 값 …` (값 안의 `/`와 헷갈리지 않게 `|`로 구분).
- 모든 row에 표 제목, 컬럼명, 각주를 같이 붙인다. 표와 row는 잘리지 않는다.
- 근거: TARGET 벤치마크에서 row 단위와 표 전체의 검색 성능은 비슷했고, 컬럼명만 넣을 때 성능이 급락해서 행 값과 표 제목이 같이 필요하다고 확인.

예 (IS01 6쪽)

```
[표] 서비스 범위와 제외 대상 — 대상별 처리 원문
접수번호: IS01-006-03 | 문의처·품명: 광진산업 산업용 패킹 | 문의 항목: 일반 건화물20GP | 범위: 부산–LA | 소관 판단: 안내 범위
```

## 4. Contextual Retrieval (`context.py`)

- 쪽 단위로 OpenAI를 1회 호출해서, 그 쪽 모든 청크의 맥락 한 줄을 JSON으로 받는다 (입력: 쪽 원문 + 문서 제목 + 머리글).
- 질문·정답·채점 기준은 절대 입력하지 않는다.
- `text`(원문)와 `context`(LLM 한 줄)를 분리 저장한다.
- 임베딩과 BM25 색인에는 `Chunk.embedding_text()` = `[문서 제목 | 섹션 | pN]` + `context` + `text`를 쓴다.
- 답변 생성에는 원문 `text`나 부모 쪽만 준다.
- 출처 `page`는 코드가 붙이고 LLM이 만들지 않는다.
- 환각 방지: 맥락 속 숫자·날짜·ID가 쪽 원문에 없거나 300자를 넘으면 버린다.
- row의 `context`는 속한 표의 것을 그대로 쓴다 (LLM 호출 없음).
- 캐시(`context_cache.jsonl`): 키 = 프롬프트 버전 + 모델 + 쪽 원문 + 청크 원문. 같으면 LLM을 다시 부르지 않는다.
- 모델: config `context.model` (기본 gpt-6-luna). 환경변수 `OPENAI_CONTEXT_MODEL`이 있으면 그 값.
- 예상 호출량 (추정, o200k 토크나이저 기준):

| 난이도 | 호출 수(쪽) | 입력 토큰 | 출력 토큰 (청크당 60 가정) |
|---|---|---|---|
| Basic | 360 | 약 0.45M | 약 0.10M |
| Intermediate | 3,750 | 약 5.7M | 약 0.7M |

## 5. ID와 payload

- `chunk_id`: 쪽은 `{book_id}-p{page}`, 나머지는 `{book_id}-p{page}-c{n}` (발제문 형식, 쪽 안에서 읽는 순서대로 번호).
- Qdrant 포인트 ID: `uuid.uuid5(NAMESPACE_URL, chunk_id)`
- jsonl 필드: `chunk_id`, `parent_id`, `level`, `chunk_type`, `book_id`, `file_name`, `page`(1부터), `doc_title`, `page_header`, `section_title`, `table_title`, `row_index`, `difficulty`, `domain`, `text`, `context`
- Qdrant payload는 LangChain `QdrantVectorStore`가 그대로 읽는 형식:
  `{"page_content": text, "metadata": {나머지 필드 전부}, "content_hash": …}`
  → 검색 결과 `Document.page_content` = 원문, `Document.metadata["book_id"]`, `["page"]` (팀 `chain.py`의 `format_docs`와 맞음)
- 페이로드 인덱스: `metadata.level`, `metadata.book_id`, `metadata.page`

## 6. 저장 (jsonl, 로컬 전용)

```
artifacts/corpus/basic/shipping/
├─ SB01.jsonl … SB15.jsonl     ← 최신만. 문서 하나 = 파일 하나 (쪽 + 문단 + 표 + row)
├─ manifest.json                ← 문서별 버전·지문·청크 수·이력, 실행 기록
├─ context_cache.jsonl          ← 맥락 캐시 (이것만 깃에 커밋해서 팀원과 공유)
└─ history/
   └─ SB02/v1.jsonl             ← 바뀌거나 지워진 문서의 이전 버전 (바뀐 문서만)
```

- 파일 안 순서: 쪽 레코드 → 그 쪽의 자식(문단, 표, row) → 다음 쪽 … 항상 같은 순서로 쓴다.
- 문서 파일이 90MB를 넘으면 `SB01_001.jsonl`, `SB01_002.jsonl`로 나눈다. 쪽 경계에서만 나눠서 한 쪽은 절대 두 파일로 갈라지지 않는다.
  (90MB는 1024 기준 MiB, GitHub 파일 한도는 100MiB. 실제 문서 하나는 최대 약 1.4MB라 거의 안 나뉜다.)
- 청크 파일·manifest·history는 깃에 올리지 않는다 (아래 .gitignore). 팀원은 같은 명령으로 똑같이 만들 수 있다.
  같은 PDF·코드·설정이면 청킹 결과가 똑같고, `context_cache.jsonl`이 있으면 OpenAI를 다시 부르지 않는다.
- `manifest.json` 문서 항목: `version`, `files`, `fingerprint`(PDF·설정·코드 해시), `chunks`, `context`(토큰·실패 수), `history`(이전 버전 목록), `updated_at`.
- 측정한 크기 (맥락 없이, 2026-11-05):

| 난이도 | 문서 | 쪽 | page | para | table | row | 전체 / 문서당 |
|---|---|---|---|---|---|---|---|
| Basic | 15 | 360 | 360 | 1,600 | 0 | 0 | 2.2MB / 약 0.15MB |
| Intermediate | 50 | 3,750 | 3,750 | 7,717 | 3,750 | 27,715 | 44.7MB / 약 0.9MB |
| Advanced | 120 | 18,000 | 18,000 | 67,851 | 21,126 | 100,889 | 171.4MB / 약 1.4MB |

`.gitignore`에 추가할 내용:

```
# 청킹 결과는 로컬 전용 (맥락 캐시만 공유)
artifacts/corpus/**/*.jsonl
!artifacts/corpus/**/context_cache.jsonl
artifacts/corpus/**/manifest.json
artifacts/corpus/**/history/
artifacts/qdrant_local/
results/chunk_coverage/
```

## 7. 설정 (config/chunking.yaml)

```yaml
chunking:
  max_chunk_tokens: 300     # 넘는 문단은 문장 경계로만 분할
  overlap_tokens: 0         # 오버랩 없음
  jsonl_levels: [page, para, table, row]   # sent는 추후
  tokenizer: BAAI/bge-m3    # 토큰 수를 셀 때 쓰는 토크나이저 (임베딩 모델과 같게)
  max_table_tokens: 8000    # 표 전체 청크가 이보다 길면 경고만 남김 (표·row는 자르지 않음)

index:
  levels: [page, para, table]   # row: off — Qdrant 무료 플랜 용량 제한 / sent: 추후
  embedding_model: BAAI/bge-m3
  embedding_batch_size: 32
  collection_prefix: kant       # 컬렉션 이름 = kant_{도메인}_{난이도}_{모델}  예) kant_shipping_basic_bge-m3
  local_path: artifacts/qdrant_local   # QDRANT_URL이 비어 있을 때만 쓰는 로컬 저장 위치

jsonl:                      # 문서 하나 = 파일 하나 (SB01.jsonl). 구버전은 history/SB01/v1.jsonl (로컬 전용)
  max_shard_mb: 90          # 문서 파일이 넘으면 SB01_001, SB01_002 … 로 분할
                            # 단, 한 쪽(쪽 + 그 문단·표)은 절대 두 파일로 나누지 않음 (쪽 경계에서만 나눔)

context:                    # Contextual Retrieval
  enabled: true
  model: gpt-6-luna
  prompt_version: v1        # 프롬프트를 고치면 올린다 (캐시가 새로 만들어짐)
  max_workers: 4
  max_chars: 300
  levels: [para, table]     # row는 속한 표의 맥락을 그대로 씀

paths:
  data_dir: data
  corpus_dir: artifacts/corpus
```

## 8. 실행 방법

### 메뉴로 실행 (추천)

```bash
uv run python -m rag_service.ingestion.menu
```

번호를 입력하고 엔터. `0` = 뒤로, `q` = 종료.

```
============================================================
 KANT RAG 청킹 관리
============================================================
 1. 청킹 실행 (PDF → jsonl → Qdrant)
 2. 이전 버전으로 되돌리기
 q. 종료
```

- **1. 청킹 실행:** 주제 → 난이도 → 바뀐 문서 미리보기(새 문서·바뀐 문서·삭제된 PDF·그대로) → 옵션(맥락 생성, Qdrant 동기화, 전부 다시) 켜고 끄기 → `s` 실행 → `y` 확인.
  `.env`에 OPENAI_API_KEY나 QDRANT_URL이 없으면 화면에 경고가 나온다.
- **2. 되돌리기:** 주제 → 난이도 → 이전 버전이 있는 문서 목록 → 버전 목록(보관 시각·이유·청크 수) → 선택한 버전과 현재의 쪽별 차이 미리보기 → `y` 확인.
  차이는 ID가 아니라 원문으로 비교해서 번호가 밀려도 "문단 1개 추가"처럼 정확히 나온다.

### 명령줄로 실행

```bash
# 한 번에: 청킹 → 맥락 생성 → 문서별 jsonl → Qdrant 동기화 (컬렉션 kant_shipping_basic_bge-m3)
uv run python -m rag_service.ingestion.build_corpus --difficulty basic

# 옵션
#   --no-context       맥락 생성 건너뜀 (API 키 불필요)
#   --no-index         jsonl까지만 (Qdrant 안 건드림)
#   --index-only       청킹 없이 지금 jsonl에 맞춰 Qdrant만 동기화
#   --restore SB02@v1  문서 하나를 history의 v1로 되돌림 (Qdrant도 맞춤)
#   --reindex          Qdrant 컬렉션을 지우고 전부 다시 임베딩
#   --force            바뀐 게 없어도 전부 다시 청킹 / --books SB01 SB02 일부 문서만

# 근거 보존 검사 (평가 전용) → results/chunk_coverage/
uv run python evaluation/retrieval/check_chunk_coverage.py --difficulty basic

# 테스트
uv run pytest tests/ingestion tests/storage -q
```

검색 코드에서 쓰는 법:

```python
from rag_service.storage.corpus import load_chunks, PageStore

chunks = list(load_chunks("basic", levels=["para", "table"]))   # BM25용 (최신 jsonl)
texts = [c.embedding_text() for c in chunks]
pages = PageStore("basic")
parent = pages.page_of(chunks[0])                               # 답변 생성에 줄 부모 쪽
```

```python
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from rag_service.storage.qdrant_store import level_filter

emb = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", encode_kwargs={"normalize_embeddings": True})
store = QdrantVectorStore.from_existing_collection(
    collection_name="kant_shipping_basic_bge-m3", embedding=emb,
    url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
docs = store.similarity_search("질문", k=5, filter=level_filter(["para", "table"]))   # 실험 B
docs[0].metadata["book_id"], docs[0].metadata["page"], docs[0].metadata["parent_id"]
```

## 9. 바뀐 것만 다시 처리 (증분)과 구버전 관리

예: SB02 3쪽에서 문장 하나가 바뀜

| 단계 | 다시 하는 범위 |
|---|---|
| 변경 감지 | PDF 해시가 달라진 SB02만 대상. 나머지 문서는 건너뜀 |
| 청킹 | SB02를 다시 읽음 (1~2초). 3쪽 말고는 결과가 이전과 똑같다 |
| 맥락 (LLM) | 캐시에 없는 3쪽만 1번 호출 |
| jsonl | 이전 `SB02.jsonl` → `history/SB02/v1.jsonl`로 옮기고 새 `SB02.jsonl`(v2)을 씀 |
| Qdrant | 3쪽에서 내용이 바뀐 청크만 임베딩·업로드, 없어진 포인트만 삭제 |

- 변경 판단은 ID가 아니라 해시로 한다. 문서는 PDF 해시(+설정·코드 해시), 청크는 `content_hash`(임베딩 문자열의 해시).
- 청크 번호는 쪽마다 c1부터라서, 문단이 끼어들어 번호가 밀려도 그 쪽 안에서만 밀린다.
  번호가 밀린 청크도 `content_hash`가 같으면 기존 벡터를 복사하고 임베딩하지 않는다.
- 코드·설정을 바꿔 모든 문서를 다시 청킹해도, 결과가 이전과 똑같은 문서는 파일·버전을 그대로 두고 지문만 갱신한다.
- 새 PDF는 v1로 추가, 지워진 PDF는 최신 파일을 history로 옮기고 manifest의 `removed`로 옮긴다 (Qdrant에서도 삭제).
- 되돌리기: `--restore SB02@v1` → 지금 파일은 history로 보관, v1 내용이 새 버전(v3)으로 최신이 되고 Qdrant도 맞춘다.
  PDF가 그대로면 다음 일반 실행 때 PDF에 맞춰 다시 청킹된다 (임시로 되돌려 보는 용도).
- Qdrant에는 최신만 둔다 (난이도별 컬렉션 하나). 구버전은 로컬 history에만 있다.
- 맥락 생성이 일부 실패한 문서는 다음 실행 때 그 문서만 다시 시도한다 (성공한 것은 캐시에서).
- Qdrant 동기화가 실패해도(네트워크 등) jsonl은 저장된다. 같은 명령을 다시 실행하면 빠진 것만 이어서 맞춘다.
- 다시 청킹 시간 (맥락 생성 제외): Basic 약 25초, Intermediate 약 3분, Advanced 약 10분.

## 10. 비교 실험

| 실험 | levels | 상태 |
|---|---|---|
| A | `page` | 진행 |
| B | `para` + `table` | 진행 |
| C | `row` + `sent` 추가 | 추후 |

- 개발용 질문(`development.json`)으로만 조정한다. `additional`, `expansion_validation`은 최종 비교용.
- 정답 인용(`quote`)은 청킹 결과 검사용이고 색인, 검색, 프롬프트에는 넣지 않는다.
- Hit@k와 Recall@k는 쪽(부모) 기준으로 중복 제거해서 계산하고, 문서 기준도 함께 기록한다.
- 최종 비교에서는 청킹을 고정하고 세 RAG 방식에 똑같이 적용한다.

## 11. 근거 보존 검사 결과 (맥락 없이 청킹한 버전)

| 난이도 | 인용 수 | 쪽 원문에 그대로 | 공백 무시하면 쪽에 있음 | 청크(para/table) 하나에 통째로 |
|---|---|---|---|---|
| Basic (전체 4종) | 102 | 99 (97%) | 102 (100%) | 86 (84%) |
| Intermediate (development·additional·security) | 127 | 116 (91%) | 127 (100%) | 127 (100%) |
| Advanced (development·additional·security) | 151 | 147 (97%) | 151 (100%) | 151 (100%) |

- Basic에서 청크 하나에 안 들어간 16개는 2~4개 문단에 걸친 인용이다. 부모 쪽을 LLM에 넘기면 보완된다.

## 12. Qdrant Cloud 용량 (추정)

- 무료 플랜: RAM 1GB, 디스크 4GB. 1주 미사용 시 중지, 4주 후 삭제.
- 벡터 1개 ≈ 4KB (1024차원 float32).
- 실제 색인 대상 (page + para + table): Basic 1,960 / Intermediate 15,217 / Advanced 106,977 → 합계 약 12.4만 벡터, 원본 벡터만 약 0.51GB.
  HNSW 인덱스 등과 여유 20%를 더하면 약 0.65GB로 1GB 안에 들어간다.
- `row`(세 난이도 합계 12.9만 개)를 켜면 벡터만 약 0.53GB가 더 늘어 합계 1GB를 넘는다. 그래서 현재 off.
- 컬렉션은 난이도별 하나만 두고 최신만 올려서, 버전이 바뀌어도 용량이 늘지 않는다.

## 13. 코드 위치

```
config/chunking.yaml                  청킹·색인·맥락 설정
src/rag_service/contracts/chunk.py    청크 형식 (pydantic), embedding_text(), point_id
src/rag_service/ingestion/            ← 청킹 전용
  ├─ pdf_reader.py                    PDF → 쪽별 머리글/제목/문단/표(제목·각주)
  ├─ chunker.py                       쪽 → page/para/table/row 청크
  ├─ context.py                       쪽 단위 맥락 생성 (캐시, 숫자 검사)
  ├─ writer.py                        문서별 jsonl 저장 (90MB 분할), history 보관, manifest
  ├─ index_sync.py                    Qdrant 포인트 단위 동기화 (바뀐 것만, 벡터 재사용)
  ├─ build_corpus.py                  명령줄 실행 진입점 (증분 처리, --restore, Qdrant 동기화 호출)
  └─ menu.py                          텍스트 메뉴 (청킹 실행, 되돌리기)
src/rag_service/storage/              ← 공통 (검색·생성 코드도 import)
  ├─ corpus.py                        최신 jsonl 읽기 (load_chunks, PageStore), 설정 읽기
  └─ qdrant_store.py                  컬렉션 이름, 접속, bge-m3 임베딩, level_filter
evaluation/retrieval/check_chunk_coverage.py   근거 보존 검사 (평가 전용)
tests/ingestion/test_chunking.py      청킹·분할·증분·추가/삭제/되돌리기·맥락(가짜 LLM) 테스트
tests/ingestion/test_menu.py          메뉴 (입력을 흘려 넣어 화면·결과 확인)
tests/ingestion/test_index_sync.py    로컬 Qdrant + 가짜 임베딩: 바뀐 것만 동기화, 번호 밀림, 되돌리기
tests/storage/test_qdrant_store.py    공통 기능: LangChain 읽기, 컬렉션 이름, level_filter, GPU→CPU 대체
```

## 14. 남은 확인사항

- [ ] Basic으로 맥락 생성 실제 비용, 시간 측정 (API 키 필요)
- [ ] bge-m3 실제 임베딩 + Qdrant Cloud 업서트 테스트 (.env의 QDRANT_URL, QDRANT_API_KEY)
- [ ] .gitignore에 6번의 내용 추가 (청크 jsonl·manifest·history는 로컬 전용, context_cache만 커밋)
- [ ] 임베딩 속도 확인: 윈도우 기본 PyTorch는 CPU 버전일 수 있음 (Advanced 약 10.7만 청크)
- [ ] Qdrant 계정 소유자 결정. 제출 전까지 클러스터 계속 사용.
- [ ] **Intermediate 작업 시작 시 가장 먼저 확인:** 표 각주 위치, 깨진 헤더 (자동 검사로는 각주가 표·row에 붙는 것 확인됨, 눈으로 몇 쪽 더 확인)
- [ ] 문장(`sent`) 레벨은 여유될 때 시도
