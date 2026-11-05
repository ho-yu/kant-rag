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

- 오버랩 없음. 문단이 300토큰을 넘으면 문장 경계로만 분할 (안전장치).
- 쪽 경계를 넘는 청크 없음 (`page` 보존).
- 임베딩: bge-m3 (dense). BM25: `rank-bm25`.
- 답변 생성 모델: gpt-6-luna.

## 2. 표 처리

- 표 전체를 청크 하나(`table`)로 저장하고, row마다 `컬럼명 값 / 컬럼명 값` 문장으로 `row` 청크를 따로 만든다.
- 모든 row에 문서 제목, 표 제목, 쪽 번호, 컬럼명, 단위, 각주를 같이 붙인다. row는 잘리지 않는다.
- 근거: TARGET 벤치마크에서 row 단위와 표 전체의 검색 성능은 비슷했고, 컬럼명만 넣을 때 성능이 급락해서 행 값과 표 제목이 같이 필요하다고 확인.

## 3. Contextual Retrieval

- 쪽 단위로 OpenAI를 1회 호출해서, 그 쪽 모든 청크의 맥락 한 줄을 JSON으로 받는다 (입력: 부모 쪽 + 문서 제목).
- `text`(원문)와 `context`(LLM 한 줄)를 분리 저장한다.
- 임베딩과 BM25 색인에는 `context + text`를 쓴다.
- 답변 생성에는 원문 `text`나 부모 쪽만 준다.
- 출처 `page`는 코드가 붙이고 LLM이 만들지 않는다.
- 환각 방지: 맥락 속 숫자와 날짜가 쪽 본문에 있는지 코드로 검사한다.
- row의 `context`는 속한 표의 것을 그대로 쓴다.
- 맥락 캐시: 키 = `청크 text 해시 + 프롬프트 버전 + 모델명`. 텍스트가 안 바뀐 청크는 LLM을 다시 부르지 않는다.

## 4. ID와 payload

- `chunk_id`: `{book_id}-p{page}-c{n}` (발제문 형식)
- Qdrant 포인트 ID: `uuid.uuid5(NAMESPACE_URL, chunk_id)`
- payload: `book_id`, `file_name`, `page`(1부터), `chunk_id`, `parent_id`, `level`, `section_title`, `chunk_type`, `difficulty`, `domain`, `text`, `context`
- 페이로드 인덱스: `level`, `book_id`, `page`

## 5. 저장 (jsonl)

```
artifacts/corpus/basic/shipping/
├─ v1/   SB_001.jsonl, manifest.json   ← old
├─ v2/   SB_001.jsonl, manifest.json   ← new
└─ CURRENT                              ← 사용 중인 버전 (예: v2)
```

- 쪽 레코드를 먼저 쓰고, 그 쪽의 자식(문단, 표, row)을 바로 뒤에 쓴다. 정렬은 `book_id`, 쪽, 청크 번호 순.
- 90MB를 넘으면 `SB_001.jsonl`, `SB_002.jsonl` 식으로 분할한다.
- 한 문서는 절대 두 파일로 나누지 않는다 (다음 문서를 넣으면 90MB를 넘길 때만 다음 파일로).
- `manifest.json`: 버전, 생성 날짜, 설정 스냅샷, 청커 코드 버전(깃 커밋), 문서별 지문, 청크 수, 파일 목록.
- 깃에 올릴지는 jsonl 실제 크기를 잰 뒤 결정 (Advanced가 클 수 있음).

## 6. 설정 (config/ingestion/chunking.yaml)

```yaml
chunking:
  max_chunk_tokens: 300     # 넘는 문단은 문장 경계로만 분할
  overlap_tokens: 0         # 오버랩 없음
  jsonl_levels: [page, para, table, row]   # sent는 추후
index:
  levels: [page, para, table]   # row: off — Qdrant 무료 플랜 용량 제한 / sent: 추후
jsonl:
  max_shard_mb: 90          # 넘으면 SB_001, SB_002 … 로 분할
                            # 단, 한 문서(예: SB09)는 절대 두 jsonl로 나누지 않음.
                            # 다음 문서를 넣으면 90MB를 넘길 때만 다음 파일로 넘어감
```

## 7. 비교 실험

| 실험 | levels | 상태 |
|---|---|---|
| A | `page` | 진행 |
| B | `para` + `table` | 진행 |
| C | `row` + `sent` 추가 | 추후 |

- 개발용 질문(`development.json`)으로만 조정한다. `additional`, `expansion_validation`은 최종 비교용.
- 정답 인용(`quote`)은 청킹 결과 검사용이고 색인, 검색, 프롬프트에는 넣지 않는다.
- Hit@k와 Recall@k는 쪽(부모) 기준으로 중복 제거해서 계산하고, 문서 기준도 함께 기록한다.
- 최종 비교에서는 청킹을 고정하고 세 RAG 방식에 똑같이 적용한다.

## 8. Qdrant Cloud 용량 (추정)

- 무료 플랜: RAM 1GB, 디스크 4GB. 1주 미사용 시 중지, 4주 후 삭제.
- 벡터 1개 ≈ 4KB (1024차원 float32).
- `page` + `para` + `table` 전체(세 난이도) ≈ 15만 벡터, RAM 약 0.65GB (여유 20% 포함 약 0.78GB).
- `row`와 `sent`까지 Advanced에 켜면 약 50만 벡터로 RAM을 넘길 수 있어서, 현재는 off.
- 컬렉션 이름에 버전 포함 (예: `rag_shipping_basic_bgem3_v2`). 두 버전을 동시에 두면 벡터가 2배라 비교 후 이전 버전은 삭제.

## 9. 코드 경로와 실행

```
config/ingestion/chunking.yaml        청킹 설정
src/rag_service/contracts/chunk.py    청크 payload 모델 (pydantic)
src/rag_service/ingestion/
  ├─ pdf_reader.py                    PDF → 쪽별 텍스트/표 추출
  ├─ chunker.py                       쪽 → 문단/표/row 청크
  ├─ context.py                       OpenAI로 쪽 단위 맥락 생성 (캐시 포함)
  ├─ writer.py                        jsonl 저장 (90MB 분할, manifest)
  └─ build_corpus.py                  실행 진입점
src/rag_service/storage/qdrant_store.py   임베딩 + Qdrant upsert
evaluation/retrieval/                 근거 인용 보존 검사
artifacts/corpus/{난이도}/{도메인}/    결과 jsonl
```

```
uv run python -m rag_service.ingestion.build_corpus --difficulty basic
  --version v2   새 버전으로 생성
  --force        지문이 같아도 강제 재생성
```

## 10. 증분 재청킹과 버전 관리

- 문서별 지문 3개를 manifest에 기록한다: PDF 해시, 청킹 설정 해시, 청커 코드 버전.
- 세 개가 모두 같은 문서는 건너뛰고, 바뀐 문서만 다시 청킹한다.
- PDF는 거의 안 바뀌어서, 실제로는 코드나 설정을 바꿔 전체를 다시 청킹하는 경우가 많다.
- 이전 버전 폴더는 비교가 끝날 때까지 남긴다.
- 추후: Qdrant 컬렉션 별칭(`..._current`)으로 버전 전환과 되돌리기.

## 11. 진행 순서

1. 청커 코드 (쪽, 문단, 표, row → jsonl + manifest)
2. 청킹 검사 (근거 인용이 청크 안에 남는지)
3. 맥락 생성 스크립트
4. 색인 스크립트 (bge-m3 임베딩, Qdrant upsert)
5. 평가 스크립트 (A와 B 비교)

## 12. 남은 확인사항

- [ ] Basic으로 맥락 생성 비용, 시간 측정
- [ ] jsonl 실제 크기 측정 → 깃 업로드 방식 결정
- [ ] Qdrant 계정 소유자, 컬렉션 이름 (모델명, 버전 포함) 결정. 제출 전까지 클러스터 계속 사용.
- [ ] **Intermediate 작업 시작 시 가장 먼저 확인:** 표 각주 위치, 깨진 헤더
- [ ] 줄바꿈 이어붙이기(Kiwi 규칙)를 더 큰 표본으로 확인
- [ ] 문장(`sent`) 레벨은 여유될 때 시도