# kant-rag

해운(shipping) 업무 PDF에서 근거를 찾아 답하는 **RAG 기반 Q&A 서비스**입니다. (KANT 3차 프로젝트, 팀 환각없조)

- 답변마다 근거 문서명과 페이지를 함께 보여 줍니다.
- 질문에 필요한 조건이 빠지면 되묻고(`needs_clarification`), 자료가 부족하거나 문서끼리 충돌하면 답변을 보류합니다(`abstained`).
- Naive RAG, 하이브리드 검색, 하이브리드 + 리랭킹 세 가지 검색 방식을 같은 질문으로 비교해 최종 방식을 고릅니다.

> 발제 문서: [프로젝트(3) 심화 프로젝트 - RAG 기반 Q&A 서비스](https://app.notion.com/p/hoyaa/3-RAG-Q-A-1-40f682609ddd809ba6abf7c3d71e1a90)
> 제출 마감: **2026-11-13(금) 13:00 KST**, 같은 날 오후 발표

---

## 목차

1. [진행 현황 (발제 필수 과제 기준)](#1-진행-현황-발제-필수-과제-기준)
2. [처리 흐름](#2-처리-흐름)
3. [모델과 검색 설정](#3-모델과-검색-설정)
4. [로컬 개발 환경 세팅](#4-로컬-개발-환경-세팅)
5. [실행 방법](#5-실행-방법)
6. [평가 방법](#6-평가-방법)
7. [Langfuse 실행 기록 확인](#7-langfuse-실행-기록-확인)
8. [결과 비교표와 최종 선택](#8-결과-비교표와-최종-선택)
9. [폴더 구조](#9-폴더-구조)
10. [팀원별 역할](#10-팀원별-역할)
11. [자주 겪는 문제](#11-자주-겪는-문제)
12. [Git 규칙](#12-git-규칙)

---

## 1. 진행 현황 (발제 필수 과제 기준)

> 기준: main `7eca27f` (2026-11-09). ✅ 구현됨 · 🟡 일부 구현 · ⬜ 아직 없음

| 필수 과제 | 항목 | 상태 | 위치 |
|---|---|---|---|
| **1. PDF 기반 RAG 서비스** | PDF 텍스트·표 추출, 청킹, 출처(book_id·page) 보존 | ✅ | `src/rag_service/ingestion/` |
| | Qdrant 색인 (난이도별 컬렉션) | ✅ Basic·Intermediate | `ingestion/index_sync.py`, `storage/qdrant_store.py` |
| | 답변 + 출처(문서명·페이지) 표시 | ✅ | `pipeline/chain.py`, `generation/sources.py` |
| | 추가 확인(`needs_clarification`)·보류(`abstained`) 상태 | ✅ | `contracts/schemas.py` (`RagAnswer`) |
| | 질문 사전 판단(차단·잡담·질문 재작성), 답변 사후 검사 | ✅ | `generation/guard.py` |
| | 채팅 화면 (Streamlit) | ✅ | `client/client.py` |
| **2. 세 가지 RAG 방식 비교** | Naive / Hybrid / Hybrid + Rerank 검색기 | ✅ 두 가지 구현 | `retrieval_a/`, `retrieval_b/` (아래 3장) |
| | 같은 질문으로 검색 성능 비교 | ✅ 검색 지표 | `evaluation/retrieval/retrieval_b_compare.py` |
| | 같은 질문으로 답변 품질 비교 | ⬜ | — |
| **3. FastAPI · LangChain · Langfuse** | LangChain 체인 (검색 → 프롬프트 → gpt-6-luna → 출처) | ✅ | `pipeline/chain.py` |
| | Langfuse 실행 기록 (질문·검색 문서·모델 입출력) | ✅ | `observability/tracing.py` |
| | FastAPI (`/health`, `/query`, 요청·응답 검증, 오류 구분) | ⬜ | `src/rag_service/api/` (예정) |
| | 처리 오류를 `error` 상태로 구분 | ⬜ | — |
| **4. 평가와 결과 설명** | 검색: Hit@k · Recall@k (+ MRR) | ✅ | `retrieval_b/metrics.py` |
| | 평가 질문을 Langfuse 데이터셋으로 등록 | ✅ | `evaluation/upload_langfuse_datasets.py` |
| | 답변: RAGAS Faithfulness | ⬜ | `evaluation/answer/` (예정) |
| | 답변: LLM-as-a-Judge (사실·조건·근거·판단) | ⬜ | `evaluation/answer/` (예정) |
| | 보안 답변 4개 (기본 vs 방어 프롬프트) | ⬜ | `evaluation/security/` (예정) |
| | 운영 지표 (응답 시간·오류율·입출력 토큰) | ⬜ | `evaluation/benchmark/` (예정) |
| **산출물** | 비교표, 최종 선택 이유·한계, 발표자료 | ⬜ | 8장 |

---

## 2. 처리 흐름

```
[문서 준비]  data/{난이도}/shipping/documents/*.pdf
              → ingestion (텍스트·표 추출 → page / para / table 청크 → 청크별 맥락 한 줄)
              → artifacts/corpus/{난이도}/shipping/*.jsonl
              → bge-m3 임베딩 → Qdrant 컬렉션 kant_shipping_{난이도}_bge-m3

[질문 처리]  client/client.py (Streamlit)
              → 질문 사전 판단 (guard): block / direct / rag, 검색용 질문 재작성, 여러 내용이면 나눠 묻기
              → pipeline/chain.py
                   검색 (retrieval_mode: naive | hybrid | hybrid_rerank)
                   → 문맥 만들기 ([S1] 파일명 / page …), 필요하면 쪽 전체로 확장(page_context)
                   → gpt-6-luna 구조화 출력 (status, answer, reason, source_ids)
                   → 코드가 source_ids를 실제 검색 결과와 맞춰 [출처] 붙이기
              → 답변 사후 검사 (guard) → 화면 표시
              → 전 과정 Langfuse 기록
```

- 모델이 문서명과 페이지를 만들지 않습니다. 모델은 `S1`, `S2` 같은 자료 번호만 고르고, 파일명·페이지는 코드가 검색 결과의 metadata에서 붙입니다.
- 정답 JSON(근거·기대 동작)은 검색어, 필터, 색인, 답변 생성 입력에 넣지 않습니다. 채점할 때만 씁니다 (발제 규칙).

---

## 3. 모델과 검색 설정

발제 요구에 따라 임베딩·리랭커·주요 설정을 기록합니다. 세 방식 비교에는 같은 임베딩과 저장소를 씁니다.

### 공통

| 항목 | 값 | 설정 위치 |
|---|---|---|
| 분야 / 난이도 | 해운(shipping) / Basic → Intermediate → Advanced 순서로 진행 | `data/README.md` |
| 답변 생성 모델 | `gpt-6-luna` | `pipeline/chain.py`, `generation/guard.py` |
| 임베딩 모델 | `BAAI/bge-m3` | `config/chunking.yaml` |
| 리랭커 | `BAAI/bge-reranker-v2-m3` | `config/retrieval_a/rerank.yaml`, `retrieval_b/rerank.py` |
| 벡터 저장소 | Qdrant Cloud (팀 공용 클러스터), `QDRANT_URL`이 비면 로컬 폴더 | `config/chunking.yaml` |
| 컬렉션 이름 | `kant_shipping_{난이도}_bge-m3` | `config/chunking.yaml` |
| 청크 | 최대 300토큰, overlap 0, 레벨 page / para / table | `config/chunking.yaml` |
| 청크 맥락 | 청크마다 gpt-6-luna가 맥락 한 줄 작성 (Contextual Retrieval) | `config/chunking.yaml`, `ingestion/context.py` |
| 검색 레벨 | para + table (page는 쪽 확장·출처용) | 각 검색기 설정 |
| 최종 청크 수 k | 5 (나눠 찾기 사용 시 합쳐서 최대 8) | 각 검색기 설정, `pipeline/chain.py` |

### 검색 방식별 설정

팀에서 두 가지 구현을 각각 만들었고, 지금 채팅 화면(`chain.py`)은 **retrieval_a**에 연결되어 있습니다.

| 방식 | retrieval_a (`config/retrieval_a/*.yaml`) | retrieval_b (`src/rag_service/retrieval_b/`) |
|---|---|---|
| Naive | bge-m3 Dense, top_k 5 | bge-m3 Dense, k 5 |
| 키워드 검색 | bge-m3 Sparse | BM25 (Kiwi 형태소 토큰화) |
| Hybrid | Dense 0.7 + Sparse 0.3, min-max 정규화 합산 | Dense·BM25 각 30개 후보 → RRF(k=60) |
| Hybrid + Rerank | Hybrid 후보 50개 → 리랭커 → 상위 5개 | Hybrid 후보 50개 → 리랭커 → 상위 5개 |
| 체인 연결 | ✅ (`naive`, `hybrid`, `hybrid_rerank`) | 검색 평가 실행기에서 사용 |

---

## 4. 로컬 개발 환경 세팅

처음 한 번만 하면 됩니다. 순서대로 따라 하세요.

### 4-1. 미리 설치할 것

| 도구 | 용도 | 확인 명령 |
|---|---|---|
| Git | 저장소 내려받기 | `git --version` |
| uv | 파이썬 버전과 패키지 관리 | `uv --version` |

Python은 따로 설치하지 않아도 됩니다. uv가 프로젝트에 맞는 Python 3.12를 자동으로 내려받습니다.

- macOS / Linux

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

- Windows (PowerShell)

  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

설치 후 터미널을 새로 열고 `uv --version`이 나오면 됩니다.

### 4-2. 저장소 내려받기

```bash
git clone https://github.com/ho-yu/kant-rag.git
cd kant-rag
```

PDF와 청크 파일이 저장소에 포함되어 있어 내려받는 데 시간이 조금 걸립니다.

### 4-3. 패키지 설치

```bash
uv sync
```

- `.venv/`에 가상환경이 만들어지고 `uv.lock`에 적힌 버전 그대로 설치됩니다. 팀원 모두 같은 버전을 씁니다.
- PyTorch가 포함되어 있어 **처음 설치는 5분 이상 걸리고 디스크를 수 GB 씁니다.**
  - Windows는 NVIDIA GPU(CUDA 13.0)용 PyTorch를, macOS·Linux는 PyPI 기본 PyTorch를 받습니다 (`pyproject.toml`의 `[tool.uv.sources]`).
- 가상환경을 직접 활성화할 필요는 없습니다. 명령 앞에 `uv run`을 붙이면 됩니다. **명령은 항상 저장소 맨 위 폴더에서 실행하세요.**

> 패키지를 추가할 때는 `pip install` 대신 `uv add 패키지이름`을 쓰고, `pyproject.toml`과 `uv.lock`을 같이 커밋하세요.

### 4-4. 환경 변수(.env) 만들기

```bash
# macOS / Linux
cp .env.example .env

# Windows (PowerShell)
Copy-Item .env.example .env
```

| 변수 | 필수 | 설명 |
|---|---|---|
| `OPENAI_API_KEY` | ✅ | 답변 생성·가드·청크 맥락에 쓰는 OpenAI 키 |
| `OPENAI_MODEL` | ✅ | `gpt-6-luna` (채팅 화면의 바로 답변에 사용) |
| `QDRANT_URL`, `QDRANT_API_KEY` | ✅ | 팀 공용 Qdrant 클러스터. **담당자(이정원)에게 개인 메시지로 받습니다** |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` | ✅ | Langfuse 실행 기록. 팀 서버 주소는 담당자에게 받습니다 |
| `OPENAI_CONTEXT_MODEL` | 선택 | 청크 맥락 모델을 바꿀 때만 (기본 gpt-6-luna) |
| `EMBEDDING_MODEL`, `RERANKER_MODEL` | 선택 | retrieval_b에서 모델을 바꿔 실험할 때만 (기본 bge-m3, bge-reranker-v2-m3) |

`.env`는 `.gitignore`에 등록되어 커밋되지 않습니다. **API 키는 절대 커밋하거나 채팅·노션에 붙여 넣지 마세요.**

### 4-5. 설치 확인

```bash
uv run python -c "import fastapi, langchain, langfuse, qdrant_client, ragas; print('설치 OK')"
uv run ruff check .
uv run pytest -m "not integration"
```

`설치 OK`가 나오고 pytest가 실패 없이 끝나면 준비 완료입니다 (약 2~3분).
- `integration` 표시가 붙은 테스트는 Qdrant Cloud·모델 다운로드가 필요해서 평소에는 뺍니다.
- ruff는 내가 고친 파일에 스타일 오류가 없는지 확인하는 용도입니다. 특정 파일만 볼 때는 `uv run ruff check 파일경로`.

---

## 5. 실행 방법

### 5-1. 채팅 화면 (서비스 시연)

```bash
uv run streamlit run client/client.py
```

브라우저가 열리면 사이드바에서 **난이도, 도메인(shipping), 검색 방식(naive / hybrid / hybrid+rerank), 쪽 확장**을 고르고 질문합니다.
답변 아래 접는 칸에서 출처(파일명, 페이지, 원문)를 확인할 수 있습니다.

### 5-2. 체인만 실행 (스크립트)

```bash
uv run python main.py
```

예시 질문 하나를 체인에 보내고 결과(`RagAnswer`)를 출력합니다. 질문·난이도·검색 방식은 `main.py` 안의 값을 바꿔서 씁니다.

### 5-3. API 서버 (예정)

FastAPI 코드(`src/rag_service/api/`)가 추가되면 아래처럼 실행하고 http://localhost:8000/docs 에서 시연합니다. 모듈 경로는 구현에 맞춰 수정하세요.

```bash
uv run uvicorn rag_service.api.main:app --reload
```

### 5-4. 검색기만 바로 확인 (retrieval_b)

```bash
uv run python -m rag_service.retrieval_b "부산에서 LA로 보내는 일반 건화물의 서류 제출 마감은 언제인가요?"
uv run python -m rag_service.retrieval_b "질문" --mode hybrid          # naive | bm25 | hybrid | rerank
uv run python -m rag_service.retrieval_b "질문" --difficulty intermediate --k 3
```

### 5-5. 청킹·색인 (문서 준비)

```bash
uv run python -m rag_service.ingestion.menu
```

번호를 골라 청킹 실행이나 되돌리기를 합니다. 자세한 내용은 [docs/chunking_guide.md](docs/chunking_guide.md)를 보세요.

> ⚠️ Qdrant는 팀이 한 클러스터를 같이 씁니다. **청킹 담당이 아니면 Qdrant 동기화(메뉴에서 동기화 켜기, `--index-only`, `--reindex`)를 돌리지 마세요.** 청크 파일은 저장소에 이미 있고, Qdrant에도 올라가 있습니다.

---

## 6. 평가 방법

발제 STEP 6·7 기준입니다. 평가 질문과 정답은 `data/{난이도}/shipping/evaluation/`에 있습니다.

| 파일 | 내용 |
|---|---|
| `development_questions.json`, `additional_questions.json` (Basic은 `expansion_validation_questions.json`도) | 실행할 질문과 ID |
| `development.json`, `additional.json` (Basic은 `expansion_validation.json`도) | 참고 정답, 근거(`evidence_groups`), 기대 동작(`expected_action`) |
| `security_questions.json`, `security_tasks.json` | 보안 비교용 질문과 사례 |

### 6-1. 검색 평가: Hit@k · Recall@k ✅

```bash
uv run python evaluation/retrieval/retrieval_b_compare.py                                   # 기본: naive, development
uv run python evaluation/retrieval/retrieval_b_compare.py --modes naive bm25 hybrid rerank
uv run python evaluation/retrieval/retrieval_b_compare.py --modes naive a_naive a_hybrid a_rerank   # retrieval_a와 같은 기준으로
uv run python evaluation/retrieval/retrieval_b_compare.py --split additional --open          # 끝나면 결과 페이지 열기
```

- 결과: `results/comparison/retrieval_b/{난이도}_{split}_{levels}.json`과 같은 이름의 `.html`
- 채점 기준 (발제): 문항의 모든 `evidence_groups`에 있는 `book_id`를 중복 없이 모은 목록과, 상위 k개 청크의 문서 ID를 비교합니다. 같은 문서는 한 번만 셉니다. 세 방식에 같은 k를 씁니다. 근거 문서가 없는 문항은 "해당 없음"으로 빼고 이유를 적습니다.
  - Hit@k: 근거 문서를 하나라도 찾으면 1
  - Recall@k: 찾은 근거 문서 수 / 전체 근거 문서 수
  - 페이지 단위 지표와 MRR은 참고용 보조 지표입니다.
- 예전에 저장한 JSON을 HTML로 바꾸기: `uv run python evaluation/reporting/retrieval_report.py <json 경로>`
- 청킹 후 근거 문장이 청크에 남았는지 점검: `uv run python evaluation/retrieval/check_chunk_coverage.py --difficulty basic`

### 6-2. 답변 평가: RAGAS Faithfulness · LLM-as-a-Judge ⬜ (예정)

- 같은 질문을 세 방식으로 실행하고 답변·상태·인용·**실제 최종 문맥**을 저장한 뒤 채점합니다.
- RAGAS Faithfulness: 질문 → `user_input`, 답변 → `response`, 실제 최종 문맥 → `retrieved_contexts`
- LLM-as-a-Judge: 사실·조건·근거·판단 네 항목을 각각 0/1로 채점하고 이유를 남깁니다 (합산하지 않음). 참고 정답과 채점 기준은 평가 단계에서만 넣습니다.
- 해당 없음·평가 불가는 평균에서 빼고 0점은 포함합니다. 평균 옆에 평가 대상 수, 해당 없음, 평가 불가, 오류 건수를 적습니다.
- 평가 질문을 Langfuse 데이터셋으로 올리기 ✅: `uv run python evaluation/upload_langfuse_datasets.py`

### 6-3. 보안 비교 ⬜ (예정)

- 검색 구성 하나에서 보안 질문 2개를 **같은 저장 문맥**으로 기본 프롬프트·방어 프롬프트에 각각 넣어 답변 4개를 확인합니다.
- 공격 문장이 실제 모델 입력에 들어갔는지, 공격을 따랐는지, 업무 답변이 맞는지, 불필요하게 거절했는지 나눠 기록합니다. 공격이 전달되지 않은 경우는 방어 성공으로 세지 않습니다.

### 6-4. 운영 지표 ⬜ (예정)

- 응답 시간(API가 질문을 받은 때부터 응답까지), 오류율(재시도 후에도 시스템 오류로 끝난 비율), 입력·출력 토큰을 방식별로 비교합니다.
- 채점 호출(RAGAS·Judge)의 시간과 토큰은 서비스 지표에 섞지 않습니다. 측정 못 한 값은 0이 아니라 "확인 불가"로 적습니다.

---

## 7. Langfuse 실행 기록 확인

발제 필수 항목입니다. 질문, 검색 문서, 실제 모델 입출력을 요청 하나로 묶어 기록합니다.

1. `.env`에 `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`를 넣습니다.
2. 채팅 화면(`client/client.py`)이나 `main.py`로 질문을 보냅니다.
3. Langfuse 웹에서 **Tracing**을 엽니다.
   - `user-question`: 질문 한 건 전체 (입력 질문, 최종 출력)
   - `rag-answer`: 체인 실행 기록. 안에서 검색 결과(docs), 모델에 넣은 문맥과 프롬프트, 모델 출력을 단계별로 볼 수 있습니다
   - metadata의 `domain`, `difficulty`로 걸러 볼 수 있습니다
4. 평가 질문은 **Datasets**에서 확인합니다 (6-2의 업로드 스크립트로 등록).

> 정정 공지를 검색이 못 찾은 경우와, 찾았지만 문맥에서 빠진 경우를 trace에서 구분할 수 있어야 합니다 (발제 STEP 7).

---

## 8. 결과 비교표와 최종 선택

> ⬜ 평가를 마친 뒤 채웁니다. 발제 산출물 2·3·4·5번에 해당합니다.

### 8-1. 세 방식 비교표 (일반 평가 질문)

| 항목 | Naive | Hybrid | Hybrid + Rerank |
|---|---|---|---|
| 평가 질문 수 / 선택 이유 | | | |
| Hit@k (k=5) | | | |
| Recall@k (k=5) | | | |
| RAGAS Faithfulness 평균 (대상 수 / 해당 없음 / 평가 불가) | | | |
| Judge 사실 / 조건 / 근거 / 판단 충족률 | | | |
| 평균 응답 시간 (정상 요청 수) | | | |
| 오류율 (실패 건수) | | | |
| 입력 / 출력 토큰 합계 (확인한 요청 수) | | | |
| 모델 호출 비용 (단가 확인 시) | | | |

### 8-2. 보안 비교 (답변 4개)

| 질문 | 프롬프트 | 공격 노출 | 공격을 따름 | 업무 답변 정확 | 불필요한 거절 |
|---|---|---|---|---|---|
| 보안 질문 1 | 기본 | | | | |
| 보안 질문 1 | 방어 | | | | |
| 보안 질문 2 | 기본 | | | | |
| 보안 질문 2 | 방어 | | | | |

### 8-3. 최종 선택 이유와 한계

- 최종 검색 방식:
- 선택 이유 (품질·시간·토큰을 함께 봄):
- 좋아지거나 나빠진 대표 사례 (원문 대조):
- 남은 한계와 확인하지 못한 부분:

---

## 9. 폴더 구조

```
kant-rag/
├── client/                    # Streamlit 채팅 화면
├── config/
│   ├── chunking.yaml          # 청킹·색인 설정
│   └── retrieval_a/           # retrieval_a 검색 설정 (naive, hybrid, rerank)
├── data/                      # 원본 데이터 (shipping만 사용)
│   └── {basic,intermediate,advanced}/shipping/
│       ├── documents/         # PDF (Basic 15, Intermediate 50, Advanced 120)
│       └── evaluation/        # 평가 질문·정답 JSON
├── artifacts/
│   └── corpus/{난이도}/shipping/   # 청크 jsonl, 맥락 캐시, manifest (Basic·Intermediate)
├── docs/                      # 청킹 가이드·스펙
├── src/rag_service/
│   ├── ingestion/             # PDF → 청크 → 맥락 → jsonl → Qdrant 동기화
│   ├── storage/               # Qdrant 접속, 청크 읽기 (공통)
│   ├── contracts/             # 청크·답변·가드 데이터 형식
│   ├── retrieval_a/           # 검색기 구현 A (체인에 연결됨)
│   ├── retrieval_b/           # 검색기 구현 B + Hit@k·Recall@k 채점
│   ├── generation/            # 프롬프트, 가드, 출처 붙이기
│   ├── pipeline/              # LangChain 체인 (create_chain)
│   ├── observability/         # Langfuse 추적
│   └── api/                   # FastAPI (예정)
├── evaluation/
│   ├── retrieval/             # 검색 비교 실행기, 청크 근거 점검
│   ├── reporting/             # 결과 JSON → HTML
│   ├── answer/ security/ benchmark/   # 답변·보안·운영 지표 평가 (예정)
│   └── upload_langfuse_datasets.py
├── results/                   # 평가 결과 (커밋하지 않음)
├── tests/                     # 테스트 (폴더별 README 참고)
├── main.py                    # 체인 실행 예시
├── pyproject.toml             # 의존성
└── uv.lock                    # 고정된 패키지 버전 (직접 수정하지 않음)
```

---

## 10. 팀원별 역할

> 커밋 기록 기준입니다. 발표 전에 각자 기여 내용을 보완해 주세요.

| 이름 | 맡은 부분 |
|---|---|
| 이정원 | PDF 추출·청킹·맥락 생성, Qdrant 색인, Streamlit 채팅 화면, 질문·답변 가드, 출처 표시와 쪽 확장, FastAPI(예정) |
| 임윤성 | LangChain 체인 구조, 답변 스키마·프롬프트, Langfuse 추적, 평가 질문 Langfuse 데이터셋 등록 |
| 최완위 | 검색기 구현 A (Naive / bge-m3 Sparse Hybrid / Rerank)와 체인 연결 |
| 유원호 | 개발 환경(uv)·README, 검색기 구현 B (Naive / BM25 / RRF Hybrid / Rerank), Hit@k·Recall@k·MRR 채점과 비교 리포트 |

---

## 11. 자주 겪는 문제

| 증상 | 해결 |
|---|---|
| `uv: command not found` | uv 설치 후 터미널을 새로 열어 주세요. |
| `uv sync`가 오래 걸림 | PyTorch 설치 때문입니다. 처음 한 번만 오래 걸립니다. |
| `ModuleNotFoundError: rag_service` | 저장소 맨 위 폴더에서 `uv run`으로 실행했는지 확인하세요. |
| `import ragas`에서 `No module named 'langchain_community.chat_models.vertexai'` | ragas 0.4.3은 langchain-community 0.4 이상과 맞지 않아 `<0.4`로 고정해 두었습니다. 최신 코드를 받은 뒤 `uv sync`를 다시 실행하세요. |
| Qdrant 접속 오류 (401/403) | `.env`의 `QDRANT_URL`, `QDRANT_API_KEY`를 확인하세요. 키는 담당자에게 받습니다. |
| 리랭커·임베딩 첫 실행이 느림 | 처음 한 번 Hugging Face에서 모델을 내려받습니다 (수 GB). |
| pytest에서 같은 이름 테스트 파일 충돌 | `tests/` 하위 폴더 사이에 테스트 파일 이름이 겹치지 않게 짓습니다 (예: `test_rerank_b.py`). |
| 패키지가 꼬인 것 같음 | `.venv` 폴더를 지우고 `uv sync`를 다시 실행하세요. |

---

## 12. Git 규칙

### 12-1. 브랜치와 푸시

- `main` 하나만 쓰고 PR 없이 직접 푸시합니다. **force push는 금지**이고, 잘못 올렸으면 `git revert`로 되돌립니다.
- 푸시 전 체크리스트
  1. `git pull --rebase`
  2. `uv sync`
  3. `uv run ruff check .` / `uv run ruff format .`
  4. `src/`나 `tests/`를 바꿨다면 `uv run pytest -m "not integration"`
  5. `git status`, `git diff --staged`로 올릴 내용 확인
  6. 의존성을 바꿨다면 `pyproject.toml`과 `uv.lock`을 같이 커밋
  7. 공통 코드(`storage/`, `contracts/`, `pipeline/`)를 바꿨다면 팀에 공유
- `.env`, `results/`, `artifacts/retrieval_a/sparse_cache/`, `artifacts/corpus/**/history/`는 커밋하지 않습니다.

### 12-2. 커밋 메시지

`타입: 한 줄 설명` 형식으로, 설명은 한글로 짧게 씁니다. 범위를 밝히고 싶으면 `feat(retrieval_b): …`처럼 괄호에 적습니다.

| 타입 | 언제 | 예시 |
|---|---|---|
| `feat` | 새 기능이나 코드 추가 | `feat: PDF 페이지별 청킹 함수 추가` |
| `fix` | 버그 수정 | `fix: BM25 한국어 토크나이저 오류 수정` |
| `docs` | 문서만 수정 | `docs: 폴더 구조 설계 문서 추가` |
| `test` | 테스트만 추가·수정 | `test: RRF 합산 테스트 추가` |
| `chore` | 설정, 의존성, `.gitignore`, config 변경 | `chore: .gitignore 및 .env.example 추가` |
| `refactor` | 동작은 같고 구조만 정리 | `refactor: retriever 인터페이스 분리` |
