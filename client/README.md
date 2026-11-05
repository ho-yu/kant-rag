# kant-rag

해운(shipping) PDF 문서를 근거로 답하는 RAG 기반 Q&A 서비스입니다.
Naive RAG, 하이브리드 검색, 하이브리드 + 리랭킹 세 가지 방식을 비교합니다.

---

## 1. 로컬 개발 환경 세팅

처음 한 번만 하면 됩니다. 순서대로 따라 하세요.

### 1-1. 미리 설치할 것

| 도구 | 용도 | 확인 명령 |
|---|---|---|
| Git | 저장소 내려받기 | `git --version` |
| uv | 파이썬 버전과 패키지 관리 | `uv --version` |

Python은 따로 설치하지 않아도 됩니다. uv가 프로젝트에 맞는 Python 3.12를 자동으로 내려받습니다.

**uv 설치**

- macOS / Linux

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

- Windows (PowerShell)

  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

설치 후 터미널을 새로 열고 `uv --version`이 나오면 됩니다.

### 1-2. 저장소 내려받기

```bash
git clone https://github.com/ho-yu/kant-rag.git
cd kant-rag
```

PDF 데이터가 저장소에 포함되어 있어 내려받는 데 시간이 조금 걸립니다 (약 70MB).

### 1-3. 패키지 설치

```bash
uv sync
```

- `.venv/` 폴더에 가상환경이 만들어지고, `uv.lock`에 적힌 버전 그대로 설치됩니다. 팀원 모두 같은 버전을 쓰게 됩니다.
- 임베딩 모델용 PyTorch가 포함되어 있어 **처음 설치는 5분 이상 걸리고 디스크를 수 GB 사용합니다** (Linux는 GPU용 패키지까지 받아 약 6GB).
- 가상환경을 직접 활성화(`activate`)할 필요는 없습니다. 아래처럼 명령 앞에 `uv run`을 붙이면 됩니다.

> 패키지를 추가할 때는 `pip install` 대신 `uv add 패키지이름`을 사용하세요. `pyproject.toml`과 `uv.lock`이 함께 갱신되니 두 파일을 같이 커밋하면 됩니다.

### 1-4. 환경 변수(.env) 만들기

API 키 같은 비밀 값은 `.env` 파일에 넣습니다. 예시 파일을 복사해서 값을 채우세요.

```bash
# macOS / Linux
cp .env.example .env

# Windows (PowerShell)
Copy-Item .env.example .env
```

| 변수 | 설명 |
|---|---|
| `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` | 답변 생성 LLM 접속 정보 |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` | Langfuse 실행 기록 (클라우드: `https://cloud.langfuse.com`) |
| `QDRANT_URL`, `QDRANT_API_KEY` | Qdrant 클라우드를 쓸 때만 입력. 로컬 모드면 비워 둡니다 |

`.env`는 `.gitignore`에 등록되어 있어 커밋되지 않습니다. **API 키는 절대 커밋하거나 채팅에 붙여 넣지 마세요.**

### 1-5. 설치 확인

```bash
uv run python -c "import fastapi, langchain, langfuse, qdrant_client; print('설치 OK')"
uv run ruff check .
uv run pytest
```

`설치 OK`가 출력되고 ruff가 `All checks passed!`를 보여 주면 준비가 끝난 것입니다.
(지금은 테스트 코드가 없어서 pytest는 `no tests ran`으로 끝납니다. 정상입니다.)

---

## 2. 실행 방법

| 하고 싶은 일 | 명령 |
|---|---|
| 코드 스타일 검사 | `uv run ruff check .` |
| 자동 정리(포맷) | `uv run ruff format .` |
| 테스트 실행 | `uv run pytest` |
| 파이썬 스크립트 실행 | `uv run python 파일경로.py` |
| API 서버 실행 (API 코드 추가 후) | `uv run uvicorn rag_service.api.main:app --reload` |

API 서버를 띄운 뒤 브라우저에서 http://localhost:8000/docs 를 열면 FastAPI 문서 화면에서 질문을 보내 볼 수 있습니다.
(API 코드는 아직 작성 전입니다. 모듈 경로는 구현할 때 맞춰 수정하세요.)

---

## 3. 폴더 구조

```
kant-rag/
├── data/                 # 원본 데이터 (shipping만 사용)
│   ├── basic/shipping/          # PDF 15개 + 평가 JSON
│   ├── intermediate/shipping/   # PDF 50개 + 평가 JSON
│   └── advanced/shipping/       # PDF 120개 + 평가 JSON
├── src/rag_service/      # 서비스 코드 (ingestion, retrieval, generation, api 등)
├── config/               # 단계별 설정 파일
├── evaluation/           # 평가 코드 (검색, 답변, 보안, 벤치마크)
├── results/              # 비교표와 평가 결과
├── artifacts/            # 추출 텍스트, 색인 등 생성물
├── tests/                # 테스트 코드
├── pyproject.toml        # 의존성 목록
└── uv.lock               # 고정된 패키지 버전 (직접 수정하지 않음)
```

데이터 구성에 대한 설명은 [data/README.md](data/README.md)를 참고하세요.

---

## 4. 자주 겪는 문제

| 증상 | 해결 |
|---|---|
| `uv: command not found` | uv 설치 후 터미널을 새로 열어 주세요. |
| `uv sync`가 오래 걸림 | PyTorch 설치 때문입니다. 처음 한 번만 오래 걸립니다. |
| `import ragas`에서 `No module named 'langchain_community.chat_models.vertexai'` | ragas 0.4.3은 langchain-community 0.4 이상과 맞지 않습니다. `pyproject.toml`에서 `<0.4`로 고정해 두었으니 최신 코드를 받은 뒤 `uv sync`를 다시 실행하세요. |
| 패키지가 꼬인 것 같음 | `.venv` 폴더를 지우고 `uv sync`를 다시 실행하세요. |

---

## 5. Git 커밋 메시지 규칙

`타입: 한 줄 설명` 형식으로 설명은 한글로 짧게 쓴다.

| 타입 | 언제 | 예시 |
|---|---|---|
| `feat` | 새 기능이나 코드 추가 | `feat: PDF 페이지별 청킹 함수 추가` |
| `fix` | 버그 수정 | `fix: BM25 한국어 토크나이저 오류 수정` |
| `docs` | 문서만 수정 | `docs: 폴더 구조 설계 문서 추가` |
| `chore` | 설정, 의존성, `.gitignore`, config 변경 | `chore: .gitignore 및 .env.example 추가` |
| `refactor` | 동작은 같고 구조만 정리 | `refactor: retriever 인터페이스 분리` |
