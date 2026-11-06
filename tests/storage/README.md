# tests/storage — 공통 기능 테스트

`src/rag_service/storage/`(검색·생성 코드가 import하는 공통 기능)가 망가지지 않았는지 확인한다.
팀원 검색 코드가 Qdrant 컬렉션을 제대로 읽을 수 있는지가 핵심이다.

## 언제 돌리나

- `storage/qdrant_store.py`나 `storage/corpus.py`를 고친 뒤
- `contracts/chunk.py`(청크 형식, payload)를 고친 뒤
- `git pull`로 다른 사람 코드를 받은 뒤, 커밋하기 전

자동으로 돌지 않는다 (CI 없음). 직접 실행한다.

## 실행

```
uv run pytest tests/storage -q
```

전부 `passed`면 정상. PyTorch가 없으면 GPU 테스트는 `skipped`로 건너뛴다.
Qdrant Cloud와 bge-m3 다운로드 없이 돈다 (로컬 Qdrant 폴더 + 가짜 임베딩). 내 `artifacts/`와 Qdrant Cloud는 건드리지 않는다.

## 파일별 내용

### test_qdrant_store.py

| 확인하는 것 | 테스트 |
|---|---|
| LangChain(`QdrantVectorStore`)으로 우리 컬렉션을 읽어 `book_id`·`page`가 담긴 Document를 받음 | `test_langchain_vectorstore_reads_our_collection` |
| 컬렉션 이름 규칙 `kant_{주제}_{난이도}_bge-m3` | `test_collection_name_follows_topic_and_difficulty` |
| `level_filter`가 고른 레벨만 남김 | `test_level_filter_keeps_only_given_levels` |
| GPU가 없으면 CPU로 대체 | `test_device_falls_back_to_cpu_without_gpu` |

`storage/corpus.py`(jsonl 읽기)는 청킹 결과가 있어야 확인할 수 있어서 `tests/ingestion/test_chunking.py`의 `test_loader_reads_latest_only`에서 같이 확인한다.

## 전체 한 번에

청킹 테스트까지 모두 돌리려면 레포 맨 위 폴더에서:

```
uv run pytest
```
