# tests/ingestion — 청킹 코드 테스트

`src/rag_service/ingestion/`(청킹 전용 코드)이 망가지지 않았는지 확인한다.
**청킹 결과(artifacts의 jsonl)를 검사하는 게 아니라**, 레포에 있는 PDF(SB01, IS01 등)를 그 자리에서 직접 청킹해 정해진 결과가 나오는지 본다.

## 언제 돌리나

- `ingestion/` 코드나 `config/chunking.yaml`을 고친 뒤
- `git pull`로 다른 사람 코드를 받은 뒤
- 커밋하기 전

자동으로 돌지 않는다 (CI 없음). 직접 실행한다.

## 실행

```
uv run pytest tests/ingestion -q                       # 이 폴더 전체 (약 1분)
uv run pytest tests/ingestion/test_chunking.py -q      # 파일 하나만
uv run pytest tests/ingestion -q -k restore            # 이름에 restore가 들어간 테스트만
```

전부 `passed`면 정상. `failed`가 나오면 방금 바꾼 코드가 기존 기능을 건드린 것이다.
`data/` PDF가 없으면 PDF가 필요한 테스트는 `skipped`로 건너뛴다.

OpenAI API 키, Qdrant Cloud, bge-m3 다운로드가 **없어도** 돈다 (가짜 LLM·가짜 임베딩·로컬 Qdrant 폴더를 쓴다).
테스트는 임시 폴더에서 돌아서 내 `artifacts/`나 Qdrant Cloud를 건드리지 않는다.

## 파일별 내용

### test_chunking.py — 청크 만들기

| 확인하는 것 | 테스트 |
|---|---|
| 줄 잇기 (띄어쓰기 유지·추가) | `test_join_keeps_existing_space`, `test_join_after_punctuation_adds_space` |
| 청크 ID와 부모 연결, 쪽마다 page 청크 1개 | `test_ids_and_parents`, `test_one_page_record_per_page` |
| 문단이 쪽을 넘지 않음, 머리글·쪽 번호 제거 | `test_paragraphs_never_cross_pages`, `test_header_and_page_number_removed` |
| 긴 문단은 문장 경계에서만 분할 | `test_long_paragraph_split_on_sentence_boundary` |
| 표 청크와 row 청크, 표 각주 보존 | `test_table_and_rows`, `test_table_footnote_kept` |
| 90MB 넘으면 쪽 경계에서 파일 분할 | `test_big_document_split_on_page_boundary` |
| Qdrant 포인트 ID가 항상 같음 | `test_point_id_is_stable_uuid` |
| 맥락 검사 (지어낸 숫자 거부, 문서 ID·쪽 번호는 허용) | `test_validate_context_rejects_new_numbers`, `test_context_may_mention_doc_id_and_page_number` |
| 맥락 생성 흐름, 캐시만 있으면 API 키 없이 동작 | `test_context_generator_with_fake_llm`, `test_cached_context_needs_no_api_key` |
| 바뀐 문서만 다시 씀, 추가·삭제·되돌리기 | `test_only_changed_documents_are_rewritten`, `test_added_removed_and_restore` |
| jsonl 읽기는 최신본만 (history 제외) | `test_loader_reads_latest_only` |

### test_menu.py — 텍스트 메뉴

키 입력을 흘려 넣어 화면과 결과를 확인한다.

| 확인하는 것 | 테스트 |
|---|---|
| 청킹 메뉴에서 옵션 켜고 끄기, 실행 | `test_chunk_menu_toggles_and_runs` |
| 되돌리기 메뉴의 차이 미리보기와 복원 | `test_rollback_menu_shows_diff_and_restores` |
| 뒤로 가기, 잘못된 입력 | `test_back_and_invalid_input` |
| 번호가 밀려도 내용으로 비교 | `test_diff_ignores_shifted_ids` |

### test_index_sync.py — Qdrant 업로드

로컬 Qdrant 폴더와 가짜 임베딩으로 동기화 흐름만 확인한다.

| 확인하는 것 | 테스트 |
|---|---|
| 한 번에 청킹+업로드, 다시 돌리면 임베딩 0, 바뀐 문서만 임베딩 | `test_one_command_and_only_changes_are_synced` |
| 문단이 끼어들어 번호가 밀려도 기존 벡터 재사용 | `test_shifted_ids_reuse_vectors_and_only_page_changes` |
| `--index-only`, 올리는 레벨 변경 | `test_index_only_and_level_change` |
| 되돌리기 후 Qdrant도 같이 맞춰짐 | `test_restore_syncs_qdrant` |
