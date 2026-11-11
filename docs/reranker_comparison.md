# 리랭커 비교 실험 (retrieval_a, 2026-11-11)

retrieval_a의 hybrid + rerank에서 리랭커 모델과 리랭커 입력 방식을 바꿔 검색 성능을 비교했다.
결론: **지금 설정(`BAAI/bge-reranker-v2-m3`, 머리말 포함 입력)을 유지한다.**

## 조건

| 항목 | 값 |
|---|---|
| 난이도 · 도메인 | basic · shipping |
| 질문 | development + additional + expansion_validation 72문항 중 근거가 있는 68문항 (근거 없는 4문항 제외: S-DEV-12, S-DEV-27, S-FINAL-12, S-FINAL-29) |
| 검색 | 임베딩·sparse bge-m3, hybrid 후보 50개 → 리랭커 → 상위 5개 |
| 지표 | `rag_service.retrieval_a.eval_retrieval` (정답 **문장**이 청크에 들어 있는지 기준) |
| 1문항 | 1/68 ≈ 0.015 |

지표 뜻
- Hit@k: 상위 k개 안에 정답 문장이 든 청크가 있는 문항 비율
- MRR: 정답 청크가 처음 나온 순위의 역수 평균
- PageHit@1: 1위 청크가 정답 쪽(page)인 문항 비율

이 지표는 발제 STEP 6의 문서 단위 Hit@k·Recall@k(`evaluation/retrieval/retrieval_scores.py`)와 다르다.
리랭커는 같은 문서 안에서 어느 청크를 위로 올리느냐를 바꾸므로, 모델 고르기에는 문장 기준 지표가 차이를 더 잘 보여 준다.

## 리랭커가 개선할 수 있는 몫

| | 값 | 문항 수 |
|---|---|---|
| hybrid Hit@50 (후보 50개 안에 정답이 있음 = 리랭커의 최대치) | 0.632 | 43 |
| rerank Hit@5 (지금 리랭커) | 0.368 | 25 |
| 후보 안에 정답이 있는데 5위 밖 → 리랭커로 되찾을 수 있는 몫 | 0.265 | 18 |
| 후보 50개에도 정답이 없음 → 검색·청킹 쪽 과제 | 0.368 | 25 |

## 결과

| 실험 | Hit@1 | Hit@3 | Hit@5 | MRR | PageHit@1 | 결과 폴더 |
|---|---|---|---|---|---|---|
| **지금: `BAAI/bge-reranker-v2-m3`, 머리말 포함** | **0.265** | **0.338** | 0.368 | 0.301 | **0.324** | `artifacts/retrieval_a/eval_bge_m3` |
| `dragonkue/bge-reranker-v2-m3-ko` (지금 모델의 한국어 파인튜닝판) | **0.265** | **0.338** | **0.382** | **0.308** | 0.309 | `artifacts/retrieval_a/eval_rr_ko` |
| `Dongjin-kr/ko-reranker` (bge-reranker-large 기반 한국어) | 0.250 | 0.324 | **0.382** | 0.299 | 0.294 | `artifacts/retrieval_a/eval_rr_dongjin` |
| 지금 모델, 머리말 빼기 (`use_context: false`) | 0.235 | 0.324 | 0.368 | 0.282 | 0.265 | `artifacts/retrieval_a/eval_rr_nocontext` |
| 참고: hybrid (리랭커 없음) | 0.235 | 0.309 | 0.397 | 0.286 | 0.279 | `artifacts/retrieval_a/eval_bge_m3` |
| 참고: hybrid 상위 50개 | | | Hit@50 0.632 | | | `artifacts/retrieval_a/eval_hit50` |

## 해석

- **모델 교체**: 두 한국어 리랭커 모두 지금 모델과 ±1~2문항 차이. 우연 범위라 바꿀 이유가 없다 (다운로드 2GB대, 라이선스 관리만 늘어난다).
- **머리말 빼기**: 리랭커 입력에서 `[문서 제목 | 섹션 | 쪽]`과 문맥 요약을 빼면 PageHit@1이 4문항, Hit@1이 2문항 떨어진다. 머리말이 맞는 쪽을 고르는 데 도움이 된다 → `config/retrieval_a/rerank.yaml`의 `use_context: true` 유지.
- **공통 현상**: 세 모델 모두 rerank Hit@5가 hybrid Hit@5(0.397)보다 낮다. hybrid 5위 안에 있던 정답을 리랭커가 밀어내는 문항이 있다는 뜻이고, 모델이나 입력 형식으로는 바뀌지 않았다.

## 다음에 볼 것

1. rerank가 hybrid보다 못한 문항을 질문별로 보기: `evaluation/retrieval/rerank_diagnose.py`, 각 폴더의 `per_question.csv`
2. 후보 50개에 정답이 없는 25문항의 원인 (검색·청킹)
3. 리랭커 점수와 hybrid 점수를 섞어 순위 매기기 (코드 수정 필요)

## 결과 폴더 구성

- `summary.json`: 전체 평균
- `per_question.csv`: 질문별 결과
- `raw/{모드}_k{k}.jsonl`: 질문별 검색 결과 (상위 청크의 book_id, page, chunk_id)

## 재현 방법

레포 맨 위에서 가상환경을 켜고 실행한다. 공통 질문 인자:

```powershell
$G = "data/basic/shipping/evaluation/development.json","data/basic/shipping/evaluation/additional.json","data/basic/shipping/evaluation/expansion_validation.json"
```

**hybrid 상위 50개**

```powershell
python -m rag_service.retrieval_a.eval_retrieval --golden $G --profile bge-m3 --modes hybrid --k 50 --out artifacts/retrieval_a/eval_hit50
```

**다른 리랭커**: `config/retrieval_a/models.yaml`에 리랭커만 다른 프로필을 잠시 추가한다 (팀 설정에는 커밋하지 않았다).

```yaml
bge-m3-rr-ko:
  embedding: BAAI/bge-m3
  query_prefix: ""
  sparse: bge-m3
  reranker: dragonkue/bge-reranker-v2-m3-ko

bge-m3-rr-dongjin:
  embedding: BAAI/bge-m3
  query_prefix: ""
  sparse: bge-m3
  reranker: Dongjin-kr/ko-reranker
```

평가 스크립트는 오프라인 모드(`HF_HUB_OFFLINE=1`)로 돌기 때문에 모델을 먼저 받아 둔다.

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download('dragonkue/bge-reranker-v2-m3-ko'); snapshot_download('Dongjin-kr/ko-reranker')"
python -m rag_service.retrieval_a.eval_retrieval --golden $G --profile bge-m3-rr-ko --modes rerank --out artifacts/retrieval_a/eval_rr_ko
python -m rag_service.retrieval_a.eval_retrieval --golden $G --profile bge-m3-rr-dongjin --modes rerank --out artifacts/retrieval_a/eval_rr_dongjin
```

**머리말 빼기**: `config/retrieval_a/rerank.yaml`을 복사해 `reranker.use_context`만 `false`로 바꾼 설정으로 `build_rerank_retriever(..., config_path=복사본)`을 만들어 같은 질문으로 평가했다 (레포 설정 파일은 바꾸지 않았다).
