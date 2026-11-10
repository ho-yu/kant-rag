"""답변 평가 1단계: 같은 질문을 세 검색 방식으로 실행하고, 질문 ID별 결과를 저장한다.

뒤 단계(faithfulness.py, judge.py, summary.py)는 모두 이 파일이 만든 jsonl만 읽는다.

- 답변은 chain.py와 같은 부품으로 만든다: 검색기 → format_docs → answer_prompt → gpt-6-luna(RagAnswer).
  chain은 검색 문서를 돌려주지 않아서, 실제 최종 문맥(모델에 넣은 청크)을 남기려고 부품을 직접 잇는다.
- 사전 판단(질문 재작성·나눠 찾기), 쪽 확장, 답변 사후 검사는 쓰지 않는다. config.json에 적어 둔다.
- 질문은 {split}_questions.json에서 id와 question만 읽는다. 정답 파일은 이 단계에서 열지 않는다 (발제 STEP 6).
- 오류가 난 질문도 status "error"로 한 줄 남긴다. 다시 실행해도 이미 저장한 질문(오류 포함)은 건너뛴다.

실행 (레포 맨 위에서)
    uv run python evaluation/answer/run_answers.py --modes naive --limit 1   # 1문항만 시험 (API 1회)
    uv run python evaluation/answer/run_answers.py                           # 30문항 × 3방식

결과 (results/는 .gitignore에 없으니 커밋하지 않는다)
    results/answer/retrieval_b/{난이도}_{split}/answers_{방식}.jsonl   한 줄 = 요청 하나
    results/answer/retrieval_b/{난이도}_{split}/config.json            세 방식에 같게 둔 설정
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.documents import Document

from rag_service.storage.corpus import repo_root

MODEL = "gpt-6-luna"
TIMEOUT_S = 60  # 발제 STEP 7 "호출 한도와 오류": 실행 전에 정해 둔다
MAX_RETRIES = 2  # ChatOpenAI 안에서 하는 재시도. 재시도는 별도 요청으로 세지 않는다
WARMUP_QUESTION = "부산에서 LA로 컨테이너를 예약할 때 필요한 정보는?"
MODES = ("naive", "hybrid", "hybrid_rerank")  # chain.py의 retrieval_mode와 같은 이름
SPLITS = ("development", "additional", "expansion_validation")
OUT_DIR = Path("results") / "answer" / "retrieval_b"
PIPELINE_NOTE = (
    "검색기 → format_docs → answer_prompt → 모델(RagAnswer). "
    "사전 판단(질문 재작성·나눠 찾기), 쪽 확장, 답변 사후 검사는 쓰지 않음"
)

Record = dict[str, Any]


# ---------- 경로와 질문 ----------


def out_dir_for(difficulty: str, split: str) -> Path:
    return repo_root() / OUT_DIR / f"{difficulty}_{split}"


def answers_path(out_dir: Path, mode: str) -> Path:
    return out_dir / f"answers_{mode}.jsonl"


def questions_path(split: str, *, domain: str, difficulty: str) -> Path:
    if split not in SPLITS:
        raise ValueError(f"split은 {SPLITS} 중 하나여야 합니다: {split!r}")
    return (
        repo_root()
        / "data"
        / difficulty
        / domain
        / "evaluation"
        / f"{split}_questions.json"
    )


def load_questions(path: str | Path) -> list[dict[str, str]]:
    """질문 파일에서 id와 question만 꺼낸다. 다른 키가 있어도 버린다."""
    path = Path(path)
    items = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(items, list):
        raise TypeError(f"{path.name}: 질문 목록(list)이어야 합니다")
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for n, item in enumerate(items, start=1):
        qid, question = item.get("id"), item.get("question")
        if not qid or not isinstance(question, str) or not question.strip():
            raise ValueError(f"{path.name} {n}번째 항목: id와 question이 필요합니다")
        if qid in seen:
            raise ValueError(f"{path.name}: id가 겹칩니다: {qid!r}")
        seen.add(qid)
        out.append({"id": qid, "question": question})
    return out


# ---------- 검색기와 답변 체인 (무거운 import는 함수 안에서) ----------


def build_retriever(mode: str, *, domain: str, difficulty: str):
    """retrieval_b 검색기. k·levels는 Settings 기본값을 써서 세 방식이 같게 한다."""
    from rag_service.retrieval_b import (
        build_hybrid_retriever,
        build_naive_retriever,
        build_rerank_retriever,
    )

    builders = {
        "naive": build_naive_retriever,
        "hybrid": build_hybrid_retriever,
        "hybrid_rerank": build_rerank_retriever,
    }
    if mode not in builders:
        raise ValueError(f"mode는 {MODES} 중 하나여야 합니다: {mode!r}")
    return builders[mode](domain=domain, difficulty=difficulty)


def make_answer_chain(model: str = MODEL):
    """chain.py create_chain의 answer 단계와 같은 것. timeout·재시도만 더한다."""
    from langchain_openai import ChatOpenAI

    from rag_service.contracts.schemas import RagAnswer
    from rag_service.generation.prompt import answer_prompt

    llm = ChatOpenAI(model=model, timeout=TIMEOUT_S, max_retries=MAX_RETRIES)
    return answer_prompt | llm.with_structured_output(RagAnswer)


def default_format_context(docs: Sequence[Document]) -> str:
    """chain.py의 format_docs를 그대로 쓴다 ([S1] SB01.pdf / page 1 ...)."""
    from rag_service.pipeline.chain import format_docs

    return format_docs(list(docs))


# ---------- 기록 한 줄 만들기 ----------


def context_records(docs: Iterable[Document]) -> list[dict[str, Any]]:
    """모델에 넣은 문맥을 S번호와 함께 남긴다. format_docs의 [S1], [S2] 순서와 같다."""
    out = []
    for i, d in enumerate(docs, start=1):
        m = d.metadata or {}
        out.append(
            {
                "sid": f"S{i}",
                "chunk_id": m.get("chunk_id"),
                "book_id": m.get("book_id"),
                "page": m.get("page"),
                "level": m.get("level"),
                "text": d.page_content,
            }
        )
    return out


def usage_totals(usage_metadata: dict[str, Any] | None) -> dict[str, int] | None:
    """get_usage_metadata_callback이 모은 모델별 사용량을 합친다. 기록이 없으면 None (= 확인 불가)."""
    if not usage_metadata:
        return None
    return {
        "input": sum(int(u.get("input_tokens") or 0) for u in usage_metadata.values()),
        "output": sum(
            int(u.get("output_tokens") or 0) for u in usage_metadata.values()
        ),
    }


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def run_one(
    qid: str,
    question: str,
    *,
    mode: str,
    retriever: Any,
    answer_chain: Any,
    format_context: Callable[[Sequence[Document]], str] = default_format_context,
) -> Record:
    """질문 하나 = 요청 하나. 성공이든 실패든 기록 한 줄을 돌려준다 (예외를 밖으로 던지지 않는다)."""
    rec: Record = {
        "id": qid,
        "mode": mode,
        "question": question,
        "status": "error",
        "answer": "",
        "reason": "",
        "source_ids": [],
        "contexts": [],
        "retrieval_ms": None,
        "latency_ms": None,
        "tokens": None,
        "error": None,
        "error_stage": None,
    }
    stage = "retrieval"
    start = time.perf_counter()
    with get_usage_metadata_callback() as cb:
        try:
            docs = list(retriever.invoke(question))
            rec["retrieval_ms"] = _ms(start)
            rec["contexts"] = context_records(docs)
            stage = "generation"
            answer = answer_chain.invoke(
                {"question": question, "context": format_context(docs)}
            )
            rec.update(
                status=answer.status,
                answer=answer.answer,
                reason=answer.reason,
                source_ids=list(answer.source_ids),
            )
        except Exception as e:  # noqa: BLE001 — 한 질문의 실패로 전체 실행을 멈추지 않는다
            rec.update(
                status="error", error=f"{type(e).__name__}: {e}", error_stage=stage
            )
        rec["tokens"] = usage_totals(cb.usage_metadata)
    rec["latency_ms"] = _ms(start)
    return rec


# ---------- 저장과 이어 하기 ----------


def read_jsonl(path: Path) -> list[Record]:
    """jsonl을 읽는다. 파일이 없으면 빈 목록, 깨진 줄(쓰다 끊긴 줄)은 건너뛴다."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def done_ids(path: Path) -> set[str]:
    """이미 저장한 질문 id. 오류 줄도 완료로 본다 (재시도 결과로 덮어쓰지 않는다)."""
    return {r["id"] for r in read_jsonl(path) if r.get("id")}


def run_mode(
    mode: str,
    questions: Sequence[dict[str, str]],
    path: Path,
    *,
    retriever_factory: Callable[[], Any],
    answer_chain: Any,
    format_context: Callable[[Sequence[Document]], str] = default_format_context,
    log: Callable[[str], None] = print,
) -> list[Record]:
    """한 방식으로 아직 안 한 질문만 실행하고, 한 줄씩 바로 저장한다. 새로 만든 기록을 돌려준다."""
    done = done_ids(path)
    todo = [q for q in questions if q["id"] not in done]
    log(f"[{mode}] 저장됨 {len(questions) - len(todo)}, 새로 실행 {len(todo)}")
    if not todo:
        return []

    retriever = retriever_factory()
    # 모델·색인 로딩과 첫 import 시간이 첫 질문의 응답 시간에 섞이지 않게 한 번 먼저 부른다 (저장하지 않음)
    format_context(list(retriever.invoke(WARMUP_QUESTION)))

    path.parent.mkdir(parents=True, exist_ok=True)
    new: list[Record] = []
    with path.open("a", encoding="utf-8") as fh:
        for i, q in enumerate(todo, start=1):
            rec = run_one(
                q["id"],
                q["question"],
                mode=mode,
                retriever=retriever,
                answer_chain=answer_chain,
                format_context=format_context,
            )
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            new.append(rec)
            log(
                f"[{mode}] {i}/{len(todo)} {q['id']} {rec['status']} {rec['latency_ms'] / 1000:.1f}s"
            )
    return new


# ---------- 실행 설정 ----------


def run_config(
    *, split: str, domain: str, difficulty: str, model: str
) -> dict[str, Any]:
    """세 방식에 같게 둔 설정 (발제 STEP 6 "비교 조건 맞추기")."""
    from rag_service.retrieval_b.dense import DEFAULT_EMBEDDING_MODEL
    from rag_service.retrieval_b.rerank import model_name_from_env
    from rag_service.retrieval_b.settings import get_settings

    s = get_settings()
    return {
        "split": split,
        "domain": domain,
        "difficulty": difficulty,
        "generation": {
            "model": model,
            "prompt": "generation/prompt.py answer_prompt",
            "output": "contracts/schemas.py RagAnswer",
            "context_format": "pipeline/chain.py format_docs",
            "timeout_s": TIMEOUT_S,
            "max_retries": MAX_RETRIES,
        },
        "retrieval": {
            "impl": "retrieval_b",
            "k": s.k,
            "levels": list(s.levels),
            "candidate_k": s.candidate_k,
            "rrf_k": s.rrf_k,
            "rerank_n": s.rerank_n,
            "rerank_max_length": s.rerank_max_length,
            "embedding_model": (os.getenv("EMBEDDING_MODEL") or "").strip()
            or DEFAULT_EMBEDDING_MODEL,
            "reranker_model": model_name_from_env(),
        },
        "pipeline": PIPELINE_NOTE,
    }


def check_or_write_config(out_dir: Path, config: dict[str, Any]) -> None:
    """처음이면 config.json을 쓰고, 이미 있으면 설정이 같은지 본다.

    설정이 바뀐 채로 이어 돌리면 한 파일 안에 다른 조건의 답변이 섞인다. 그래서 멈춘다.
    """
    path = out_dir / "config.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        old_core = {k: v for k, v in old.items() if k != "created_at"}
        if old_core != config:
            raise SystemExit(
                f"{path}의 설정이 지금 설정과 다릅니다. "
                "같은 조건으로 이어 돌리려면 설정을 되돌리고, 새로 돌리려면 이 폴더를 지우세요."
            )
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    stamped = {"created_at": datetime.now(UTC).isoformat(timespec="seconds"), **config}
    path.write_text(json.dumps(stamped, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------- 진입점 ----------


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv(repo_root() / ".env")
    p = argparse.ArgumentParser(
        description="답변 평가 1단계: 세 방식으로 답변 생성·저장"
    )
    p.add_argument("--split", default="development", choices=SPLITS)
    p.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    p.add_argument("--domain", default="shipping")
    p.add_argument("--difficulty", default="basic")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--limit", type=int, default=0, help="앞에서 N문항만 (0이면 전부)")
    args = p.parse_args(argv)

    questions = load_questions(
        questions_path(args.split, domain=args.domain, difficulty=args.difficulty)
    )
    if args.limit:
        questions = questions[: args.limit]
    out_dir = out_dir_for(args.difficulty, args.split)
    check_or_write_config(
        out_dir,
        run_config(
            split=args.split,
            domain=args.domain,
            difficulty=args.difficulty,
            model=args.model,
        ),
    )
    answer_chain = make_answer_chain(args.model)
    for mode in args.modes:
        path = answers_path(out_dir, mode)
        run_mode(
            mode,
            questions,
            path,
            retriever_factory=lambda m=mode: build_retriever(
                m, domain=args.domain, difficulty=args.difficulty
            ),
            answer_chain=answer_chain,
        )
        print(f"저장: {path.relative_to(repo_root())}")


if __name__ == "__main__":
    main()
