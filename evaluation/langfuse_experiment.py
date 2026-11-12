"""Langfuse 실험 한 번: 데이터셋 질문 → 팀 chain(retrieval_a 또는 --impl retrieval_b) → trace + 검색·RAGAS 점수 → Langfuse.

답변 점수는 나눠서 매긴다 (발제 STEP 6 "RAGAS와 LLM-as-a-Judge 실행 방법")
- RAGAS 4개: 이 스크립트가 quality.py로 직접 계산해 올린다. Faithfulness는 RAGAS 공식 예제대로
  user_input=질문, response=답변, retrieved_contexts=실제 최종 문맥.
- Judge 4항목: Langfuse 서버 evaluator가 채점한다 (팀 기획). 서버 규칙이 이 스크립트가 만든
  experiment-item-run observation을 골라 judge_* 점수를 붙인다. 집계는 evaluation/answer/server_judge_summary.py.
- --code-answer-scores면 Judge도 코드로 채점한다 (서버 결과와 비교할 때만. 그동안 서버 규칙을 끈다).

누가 무엇을 부르나
    이 스크립트(노트북) ─ Langfuse SDK run_experiment
        ├─ 데이터셋 문항마다 task          = 팀 chain 실행 (pipeline/chain.py create_chain 그대로, 최종 문맥은 콜백으로 받음)
        ├─ 문항마다 evaluators              = retrieval_judgement (evaluation/retrieval/retrieval_scores.py)
        │                                     answer_quality   (evaluation/answer/quality.py: RAGAS 4개, --code-answer-scores면 Judge도)
        ├─ 실행이 끝나면 run_evaluators    = run_summary (지표별 평균 + 건수 + 채점 토큰)
        └─ trace와 점수를 Langfuse 서버로 보낸다. Judge 점수는 서버 evaluator 규칙이 따로 붙인다.

Langfuse에 올라가는 점수 이름
    문항 점수 (scored일 때, NUMERIC)
        hit_at_5, recall_at_5, mrr_at_5                                            retrieval_scores.py (문서 단위)
        faithfulness, answer_relevancy, context_precision, context_recall          0~1 (--no-ragas면 없음)
        judge_facts, judge_conditions, judge_evidence, judge_decision              0 또는 1 (--code-answer-scores일 때만.
                                                                                    평소에는 서버 evaluator가 같은 이름으로 붙임)
    해당 없음·평가 불가 (CATEGORICAL)
        <이름>_state = "na" | "unavailable", comment에 이유. 0이나 1로 바꿔 올리지 않는다
    실행 점수 (run 단위)
        <이름>_mean, comment에 "평가 대상 / 해당 없음 / 평가 불가" 건수와 답변·되묻기·보류별 평균
        grading_tokens_input, grading_tokens_output (코드 채점 호출만. 서비스 토큰은 chain trace에, 서버 Judge는 evaluator 실행 기록에 따로 남는다)

데이터셋: evaluation/upload_langfuse_datasets.py가 올린 {domain}-{difficulty}-{split}-v1 (예: shipping-basic-development-v1)
    input = {"question"}, expected_output = 정답 파일 칸 (채점에만 쓴다. chain에는 질문만 들어간다)

실행 (레포 맨 위에서, .env에 OPENAI_*, LANGFUSE_*, QDRANT_* 필요)
    uv run python evaluation/langfuse_experiment.py --modes hybrid_rerank --limit 2      # 먼저 2문항으로 확인
    uv run python evaluation/langfuse_experiment.py                                      # 세 방식 × development 전체
    uv run python evaluation/langfuse_experiment.py --split additional --page-context --tag page
    uv run python evaluation/langfuse_experiment.py --impl retrieval_b                   # retrieval_b 검색기로 같은 비교
    uv run python evaluation/langfuse_experiment.py --no-ragas --limit 2                 # 채점 비용 없이 검색 점수만
    uv run python evaluation/langfuse_experiment.py --code-answer-scores --limit 2       # Judge도 코드로 (서버 규칙을 끄고 비교할 때만)

로컬 사본: results/langfuse/{run_name}.json (요약·채점 토큰·Langfuse 링크 url). results/는 커밋하지 않는다.

한계: client.py의 사전 판단(query guard)·답변 검사(answer guard)는 거치지 않고 chain만 부른다
(검색 질문 = 원래 질문, 나눠 찾기 없음).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODES = ("naive", "hybrid", "hybrid_rerank")
IMPLS = ("retrieval_a", "retrieval_b")  # pipeline/chain.py IMPLS와 같게
DEFAULT_IMPL = "retrieval_a"
IMPL_PREFIX = {"retrieval_a": "a", "retrieval_b": "b"}
SPLITS = ("development", "additional", "expansion_validation")
EXPERIMENT = "kant-rag-eval"
MAX_CONCURRENCY = 4
SOURCES_MARK = "\n\n\n[출처]"

Record = dict[str, Any]


def _load(rel: str, name: str):
    """evaluation/ 아래 파일을 불러온다 (evaluation/은 패키지가 아니라서)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "evaluation" / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


quality = _load("answer/quality.py", "answer_quality")
retrieval_scores = _load("retrieval/retrieval_scores.py", "retrieval_scores")
score_configs = _load("register_score_configs.py", "register_score_configs")


def dataset_name(domain: str, difficulty: str, split: str) -> str:
    """upload_langfuse_datasets.py와 같은 규칙."""
    return f"{domain}-{difficulty}-{split.replace('_', '-')}-v1"


def _field(item: Any, key: str) -> Any:
    """Langfuse DatasetItem(속성)과 로컬 dict 문항 둘 다 받는다."""
    return item.get(key) if isinstance(item, dict) else getattr(item, key, None)


# ---------- task: chain 실행 → 답변 기록 ----------


DOCS_STEP = "retrieve_docs"  # pipeline/chain.py create_chain 안의 검색 단계 함수 이름


def _docs_capture() -> Any:
    """chain을 고치지 않고, 검색 단계(retrieve_docs)가 돌려준 최종 문맥을 콜백으로 받아 둔다.

    chain.py는 답(RagAnswer)만 돌려주고 docs는 안에서만 쓴다. LangChain은 chain 안의 단계마다
    on_chain_start(name=함수 이름)·on_chain_end(출력)를 부르므로, 이름이 retrieve_docs인 단계의 출력을 잡는다.
    나눠 찾기·page 확장 뒤의 결과이고, 모델 프롬프트에 들어간 문맥과 같다.
    """
    from langchain_core.callbacks import BaseCallbackHandler

    class DocsCapture(BaseCallbackHandler):
        def __init__(self) -> None:
            self.run_ids: set[Any] = set()
            self.docs: list[Any] | None = None

        def on_chain_start(self, serialized, inputs, *, run_id, **kwargs):
            if kwargs.get("name") == DOCS_STEP:
                self.run_ids.add(run_id)

        def on_chain_end(self, outputs, *, run_id, **kwargs):
            if run_id in self.run_ids:
                self.docs = list(outputs or [])

    return DocsCapture()


async def invoke_with_docs(chain: Any, request: Mapping[str, Any]) -> dict[str, Any]:
    """팀 chain을 그대로 부르고 {"answer": RagAnswer, "docs": 최종 문맥}을 돌려준다.

    콜백은 chain에 이미 붙은 Langfuse 콜백에 더해진다 (서비스 trace는 그대로 남는다).
    검색 단계를 못 찾으면(chain.py에서 함수 이름이 바뀐 경우) 빈 문맥으로 채점하지 않고 오류로 남긴다.
    """
    capture = _docs_capture()
    answer = await chain.ainvoke(dict(request), config={"callbacks": [capture]})
    if capture.docs is None:
        raise RuntimeError(
            f"chain에서 검색 단계({DOCS_STEP})의 결과를 찾지 못함. "
            "pipeline/chain.py의 함수 이름이 바뀌었는지 확인"
        )
    return {"answer": answer, "docs": capture.docs}


def record_from_chain(question: str, output: Mapping[str, Any]) -> Record:
    """invoke_with_docs 결과 → 채점용 기록 (run_answers.py 기록과 같은 모양).

    answer는 [출처] 목록을 뗀 본문. contexts는 모델이 실제로 받은 순서 그대로 S1, S2 ….
    """
    answer = output["answer"]
    contexts = [
        {
            "sid": f"S{i}",
            "book_id": doc.metadata.get("book_id"),
            "page": doc.metadata.get("page"),
            "chunk_id": doc.metadata.get("chunk_id"),
            "text": doc.page_content,
        }
        for i, doc in enumerate(output.get("docs") or [], start=1)
    ]
    return {
        "question": question,
        "status": answer.status,
        "answer": (answer.answer or "").split(SOURCES_MARK)[0].strip(),
        "reason": answer.reason or "",
        "source_ids": list(answer.source_ids or []),
        "contexts": contexts,
        "error": None,
    }


def error_record(question: str, error: BaseException) -> Record:
    """서비스 오류. 문항을 버리지 않고 남겨서 채점에서 평가 불가로 센다 (발제 "해당 없음과 오류")."""
    return {
        "question": question,
        "status": "error",
        "answer": "",
        "reason": "",
        "source_ids": [],
        "contexts": [],
        "error": f"{type(error).__name__}: {error}",
    }


def make_task(
    chain: Any, *, mode: str, page_context: bool = False
) -> Callable[..., Any]:
    """Langfuse task: 데이터셋 문항의 질문만 chain에 넣는다 (정답 칸은 넣지 않는다)."""

    async def task(*, item: Any, **kwargs: Any) -> Record:
        question = (_field(item, "input") or {}).get("question", "")
        start = time.perf_counter()
        request = {
            "question": question,
            "retrieval_mode": mode,
            "page_context": page_context,
        }
        try:
            rec = record_from_chain(question, await invoke_with_docs(chain, request))
        except Exception as e:  # noqa: BLE001 — 서비스 오류도 기록으로 남긴다
            rec = error_record(question, e)
        rec["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
        return rec

    return task


# ---------- 점수 ↔ Langfuse Evaluation ----------


def to_evaluations(
    scores: Iterable[Mapping[str, Any]],
    *,
    status: str | None,
    config_ids: Mapping[str, str] | None = None,
) -> list[Any]:
    """점수 dict → Langfuse Evaluation. scored면 NUMERIC 값, 아니면 <이름>_state CATEGORICAL.

    config_ids({점수 이름: config_id})에 이름이 있으면 점수 설정과 연결한다 (register_score_configs.py).
    """
    from langfuse import Evaluation

    ids = config_ids or {}
    out = []
    for s in scores:
        meta = {"state": s["state"], "cause": s.get("cause"), "status": status}
        if s["state"] == "scored":
            out.append(
                Evaluation(
                    name=s["name"],
                    value=float(s["value"]),
                    comment=s.get("reason") or None,
                    metadata=meta,
                    data_type="NUMERIC",
                    config_id=ids.get(s["name"]),
                )
            )
        else:
            comment = f"{s.get('cause')}: {s.get('reason') or ''}".strip()
            out.append(
                Evaluation(
                    name=f"{s['name']}_state",
                    value=s["state"],
                    comment=comment,
                    metadata=meta,
                    data_type="CATEGORICAL",
                    config_id=ids.get(f"{s['name']}_state"),
                )
            )
    return out


def from_evaluations(evaluations: Iterable[Any]) -> list[Record]:
    """to_evaluations의 반대 (run_summary가 문항 점수를 다시 모을 때)."""
    out = []
    for e in evaluations:
        meta = getattr(e, "metadata", None) or {}
        state = meta.get("state")
        if state not in ("scored", "na", "unavailable"):
            continue  # 우리 평가기가 만든 점수가 아님
        name = e.name if state == "scored" else e.name.removesuffix("_state")
        value = e.value if state == "scored" else None
        out.append(
            {
                "name": name,
                "value": value,
                "state": state,
                "cause": meta.get("cause"),
                "reason": getattr(e, "comment", "") or "",
            }
        )
    return out


def make_evaluators(
    scorer: Any | None,
    *,
    retrieval_k: int | None = retrieval_scores.K,
    config_ids: Mapping[str, str] | None = None,
) -> list[Callable[..., Any]]:
    """문항 평가기 목록. 함수 이름이 Langfuse 화면의 평가기 이름이 된다.

    scorer가 None이면(--no-ragas) 답변 점수를 매기지 않는다. retrieval_k가 None이면 검색 점수도 없다.
    """

    async def answer_quality(
        *, input: Any, output: Any, expected_output: Any, metadata: Any, **kwargs: Any
    ):
        rec = dict(output or {})
        rec.setdefault("question", (input or {}).get("question", ""))
        scores = await scorer.score(rec, expected_output or None)
        return to_evaluations(scores, status=rec.get("status"), config_ids=config_ids)

    def retrieval_judgement(
        *, input: Any, output: Any, expected_output: Any, metadata: Any, **kwargs: Any
    ):
        rec = dict(output or {})
        scores = retrieval_scores.score_retrieval(
            rec, expected_output or None, k=retrieval_k
        )
        return to_evaluations(scores, status=rec.get("status"), config_ids=config_ids)

    evaluators: list[Callable[..., Any]] = [] if scorer is None else [answer_quality]
    if retrieval_k is not None:  # --no-retrieval이면 검색 판정을 부르지 않는다
        evaluators.append(retrieval_judgement)
    return evaluators


def summarize_items(item_results: Iterable[Any]) -> dict[str, dict[str, Any]]:
    """실험 결과의 문항들 → 지표별 요약 (quality.summarize)."""
    rows = []
    for r in item_results:
        output = r.output if isinstance(r.output, dict) else {}
        rows.append(
            (output.get("status") or "error", from_evaluations(r.evaluations or []))
        )
    return quality.summarize(rows)


def run_comment(m: Mapping[str, Any]) -> str:
    def fmt(st: str) -> str:
        v = (m["by_status"].get(st) or {}).get("mean")
        return "-" if v is None else f"{v:.3f}"

    return (
        f"평가 대상 {m['n_scored']} · 해당 없음 {m['n_na']} · 평가 불가 {m['n_unavailable']} | "
        f"답변 {fmt('answered')} · 되묻기 {fmt('needs_clarification')} · 보류 {fmt('abstained')}"
    )


def make_run_evaluator(scorer: Any | None) -> Callable[..., Any]:
    """실행 단위 점수: 지표별 평균(평가 대상만) + 건수 comment + 채점 토큰 (코드 채점을 했을 때만)."""
    from langfuse import Evaluation

    def run_summary(*, item_results: Sequence[Any], **kwargs: Any) -> list[Any]:
        out = []
        for name, m in summarize_items(item_results).items():
            if (
                m["mean"] is None
            ):  # 평가 대상이 0건이면 평균을 만들지 않는다 (0으로 채우지 않음)
                continue
            out.append(
                Evaluation(
                    name=f"{name}_mean",
                    value=m["mean"],
                    comment=run_comment(m),
                    metadata=m,
                    data_type="NUMERIC",
                )
            )
        if scorer is None:
            return out
        usage = scorer.grading_usage()
        out.append(
            Evaluation(
                name="grading_tokens_input",
                value=usage["total"]["input"],
                metadata=usage,
            )
        )
        out.append(
            Evaluation(
                name="grading_tokens_output",
                value=usage["total"]["output"],
                metadata=usage,
            )
        )
        return out

    return run_summary


# ---------- 실행 ----------


def run_name_for(
    mode: str,
    *,
    page_context: bool,
    tag: str = "",
    now: datetime | None = None,
    impl: str = DEFAULT_IMPL,
) -> str:
    """실행 이름. retrieval_b면 앞에 b_를 붙인다 (retrieval_a는 예전 이름 그대로)."""
    stamp = (now or datetime.now().astimezone()).strftime("%Y%m%d-%H%M")
    prefix = "" if impl == DEFAULT_IMPL else f"{IMPL_PREFIX[impl]}_"
    name = f"{prefix}{mode}{'_page' if page_context else ''}_{stamp}"
    return f"{name}_{tag}" if tag else name


def save_local(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def run_mode(
    langfuse: Any, items: Sequence[Any], args: argparse.Namespace, mode: str
) -> dict[str, Any]:
    """방식 하나를 실험 한 번으로 돌린다. chain·채점기는 방식마다 새로 만든다 (실험마다 이벤트 루프가 새로 생김)."""
    from rag_service.pipeline import create_chain

    chain = create_chain(domain=args.domain, difficulty=args.difficulty, impl=args.impl,
                         top_k=getattr(args, "top_k", None))
    code_judge = bool(args.code_answer_scores)
    scorer = (
        None
        if args.no_ragas
        else quality.QualityScorer.create(args.model, with_judge=code_judge)
    )
    run_name = run_name_for(
        mode, page_context=args.page_context, tag=args.tag, impl=args.impl
    )
    k = None if args.no_retrieval else args.k
    result = langfuse.run_experiment(
        name=EXPERIMENT,
        run_name=run_name,
        description=f"{args.dataset} · {args.impl} · {mode} · "
        + (f"RAGAS 코드 채점 {args.model}" if scorer else "RAGAS 채점 안 함")
        + (" · Judge 코드 채점" if code_judge else " · Judge는 Langfuse evaluator"),
        data=list(items),
        task=make_task(chain, mode=mode, page_context=args.page_context),
        evaluators=make_evaluators(
            scorer, retrieval_k=k, config_ids=score_configs.config_ids(langfuse)
        ),
        run_evaluators=[make_run_evaluator(scorer)],
        max_concurrency=args.concurrency,
        metadata={
            "impl": args.impl,
            "top_k": str(getattr(args, "top_k", None) or "default"),
            "mode": mode,
            "page_context": str(args.page_context),
            "ragas": "code" if scorer else "off",
            "judge": "code" if code_judge else "langfuse_evaluator",
            "grading_model": args.model if scorer else "",
            "dataset": args.dataset,
        },
    )
    summary = summarize_items(getattr(result, "item_results", []) or [])
    payload = {
        "run_name": run_name,
        "dataset": args.dataset,
        "impl": args.impl,
        "top_k": getattr(args, "top_k", None),
        "mode": mode,
        "page_context": args.page_context,
        "ragas": "code" if scorer else "off",
        "judge": "code" if code_judge else "langfuse_evaluator",
        "grading_model": args.model if scorer else None,
        "n_items": len(items),
        "summary": summary,
        "grading_usage": scorer.grading_usage() if scorer else None,
        "url": getattr(result, "dataset_run_url", None)
        or getattr(result, "experiment_url", None),
    }
    save_local(ROOT / "results" / "langfuse" / f"{run_name}.json", payload)
    print(f"\n[{args.impl} · {mode}] {run_name}")
    print(quality.report(summary))
    print(f"Langfuse: {payload['url'] or '(링크 없음)'}")
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    p = argparse.ArgumentParser(
        description="Langfuse 실험: 팀 chain 실행 trace와 검색 점수를 올린다 (답변 점수는 Langfuse 서버 evaluator)"
    )
    p.add_argument("--domain", default="shipping")
    p.add_argument("--difficulty", default="basic")
    p.add_argument("--split", default="development", choices=SPLITS)
    p.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    p.add_argument(
        "--impl",
        default=DEFAULT_IMPL,
        choices=IMPLS,
        help="검색기 구현 (같은 chain·프롬프트·채점으로 retrieval_a와 retrieval_b를 비교)",
    )
    p.add_argument(
        "--page-context",
        action="store_true",
        help="찾은 문단·표 대신 그 쪽 전체를 문맥으로 (chain 옵션)",
    )
    p.add_argument(
        "--code-answer-scores",
        action="store_true",
        help="Judge 4항목도 코드로 채점해 올린다. 평소에는 서버 evaluator가 채점. 서버 규칙을 끄고 비교할 때만",
    )
    p.add_argument(
        "--no-ragas",
        action="store_true",
        help="RAGAS 채점을 하지 않는다 (채점 비용 없이 검색 점수만 볼 때)",
    )
    p.add_argument(
        "--model",
        default=quality.MODEL,
        help="코드 채점 LLM (RAGAS, --code-answer-scores면 Judge도. 모든 방식에 같게)",
    )
    p.add_argument(
        "--k",
        type=int,
        default=None,
        help="검색 판정 k (모든 방식에 같게). 기본: --top-k가 있으면 그 값, 없으면 5",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="답변에 넣을 최종 문맥 수 (검색기 top_k 덮어쓰기, k 비교 실험용). 기본: 설정 파일·Settings 값",
    )
    p.add_argument(
        "--no-retrieval", action="store_true", help="검색 판정 점수를 올리지 않는다"
    )
    p.add_argument("--concurrency", type=int, default=MAX_CONCURRENCY)
    p.add_argument("--limit", type=int, default=0, help="앞에서 N문항만 (0이면 전부)")
    p.add_argument("--tag", default="", help="실행 이름 끝에 붙일 말")
    args = p.parse_args(argv)
    if args.no_ragas and args.code_answer_scores:
        p.error("--no-ragas와 --code-answer-scores는 같이 쓸 수 없습니다")
    if args.k is None:  # 검색 판정 k는 실제로 넣은 문맥 수와 같게
        args.k = args.top_k or retrieval_scores.K
    args.dataset = dataset_name(args.domain, args.difficulty, args.split)

    from langfuse import get_client

    langfuse = get_client()
    try:
        if not langfuse.auth_check():
            raise SystemExit("Langfuse 연결 설정(.env의 LANGFUSE_*)을 확인해 주세요.")
        items = list(langfuse.get_dataset(name=args.dataset).items)
        if args.limit:
            items = items[: args.limit]
        print(
            f"[데이터셋] {args.dataset} {len(items)}문항 · {args.impl} · 방식 {', '.join(args.modes)}",
            flush=True,
        )
        for mode in args.modes:
            run_mode(langfuse, items, args, mode)
    finally:
        langfuse.flush()


if __name__ == "__main__":
    main()
