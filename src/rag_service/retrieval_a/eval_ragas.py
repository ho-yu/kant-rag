from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path

from rag_service.retrieval_a.naive import REPO_ROOT

log = logging.getLogger("eval_ragas")

MODES = ("naive", "hybrid", "rerank")
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
METRIC_LABEL = {
    "faithfulness": "Faithfulness (자료에 근거했나)",
    "answer_relevancy": "Answer Relevancy (질문에 맞는 답인가)",
    "context_precision": "Context Precision (쓸모 있는 자료가 위에 있나)",
    "context_recall": "Context Recall (필요한 내용을 다 가져왔나)",
}
SOURCES_MARK = "\n\n\n[출처]"
RAGAS_COLUMN = {"context_precision": "llm_context_precision_with_reference"}


def load_items(paths: list[str]) -> tuple[list[dict], list[str]]:
    items, skipped, seen = [], [], set()
    for path in paths:
        text = Path(path).read_text(encoding="utf-8-sig").strip()
        rows = json.loads(text) if text.startswith("[") else [json.loads(l) for l in text.splitlines() if l.strip()]
        for q in rows:
            if q["id"] in seen:
                continue
            seen.add(q["id"])
            if not [g for g in q.get("evidence_groups") or [] if g]:
                skipped.append(q["id"])
                continue
            items.append({"id": q["id"], "question": q["question"], "reference": q.get("reference_answer", "")})
    return items, skipped


def read_jsonl(path: Path) -> dict[str, dict]:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[rec["id"]] = rec
    return out


def strip_sources(answer: str) -> str:
    return answer.split(SOURCES_MARK)[0].strip()


def make_generator(mode: str, *, domain: str, difficulty: str, profile: str, model: str, k: int):
    from langchain_openai import ChatOpenAI

    from rag_service.contracts.schemas import RagAnswer
    from rag_service.generation.prompt import answer_prompt
    from rag_service.pipeline.chain import format_docs
    from rag_service.retrieval_a.eval_retrieval import _build

    retriever = _build(mode, k, profile, difficulty, domain)
    answer_chain = answer_prompt | ChatOpenAI(model=model).with_structured_output(RagAnswer)

    def search(question: str, tries: int = 5):
        import time

        for i in range(tries):
            try:
                return retriever.invoke(question)
            except Exception as e:
                if i == tries - 1:
                    raise
                log.warning("검색 실패, %d초 뒤 다시 시도 (%s)", 5 * (i + 1), type(e).__name__)
                time.sleep(5 * (i + 1))

    def generate(question: str) -> dict:
        docs = search(question)
        answer = answer_chain.invoke({"question": question, "context": format_docs(docs)})
        return {
            "status": answer.status,
            "answer": strip_sources(answer.answer),
            "contexts": [d.page_content for d in docs],
            "chunk_ids": [d.metadata.get("chunk_id") for d in docs],
        }

    return generate


def generate_answers(mode: str, items: list[dict], path: Path, args, usage: dict) -> dict[str, dict]:
    from langchain_core.callbacks import get_usage_metadata_callback

    done = read_jsonl(path)
    todo = [q for q in items if q["id"] not in done]
    log.info("[%s] 답변 저장됨 %d, 새로 생성 %d", mode, len(items) - len(todo), len(todo))
    if not todo:
        return done
    generate = make_generator(mode, domain=args.domain, difficulty=args.difficulty, profile=args.profile, model=args.model, k=args.k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh, get_usage_metadata_callback() as cb:
        for i, q in enumerate(todo, start=1):
            rec = {"id": q["id"], **generate(q["question"])}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            done[q["id"]] = rec
            if i % 10 == 0 or i == len(todo):
                log.info("[%s] 답변 %d/%d", mode, i, len(todo))
        for u in cb.usage_metadata.values():
            usage["answer_in"] += u.get("input_tokens", 0)
            usage["answer_out"] += u.get("output_tokens", 0)
    return done


def score_answers(mode: str, items: list[dict], answers: dict[str, dict], path: Path, args, usage: dict) -> dict[str, dict]:
    done = read_jsonl(path)
    todo = [q for q in items if q["id"] not in done]
    log.info("[%s] 채점 저장됨 %d, 새로 채점 %d", mode, len(items) - len(todo), len(todo))
    if not todo:
        return done

    from langchain_openai import ChatOpenAI
    from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
    from ragas.cost import get_token_usage_for_openai
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy

    from rag_service.retrieval_a.naive import _embeddings

    llm = LangchainLLMWrapper(ChatOpenAI(model=args.model), bypass_temperature=True, bypass_n=True)
    emb = LangchainEmbeddingsWrapper(_embeddings("BAAI/bge-m3", "auto", True, ""))
    metrics = [Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(), LLMContextRecall()]
    path.parent.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(todo), args.batch):
        part = todo[start : start + args.batch]
        samples = [
            SingleTurnSample(
                user_input=q["question"],
                response=answers[q["id"]]["answer"],
                retrieved_contexts=answers[q["id"]]["contexts"],
                reference=q["reference"],
            )
            for q in part
        ]
        result = evaluate(
            dataset=EvaluationDataset(samples=samples),
            metrics=metrics,
            llm=llm,
            embeddings=emb,
            run_config=RunConfig(max_workers=args.workers, timeout=300, max_retries=5),
            token_usage_parser=get_token_usage_for_openai,
            show_progress=False,
            raise_exceptions=False,
        )
        try:
            tu = result.total_tokens()
            usage["judge_in"] += tu.input_tokens
            usage["judge_out"] += tu.output_tokens
        except Exception:
            pass
        df = result.to_pandas()
        with open(path, "a", encoding="utf-8") as fh:
            for q, (_, row) in zip(part, df.iterrows()):
                rec = {"id": q["id"]}
                for m in METRICS:
                    v = row.get(RAGAS_COLUMN.get(m, m))
                    rec[m] = None if v is None or v != v else float(v)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done[q["id"]] = rec
        log.info("[%s] 채점 %d/%d", mode, min(start + args.batch, len(todo)), len(todo))
    return done


def mean(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def run(args) -> dict:
    items, skipped = load_items(args.golden)
    if args.limit:
        items = items[: args.limit]
    out = Path(args.out) if os.path.isabs(args.out) else REPO_ROOT / args.out
    log.info("채점 문항 %d개, 근거 없어 제외 %d개 · 모델 설정 %s · 난이도 %s · LLM %s", len(items), len(skipped), args.profile, args.difficulty, args.model)
    usage = {"answer_in": 0, "answer_out": 0, "judge_in": 0, "judge_out": 0}
    summary = {"profile": args.profile, "difficulty": args.difficulty, "llm": args.model, "k": args.k,
               "questions": len(items), "skipped": skipped, "modes": {}}
    rows = []
    for mode in args.modes:
        answers = generate_answers(mode, items, out / f"answers_{mode}.jsonl", args, usage)
        scores = score_answers(mode, items, answers, out / f"scores_{mode}.jsonl", args, usage)
        summary["modes"][mode] = {m: mean([scores[q["id"]].get(m) for q in items]) for m in METRICS}
        summary["modes"][mode]["status"] = {
            s: sum(1 for q in items if answers[q["id"]]["status"] == s) for s in ("answered", "needs_clarification", "abstained")
        }
        for q in items:
            rows.append({"id": q["id"], "mode": mode, "status": answers[q["id"]]["status"],
                         **{m: scores[q["id"]].get(m) for m in METRICS}, "question": q["question"]})
    summary["tokens_this_run"] = usage
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out / "per_question.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "mode", "status", *METRICS, "question"])
        w.writeheader()
        w.writerows(rows)
    summary["_out"] = str(out)
    return summary


def print_report(summary: dict) -> None:
    print(f"\n[{summary['profile']} · {summary['difficulty']} · {summary['llm']}] RAGAS {summary['questions']}문항")
    print("| 모드 | " + " | ".join(METRICS) + " | answered / 되묻기 / 보류 |")
    print("|---|" + "---|" * (len(METRICS) + 1))
    for mode, m in summary["modes"].items():
        vals = " | ".join("-" if m[k] is None else f"{m[k]:.3f}" for k in METRICS)
        s = m["status"]
        print(f"| {mode} | {vals} | {s['answered']} / {s['needs_clarification']} / {s['abstained']} |")
    u = summary["tokens_this_run"]
    print(f"\n이번 실행 토큰: 답변 생성 입력 {u['answer_in']:,} / 출력 {u['answer_out']:,}, 채점 입력 {u['judge_in']:,} / 출력 {u['judge_out']:,}")
    print(f"저장: {summary['_out']}")


def _main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("httpx", "huggingface_hub", "sentence_transformers", "transformers", "ragas", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    p = argparse.ArgumentParser(description="RAGAS 기본 지표 채점 (검색 + LLM 답변)")
    p.add_argument("--golden", nargs="+", required=True)
    p.add_argument("--profile", default="bge-m3")
    p.add_argument("--difficulty", default="basic")
    p.add_argument("--domain", default="shipping")
    p.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    p.add_argument("--model", default="gpt-6-luna", help="답변 생성과 채점에 쓸 OpenAI 모델")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch", type=int, default=10)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default=None, help="기본: artifacts/retrieval_a/ragas_<모델 설정>[_<난이도>]_k<k>")
    args = p.parse_args()
    if not args.out:
        # 폴더에 k를 넣는다: 같은 폴더면 이미 만든 답변을 재사용하므로 --k만 바꾸면 예전 k의 답변으로 채점된다
        name = args.profile if args.difficulty == "basic" else f"{args.profile}_{args.difficulty}"
        args.out = f"artifacts/retrieval_a/ragas_{name}_k{args.k}"
    print_report(run(args))


if __name__ == "__main__":
    _main()
