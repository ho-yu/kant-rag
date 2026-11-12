"""서버 Judge 점수 모으기: Langfuse 실험 실행의 judge_* 점수 → 평균과 건수 (발제 STEP 6).

Judge 4항목은 Langfuse 서버 evaluator가 채점한다 (langfuse_experiment.py는 RAGAS와 검색 점수만 올린다).
서버 evaluator는 null을 낼 수 없어서, 해당 없음과 평가 불가는 0점에 이유 첫머리 표시로 남긴다.
    [해당 없음]  평가할 대상이 없음 (예: 문서 내용 없이 되묻기만 한 답변의 사실·근거)
    [평가 불가]  확인할 출력이 없음 (서비스 오류)
이 스크립트는 그 표시를 읽어 quality.py와 같은 상태(scored / na / unavailable)로 바꾸고,
quality.summarize로 평균(해당 없음·평가 불가는 빼고 0점은 넣음)과 건수를 낸다.
Langfuse 화면의 기본 평균은 표시가 붙은 0점까지 섞으므로 보고서 수치로 쓰지 않는다.

규칙 (quality.py judge_precheck·to_scores와 같게)
- 답변 상태가 error면 사실·조건·근거는 평가 불가, 판단은 0 (서버 점수와 상관없이)
- 판단은 항상 채점해야 하므로 판단에 표시가 붙으면 평가 불가
- 점수가 없으면 평가 불가 (missing). 서버 채점은 실행이 끝나고 몇 분 뒤에 붙을 수 있다
- 같은 이름 점수가 둘 이상이면 가장 최근 것을 쓰고 duplicates에 센다 (켜진 규칙이 둘인지 확인)
- 표시는 이유 맨 앞에서 찾는다. "판단:" 뒤에 바로 붙은 표시도 인정한다

읽는 곳 (Langfuse v4 서버. 예전 datasets/runs·traces·scores v1/v2 조회는 404)
- 문항과 답변 상태: 실험 문항 목록 GET /api/public/experiment-items (SDK api.experiments.list_items)
- Judge 점수와 이유: 점수 목록 v3 GET /api/public/v3/scores (SDK api.scores.get_many),
  source=EVAL(서버 evaluator)만, 문항 trace id로 거른다

실행 (레포 맨 위에서, langfuse_experiment.py 실행이 끝나고 몇 분 뒤. .env의 LANGFUSE_*를 읽는다)
    uv run python evaluation/answer/server_judge_summary.py                    # results/langfuse/의 최근 실행 3개
    uv run python evaluation/answer/server_judge_summary.py naive_20261112-1030 hybrid_20261112-1041
결과: 화면 표 + results/langfuse/{run_name}_judge.json (results/는 커밋하지 않는다)
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / "results" / "langfuse"
NA_MARK = "[해당 없음]"
UNAVAILABLE_MARK = "[평가 불가]"
DEFAULT_LATEST = 3  # 기본 비교 한 번 = 세 방식 실행 3개
# 예전 선택형 Score output(pass/fail/na)으로 저장된 점수. 지금 설정은 number 0/1
OLD_CATEGORIES = {"pass": 1, "fail": 0}
OLD_NA = "na"

Record = dict[str, Any]


def _load_quality():
    """같은 폴더의 quality.py (evaluation/은 패키지가 아니라서 파일로 불러온다)."""
    spec = importlib.util.spec_from_file_location(
        "answer_quality", Path(__file__).with_name("quality.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


quality = _load_quality()
JUDGE_NAMES = tuple(quality.JUDGE_NAMES.values())  # judge_facts, judge_conditions, ...
DECISION = quality.JUDGE_NAMES["decision"]


# ---------- 서버 점수 → quality.py 점수 ----------


def mark_of(comment: str | None) -> str | None:
    """이유 맨 앞의 표시. "[해당 없음] 판단: …"과 "판단: [해당 없음] …" 둘 다 본다."""
    text = (comment or "").strip()
    if text.startswith("판단:"):
        rest = text.removeprefix("판단:").strip()
        if rest.startswith((NA_MARK, UNAVAILABLE_MARK)):
            text = rest
    for mark in (NA_MARK, UNAVAILABLE_MARK):
        if text.startswith(mark):
            return mark
    return None


def from_server(name: str, hits: Sequence[Mapping[str, Any]], status: str) -> Record:
    """한 문항의 같은 이름 서버 점수들 → quality.score 모양 점수 하나.

    hits: [{"value": 0 | 1, "comment": "...", "timestamp": "..."}] (같은 이름이 여러 개일 수 있음)
    value가 예전 선택형 글자면 pass → 1, fail → 0, na → 해당 없음으로 읽는다.
    """
    if status == "error":
        if name == DECISION:
            return quality.score(
                name, 0, "scored", None, "서비스 오류로 답변을 만들지 못함"
            )
        return quality.score(
            name, None, "unavailable", "service_error", "서비스 오류로 출력이 없음"
        )
    if not hits:
        return quality.score(
            name, None, "unavailable", "missing", "서버 Judge 점수가 아직 없음"
        )
    latest = max(hits, key=lambda h: str(h.get("timestamp") or ""))
    comment = (latest.get("comment") or "").strip()
    value = latest.get("value")
    word = value.strip().lower() if isinstance(value, str) else None
    mark = NA_MARK if word == OLD_NA else mark_of(comment)
    if mark == NA_MARK:
        if name == DECISION:  # 판단은 해당 없음이 없다
            return quality.score(name, None, "unavailable", "judge_na", comment)
        return quality.score(name, None, "na", "judge_na", comment)
    if mark == UNAVAILABLE_MARK:
        return quality.score(name, None, "unavailable", "judge_unavailable", comment)
    if word is not None:
        value = OLD_CATEGORIES.get(word, value)
    if value not in (0, 1, 0.0, 1.0):
        return quality.score(
            name, None, "unavailable", "bad_value", f"0이나 1이 아닌 값: {value!r}"
        )
    return quality.score(name, int(value), "scored", None, comment)


def item_scores(
    status: str, by_name: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[Record]:
    """한 문항의 Judge 점수 4개 (순서는 JUDGE_NAMES)."""
    return [from_server(name, by_name.get(name) or [], status) for name in JUDGE_NAMES]


def duplicates(items: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """이름별로 점수가 둘 이상 붙은 문항 수."""
    out: dict[str, int] = {}
    for it in items:
        for name in JUDGE_NAMES:
            if len(it["scores"].get(name) or []) > 1:
                out[name] = out.get(name, 0) + 1
    return out


def absent_names(items: Iterable[Mapping[str, Any]]) -> list[str]:
    """실행 전체에서 점수가 한 건도 없는 Judge 이름 (evaluator 설정이나 규칙 연결 문제)."""
    seen = {name for it in items for name in it["scores"] if it["scores"][name]}
    return [name for name in JUDGE_NAMES if name not in seen]


def summarize_run(items: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """[{status, scores: {이름: [서버 점수]}}] → quality.summarize 결과."""
    return quality.summarize(
        (it["status"], item_scores(it["status"], it["scores"])) for it in items
    )


# ---------- Langfuse에서 읽기 ----------
# 읽기만 하고 아무것도 올리지 않는다.
TRACE_BATCH = 10  # 점수 조회 한 번에 넣을 trace id 수 (주소 길이를 짧게)


def _output(value: Any) -> Any:
    """io 필드의 output (dict, 또는 JSON 문자열로 올 수 있음)."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return None
    return value


def load_items(api: Any, run_name: str, since: dt.datetime) -> list[Record]:
    """실험 실행(run_name)의 문항마다 {trace_id, status}.

    api: Langfuse().api. since: 이 시각 이후에 시작한 문항만 찾는다 (API가 하한을 요구함).
    """
    items: list[Record] = []
    cursor = None
    while True:
        page = api.experiments.list_items(
            from_start_time=since,
            experiment_name=run_name,
            fields="core,io",
            limit=100,
            cursor=cursor,
        )
        for it in page.data:
            output = _output(getattr(it, "output", None))
            status = output.get("status") if isinstance(output, Mapping) else None
            items.append({"trace_id": it.trace_id, "status": status or "unknown"})
        cursor = getattr(page.meta, "cursor", None)
        if not cursor or not page.data:
            return items


def _trace_of(score: Any) -> str | None:
    """점수가 붙은 trace id (observation에 붙은 점수는 subject.trace_id)."""
    subject = getattr(score, "subject", None)
    if subject is None:
        return None
    if getattr(subject, "kind", None) == "trace":
        return subject.id
    return getattr(subject, "trace_id", None)


def _timestamp(value: Any) -> str:
    return value.isoformat() if isinstance(value, dt.datetime) else str(value or "")


def load_judge_scores(
    api: Any, trace_ids: Sequence[str]
) -> dict[str, dict[str, list[Record]]]:
    """문항 trace id들 → {trace_id: {judge 이름: [{value, comment, timestamp}]}}. 서버 evaluator(EVAL) 점수만."""
    wanted = set(trace_ids)
    out: dict[str, dict[str, list[Record]]] = {}
    for i in range(0, len(trace_ids), TRACE_BATCH):
        batch = trace_ids[i : i + TRACE_BATCH]
        cursor = None
        while True:
            page = api.scores.get_many(
                trace_id=",".join(batch),
                name=",".join(JUDGE_NAMES),
                source="EVAL",
                fields="details,subject",
                limit=100,
                cursor=cursor,
            )
            for s in page.data:
                trace_id = _trace_of(s)
                if trace_id not in wanted or s.name not in JUDGE_NAMES:
                    continue
                out.setdefault(trace_id, {}).setdefault(s.name, []).append(
                    {
                        "value": s.value,
                        "comment": s.comment,
                        "timestamp": _timestamp(getattr(s, "timestamp", None)),
                    }
                )
            cursor = getattr(page.meta, "cursor", None)
            if not cursor or not page.data:
                break
    return out


def load_run(api: Any, run_name: str, since: dt.datetime) -> list[Record]:
    """실험 실행(run_name)의 문항마다 {trace_id, status, scores: {judge 이름: [점수]}}."""
    items = load_items(api, run_name, since)
    scores = load_judge_scores(api, [it["trace_id"] for it in items])
    for it in items:
        it["scores"] = scores.get(it["trace_id"], {})
    return items


def local_runs(names: Sequence[str], latest: int) -> list[str]:
    """실행 이름. 빼면 langfuse_experiment.py가 남긴 results/langfuse/{run}.json 중 최근 것."""
    if names:
        return list(names)
    files = sorted(
        (p for p in LOCAL.glob("*.json") if not p.stem.endswith("_judge")),
        key=lambda p: p.stat().st_mtime,
    )[-latest:]
    if not files:
        raise SystemExit(
            f"{LOCAL} 에 실행 기록이 없습니다. 실행 이름을 직접 적어 주세요."
        )
    return [json.loads(p.read_text(encoding="utf-8"))["run_name"] for p in files]


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "runs", nargs="*", help="실행 이름 (빼면 results/langfuse/의 최근 실행)"
    )
    p.add_argument(
        "--latest", type=int, default=DEFAULT_LATEST, help="실행 이름을 빼면 최근 몇 개"
    )
    p.add_argument(
        "--days", type=int, default=7, help="며칠 전부터 시작한 실행을 찾을지"
    )
    args = p.parse_args(argv)

    from dotenv import load_dotenv
    from langfuse import get_client

    load_dotenv(ROOT / ".env")

    langfuse = get_client()
    if not langfuse.auth_check():
        raise SystemExit("Langfuse 연결 설정(.env의 LANGFUSE_*)을 확인해 주세요.")
    since = dt.datetime.now(dt.UTC) - dt.timedelta(days=args.days)
    LOCAL.mkdir(parents=True, exist_ok=True)
    for run_name in local_runs(args.runs, args.latest):
        items = load_run(langfuse.api, run_name, since)
        if not items:
            print(
                f"\n[서버 Judge] {run_name}: 문항을 찾지 못했어요 (실행 이름과 --days 확인)"
            )
            continue
        summary = summarize_run(items)
        dup = duplicates(items)
        absent = absent_names(items)
        payload = {
            "run_name": run_name,
            "n_items": len(items),
            "summary": summary,
            "duplicates": dup,
            "absent": absent,
        }
        (LOCAL / f"{run_name}_judge.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n[서버 Judge] {run_name} · {len(items)}문항")
        print(quality.report(summary))
        missing = sum(
            (m.get("causes") or {}).get("missing", 0) for m in summary.values()
        )
        if missing:
            print(
                f"점수 없음 {missing}건: 서버 채점이 늦게 붙을 수 있어요. 몇 분 뒤 다시 실행해 보세요."
            )
        if absent:
            print(
                f"점수가 한 건도 없는 항목: {absent} (그 evaluator의 Score output과 규칙 연결 확인)"
            )
        if dup:
            print(
                f"같은 이름 점수가 두 번 이상 붙은 문항: {dup} (켜진 규칙이 둘인지 확인)"
            )


if __name__ == "__main__":
    main()
