"""실험 비교판: results/ 아래에 쌓인 실행 결과를 한 페이지에 모아 기준 실행과 비교한다.

읽기만 한다. 검색이나 API를 다시 돌리지 않아서 몇 초면 끝난다.

읽는 것
- 검색 비교: results/comparison/retrieval_b/*.json   (evaluation/retrieval/retrieval_b_compare.py)
- 리랭크 진단: results/diagnose/retrieval_b/*.json   (evaluation/retrieval/rerank_diagnose.py)
- 답변 평가: results/answer/retrieval_b/<폴더>/      (evaluation/answer/run_answers.py, faithfulness.py)
  답변 평가 폴더 이름은 실험 이름으로 그대로 보인다 (예: basic_development_k3).

만드는 것
- results/experiment_board.html: 표 세 개. 기준 실행을 고르면 같은 방식끼리 차이(+0.036)를 색으로 보여 준다.
  열 제목을 누르면 정렬, 검색 칸으로 거르기, 검색 표는 k를 바꿀 수 있다.
- results/experiment_board_{search,diagnose,answer}.csv: 엑셀·노션에 붙이기용 (검색 표는 --k 기준)

실행 (레포 맨 위에서)
    uv run python evaluation/reporting/experiment_board.py --open
    uv run python evaluation/reporting/experiment_board.py --k 3

검색 비교(retrieval_b_compare.py)·리랭크 진단·답변 평가(run_answers.py, faithfulness.py)가 끝나면 이 비교판을 자동으로 다시 만든다.
브라우저에 열어 둔 비교판은 다시 만들어지면 몇 초 안에 저절로 바뀐다 (페이지 맨 위 "자동 갱신" 버튼으로 끌 수 있음).

기준 실행은 페이지에서 고른다. 처음에는 "기본 설정으로 돌린 development 실행 중 가장 먼저 만든 것"이 기준이다.
기대 동작 일치는 골든셋(data/{난이도}/{도메인}/evaluation/{split}.json)의 expected_action과 비교한다.
채점 단계라서 정답 파일을 읽어도 된다 (답변 생성에는 쓰지 않았다).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import webbrowser
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from statistics import mean, median
from typing import Any

SEARCH_DIR = Path("results") / "comparison" / "retrieval_b"
DIAGNOSE_DIR = Path("results") / "diagnose" / "retrieval_b"
ANSWER_DIR = Path("results") / "answer" / "retrieval_b"
OUT_HTML = Path("results") / "experiment_board.html"

# rag_service.retrieval_b.settings.Settings 기본값과 같아야 한다 (tests에서 확인).
# 이 스크립트는 무거운 검색 모듈을 불러오지 않으려고 값을 따로 적어 둔다.
DEFAULTS: dict[str, Any] = {
    "k": 5,
    "levels": "para+table",
    "candidate_k": 30,
    "rrf_k": 60,
    "rerank_n": 50,
    "rerank_max_length": 1024,
}
# 비교에 영향을 주는 값만 "바꾼 설정"에 보인다 (배치 크기·모델 이름 등은 뺀다)
TUNING_KEYS = ("candidate_k", "rrf_k", "rerank_n", "rerank_max_length")
METRICS = (
    "page_hit",
    "page_recall",
    "page_group_recall",
    "page_mrr",
    "doc_hit",
    "doc_recall",
)
ACTION_TO_STATUS = {
    "answer": "answered",
    "clarify": "needs_clarification",
    "abstain": "abstained",
}
STATUSES = ("answered", "needs_clarification", "abstained", "error")

Record = dict[str, Any]


# ---------- 공통 ----------


def repo_root(start: Path | None = None) -> Path:
    """pyproject.toml이 있는 폴더 (storage.corpus.repo_root와 같은 규칙)."""
    p = (start or Path.cwd()).resolve()
    for d in [p, *p.parents]:
        if (d / "pyproject.toml").exists():
            return d
    return p


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[Record]:
    """없는 파일은 빈 목록, 깨진 줄은 건너뛴다 (실행 중에 끊긴 파일도 읽히게)."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _round(v: float | None, n: int = 4) -> float | None:
    return None if v is None else round(v, n)


def _levels(levels: Iterable[str] | str | None) -> str | None:
    if levels is None:
        return None
    if isinstance(levels, str):
        return levels
    return "+".join(levels)


def changed_settings(values: dict[str, Any] | None, levels: str | None, k: Any = None):
    """기본값과 다른 설정만 {"rerank_n": 20} 모양으로. 기록이 없으면 None."""
    out: dict[str, Any] = {}
    if k is not None and k != DEFAULTS["k"]:
        out["k"] = k
    if levels is not None and levels != DEFAULTS["levels"]:
        out["levels"] = levels
    if values is None:
        return None if not out else {**out, "_unrecorded": True}
    for key in TUNING_KEYS:
        if key in values and values[key] != DEFAULTS[key]:
            out[key] = values[key]
    return out


# ---------- 검색 비교 ----------


def collect_search(root: Path) -> tuple[list[Record], list[str]]:
    """검색 비교 JSON마다, 방식마다 한 줄."""
    rows: list[Record] = []
    skipped: list[str] = []
    for path in sorted((root / SEARCH_DIR).glob("*.json")):
        rel = path.relative_to(root).as_posix()
        try:
            report = read_json(path)
        except (OSError, json.JSONDecodeError) as e:
            skipped.append(f"{rel}: 읽기 실패 ({type(e).__name__})")
            continue
        if not isinstance(report, dict) or not isinstance(report.get("results"), dict):
            skipped.append(f"{rel}: 검색 비교 결과 형식이 아님")
            continue
        s = report.get("settings") or {}
        tuning = s.get("retrieval_b")
        levels = _levels(s.get("levels"))
        for mode, res in report["results"].items():
            impl = "a" if mode.startswith("a_") else "b"
            row: Record = {
                "run": path.stem,
                "experiment": experiment_id(path.stem),
                "file": rel,
                "created_at": report.get("created_at"),
                "commit": report.get("git_commit"),
                "difficulty": s.get("difficulty"),
                "split": s.get("split"),
                "levels": levels,
                "limit": s.get("limit"),
                "mode": mode,
                "impl": impl,
                # a 방식은 retrieval_b 튜닝 값을 쓰지 않으므로 levels만 본다
                "changed": changed_settings(tuning if impl == "b" else {}, levels),
                "error": res.get("error") if isinstance(res, dict) else "형식 오류",
            }
            if row["error"] is None and isinstance(res, dict):
                lat = res.get("latency_ms") or {}
                row.update(
                    {
                        "ks": [int(k) for k in (res.get("summary") or {})],
                        "by_k": {
                            k: {m: v.get(m) for m in (*METRICS, "n")}
                            for k, v in (res.get("summary") or {}).items()
                        },
                        "latency_mean": lat.get("mean"),
                        "latency_median": lat.get("median"),
                        "latency_max": lat.get("max"),
                        "n_errors": res.get("n_errors"),
                        "build_seconds": res.get("build_seconds"),
                    }
                )
            rows.append(row)
    return rows, skipped


K_SUFFIX = re.compile(r"_k(\d+)$")


def experiment_id(stem: str) -> str:
    """실험 묶음 이름. k만 바꿔 따로 돌린 파일(…_k3, …_k5)은 "…_k*" 하나로 묶고, 나머지는 파일 이름 그대로."""
    m = K_SUFFIX.search(stem)
    return f"{stem[: m.start()]}_k*" if m else stem


def search_experiments(rows: Sequence[Record]) -> list[Record]:
    """검색 실험 목록. 최근에 만든 실험이 먼저. 실험마다 파일·k·가장 늦은 created_at."""
    groups: dict[str, Record] = {}
    for r in rows:
        g = groups.setdefault(
            r["experiment"],
            {"id": r["experiment"], "files": [], "ks": set(), "latest": ""},
        )
        if r["run"] not in g["files"]:
            g["files"].append(r["run"])
        g["ks"].update(int(k) for k in r.get("by_k") or {})
        g["latest"] = max(g["latest"], r.get("created_at") or "")
    out = []
    for g in groups.values():
        ks = sorted(g["ks"])
        files = sorted(g["files"])
        if g["id"].endswith("_k*") and ks:
            label = f"{g['id']} (k{ks[0]}~k{ks[-1]}, {len(files)}개 파일)"
        else:
            label = g["id"]
        out.append(
            {
                "id": g["id"],
                "label": label,
                "files": files,
                "ks": ks,
                "latest": g["latest"],
            }
        )
    return sorted(out, key=lambda e: e["latest"], reverse=True)


def line_label(row: Record) -> str:
    """k 그래프 선 이름: 방식 · 바꾼 설정 (예: "naive · 기본값", "rerank · rerank_n=20")."""
    changed = row.get("changed")
    if changed is None:
        setting = "기록 없음"
    else:
        keys = [k for k in changed if not k.startswith("_")]
        setting = ", ".join(f"{k}={changed[k]}" for k in keys) or "기본값"
    label = f"{row['mode']} · {setting}"
    if row.get("split") and row["split"] != "development":
        label += f" · {row['split']}"
    if row.get("difficulty") and row["difficulty"] != "basic":
        label += f" · {row['difficulty']}"
    return label


def search_lines(rows: Sequence[Record]) -> list[Record]:
    """k만 다르게 따로 돌린 실행(…_k3.json, …_k5.json)을 방식·설정이 같으면 한 선으로 잇는다.

    묶는 기준: 방식, 난이도, split, levels, 바꾼 튜닝 값(k 제외).
    같은 k가 여러 파일에 있으면 created_at이 가장 늦은 파일 값을 쓰고, 점마다 출처 파일을 남긴다.
    """
    groups: dict[tuple, Record] = {}
    for r in sorted(rows, key=lambda r: r.get("created_at") or ""):
        if r.get("error") or not r.get("by_k"):
            continue
        key = (
            r["mode"],
            r.get("difficulty"),
            r.get("split"),
            r.get("levels"),
            json.dumps(r.get("changed"), sort_keys=True, ensure_ascii=False),
        )
        line = groups.setdefault(
            key,
            {
                "label": line_label(r),
                "mode": r["mode"],
                "split": r.get("split"),
                "difficulty": r.get("difficulty"),
                "levels": r.get("levels"),
                "changed": r.get("changed"),
                "by_k": {},
                "source": {},
                "latency": {},
            },
        )
        for k, values in r["by_k"].items():
            line["by_k"][str(k)] = values
            line["source"][str(k)] = r["run"]
            line["latency"][str(k)] = r.get("latency_median")
    return list(groups.values())


# k별 표: 줄 = k, 열 묶음 = 지표, 묶음 안 = 선(방식·설정). (지표 키, 이름, 좋은 방향)
KTABLE_METRICS: tuple[tuple[str, str, str], ...] = (
    ("page_hit", "쪽 Hit", "up"),
    ("page_group_recall", "묶음 Recall", "up"),
    ("page_mrr", "쪽 MRR", "up"),
    ("doc_hit", "문서 Hit", "up"),
    ("latency", "검색 ms 중앙값", "down"),
    ("page_recall", "쪽 Recall", "up"),
    ("doc_recall", "문서 Recall", "up"),
)
KTABLE_DEFAULT = ("page_hit", "page_group_recall", "page_mrr", "doc_hit", "latency")
MODE_ORDER = ("naive", "bm25", "hybrid", "rerank", "a_naive", "a_hybrid", "a_rerank")


def _line_value(line: Record, k: str, key: str) -> float | None:
    if key == "latency":
        return (line.get("latency") or {}).get(k)
    return ((line.get("by_k") or {}).get(k) or {}).get(key)


def by_k_table(lines: Sequence[Record]) -> Record:
    """k별 표 데이터. 칸마다 값, 출처 파일, 같은 k·같은 지표에서 최고인지(동점은 모두), 이전 k 대비 차이."""
    order = {m: i for i, m in enumerate(MODE_ORDER)}
    ordered = sorted(
        lines, key=lambda l: (order.get(l["mode"], len(order)), l["label"])
    )
    ks = sorted({int(k) for line in ordered for k in line.get("by_k") or {}})
    cells: dict[str, dict[str, list[Record]]] = {}
    for key, _, better in KTABLE_METRICS:
        per_k: dict[str, list[Record]] = {}
        prev: dict[int, float | None] = {}
        for k in ks:
            ks_ = str(k)
            vals = [_line_value(line, ks_, key) for line in ordered]
            nums = [v for v in vals if isinstance(v, (int, float))]
            best = (
                (min(nums) if better == "down" else max(nums))
                if len(nums) > 1
                else None
            )
            row = []
            for i, (line, v) in enumerate(zip(ordered, vals, strict=True)):
                before = prev.get(i)
                ok = isinstance(v, (int, float))
                row.append(
                    {
                        "v": v if ok else None,
                        "src": (line.get("source") or {}).get(ks_),
                        "best": ok and best is not None and abs(v - best) < 1e-9,
                        "delta": round(v - before, 4)
                        if ok and isinstance(before, (int, float))
                        else None,
                    }
                )
                if ok:
                    prev[i] = v
            per_k[ks_] = row
        cells[key] = per_k
    return {
        "lines": [{"label": line["label"], "mode": line["mode"]} for line in ordered],
        "ks": ks,
        "cells": cells,
    }


def search_by_k_csv_rows(data: Record) -> list[Record]:
    """실험 하나 × k 하나가 한 줄. 열은 "{지표} {선}" (기본 지표만)과 출처 파일."""
    names = {key: name for key, name, _ in KTABLE_METRICS}
    out = []
    for exp in data.get("search_experiments") or []:
        table = (data.get("search_ktables") or {}).get(exp["id"])
        if not table:
            continue
        for k in table["ks"]:
            row: Record = {"실험": exp["label"], "k": k}
            files: set[str] = set()
            for key in KTABLE_DEFAULT:
                for line, cell in zip(
                    table["lines"], table["cells"][key][str(k)], strict=True
                ):
                    row[f"{names[key]} {line['label']}"] = cell["v"]
                    if cell["src"]:
                        files.add(cell["src"])
            row["파일"] = ", ".join(sorted(files))
            out.append(row)
    return out


# ---------- 리랭크 진단 ----------


def collect_diagnose(root: Path) -> tuple[list[Record], list[str]]:
    rows: list[Record] = []
    skipped: list[str] = []
    for path in sorted((root / DIAGNOSE_DIR).glob("*.json")):
        rel = path.relative_to(root).as_posix()
        try:
            report = read_json(path)
        except (OSError, json.JSONDecodeError) as e:
            skipped.append(f"{rel}: 읽기 실패 ({type(e).__name__})")
            continue
        summary = report.get("summary") if isinstance(report, dict) else None
        if not isinstance(summary, dict):
            skipped.append(f"{rel}: 리랭크 진단 결과 형식이 아님")
            continue
        s = report.get("settings") or {}
        levels = _levels(s.get("levels"))
        changed = changed_settings({"rerank_n": s.get("rerank_n")}, levels, s.get("k"))
        rows.append(
            {
                "run": path.stem,
                "file": rel,
                "created_at": report.get("created_at"),
                "difficulty": s.get("difficulty"),
                "split": s.get("split"),
                "levels": levels,
                "k": s.get("k"),
                "rerank_n": s.get("rerank_n"),
                "limit": s.get("limit"),
                "mode": "rerank",
                "changed": changed,
                "n": summary.get("n"),
                "labels": summary.get("labels") or {},
                "same_book": summary.get("same_book"),
                "hybrid_page_hit": summary.get("hybrid_page_hit"),
                "rerank_page_hit": summary.get("rerank_page_hit"),
                "candidate_page_hit": summary.get("candidate_page_hit"),
            }
        )
    return rows, skipped


# ---------- 답변 평가 ----------


def load_expected(
    root: Path, difficulty: str, domain: str, split: str
) -> dict[str, str]:
    """골든셋 id → 기대 상태(answered 등). 파일이 없으면 빈 사전."""
    path = root / "data" / difficulty / domain / "evaluation" / f"{split}.json"
    if not path.exists():
        return {}
    try:
        items = read_json(path)
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        x["id"]: ACTION_TO_STATUS.get(
            x.get("expected_action"), x.get("expected_action")
        )
        for x in items
        if isinstance(x, dict) and "id" in x
    }


def _guess_scope(name: str) -> tuple[str | None, str | None]:
    """config.json이 없을 때 폴더 이름(basic_development_k3)에서 난이도와 split을 짐작한다."""
    for difficulty in ("basic", "intermediate", "advanced"):
        for split in ("expansion_validation", "development", "additional"):
            if name.startswith(f"{difficulty}_{split}"):
                return difficulty, split
    return None, None


def answer_mode_row(
    answers: Sequence[Record],
    faith: Sequence[Record],
    expected: dict[str, str],
    common_ids: set[str] | None,
    usage: Sequence[Record],
    mode: str,
) -> Record:
    """한 폴더의 한 방식 요약."""
    counts = {s: sum(1 for a in answers if a.get("status") == s) for s in STATUSES}
    ok = [a for a in answers if a.get("status") != "error"]
    lat = [a["latency_ms"] for a in ok if isinstance(a.get("latency_ms"), (int, float))]
    ret = [
        a["retrieval_ms"] for a in ok if isinstance(a.get("retrieval_ms"), (int, float))
    ]
    known = [a["tokens"] for a in answers if isinstance(a.get("tokens"), dict)]
    tok_in = sum(int(t.get("input") or 0) for t in known)
    tok_out = sum(int(t.get("output") or 0) for t in known)
    judged = [a for a in answers if a.get("id") in expected]
    match = sum(1 for a in judged if a.get("status") == expected[a["id"]])

    scored = {
        f["id"]: f["faithfulness"]
        for f in faith
        if f.get("state") == "scored"
        and isinstance(f.get("faithfulness"), (int, float))
    }
    answered_ids = {a["id"] for a in answers if a.get("status") == "answered"}
    answered_scores = [v for i, v in scored.items() if i in answered_ids]
    common_scores = (
        [v for i, v in scored.items() if i in common_ids] if common_ids else []
    )
    grading = [
        u for u in usage if u.get("mode") == mode and u.get("step") == "faithfulness"
    ]
    return {
        "mode": mode,
        "n": len(answers),
        **{f"n_{s}": c for s, c in counts.items()},
        "error_rate": _round(counts["error"] / len(answers)) if answers else None,
        "status_match": match if judged else None,
        "status_total": len(judged) if judged else None,
        "latency_mean": _round(mean(lat), 1) if lat else None,
        "latency_median": _round(median(lat), 1) if lat else None,
        "latency_max": _round(max(lat), 1) if lat else None,
        "retrieval_mean": _round(mean(ret), 1) if ret else None,
        "tokens_in": tok_in if known else None,
        "tokens_out": tok_out if known else None,
        "tokens_known": len(known),
        "faith_n_rows": len(faith),
        "faith_mean": _round(mean(scored.values())) if scored else None,
        "faith_n_scored": len(scored),
        "faith_n_na": sum(1 for f in faith if f.get("state") == "na"),
        "faith_n_unavailable": sum(1 for f in faith if f.get("state") == "unavailable"),
        "faith_answered": _round(mean(answered_scores)) if answered_scores else None,
        "faith_answered_n": len(answered_scores),
        "faith_common": _round(mean(common_scores)) if common_scores else None,
        "faith_common_n": len(common_scores),
        "grading_in": sum(int(u.get("input") or 0) for u in grading)
        if grading
        else None,
        "grading_out": sum(int(u.get("output") or 0) for u in grading)
        if grading
        else None,
    }


def collect_answers(root: Path) -> tuple[list[Record], list[str]]:
    """답변 평가 폴더마다, 방식마다 한 줄."""
    rows: list[Record] = []
    skipped: list[str] = []
    base = root / ANSWER_DIR
    if not base.exists():
        return rows, skipped
    for folder in sorted(p for p in base.iterdir() if p.is_dir()):
        rel = folder.relative_to(root).as_posix()
        answer_files = sorted(folder.glob("answers_*.jsonl"))
        if not answer_files:
            skipped.append(f"{rel}: answers_*.jsonl 없음")
            continue
        config: Record = {}
        if (folder / "config.json").exists():
            try:
                config = read_json(folder / "config.json")
            except (OSError, json.JSONDecodeError):
                skipped.append(
                    f"{rel}/config.json: 읽기 실패 (설정은 기록 없음으로 표시)"
                )
        guess = _guess_scope(folder.name)
        difficulty = config.get("difficulty") or guess[0]
        split = config.get("split") or guess[1]
        domain = config.get("domain") or "shipping"
        retrieval = config.get("retrieval")
        levels = _levels(retrieval.get("levels")) if retrieval else None
        changed = changed_settings(
            retrieval, levels, retrieval.get("k") if retrieval else None
        )
        expected = (
            load_expected(root, difficulty, domain, split)
            if difficulty and split
            else {}
        )
        by_mode = {p.stem.removeprefix("answers_"): read_jsonl(p) for p in answer_files}
        # 세 방식 모두 answered로 답한 문항: 방식마다 문항이 달라지지 않게 같은 문항끼리 비교한다
        answered_sets = [
            {a["id"] for a in rows_ if a.get("status") == "answered"}
            for rows_ in by_mode.values()
        ]
        common_ids = (
            set.intersection(*answered_sets) if len(answered_sets) > 1 else None
        )
        usage = read_jsonl(folder / "grading_usage.jsonl")
        for mode, answers in by_mode.items():
            faith = read_jsonl(folder / f"faithfulness_{mode}.jsonl")
            rows.append(
                {
                    "run": folder.name,
                    "file": rel,
                    "created_at": config.get("created_at"),
                    "difficulty": difficulty,
                    "split": split,
                    "levels": levels,
                    "k": retrieval.get("k") if retrieval else None,
                    "model": (config.get("generation") or {}).get("model"),
                    "changed": changed,
                    "common_n": len(common_ids) if common_ids is not None else None,
                    **answer_mode_row(
                        answers, faith, expected, common_ids, usage, mode
                    ),
                }
            )
    return rows, skipped


# ---------- CSV ----------


def _fmt_changed(changed: dict[str, Any] | None) -> str:
    if changed is None:
        return "기록 없음"
    shown = {k: v for k, v in changed.items() if not k.startswith("_")}
    text = ", ".join(f"{k}={v}" for k, v in shown.items()) or "기본값"
    return text + (" (튜닝 값 기록 없음)" if changed.get("_unrecorded") else "")


def search_csv_rows(rows: Sequence[Record], k: int) -> list[Record]:
    out = []
    for r in rows:
        at = (r.get("by_k") or {}).get(str(k)) or {}
        at1 = (r.get("by_k") or {}).get("1") or {}
        out.append(
            {
                "실행": r["run"],
                "만든 시각": r["created_at"],
                "split": r["split"],
                "levels": r["levels"],
                "방식": r["mode"],
                "바꾼 설정": _fmt_changed(r["changed"]),
                "n": at.get("n"),
                f"쪽 Hit@{k}": at.get("page_hit"),
                f"쪽 Recall@{k}": at.get("page_recall"),
                f"묶음 Recall@{k}": at.get("page_group_recall"),
                f"쪽 MRR@{k}": at.get("page_mrr"),
                "쪽 Hit@1": at1.get("page_hit"),
                f"문서 Hit@{k}": at.get("doc_hit"),
                "검색 ms 중앙값": r.get("latency_median"),
                "오류": r.get("error") or r.get("n_errors"),
            }
        )
    return out


def diagnose_csv_rows(rows: Sequence[Record]) -> list[Record]:
    return [
        {
            "실행": r["run"],
            "만든 시각": r["created_at"],
            "split": r["split"],
            "k": r["k"],
            "후보 수": r["rerank_n"],
            "n": r["n"],
            **{name: c for name, c in r["labels"].items()},
            "같은 책": r["same_book"],
            "hybrid 쪽 Hit": r["hybrid_page_hit"],
            "rerank 쪽 Hit": r["rerank_page_hit"],
            "후보 안 정답": r["candidate_page_hit"],
        }
        for r in rows
    ]


def answer_csv_rows(rows: Sequence[Record]) -> list[Record]:
    return [
        {
            "실행": r["run"],
            "만든 시각": r["created_at"],
            "split": r["split"],
            "방식": r["mode"],
            "바꾼 설정": _fmt_changed(r["changed"]),
            "답변 수": r["n"],
            "answered": r["n_answered"],
            "needs_clarification": r["n_needs_clarification"],
            "abstained": r["n_abstained"],
            "error": r["n_error"],
            "기대 동작 일치": r["status_match"],
            "Faithfulness": r["faith_mean"],
            "채점 수": r["faith_n_scored"],
            "해당 없음": r["faith_n_na"],
            "평가 불가": r["faith_n_unavailable"],
            "answered만": r["faith_answered"],
            "공통 answered": r["faith_common"],
            "공통 문항 수": r["faith_common_n"],
            "응답 ms 중앙값": r["latency_median"],
            "응답 ms 평균": r["latency_mean"],
            "검색 ms 평균": r["retrieval_mean"],
            "입력 토큰": r["tokens_in"],
            "출력 토큰": r["tokens_out"],
            "채점 입력 토큰": r["grading_in"],
            "채점 출력 토큰": r["grading_out"],
        }
        for r in rows
    ]


def write_csv(path: Path, rows: Sequence[Record]) -> None:
    """엑셀에서 한글이 깨지지 않게 BOM을 붙인다. 행이 없으면 파일을 만들지 않는다."""
    if not rows:
        return
    fields: list[str] = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# ---------- HTML ----------

CSS = """
:root { --fg:#1f2328; --muted:#656d76; --bg:#ffffff; --panel:#f6f8fa; --line:#d0d7de;
  --accent:#0969da; --up:#1a7f37; --down:#cf222e; --base:#ddf4ff; --chg:#fff8c5; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e6edf3; --muted:#9198a1; --bg:#0d1117; --panel:#161b22; --line:#30363d;
    --accent:#4493f8; --up:#3fb950; --down:#f85149; --base:#0c2d48; --chg:#3a3000; } }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
  font: 14px/1.5 -apple-system, "Segoe UI", "Malgun Gothic", "Apple SD Gothic Neo", sans-serif; }
main { max-width: 96rem; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 32px 0 6px; }
.meta, .note { color: var(--muted); font-size: 13px; margin: 0 0 8px; }
.controls { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin: 8px 0 10px; font-size: 13px; }
.controls select, .controls input { font: inherit; font-size: 13px; padding: 3px 6px; color: var(--fg);
  background: var(--bg); border: 1px solid var(--line); border-radius: 6px; }
.controls input { width: 14rem; }
.box { overflow-x: auto; border: 1px solid var(--line); border-radius: 6px; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { padding: 5px 9px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; }
th { background: var(--panel); font-weight: 600; cursor: pointer; user-select: none; position: sticky; top: 0; }
th.sorted::after { content: " ▾"; } th.sorted.asc::after { content: " ▴"; }
td.num, th.num { text-align: right; }
tr.base td { background: var(--base); }
tr:last-child td { border-bottom: 0; }
.chg { background: var(--chg); font-weight: 700; padding: 0 4px; border-radius: 4px; }
.d { display: block; font-size: 11px; line-height: 1.2; }
.up { color: var(--up); } .down { color: var(--down); } .same { color: var(--muted); }
.err { color: var(--down); font-weight: 600; }
.empty { color: var(--muted); padding: 10px 12px; }
details { margin-top: 28px; color: var(--muted); font-size: 13px; }
code { font: 12px ui-monospace, SFMono-Regular, Consolas, monospace; }
.views { display: inline-flex; border: 1px solid var(--line); border-radius: 6px; overflow: hidden; }
.views button { font: inherit; font-size: 13px; padding: 4px 10px; border: 0; background: var(--bg); color: var(--fg); cursor: pointer; }
.views button + button { border-left: 1px solid var(--line); }
.views button.on { background: var(--accent); color: #fff; }
.chart { padding: 8px 12px 12px; }
.plot { width: 100%; max-width: 980px; height: auto; display: block; }
.plot .grid { stroke: var(--line); stroke-width: 1; }
.plot .ax { fill: var(--muted); font-size: 11px; }
.plot .lbl { font-size: 12px; }
.plot .pt { fill: var(--fg); font-size: 10px; }
.legend { list-style: none; display: flex; flex-wrap: wrap; gap: 6px 14px; padding: 0; margin: 8px 0 0; font-size: 12px; }
.legend li { cursor: pointer; display: inline-flex; align-items: center; gap: 5px; }
.legend li.off { opacity: 0.35; text-decoration: line-through; }
.pivot { margin-top: 12px; }
.refresh-toggle { font: inherit; font-size: 12px; padding: 1px 8px; border: 1px solid var(--line); border-radius: 10px; background: var(--bg); color: var(--muted); cursor: pointer; }
.refresh-toggle.on { color: var(--up); border-color: var(--up); }
.ktable th.grp { text-align: center; }
.ktable th.grp, .ktable .gstart { border-left: 2px solid var(--fg); }
.ktable .kcol { position: sticky; left: 0; background: var(--panel); z-index: 1; }
.kchecks label { display: inline-flex; gap: 4px; align-items: center; }
.missing-note { padding: 6px 10px; font-size: 12px; color: var(--muted); border-bottom: 1px solid var(--line); }
.missing-note button { font: inherit; font-size: 12px; padding: 1px 8px; border: 1px solid var(--line); border-radius: 4px; background: var(--bg); color: var(--accent); cursor: pointer; }
.sw { width: 10px; height: 10px; border-radius: 2px; display: inline-block; margin-right: 6px; }
.legend i { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
"""

SCRIPT = r"""
const DATA = JSON.parse(document.getElementById('board-data').textContent);
// 자동 새로고침: 켜 둔 상태와 스크롤 위치를 기억한다 (저장이 막힌 환경이면 그냥 넘어간다)
// 자동 갱신: 비교판을 다시 만들 때마다 experiment_board_version.js(버전 한 줄)도 새로 쓴다.
// file:// 페이지에서는 fetch가 막히지만 <script> 불러오기는 되므로, 3초마다 그 파일을 다시 불러
// 버전이 바뀌었을 때만 새로고침한다. 고른 실험·보기·지표는 localStorage에, 스크롤은 새로고침 직전에 저장한다.
const store = {
  get: (k) => { try { return localStorage.getItem(k); } catch (e) { return null; } },
  set: (k, v) => { try { localStorage.setItem(k, v); } catch (e) { /* 저장이 막힌 환경 */ } },
};
(() => {
  const btn = document.getElementById('autorefresh');
  const last = document.getElementById('last-refresh');
  const t = new Date(DATA.created_at);
  const two = (n) => String(n).padStart(2, '0');
  if (last && !isNaN(t)) last.textContent = `마지막 갱신 ${two(t.getHours())}:${two(t.getMinutes())}:${two(t.getSeconds())}`;
  const y = Number(store.get('board-scroll') || 0);
  if (y) { store.set('board-scroll', ''); window.addEventListener('load', () => window.scrollTo(0, y)); }
  let on = store.get('board-autorefresh') !== '0';  // 기본은 켬
  const show = () => { btn.textContent = on ? '자동 갱신 켬' : '자동 갱신 끔'; btn.classList.toggle('on', on); };
  btn.addEventListener('click', () => { on = !on; store.set('board-autorefresh', on ? '1' : '0'); show(); });
  show();
  let tag = null;
  setInterval(() => {
    if (!on) return;
    if (tag) tag.remove();
    tag = document.createElement('script');
    tag.src = `experiment_board_version.js?t=${Date.now()}`;
    tag.onload = () => {
      const v = window.BOARD_VERSION;
      if (v && v !== DATA.version) { store.set('board-scroll', String(window.scrollY)); location.reload(); }
    };
    tag.onerror = () => { /* 버전 파일이 없으면 조용히 넘어간다 */ };
    document.head.appendChild(tag);
  }, 3000);
})();
const fmt = (v, d = 3) => (v === null || v === undefined) ? '-' : (typeof v === 'number' ? (Number.isInteger(v) && d !== 3 ? String(v) : v.toFixed(d)) : String(v));
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
const changedText = (c) => {
  if (c === null || c === undefined) return '<span class="same">기록 없음</span>';
  const keys = Object.keys(c).filter((k) => !k.startsWith('_'));
  const body = keys.length ? keys.map((k) => `<span class="chg">${esc(k)}=${esc(c[k])}</span>`).join(' ') : '<span class="same">기본값</span>';
  return body + (c._unrecorded ? ' <span class="same">(튜닝 값 기록 없음)</span>' : '');
};
const isDefault = (r) => r.changed !== null && r.changed !== undefined && Object.keys(r.changed).filter((k) => !k.startsWith('_')).length === 0;

// 처음 기준: 기본 설정이 기록된 development 실행 > 기록 없는 실행 > 아무거나. 그 안에서 가장 먼저 만든 것.
function defaultBase(rows) {
  const dev = (rs) => rs.filter((r) => r.split === 'development' && !r.limit);
  const recorded = rows.filter(isDefault);
  const unrecorded = rows.filter((r) => r.changed === null);
  const pools = [dev(recorded), dev(unrecorded), recorded, unrecorded, rows];
  const pool = pools.find((p) => p.length) || [];
  const t = (r) => r.created_at || '9999';  // 시각이 없으면 뒤로
  const sorted = [...pool].sort((a, b) => t(a).localeCompare(t(b)));
  return sorted.length ? sorted[0].run : null;
}

// cols: {key, label, get(row, ctx), num, better: 'up'|'down'|null, digits}
function makeTable(sec) {
  const el = document.getElementById(sec.id);
  const rows = sec.rows;
  if (!rows.length) { el.innerHTML = `<div class="empty">${esc(sec.empty)}</div>`; return; }
  const runs = [...new Set(rows.map((r) => r.run))];
  const tableView = { id: 'table', label: sec.tableLabel || '표' };
  const views = sec.tableLast ? [...(sec.charts || []), tableView] : [tableView, ...(sec.charts || [])];
  // 보기 선택은 섹션마다 기억한다. 기억한 값이 없으면 섹션 기본 보기(검색은 k별 그래프)
  const viewKey = `board-view2-${sec.id}`;  // 키 이름을 바꿔 예전 "표" 선택은 한 번 새 기본 보기로 열린다
  let savedView = null;
  try { savedView = localStorage.getItem(viewKey); } catch (e) { /* 저장이 막힌 환경 */ }
  const startView = views.some((v) => v.id === savedView) ? savedView : (sec.defaultView || 'table');
  // 실험 선택 (검색 섹션만): 기억한 값이 없거나 사라졌으면 가장 최근 실험
  const exps = sec.experiments || null;
  let savedExp = null;
  try { savedExp = localStorage.getItem(`board-exp-${sec.id}`); } catch (e) { /* 저장이 막힌 환경 */ }
  const startExp = exps ? ((savedExp === '' || exps.some((x) => x.id === savedExp)) ? savedExp : (exps[0] ? exps[0].id : '')) : '';
  const state = { base: defaultBase(rows), q: '', sort: null, asc: false, k: sec.k ? String(sec.k) : null, exp: startExp,
    view: startView, hidden: new Set(), showMissing: false, ...(sec.chartState || {}) };
  const ctrl = document.createElement('div');
  ctrl.className = 'controls';
  ctrl.innerHTML = (views.length > 1 ? `<span class="views">${views.map((v) => `<button type="button" data-view="${v.id}"${v.id === state.view ? ' class="on"' : ''}>${esc(v.label)}</button>`).join('')}</span>` : '')
    + (exps ? `<label>실험 <select class="exp"><option value=""${state.exp === '' ? ' selected' : ''}>전체</option>${exps.map((x) => `<option value="${esc(x.id)}"${x.id === state.exp ? ' selected' : ''}>${esc(x.label)}</option>`).join('')}</select></label>` : '')
    + `<label>기준 실행 <select class="base">${['', ...runs].map((r) => `<option value="${esc(r)}"${r === state.base ? ' selected' : ''}>${r ? esc(r) : '(비교 안 함)'}</option>`).join('')}</select></label>`
    + (sec.ks ? `<label>k <select class="k">${sec.ks.map((k) => `<option${String(k) === state.k ? ' selected' : ''}>${k}</option>`).join('')}</select></label>` : '')
    + `<label>거르기 <input class="q" placeholder="실행·방식·split·설정"></label>`;
  const box = document.createElement('div');
  box.className = 'box';
  el.append(ctrl, box);

  function baseOf(r) {
    if (!state.base || r.run === state.base) return null;
    return rows.find((b) => b.run === state.base && b.mode === r.mode && b.split === r.split && b.difficulty === r.difficulty) || null;
  }
  function cell(c, r) {
    const v = c.get(r, state);
    if (!c.num) return `<td>${c.html ? v : esc(fmt(v, 0))}</td>`;
    let delta = '';
    const b = baseOf(r);
    if (c.better && b && typeof v === 'number') {
      const bv = c.get(b, state);
      if (typeof bv === 'number') {
        const dv = v - bv;
        const eps = (c.digits ?? 3) === 3 ? 0.0005 : 0.5;
        const cls = Math.abs(dv) < eps ? 'same' : ((dv > 0) === (c.better === 'up') ? 'up' : 'down');
        delta = `<span class="d ${cls}">${dv > 0 ? '+' : ''}${dv.toFixed(c.digits ?? 3)}</span>`;
      }
    }
    return `<td class="num">${esc(fmt(v, c.digits ?? 3))}${delta}</td>`;
  }
  function render() {
    const cols = sec.cols(state);
    const q = state.q.trim().toLowerCase();
    let shown = rows.filter((r) => (!state.exp || r.experiment === state.exp) && (!q || [r.run, r.mode, r.split, r.levels, JSON.stringify(r.changed)].join(' ').toLowerCase().includes(q)));
    ctrl.querySelectorAll('[data-view]').forEach((b) => b.classList.toggle('on', b.dataset.view === state.view));
    if (state.view !== 'table') {
      const chart = views.find((v) => v.id === state.view);
      box.innerHTML = `<div class="chart">${chart.render(shown, state)}</div>`;
      box.querySelectorAll('select[data-key]').forEach((s) => s.addEventListener('change', (e) => { state[s.dataset.key] = e.target.value; render(); }));
      box.querySelectorAll('input[data-kmetric]').forEach((cb) => cb.addEventListener('change', () => {
        if (cb.checked) state.kmetrics.add(cb.dataset.kmetric); else state.kmetrics.delete(cb.dataset.kmetric);
        store.set('board-kmetrics', JSON.stringify([...state.kmetrics]));
        render();
      }));
      box.querySelectorAll('[data-series]').forEach((li) => li.addEventListener('click', () => {
        const name = li.dataset.series;
        if (state.hidden.has(name)) state.hidden.delete(name); else state.hidden.add(name);
        render();
      }));
      return;
    }
    if (state.sort !== null) {
      const c = cols[state.sort];
      shown = [...shown].sort((a, b) => {
        const x = c.get(a, state), y = c.get(b, state);
        if (x === y) return 0;
        if (x === null || x === undefined) return 1;
        if (y === null || y === undefined) return -1;
        return (x < y ? -1 : 1) * (state.asc ? 1 : -1);
      });
    }
    // 검색 표: k 하나씩 따로 돌린 파일은 다른 k 값이 없다. 고른 k 값이 없는 줄은 숨긴다 (오류 줄은 메시지 때문에 남김)
    let note = '';
    if (sec.hideMissingK && state.k) {
      const missing = shown.filter((r) => !r.error && !((r.by_k || {})[state.k]));
      if (missing.length) {
        if (!state.showMissing) shown = shown.filter((r) => !missing.includes(r));
        note = `<div class="missing-note">k=${esc(state.k)} 값이 없는 줄 ${missing.length}개 ${state.showMissing ? '보이는 중' : '숨김'} · <button type="button" class="toggle-missing">${state.showMissing ? '숨기기' : '보이기'}</button></div>`;
      }
    }
    box.innerHTML = note + `<table><thead><tr>${cols.map((c, i) => `<th data-i="${i}" class="${c.num ? 'num' : ''}${state.sort === i ? ' sorted' + (state.asc ? ' asc' : '') : ''}">${esc(c.label(state))}</th>`).join('')}</tr></thead><tbody>`
      + shown.map((r) => `<tr class="${r.run === state.base ? 'base' : ''}">${cols.map((c) => cell(c, r)).join('')}</tr>`).join('')
      + '</tbody></table>';
    const tm = box.querySelector('.toggle-missing');
    if (tm) tm.addEventListener('click', () => { state.showMissing = !state.showMissing; render(); });
    box.querySelectorAll('th').forEach((th) => th.addEventListener('click', () => {
      const i = Number(th.dataset.i);
      if (state.sort === i) state.asc = !state.asc; else { state.sort = i; state.asc = !cols[i].num; }
      render();
    }));
  }
  ctrl.querySelectorAll('[data-view]').forEach((b) => b.addEventListener('click', () => {
    state.view = b.dataset.view;
    try { localStorage.setItem(viewKey, state.view); } catch (e) { /* 저장이 막힌 환경 */ }
    render();
  }));
  ctrl.querySelector('.base').addEventListener('change', (e) => { state.base = e.target.value || null; render(); });
  const expSel = ctrl.querySelector('.exp');
  if (expSel) expSel.addEventListener('change', (e) => {
    state.exp = e.target.value;
    try { localStorage.setItem(`board-exp-${sec.id}`, state.exp); } catch (err) { /* 저장이 막힌 환경 */ }
    render();
  });
  ctrl.querySelector('.q').addEventListener('input', (e) => { state.q = e.target.value; render(); });
  const ks = ctrl.querySelector('.k');
  if (ks) ks.addEventListener('change', (e) => { state.k = e.target.value; render(); });
  render();
}

// ---------- 그래프 (외부 라이브러리 없이 SVG로 그린다) ----------
const PALETTE = ['#0969da', '#cf222e', '#1a7f37', '#8250df', '#bf8700', '#1b7c83', '#e16f24', '#bf3989', '#57606a', '#0550ae', '#6e7781', '#4d2d00'];
const runLabel = (run) => run.replace(/^rerank_/, '').replace(/^(basic|intermediate|advanced)_(development|additional|expansion_validation)_?/, '').replace(/^para-table_?/, '') || '기본';
const seriesName = (r) => `${runLabel(r.run)} · ${r.mode}`;
const colorOf = (names) => { const m = new Map(); names.forEach((n, i) => m.set(n, PALETTE[i % PALETTE.length])); return m; };
const optSelect = (key, label, options, value) => `<label>${esc(label)} <select data-key="${key}">${options.map(([v, t]) => `<option value="${esc(v)}"${String(v) === String(value) ? ' selected' : ''}>${esc(t)}</option>`).join('')}</select></label>`;
const legend = (names, color, hidden) => `<ul class="legend">${names.map((n) => `<li data-series="${esc(n)}" class="${hidden.has(n) ? 'off' : ''}" title="눌러서 숨기기/보이기"><i style="background:${color.get(n)}"></i>${esc(n)}</li>`).join('')}</ul>`;
const niceMax = (v) => { if (!(v > 0) || v <= 1) return 1; const p = 10 ** Math.floor(Math.log10(v)); return Math.ceil((v * 1.1) / p) * p; };
const tick = (v) => (Math.abs(v) >= 100 ? Math.round(v).toLocaleString() : (Number.isInteger(v) ? String(v) : v.toFixed(2)));

// 공통 축. xs가 주어지면 범주(같은 간격), 아니면 [xMin, xMax] 선형
// 산점도용: 값이 몰려 있어도 퍼져 보이게 최솟값~최댓값에 여유를 둔 범위
const span = (vals) => {
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const pad = hi > lo ? (hi - lo) * 0.15 : Math.max(Math.abs(hi) * 0.05, 0.05);
  return [Math.max(0, lo - pad), hi + pad];
};

function frame({ w = 760, h = 380, yMax, yMin = 0, yLabel, xLabel, xs = null, xMax = null, xMin = 0, xTicks = null }) {
  const rot = !!xs && xs.some((v) => String(v).length > 4);  // 긴 이름은 비스듬히
  const m = { l: 56, r: 16, t: 14, b: rot ? 110 : 44 };
  const pw = w - m.l - m.r, ph = h - m.t - m.b;
  const X = xs ? ((v) => m.l + (xs.length === 1 ? pw / 2 : (xs.indexOf(v) / (xs.length - 1)) * pw)) : ((v) => m.l + ((v - xMin) / (xMax - xMin)) * pw);
  const Y = (v) => m.t + ph - ((v - yMin) / (yMax - yMin)) * ph;
  let g = '';
  for (let i = 0; i <= 5; i++) {
    const v = yMin + ((yMax - yMin) / 5) * i, y = Y(v);
    g += `<line x1="${m.l}" x2="${w - m.r}" y1="${y}" y2="${y}" class="grid"/><text x="${m.l - 6}" y="${y + 4}" class="ax" text-anchor="end">${tick(v)}</text>`;
  }
  const xt = xs || xTicks || [...Array(6).keys()].map((i) => xMin + ((xMax - xMin) / 5) * i);
  xt.forEach((v) => {
    const x = X(v), y = h - m.b + 16;
    g += rot
      ? `<text x="${x}" y="${y}" class="ax" text-anchor="end" transform="rotate(-35 ${x} ${y})">${esc(v)}</text>`
      : `<text x="${x}" y="${y}" class="ax" text-anchor="middle">${esc(xs ? v : tick(v))}</text>`;
  });
  g += `<text x="${m.l + pw / 2}" y="${h - 6}" class="ax lbl" text-anchor="middle">${esc(xLabel)}</text>`;
  g += `<text x="14" y="${m.t + ph / 2}" class="ax lbl" text-anchor="middle" transform="rotate(-90 14 ${m.t + ph / 2})">${esc(yLabel)}</text>`;
  return { w, h, m, pw, ph, X, Y, g };
}
// 막대가 몇 개 없을 때 화면 폭만큼 커지지 않게 원래 크기까지만 늘린다
const svg = (f, body) => `<svg viewBox="0 0 ${f.w} ${f.h}" class="plot" style="max-width:${f.w}px" role="img">${f.g}${body}</svg>`;

// 방식마다 색을 고정한다. 같은 방식의 다른 설정은 점선 모양으로 구분한다.
const MODE_ORDER = ['naive', 'bm25', 'hybrid', 'rerank', 'a_naive', 'a_hybrid', 'a_rerank'];
const modeColor = (mode) => PALETTE[(MODE_ORDER.indexOf(mode) + MODE_ORDER.length) % MODE_ORDER.length] || PALETTE[0];
const DASHES = ['', '6 4', '2 3', '10 3 2 3', '1 5'];

// k 그래프: Python에서 묶은 DATA.search_lines(방식·설정이 같으면 한 선)를 그린다
function lineChart(shown, state, metrics) {
  const metric = state.metric || metrics[0][0];
  const mlabel = (metrics.find((m) => m[0] === metric) || metrics[0])[1];
  const q = state.q.trim().toLowerCase();
  const pool = state.exp ? ((DATA.search_lines_by_experiment || {})[state.exp] || []) : (DATA.search_lines || []);
  const lines = pool.filter((l) => !q || [l.label, l.mode, l.split, l.levels, JSON.stringify(l.changed)].join(' ').toLowerCase().includes(q));
  if (!lines.length) return '<div class="empty">그릴 결과가 없어요.</div>';
  const xs = [...new Set(lines.flatMap((l) => Object.keys(l.by_k).map(Number)))].sort((a, b) => a - b);
  const names = lines.map((l) => l.label);
  const color = new Map(), dash = new Map(), perMode = {};
  lines.forEach((l) => { const i = (perMode[l.mode] = (perMode[l.mode] ?? -1) + 1); color.set(l.label, modeColor(l.mode)); dash.set(l.label, DASHES[i % DASHES.length]); });
  // k는 실제 간격대로 (1, 3, 5, 10, 15가 같은 간격으로 보이지 않게)
  const f = frame({ yMax: 1, yLabel: mlabel, xLabel: 'k (상위 몇 개)', xMax: Math.max(...xs), xTicks: xs });
  const val = (l, k) => ((l.by_k[k] || {})[metric]);
  let body = '';
  lines.forEach((l) => {
    const n = l.label;
    if (state.hidden.has(n)) return;
    const pts = xs.map((k) => [k, val(l, k)]).filter(([, v]) => typeof v === 'number');
    if (!pts.length) return;
    const c = color.get(n);
    body += `<polyline fill="none" stroke="${c}" stroke-width="2" stroke-dasharray="${dash.get(n)}" points="${pts.map(([k, v]) => `${f.X(k)},${f.Y(v)}`).join(' ')}"/>`;
    body += pts.map(([k, v]) => `<circle cx="${f.X(k)}" cy="${f.Y(v)}" r="4" fill="${c}"><title>${esc(n)}\nk=${k} ${esc(mlabel)} ${v.toFixed(3)}\n파일: ${esc(l.source[k] || '')}</title></circle>`).join('');
  });
  // 같은 데이터를 표로: 선마다 한 줄, k마다 한 칸. 열마다 가장 좋은 값은 굵게(색만으로 표시하지 않음)
  const visible = lines.filter((l) => !state.hidden.has(l.label));
  const best = Object.fromEntries(xs.map((k) => [k, Math.max(...visible.map((l) => val(l, k)).filter((v) => typeof v === 'number'))]));
  const pivot = `<div class="box pivot"><table><thead><tr><th>선 (방식 · 설정)</th>${xs.map((k) => `<th class="num">k=${k}</th>`).join('')}</tr></thead><tbody>`
    + visible.map((l) => `<tr><td><i class="sw" style="background:${color.get(l.label)}"></i>${esc(l.label)}</td>${xs.map((k) => {
      const v = val(l, k);
      if (typeof v !== 'number') return '<td class="num same">-</td>';
      const top = Math.abs(v - best[k]) < 1e-9;
      return `<td class="num" title="파일: ${esc(l.source[k] || '')}">${top ? `<b>${v.toFixed(3)} ▲</b>` : v.toFixed(3)}</td>`;
    }).join('')}</tr>`).join('')
    + `</tbody></table></div><p class="note">${esc(mlabel)} · 굵은 글씨와 ▲가 그 k에서 가장 높은 값이에요. 칸에 마우스를 올리면 출처 파일이 보여요.</p>`;
  return `<div class="controls">${optSelect('metric', '지표', metrics, metric)}<span class="note">k만 바꿔 따로 돌린 실행은 방식·설정이 같으면 한 선으로 이어져요. 같은 k가 여러 파일에 있으면 가장 최근 값을 써요.</span></div>`
    + `${svg(f, body)}${legend(names, color, state.hidden)}${pivot}`;
}

// 좌표(산점도): x축·y축 지표를 골라 한 실행·방식을 점 하나로
// k별 표: 줄 = k, 열 묶음 = 지표, 묶음 안 = 선(방식·설정). 데이터는 Python의 by_k_table (DATA.search_ktables)
function kTable(shown, state) {
  const metrics = DATA.ktable_metrics || [];
  if (!state.kmetrics) {
    let saved = null;
    try { saved = JSON.parse(store.get('board-kmetrics') || 'null'); } catch (e) { saved = null; }
    state.kmetrics = new Set(Array.isArray(saved) ? saved : (DATA.ktable_default || []));
  }
  const checks = `<div class="controls kchecks"><span>지표</span>${metrics.map(([key, name]) => `<label><input type="checkbox" data-kmetric="${esc(key)}"${state.kmetrics.has(key) ? ' checked' : ''}> ${esc(name)}</label>`).join('')}</div>`;
  const table = (DATA.search_ktables || {})[state.exp || ''];
  if (!table || !table.ks.length) return checks + '<div class="empty">표로 보일 결과가 없어요.</div>';
  let chosen = metrics.filter(([key]) => state.kmetrics.has(key));
  let hint = '<p class="note">같은 k·같은 지표에서 가장 좋은 값은 굵은 글씨와 ▲ (검색 ms는 가장 빠른 값, 동점은 모두). 값 아래 작은 숫자는 같은 선의 이전 k 대비 차이예요. 칸에 마우스를 올리면 출처 파일이 보여요.</p>';
  if (!state.exp && chosen.length > 1) {
    chosen = chosen.slice(0, 1);
    hint = '<p class="note">"전체"에서는 선이 많아 첫 번째로 고른 지표만 보여요. 위 "실험" 칸에서 실험을 고르면 모든 지표가 한 표에 나와요.</p>' + hint;
  }
  if (!chosen.length) return checks + '<div class="empty">보여 줄 지표를 하나 이상 고르세요.</div>';
  const lines = table.lines;
  const head1 = `<tr><th rowspan="2" class="kcol">k</th>${chosen.map(([, name]) => `<th colspan="${lines.length}" class="grp">${esc(name)}</th>`).join('')}</tr>`;
  const head2 = `<tr>${chosen.map(() => lines.map((l, i) => `<th class="num${i === 0 ? ' gstart' : ''}"><i class="sw" style="background:${modeColor(l.mode)}"></i>${esc(l.label)}</th>`).join('')).join('')}</tr>`;
  const body = table.ks.map((k) => `<tr><td class="kcol"><b>k=${k}</b></td>${chosen.map(([key, , better]) => (table.cells[key][String(k)] || []).map((c, i) => {
    const cls = `num${i === 0 ? ' gstart' : ''}`;
    if (c.v === null || c.v === undefined) return `<td class="${cls} same">-</td>`;
    const digits = key === 'latency' ? 1 : 3;
    const text = c.v.toFixed(digits);
    let d = '<span class="d same">–</span>';
    if (typeof c.delta === 'number') {
      const eps = key === 'latency' ? 0.05 : 0.0005;
      const dc = Math.abs(c.delta) < eps ? 'same' : ((c.delta > 0) === (better === 'up') ? 'up' : 'down');
      d = `<span class="d ${dc}">${c.delta >= 0 ? '+' : '−'}${Math.abs(c.delta).toFixed(digits)}</span>`;
    }
    return `<td class="${cls}" title="파일: ${esc(c.src || '')}">${c.best ? `<b>${text} ▲</b>` : text}${d}</td>`;
  }).join('')).join('')}</tr>`).join('');
  return checks + `<div class="box ktable"><table><thead>${head1}${head2}</thead><tbody>${body}</tbody></table></div>` + hint;
}

function scatterChart(shown, state, axes, defX, defY) {
  const xKey = state.sx || defX, yKey = state.sy || defY;
  const ax = (key) => axes.find((a) => a.key === key) || axes[0];
  const xa = ax(xKey), ya = ax(yKey);
  const pts = shown.map((r) => ({ r, n: seriesName(r), x: xa.get(r, state), y: ya.get(r, state) })).filter((p) => typeof p.x === 'number' && typeof p.y === 'number');
  const opts = axes.map((a) => [a.key, a.label(state)]);
  const ctl = `<div class="controls">${optSelect('sx', 'x축', opts, xa.key)}${optSelect('sy', 'y축', opts, ya.key)}<span class="note">오른쪽 위일수록 x·y가 모두 큰 실험이에요. 시간·토큰은 왼쪽일수록 좋아요.</span></div>`;
  if (!pts.length) return ctl + '<div class="empty">두 지표가 모두 있는 결과가 없어요.</div>';
  const names = pts.map((p) => p.n), color = colorOf(names);
  const [xMin, xMax] = span(pts.map((p) => p.x)), [yMin, yMax] = span(pts.map((p) => p.y));
  const f = frame({ xMin, xMax, yMin, yMax, yLabel: ya.label(state), xLabel: xa.label(state) });
  const visible = pts.filter((p) => !state.hidden.has(p.n));
  const labels = visible.length <= 8;  // 점이 많으면 이름은 마우스를 올렸을 때만
  const body = visible.map((p) => `<circle cx="${f.X(p.x)}" cy="${f.Y(p.y)}" r="6" fill="${color.get(p.n)}" fill-opacity="0.85"><title>${esc(p.n)}\n${esc(xa.label(state))} ${tick(p.x)} · ${esc(ya.label(state))} ${tick(p.y)}</title></circle>`
    + (labels ? (f.X(p.x) > f.w * 0.75  // 오른쪽 끝 점은 이름을 왼쪽에 써서 잘리지 않게
      ? `<text x="${f.X(p.x) - 8}" y="${f.Y(p.y) + 4}" class="pt" text-anchor="end">${esc(p.n)}</text>`
      : `<text x="${f.X(p.x) + 8}" y="${f.Y(p.y) + 4}" class="pt">${esc(p.n)}</text>`) : '')).join('');
  return ctl + svg(f, body) + legend(names, color, state.hidden);
}

// 막대: 실행·방식마다 막대 하나 (stack이면 여러 값을 쌓는다)
function barChart(shown, state, metrics, stack = null) {
  const rs = shown.filter((r) => !r.error);
  if (!rs.length) return '<div class="empty">그릴 결과가 없어요.</div>';
  const names = rs.map((r) => (stack ? runLabel(r.run) : seriesName(r)));
  const w = Math.max(520, 90 + rs.length * 64);
  let body = '', ctl = '', leg = '';
  if (stack) {
    const total = Math.max(...rs.map((r) => stack.keys.reduce((s, k) => s + (stack.get(r, k) || 0), 0)));
    const f = frame({ w, yMax: niceMax(total), yLabel: stack.label, xLabel: '실행', xs: names });
    const bw = Math.min(44, (f.pw / rs.length) * 0.6);
    rs.forEach((r, i) => {
      let acc = 0;
      stack.keys.forEach((k, j) => {
        const v = stack.get(r, k) || 0;
        if (!v) return;
        body += `<rect x="${f.X(names[i]) - bw / 2}" y="${f.Y(acc + v)}" width="${bw}" height="${f.Y(acc) - f.Y(acc + v)}" fill="${PALETTE[j % PALETTE.length]}"><title>${esc(names[i])}\n${esc(k)} ${v}</title></rect>`;
        acc += v;
      });
    });
    leg = `<ul class="legend">${stack.keys.map((k, j) => `<li><i style="background:${PALETTE[j % PALETTE.length]}"></i>${esc(k)}</li>`).join('')}</ul>`;
    return svg(f, body) + leg;
  }
  const metric = state.bm || metrics[0][0];
  const mlabel = (metrics.find((m) => m[0] === metric) || metrics[0])[1];
  const vals = rs.map((r) => r[metric]);
  const max = Math.max(...vals.filter((v) => typeof v === 'number'), 0);
  const f = frame({ w, yMax: max <= 1 ? 1 : niceMax(max), yLabel: mlabel, xLabel: '실행 · 방식', xs: names });
  const color = colorOf([...new Set(rs.map((r) => r.mode))]);
  const bw = Math.min(44, (f.pw / rs.length) * 0.6);
  rs.forEach((r, i) => {
    const v = vals[i];
    if (typeof v !== 'number') return;
    body += `<rect x="${f.X(names[i]) - bw / 2}" y="${f.Y(v)}" width="${bw}" height="${f.Y(0) - f.Y(v)}" fill="${color.get(r.mode)}"><title>${esc(names[i])}\n${esc(mlabel)} ${tick(v)}</title></rect>`
      + `<text x="${f.X(names[i])}" y="${f.Y(v) - 4}" class="pt" text-anchor="middle">${tick(v)}</text>`;
  });
  ctl = `<div class="controls">${optSelect('bm', '지표', metrics, metric)}<span class="note">색은 방식이에요.</span></div>`;
  leg = `<ul class="legend">${[...color.keys()].map((k) => `<li><i style="background:${color.get(k)}"></i>${esc(k)}</li>`).join('')}</ul>`;
  return ctl + svg(f, body) + leg;
}

const txt = (key, label) => ({ label: () => label, get: (r) => r[key] });
const num = (key, label, better, digits) => ({ label: () => label, get: (r) => r[key] ?? null, num: true, better, digits });
const at = (metric, label, better = 'up', fixedK = null) => ({
  label: (s) => `${label}@${fixedK ?? s.k}`, num: true, better,
  get: (r, s) => (((r.by_k || {})[fixedK ?? s.k] || {})[metric] ?? null),
});
const common = [
  { label: () => '실행', get: (r) => r.run },
  { label: () => '만든 시각', get: (r) => (r.created_at || '').replace('T', ' ').replace('+00:00', ' UTC') },
  txt('split', 'split'),
];
const changedCol = { label: () => '바꾼 설정', get: (r) => changedText(r.changed), html: true };

const searchKs = [...new Set(DATA.search.flatMap((r) => r.ks || []))].sort((a, b) => a - b);
const SEARCH_METRICS = [['page_hit', '쪽 Hit'], ['page_recall', '쪽 Recall'], ['page_group_recall', '묶음 Recall'], ['page_mrr', '쪽 MRR'], ['doc_hit', '문서 Hit']];
const ANSWER_METRICS = [['faith_mean', 'Faithfulness'], ['faith_common', '공통 answered'], ['faith_answered', 'answered만'],
  ['status_match', '기대 동작 일치'], ['latency_median', '응답 ms 중앙값'], ['tokens_in', '입력 토큰'], ['tokens_out', '출력 토큰']];
makeTable({
  id: 'search', rows: DATA.search, empty: '검색 비교 결과가 없어요. retrieval_b_compare.py를 먼저 돌리세요.',
  ks: searchKs.length ? searchKs : null, k: searchKs.includes(DATA.k) ? DATA.k : (searchKs[searchKs.length - 1] ?? null),
  defaultView: 'ktable', hideMissingK: true, tableLabel: '파일별 표', tableLast: true, experiments: DATA.search_experiments || [],
  charts: [
    { id: 'ktable', label: 'k별 표', render: (rs, s) => kTable(rs, s) },
    { id: 'line', label: 'k별 그래프', render: (rs, s) => lineChart(rs, s, SEARCH_METRICS, searchKs) },
    { id: 'scatter', label: '좌표 (속도·정확도)', render: (rs, s) => scatterChart(rs, s, [
      { key: 'latency_median', label: () => '검색 ms 중앙값', get: (r) => r.latency_median ?? null },
      ...SEARCH_METRICS.map(([key, label]) => ({ key, label: (st) => `${label}@${st.k}`, get: (r, st) => (((r.by_k || {})[st.k] || {})[key] ?? null) })),
    ], 'latency_median', 'page_hit') },
  ],
  cols: () => [
    ...common, txt('mode', '방식'), changedCol,
    { label: (s) => 'n', get: (r, s) => (((r.by_k || {})[s.k] || {}).n ?? null), num: true, digits: 0 },
    at('page_hit', '쪽 Hit'), at('page_recall', '쪽 Recall'), at('page_group_recall', '묶음 Recall'),
    at('page_mrr', '쪽 MRR'), { ...at('page_hit', '', 'up', '1'), label: () => '1위 쪽 Hit' }, at('doc_hit', '문서 Hit'),
    num('latency_median', '검색 ms 중앙값', 'down', 1),
    { label: () => '오류', get: (r) => r.error ? `<span class="err">${esc(r.error)}</span>` : esc(fmt(r.n_errors, 0)), html: true },
  ],
});

makeTable({
  id: 'diagnose', rows: DATA.diagnose, empty: '리랭크 진단 결과가 없어요. rerank_diagnose.py를 먼저 돌리세요.',
  charts: [
    { id: 'bar', label: '분류 막대', render: (rs, s) => barChart(rs, s, [], { label: '질문 수', keys: DATA.labels, get: (r, k) => (r.labels || {})[k] ?? 0 }) },
  ],
  cols: () => [
    ...common, changedCol, num('k', 'k', null, 0), num('rerank_n', '후보 수', null, 0), num('n', 'n', null, 0),
    ...DATA.labels.map((name) => ({ label: () => name, get: (r) => (r.labels || {})[name] ?? null, num: true, digits: 0,
      better: name === '밀림' || name === '후보 없음' || name === '둘 다 놓침' ? 'down' : (name === '이김' || name === '둘 다 맞힘' ? 'up' : null) })),
    num('same_book', '같은 책', null, 0),
    num('hybrid_page_hit', 'hybrid 쪽 Hit', 'up'), num('rerank_page_hit', 'rerank 쪽 Hit', 'up'),
    num('candidate_page_hit', '후보 안 정답', 'up'),
  ],
});

makeTable({
  id: 'answer', rows: DATA.answer, empty: '답변 평가 결과가 없어요. run_answers.py와 faithfulness.py를 먼저 돌리세요.',
  charts: [
    { id: 'bar', label: '막대 그래프', render: (rs, s) => barChart(rs, s, ANSWER_METRICS) },
    { id: 'scatter', label: '좌표 (비용·품질)', render: (rs, s) => scatterChart(rs, s,
      ANSWER_METRICS.map(([key, label]) => ({ key, label: () => label, get: (r) => r[key] ?? null })), 'tokens_in', 'faith_mean') },
  ],
  cols: () => [
    ...common, txt('mode', '방식'), changedCol, num('n', '답변 수', null, 0),
    num('n_answered', 'answered', null, 0), num('n_needs_clarification', '되묻기', null, 0),
    num('n_abstained', '보류', null, 0), num('n_error', '오류', 'down', 0),
    { label: () => '기대 동작 일치', get: (r) => r.status_match ?? null, num: true, digits: 0, better: 'up' },
    num('faith_mean', 'Faithfulness', 'up'),
    num('faith_common', '공통 answered', 'up'), num('faith_common_n', '공통 문항', null, 0),
    num('faith_answered', 'answered만', 'up'), num('faith_n_scored', '채점 수', null, 0),
    num('faith_n_na', '해당 없음', null, 0), num('faith_n_unavailable', '평가 불가', 'down', 0),
    num('latency_median', '응답 ms 중앙값', 'down', 1), num('retrieval_mean', '검색 ms 평균', 'down', 1),
    num('tokens_in', '입력 토큰', 'down', 0), num('tokens_out', '출력 토큰', 'down', 0),
  ],
});
"""

DIAGNOSE_LABELS = ("둘 다 맞힘", "밀림", "이김", "후보 없음", "둘 다 놓침", "정답 없음")


def build_html(data: Record) -> str:
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    skipped = (
        "".join(f"<li>{escape(s)}</li>" for s in data["skipped"]) or "<li>없음</li>"
    )
    files = (
        "".join(f"<li><code>{escape(f)}</code></li>" for f in data["files"])
        or "<li>없음</li>"
    )
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>실험 비교판</title><style>{CSS}</style></head>
<body><main>
<h1>실험 비교판</h1>
<p class="meta">만든 시각 {escape(data["created_at"])} · 검색 {data["counts"]["search"]}줄 · 리랭크 진단 {data["counts"]["diagnose"]}줄 · 답변 평가 {data["counts"]["answer"]}줄
· <button type="button" id="autorefresh" class="refresh-toggle" title="실험이 끝나 비교판이 다시 만들어지면 이 페이지도 몇 초 안에 저절로 바뀌어요">자동 갱신 켬</button> <span id="last-refresh"></span></p>
<p class="note">각 표 위의 [표] · [그래프] · [좌표] 버튼으로 보기를 바꿔요. 그래프의 점·막대에 마우스를 올리면 값이 보이고, 범례를 누르면 그 줄이 숨겨져요. 거르기 칸은 그래프에도 적용돼요.</p>
<p class="note">기준 실행(파란 줄)과 같은 방식·같은 split끼리 차이를 아래 작은 숫자로 보여 줘요. 초록은 좋아짐, 빨강은 나빠짐이에요(시간·토큰은 낮을수록 좋음).
노란 칸이 기본값과 다르게 바꾼 설정이에요. 질문이 30개 안팎이라 0.05보다 작은 차이는 우연일 수 있어요.</p>

<h2>1. 검색 비교</h2>
<p class="note">retrieval_b_compare.py 결과. 같은 실행 안에서도 a_ 방식은 retrieval_a 설정을 써요. "1위 쪽 Hit"은 k와 상관없이 1위만 봐요.</p>
<div id="search"></div>

<h2>2. 리랭크 진단</h2>
<p class="note">rerank_diagnose.py 결과. 밀림·후보 없음·둘 다 놓침은 적을수록 좋아요.</p>
<div id="diagnose"></div>

<h2>3. 답변 평가</h2>
<p class="note">run_answers.py·faithfulness.py 결과 폴더마다 한 묶음이에요. Faithfulness 평균은 채점된 답변만(0점 포함, 해당 없음·평가 불가 제외).
"공통 answered"는 그 폴더의 모든 방식이 answered로 답한 같은 문항만 평균 낸 값이라, 방식끼리 비교할 때는 이 값이 공정해요.
기대 동작 일치는 골든셋 expected_action과 status가 같은 문항 수예요. 응답 시간 평균과 채점 토큰은 CSV에 있어요.</p>
<div id="answer"></div>

<details><summary>읽은 파일 {len(data["files"])}개 · 건너뛴 파일 {len(data["skipped"])}개</summary>
<p>읽은 파일</p><ul>{files}</ul><p>건너뛴 파일</p><ul>{skipped}</ul></details>
</main>
<script type="application/json" id="board-data">{payload}</script>
<script>{SCRIPT}</script>
</body></html>
"""


def collect(root: Path, k: int = 5) -> Record:
    search, s_skip = collect_search(root)
    diagnose, d_skip = collect_diagnose(root)
    answer, a_skip = collect_answers(root)
    files = sorted({r["file"] for r in (*search, *diagnose, *answer)})
    data = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "k": k,
        "labels": list(DIAGNOSE_LABELS),
        "search": search,
        "search_lines": search_lines(search),
        "search_experiments": search_experiments(search),
        "ktable_metrics": [list(m) for m in KTABLE_METRICS],
        "ktable_default": list(KTABLE_DEFAULT),
        # 실험을 하나 고르면 그 실험 파일끼리만 선을 잇는다
        "search_lines_by_experiment": {
            e: search_lines([r for r in search if r["experiment"] == e])
            for e in {r["experiment"] for r in search}
        },
        "diagnose": diagnose,
        "answer": answer,
        "files": files,
        "skipped": [*s_skip, *d_skip, *a_skip],
        "counts": {
            "search": len(search),
            "diagnose": len(diagnose),
            "answer": len(answer),
        },
    }
    data["version"] = str(
        time.time_ns()
    )  # 페이지가 바뀌었는지 알아보는 값 (experiment_board_version.js)
    lines_by_exp = data["search_lines_by_experiment"]
    data["search_ktables"] = {"": by_k_table(data["search_lines"])} | {
        e: by_k_table(lines) for e, lines in lines_by_exp.items()
    }
    return data


def write_board(root: Path, out: Path, k: int = 5) -> Record:
    data = collect(root, k)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(data), encoding="utf-8")
    stem = out.with_suffix("")
    write_csv(Path(f"{stem}_search.csv"), search_csv_rows(data["search"], k))
    write_csv(Path(f"{stem}_diagnose.csv"), diagnose_csv_rows(data["diagnose"]))
    write_csv(Path(f"{stem}_answer.csv"), answer_csv_rows(data["answer"]))
    write_csv(Path(f"{stem}_search_by_k.csv"), search_by_k_csv_rows(data))
    # 열어 둔 페이지가 3초마다 이 파일을 다시 불러 버전이 바뀌면 새로고침한다 (HTML을 다 쓴 뒤에 쓴다)
    Path(f"{stem}_version.js").write_text(
        f"window.BOARD_VERSION = {json.dumps(data['version'])};\n",
        encoding="utf-8",
    )
    return data


def refresh(root: Path) -> Path | None:
    """실험 스크립트가 끝날 때 불러 비교판을 다시 만든다. 실패해도 경고 한 줄만 찍고 실험은 그대로 끝낸다."""
    out = root / OUT_HTML
    try:
        write_board(root, out)
    except Exception as e:  # noqa: BLE001 — 비교판 문제로 실험 결과를 버리지 않는다
        print(f"[경고] 비교판 갱신 실패: {type(e).__name__}: {e}")
        return None
    print(f"비교판 갱신: {out.resolve().as_uri()}")
    return out


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(
        description="results/의 실험 결과를 한 페이지로 비교"
    )
    parser.add_argument(
        "--root", type=Path, default=None, help="레포 맨 위 (기본: 자동으로 찾음)"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="HTML 경로 (기본: results/experiment_board.html)",
    )
    parser.add_argument(
        "--k", type=int, default=5, help="처음 보여 줄 k와 CSV의 k (기본 5)"
    )
    parser.add_argument(
        "--open", action="store_true", help="만든 뒤 브라우저로 바로 연다"
    )
    args = parser.parse_args(argv)

    root = (args.root or repo_root()).resolve()
    out = args.out or root / OUT_HTML
    data = write_board(root, out, args.k)
    c = data["counts"]
    print(
        f"검색 {c['search']}줄, 리랭크 진단 {c['diagnose']}줄, 답변 평가 {c['answer']}줄"
    )
    for s in data["skipped"]:
        print(f"  건너뜀: {s}")
    print(f"저장: {out}")
    uri = out.resolve().as_uri()
    print(f"결과 페이지: {uri}")
    if args.open:
        webbrowser.open(uri)
    return out


if __name__ == "__main__":
    # Windows 터미널 기본(cp949)으로는 본문의 "–" 같은 글자를 못 찍는다
    sys.stdout.reconfigure(encoding="utf-8")
    main()
