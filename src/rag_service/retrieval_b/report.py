"""검색 비교 결과(JSON)를 브라우저로 보는 HTML 한 장으로 바꾼다.

retrieval_b_compare.py가 저장하는 report 딕셔너리를 그대로 받는다.
CSS와 스크립트가 파일 안에 들어 있어서 인터넷 없이 더블클릭으로 열린다.

    from rag_service.retrieval_b.report import write_html
    write_html(report, out.with_suffix(".html"), items=items)

page_link()는 터미널에서 Ctrl+클릭으로 여는 file:// 주소를 돌려주고,
open_browser=True면 기본 브라우저로 바로 연다.
"""

from __future__ import annotations

import webbrowser
from collections.abc import Sequence
from html import escape
from pathlib import Path
from typing import Any

METRIC_LABELS = {
    "doc_hit": "문서 Hit",
    "doc_recall": "문서 Recall",
    "doc_group_recall": "문서 묶음 Recall",
    "page_hit": "쪽 Hit",
    "page_recall": "쪽 Recall",
    "page_group_recall": "쪽 묶음 Recall",
    "doc_mrr": "문서 MRR",
    "page_mrr": "쪽 MRR",
}
ACTION_LABELS = {"answer": "답변", "clarify": "되묻기", "abstain": "답변 보류"}
GROUP_LABELS = {
    "all_missed": "모든 방식이 놓침",
    "split": "방식마다 갈림",
    "all_found": "모든 방식이 찾음",
    "no_gold": "채점 제외",
}

CSS = """
:root { --fg:#1f2328; --muted:#656d76; --bg:#ffffff; --panel:#f6f8fa; --line:#d0d7de;
  --bar:#0969da; --ok:#1a7f37; --bad:#cf222e; --best:#fff8c5; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e6edf3; --muted:#9198a1; --bg:#0d1117; --panel:#161b22; --line:#30363d;
    --bar:#4493f8; --ok:#3fb950; --bad:#f85149; --best:#3a3000; } }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
  font: 14px/1.55 -apple-system, "Segoe UI", "Malgun Gothic", "Apple SD Gothic Neo", sans-serif; }
main { max-width: 76rem; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 32px 0 6px; }
.meta, .note { color: var(--muted); font-size: 13px; margin: 0 0 10px; }
.fail { border: 1px solid var(--bad); color: var(--bad); border-radius: 6px; padding: 8px 12px; margin: 8px 0; }
.box { overflow-x: auto; border: 1px solid var(--line); border-radius: 6px; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { padding: 6px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
th { background: var(--panel); font-weight: 600; white-space: nowrap; }
tr:last-child td { border-bottom: 0; }
td.num, th.num { text-align: right; white-space: nowrap; }
tr.sep td { border-top: 2px solid var(--line); }
td.best { background: var(--best); font-weight: 700; }
.bar { display: inline-block; vertical-align: middle; width: 48px; height: 6px; margin-left: 6px;
  background: var(--panel); border: 1px solid var(--line); border-radius: 3px; overflow: hidden; }
.bar i { display: block; height: 100%; background: var(--bar); }
.ok { color: var(--ok); font-weight: 600; } .bad { color: var(--bad); font-weight: 600; } .na { color: var(--muted); }
td.ok, td.bad, td.na, td.nw { white-space: nowrap; }
td.more { min-width: 16rem; }
.filters { display: flex; flex-wrap: wrap; gap: 6px; margin: 0 0 10px; }
.filters button { font: inherit; font-size: 13px; padding: 4px 10px; border-radius: 999px; cursor: pointer;
  border: 1px solid var(--line); background: var(--bg); color: var(--fg); }
.filters button[aria-pressed="true"] { background: var(--bar); border-color: var(--bar); color: #fff; }
details summary { cursor: pointer; color: var(--muted); }
.detail { margin: 6px 0 0; font-size: 12px; }
.detail dt { font-weight: 600; margin-top: 4px; }
.detail dd { margin: 0; }
code { font: 12px ui-monospace, SFMono-Regular, Consolas, monospace; color: var(--muted); }
td.q { min-width: 18rem; }
.defs dt { font-weight: 600; margin-top: 6px; } .defs dd { margin: 0 0 0 0; color: var(--muted); }
"""

SCRIPT = """
document.querySelectorAll('.filters button').forEach((b) => b.addEventListener('click', () => {
  document.querySelectorAll('.filters button').forEach((o) => o.setAttribute('aria-pressed', o === b));
  const want = b.dataset.group;
  document.querySelectorAll('tr[data-group]').forEach((tr) => {
    tr.hidden = want !== 'all' && tr.dataset.group !== want;
  });
}));
"""


def _num(v: float | None) -> str:
    return "-" if v is None else f"{v:.3f}"


def _bar(v: float | None) -> str:
    if v is None:
        return ""
    pct = max(0.0, min(1.0, float(v))) * 100
    return f'<span class="bar"><i style="width:{pct:.0f}%"></i></span>'


def _pages(pairs: Sequence[Sequence[Any]]) -> str:
    return ", ".join(f"{b} p{p}" for b, p in pairs) or "(없음)"


def _metric_rows(
    ok: dict[str, Any], ks: Sequence[int], metrics: Sequence[str]
) -> list[str]:
    """k마다 방식별 행. 같은 k 안에서 가장 높은 값(동점 포함)을 강조한다."""
    rows = []
    for i, k in enumerate(ks):
        summaries = {m: r["summary"][str(k)] for m, r in ok.items()}
        best = {}
        for name in metrics:
            # 예전 결과 JSON에는 MRR 같은 새 지표가 없을 수 있어 .get으로 읽는다
            values = [
                s.get(name) for s in summaries.values() if s.get(name) is not None
            ]
            if len(values) > 1 and max(values) != min(values):
                best[name] = max(values)
        for j, (mode, s) in enumerate(summaries.items()):
            cls = ' class="sep"' if i and not j else ""
            cells = []
            for name in metrics:
                value = s.get(name)
                mark = " best" if name in best and value == best[name] else ""
                cells.append(f'<td class="num{mark}">{_num(value)}{_bar(value)}</td>')
            rows.append(
                f"<tr{cls}><td>{escape(mode)}</td><td class='num'>{k}</td>"
                f"<td class='num'>{s['n']}</td>{''.join(cells)}</tr>"
            )
    return rows


def _question_group(marks: Sequence[int | None]) -> str:
    scored = [m for m in marks if m is not None]
    if not scored:
        return "no_gold"
    if all(scored):
        return "all_found"
    if not any(scored):
        return "all_missed"
    return "split"


def _question_rows(
    ok: dict[str, Any], by_id: dict[str, dict[str, Any]], k: int
) -> tuple[list[str], dict[str, int]]:
    per_mode = {m: {it["id"]: it for it in r["items"]} for m, r in ok.items()}
    first = next(iter(ok.values()))["items"]
    counts = dict.fromkeys(GROUP_LABELS, 0)
    rows = []
    for it in first:
        qid = it["id"]
        cells, marks, detail = [], [], []
        for mode in ok:
            row = per_mode[mode].get(qid)
            s = row["scores"][str(k)] if row else None
            if not s or s["page_hit"] is None:
                marks.append(None)
                cells.append('<td class="na">-</td>')
                continue
            marks.append(s["page_hit"])
            n_groups = row.get("n_groups") or 0
            filled = round((s["page_group_recall"] or 0) * n_groups)
            sym, cls = ("○", "ok") if s["page_hit"] else ("×", "bad")
            frac = f" ({filled}/{n_groups})" if n_groups else ""
            cells.append(f'<td class="{cls}">{sym}{frac}</td>')
            ids = ", ".join(map(str, s["found_chunk_ids"]))
            detail.append(
                f"<dt>{escape(mode)}</dt>"
                f"<dd>놓친 정답 쪽: {escape(_pages(s['missing_pages']))}</dd>"
                f"<dd>상위 {k}: <code>{escape(ids)}</code></dd>"
            )
            if row.get("error"):
                detail.append(f"<dd class='bad'>오류: {escape(str(row['error']))}</dd>")
        group = _question_group(marks)
        counts[group] += 1
        gold = by_id.get(qid, {})
        action = ACTION_LABELS.get(
            it.get("expected_action"), it.get("expected_action") or "-"
        )
        more = (
            f"<details><summary>보기</summary><dl class='detail'>{''.join(detail)}</dl></details>"
            if detail
            else ""
        )
        rows.append(
            f'<tr data-group="{group}"><td class="nw">{escape(qid)}</td>'
            f"<td class='nw'>{escape(action)}</td>"
            f"<td class='q'>{escape(gold.get('question', '-'))}</td>{''.join(cells)}"
            f"<td class='more'>{more}</td></tr>"
        )
    return rows, counts


def render_html(
    report: dict[str, Any],
    items: Sequence[dict[str, Any]] | None = None,
    detail_k: int = 5,
) -> str:
    """report를 HTML 문자열로 만든다.

    items(골든셋 질문 목록)를 넘기면 질문별 표에 질문 문장을 같이 적는다.
    detail_k: 요약과 질문별 표를 몇 개 기준으로 볼지. ks에 없으면 가장 큰 k를 쓴다.
    """
    settings = report.get("settings", {})
    results: dict[str, Any] = report.get("results", {})
    ks = [int(k) for k in settings.get("ks", [])]
    ok = {m: r for m, r in results.items() if "error" not in r}
    failed = {m: r["error"] for m, r in results.items() if "error" in r}
    if ks and detail_k not in ks:
        detail_k = ks[-1]
    by_id = {it["id"]: it for it in items or []}

    title = " / ".join(
        [
            str(settings.get("difficulty", "?")),
            str(settings.get("split", "?")),
            "+".join(settings.get("levels", [])),
        ]
    )
    n = next(iter(ok.values()))["summary"][str(ks[0])]["n"] if ok and ks else "-"
    meta = (
        f"실행 {escape(str(report.get('created_at', '-')))} (UTC) · 커밋 "
        f"{escape(str(report.get('git_commit') or '-'))} · 방식 {escape(', '.join(results))} · "
        f"k {', '.join(map(str, ks))} · 채점 {n}문항 (정답 쪽이 없는 질문은 평균에서 뺌)"
    )
    body = [f"<h1>검색 비교 결과: {escape(title)}</h1>", f"<p class='meta'>{meta}</p>"]
    body += [
        f"<div class='fail'><b>{escape(m)} 실패:</b> {escape(str(e))}</div>"
        for m, e in failed.items()
    ]

    if ok:
        metrics = list(METRIC_LABELS)
        head = "".join(
            f"<th class='num'>{escape(v)}</th>" for v in METRIC_LABELS.values()
        )
        table_head = (
            f"<tr><th>방식</th><th class='num'>k</th><th class='num'>n</th>{head}</tr>"
        )

        body += [
            f"<h2>한눈에 보기 (k={detail_k})</h2>",
            "<p class='note'>같은 k 안에서 가장 높은 값을 강조했다. 막대는 0~1 사이 값의 크기다.</p>",
            f"<div class='box'><table>{table_head}{''.join(_metric_rows(ok, [detail_k], metrics))}</table></div>",
            "<h2>k별 점수</h2>",
            f"<div class='box'><table>{table_head}{''.join(_metric_rows(ok, ks, metrics))}</table></div>",
        ]

        speed = "".join(
            f"<tr><td>{escape(m)}</td><td class='num'>{r['latency_ms']['mean']} ms</td>"
            f"<td class='num'>{r['latency_ms']['median']} ms</td><td class='num'>{r['latency_ms']['max']} ms</td>"
            f"<td class='num'>{r.get('build_seconds', '-')} 초</td><td class='num'>{r['n_errors']}</td>"
            f"<td class='num'>{r.get('short_results', '-')}</td></tr>"
            for m, r in ok.items()
        )
        body += [
            "<h2>속도와 오류</h2>",
            "<p class='note'>검색 시간은 질문 1개당. 준비 시간은 모델·인덱스를 처음 불러오는 데 걸린 시간이다.</p>",
            (
                "<div class='box'><table><tr><th>방식</th><th class='num'>평균</th><th class='num'>중앙값</th>"
                "<th class='num'>최대</th><th class='num'>준비 시간</th><th class='num'>오류</th>"
                f"<th class='num'>k개 미만 반환</th></tr>{speed}</table></div>"
            ),
        ]

        q_rows, counts = _question_rows(ok, by_id, detail_k)
        buttons = [
            f"<button data-group='all' aria-pressed='true'>전체 {sum(counts.values())}</button>"
        ]
        buttons += [
            f"<button data-group='{g}' aria-pressed='false'>{label} {counts[g]}</button>"
            for g, label in GROUP_LABELS.items()
        ]
        mode_heads = "".join(f"<th>{escape(m)}</th>" for m in ok)
        body += [
            f"<h2>질문별 결과 (k={detail_k})</h2>",
            (
                "<p class='note'>○ = 상위 k개 안에 정답 쪽이 있음, × = 없음, - = 정답 쪽이 없어 채점 제외. "
                "괄호는 채운 근거 묶음 수 / 전체 묶음 수. '보기'를 누르면 놓친 정답 쪽과 찾은 청크가 나온다.</p>"
            ),
            f"<div class='filters'>{''.join(buttons)}</div>",
            (
                "<div class='box'><table><tr><th>ID</th><th>기대 행동</th><th>질문</th>"
                f"{mode_heads}<th>자세히</th></tr>{''.join(q_rows)}</table></div>"
            ),
        ]

    body += [
        "<h2>지표 뜻</h2>",
        (
            "<dl class='defs'>"
            "<dt>Hit@k</dt><dd>상위 k개 안에 정답을 하나라도 찾았으면 1, 아니면 0. 질문들의 평균.</dd>"
            "<dt>Recall@k</dt><dd>골든셋 근거(모든 evidence_groups 합집합) 중 상위 k개에서 찾은 비율.</dd>"
            "<dt>묶음 Recall@k</dt><dd>evidence_groups 묶음 하나는 '이 중 하나면 충분'이다. 후보를 하나라도 찾은 묶음의 비율.</dd>"
            "<dt>문서 / 쪽</dt><dd>문서는 book_id만 맞으면 정답, 쪽은 (book_id, page)까지 맞아야 정답.</dd>"
            "<dt>주의</dt><dd>골든셋에 없는 쪽이라도 실제 근거면 정답으로 친다(11/6 튜터 답변). "
            "이 점수는 그걸 반영하지 못해 실제보다 낮게 나올 수 있다.</dd>"
            "</dl>"
        ),
    ]
    return (
        "<!doctype html>\n<html lang='ko'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>검색 비교 결과: {escape(title)}</title><style>{CSS}</style></head>"
        f"<body><main>{''.join(body)}</main><script>{SCRIPT}</script></body></html>\n"
    )


def write_html(
    report: dict[str, Any],
    path: str | Path,
    items: Sequence[dict[str, Any]] | None = None,
    detail_k: int = 5,
) -> Path:
    """render_html 결과를 UTF-8 파일로 저장하고 경로를 돌려준다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(report, items, detail_k), encoding="utf-8")
    return path


def page_link(path: str | Path, open_browser: bool = False) -> str:
    """file:// 주소를 돌려준다. VS Code·Windows 터미널에서 Ctrl+클릭하면 브라우저로 열린다.

    open_browser=True면 기본 브라우저(크롬 등)로 바로 연다.
    """
    uri = Path(path).resolve().as_uri()
    if open_browser:
        webbrowser.open(uri)
    return uri
