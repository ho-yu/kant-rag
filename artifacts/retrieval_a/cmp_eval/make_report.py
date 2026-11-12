"""collected.json → retrieval_a(기존·Qdrant) / retrieval_b 비교 평가 보고서 PDF (세 난이도, 차트 포함).

실행: uv run --no-project --with matplotlib --with reportlab --with numpy python artifacts/retrieval_a/cmp_eval/make_report.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from statistics import mean, median
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.lib.utils import ImageReader  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle  # noqa: E402

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
_ap = argparse.ArgumentParser()
_ap.add_argument("--difficulty", nargs="*", default=None, help="넣을 난이도 (기본: 결과가 있는 전부)")
_ap.add_argument("--out", default=None)
ARGS = _ap.parse_args()
DATA = json.loads((HERE / "collected.json").read_text(encoding="utf-8"))
_tag = "_".join(ARGS.difficulty) if ARGS.difficulty else "all"
OUT_PDF = Path(ARGS.out) if ARGS.out else HERE / f"retrieval_a_vs_b_{_tag}.pdf"
FIG = HERE / "figs" / _tag
FIG.mkdir(parents=True, exist_ok=True)

_spec = importlib.util.spec_from_file_location("retrieval_scores", ROOT / "evaluation" / "retrieval" / "retrieval_scores.py")
RS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RS)
KS = (1, 3, 5)


def golden(difficulty):
    out = {}
    for split in ("development", "additional", "expansion_validation"):
        path = ROOT / "data" / difficulty / "shipping" / "evaluation" / f"{split}.json"
        if path.exists():
            out.update({q["id"]: q for q in json.loads(path.read_text(encoding="utf-8-sig"))})
    return out

FONT, FONT_B = "C:/Windows/Fonts/NanumGothic.ttf", "C:/Windows/Fonts/NanumGothicBold.ttf"
font_manager.fontManager.addfont(FONT)
font_manager.fontManager.addfont(FONT_B)
plt.rcParams.update({"font.family": "NanumGothic", "axes.unicode_minus": False, "font.size": 9})
pdfmetrics.registerFont(TTFont("NG", FONT))
pdfmetrics.registerFont(TTFont("NGB", FONT_B))
pdfmetrics.registerFontFamily("NG", normal="NG", bold="NGB", italic="NG", boldItalic="NGB")

DIFFS = ("basic", "intermediate", "advanced")
CONFIGS = ("A-기존", "A-Qdrant", "B")
LABEL = {"A-기존": "A · 기존 (청크 전체 + min-max)", "A-Qdrant": "A · Qdrant sparse 후보 + min-max", "B": "B · BM25 + RRF"}
COLOR = {"A-기존": "#2a78d6", "A-Qdrant": "#eb6834", "B": "#1baf7a"}  # 검증한 범주형 1~3번 (청록은 대비가 낮아 값을 직접 표시)
MODES = ("naive", "hybrid", "hybrid_rerank")
MODE_LABEL = {"naive": "naive", "hybrid": "hybrid", "hybrid_rerank": "hybrid + rerank"}
RET = ("hit_at_5", "recall_at_5", "mrr_at_5")
JUDGE = ("judge_facts", "judge_conditions", "judge_evidence", "judge_decision")
LLM = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")  # RAGAS (코드 채점, ragas 라이브러리)
NAME = {"hit_at_5": "Hit@5", "recall_at_5": "Recall@5", "mrr_at_5": "MRR@5",
        "judge_facts": "사실", "judge_conditions": "조건", "judge_evidence": "근거", "judge_decision": "판단",
        "faithfulness": "Faithfulness", "answer_relevancy": "Answer Relevancy",
        "context_precision": "Context Precision", "context_recall": "Context Recall"}
INK, MUTED, GRID, SURFACE = "#1f2933", "#52606d", "#e4e7eb", "#fcfcfb"

# ---------------- 데이터 정리 ----------------
runs = {}
for r in DATA["runs"]:  # 같은 난이도·구성·방식의 분할(development·additional·expansion)을 합친다
    key = (r["difficulty"], r["config"], r["mode"])
    if key not in runs:
        runs[key] = {**r, "items": list(r["items"]), "splits": [r.get("split", "development")]}
    else:
        runs[key]["items"] += r["items"]
        runs[key]["splits"].append(r.get("split", "development"))
for d in DIFFS:  # naive는 sparse를 안 써서 A-기존 = A-Qdrant (한 번만 실행)
    if (d, "A-Qdrant", "naive") in runs and (d, "A-기존", "naive") not in runs:
        runs[(d, "A-기존", "naive")] = {**runs[(d, "A-Qdrant", "naive")], "shared": True}
DONE = [d for d in DIFFS if any(k[0] == d for k in runs) and (not ARGS.difficulty or d in ARGS.difficulty)]
GOLD = {d: golden(d) for d in DONE}


def cells(d):
    return [(d, c, m) for c in CONFIGS for m in MODES if (d, c, m) in runs]


def metric(cell, name):
    items = runs[cell]["items"]
    vals = [it["scores"][name] for it in items if name in it["scores"]]
    st = [str(it["states"].get(name, "")) for it in items if name not in it["scores"]]
    return {"mean": mean(vals) if vals else None, "n": len(vals), "na": sum(s == "na" for s in st),
            "unavailable": sum(s == "unavailable" for s in st), "missing": sum(s == "" for s in st)}


def ops(cell):
    items = runs[cell]["items"]
    ok = [it for it in items if it["status"] != "error"]
    lat = [it["latency_ms"] for it in ok if it["latency_ms"] is not None]
    tok = [it["tokens"] for it in items if it["tokens"]["with_usage"]]
    return {"n": len(items), "errors": len(items) - len(ok), "latency": mean(lat) / 1000 if lat else None,
            "latency_median": median(lat) / 1000 if lat else None,
            "tok_in": sum(t["input"] for t in tok), "tok_out": sum(t["output"] for t in tok),
            "cost": sum(t.get("cost", 0) for t in tok), "n_tok": len(tok)}


ALL = [c for d in DONE for c in cells(d)]
M = {c: {n: metric(c, n) for n in (*RET, *JUDGE, *LLM)} for c in ALL}
O = {c: ops(c) for c in ALL}
N_Q = {d: max((len(runs[c]["items"]) for c in cells(d)), default=0) for d in DONE}
NOISE = {d: (3 / N_Q[d] if N_Q[d] else 0.1) for d in DONE}  # 3문항 미만 차이는 우연 범위


def retrieval_at_k(cell):
    """최종 문맥 순서로 k=1,3,5의 Hit·Recall·MRR (retrieval_scores.py와 같은 계산, 문서 단위)."""
    gold = GOLD[cell[0]]
    out = {}
    for k in KS:
        for metric, fn in (("hit", RS.hit_at_k), ("recall", RS.recall_at_k), ("mrr", RS.mrr_at_k)):
            vals, na = [], 0
            for it in runs[cell]["items"]:
                g = gold.get(it["question_id"])
                rec = {"status": it["status"], "contexts": [{"book_id": b} for b in it.get("context_books") or []]}
                v = fn(rec, g, k)
                if v is None:
                    na += 1
                else:
                    vals.append(v)
            out[f"{metric}@{k}"] = {"mean": mean(vals) if vals else None, "n": len(vals), "skip": na}
    return out


RK = {c: retrieval_at_k(c) for c in ALL}


def fmt(v, d=3):
    return "-" if v is None else f"{v:.{d}f}"


def cell_name(cell, shared_mark=False):
    mark = " ※" if shared_mark and runs[cell].get("shared") else ""
    return f"{cell[1]} · {MODE_LABEL[cell[2]]}{mark}"


# ---------------- 차트 ----------------
def style_axes(ax, ymax=1.0):
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    ax.set_ylim(0, ymax * 1.14)


def bars(ax, groups, getter, fmt_value, top):
    """x: groups, 막대: 구성 3개. 막대 위에 값을 적는다 (색만으로 구분하지 않게)."""
    x = np.arange(len(groups))
    width = 0.26
    for k, cfg in enumerate(CONFIGS):
        vals = [getter(g, cfg) for g in groups]
        bs = ax.bar(x + (k - 1) * width, [v or 0 for v in vals], width - 0.03, color=COLOR[cfg], label=LABEL[cfg], zorder=2)
        for b, v in zip(bs, vals, strict=True):
            if v is not None:
                ax.text(b.get_x() + b.get_width() / 2, v + top * 0.02, fmt_value(v), ha="center", va="bottom",
                        fontsize=6.2, color=INK)
    return x


def save_legend(fig, axes, path):
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def metric_chart(d, names, path):
    fig, axes = plt.subplots(1, len(names), figsize=(3.1 * len(names), 2.8), dpi=200, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names, strict=True):
        x = bars(ax, MODES, lambda m, c, n=name: M.get((d, c, m), {}).get(n, {}).get("mean"), lambda v: f"{v:.2f}", 1.0)
        ax.set_xticks(x, [MODE_LABEL[m] for m in MODES], fontsize=7.5)
        ax.set_title(NAME[name], fontsize=10, color=INK, loc="left", fontweight="bold")
        style_axes(ax)
    save_legend(fig, axes, path)


def k_chart(d, metric, title, path):
    """검색 방식마다 한 칸: x = k (1, 3, 5), 막대 = 구성. 같은 지표를 k만 바꿔 본다."""
    fig, axes = plt.subplots(1, len(MODES), figsize=(9.6, 2.7), dpi=200, sharey=True)
    for ax, m in zip(axes, MODES, strict=True):
        x = bars(ax, KS, lambda k, c, mm=m: RK.get((d, c, mm), {}).get(f"{metric}@{k}", {}).get("mean"),
                 lambda v: f"{v:.2f}", 1.0)
        ax.set_xticks(x, [f"k={k}" for k in KS], fontsize=7.5)
        ax.set_title(f"{title} · {MODE_LABEL[m]}", fontsize=9.5, color=INK, loc="left", fontweight="bold")
        style_axes(ax)
    save_legend(fig, axes, path)


def ops_chart(d, path):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 2.8), dpi=200)
    for ax, key, title, f in ((axes[0], "latency", "평균 응답 시간 (초, 정상 요청)", lambda v: f"{v:.1f}"),
                              (axes[1], "tok_in", "답변 생성 입력 토큰 합계", lambda v: f"{v / 1000:.0f}k")):
        top = max([O[c][key] or 0 for c in cells(d)] or [1]) or 1
        x = bars(ax, MODES, lambda m, c, k=key: O.get((d, c, m), {}).get(k), f, top)
        ax.set_xticks(x, [MODE_LABEL[m] for m in MODES], fontsize=7.5)
        ax.set_title(title, fontsize=10, color=INK, loc="left", fontweight="bold")
        style_axes(ax, top)
    save_legend(fig, axes, path)


def cross_chart(path, mode="hybrid_rerank"):
    """난이도별 비교 (한 방식 고정): Hit@5, 사실, 판단, 응답 시간. 지표마다 따로 그린다."""
    panels = (("hit_at_5", "Hit@5", 1.0), ("judge_facts", "사실 (Judge)", 1.0), ("judge_decision", "판단 (Judge)", 1.0),
              ("latency", "평균 응답 시간 (초)", None))
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 2.9), dpi=200)
    for ax, (key, title, ymax) in zip(axes, panels, strict=True):
        if key == "latency":
            get = lambda dd, c: O.get((dd, c, mode), {}).get("latency")  # noqa: E731
            top = max([O[x]["latency"] or 0 for x in ALL if x[2] == mode] or [1]) or 1
            f = lambda v: f"{v:.1f}"  # noqa: E731
        else:
            get = lambda dd, c, k=key: M.get((dd, c, mode), {}).get(k, {}).get("mean")  # noqa: E731
            top, f = 1.0, (lambda v: f"{v:.2f}")
        x = bars(ax, DONE, get, f, top)
        ax.set_xticks(x, DONE, fontsize=7.5)
        ax.set_title(title, fontsize=10, color=INK, loc="left", fontweight="bold")
        style_axes(ax, top if ymax is None else ymax)
    save_legend(fig, axes, path)


def heatmap(d, name, path):
    rows = cells(d)
    qids = sorted({it["question_id"] for c in rows for it in runs[c]["items"] if it["question_id"]})
    grid = np.full((len(rows), len(qids)), np.nan)
    for i, cell in enumerate(rows):
        by_q = {it["question_id"]: it for it in runs[cell]["items"]}
        for j, q in enumerate(qids):
            it = by_q.get(q)
            if it and name in it["scores"]:
                grid[i, j] = it["scores"][name]
    fig, ax = plt.subplots(figsize=(9.6, 0.34 * len(rows) + 1.1), dpi=200)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#e8f1fc", "#2a78d6", "#0b3a75"])
    cmap.set_bad("#eef0f2")
    ax.imshow(np.ma.masked_invalid(grid), aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    for i in range(len(rows)):
        for j in range(len(qids)):
            if np.isnan(grid[i, j]):
                ax.add_patch(matplotlib.patches.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="////",
                                                          edgecolor="#b8bec5", linewidth=0))
    ax.set_yticks(range(len(rows)), [cell_name(r) for r in rows], fontsize=6.8)
    ax.set_xticks(range(len(qids)), [q.split("-")[-1] for q in qids], fontsize=6)
    ax.set_xticks(np.arange(-0.5, len(qids)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="both", length=0, colors=MUTED)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel("문항 번호", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return qids, grid


# ---------------- 해석 (숫자로 자동 작성) ----------------
def group_best(d, names):
    avg = {}
    for c in cells(d):
        vals = [M[c][n]["mean"] for n in names if M[c][n]["mean"] is not None]
        if vals:
            avg[c] = mean(vals)
    return avg


def findings(d):
    out = []
    for title, names in (("검색(Hit·Recall·MRR)", RET), ("답변 Judge 4항목", JUDGE), ("답변 RAGAS 4지표", LLM)):
        avg = group_best(d, names)
        if not avg:
            continue
        top = max(avg, key=avg.get)
        rest = sorted(avg.values(), reverse=True)
        gap = rest[0] - rest[1] if len(rest) > 1 else None
        tail = ""
        if gap is not None:
            tail = (f" 2위와 차이 {gap:.3f}로 우연 범위({NOISE[d]:.2f}) 안이라 확실한 1위라고 보기 어렵습니다."
                    if gap < NOISE[d] else f" 2위와 차이 {gap:.3f}.")
        out.append(f"<b>{title}</b>: 평균이 가장 높은 구성은 <b>{cell_name(top)}</b> ({avg[top]:.3f}).{tail}")
    ds = diffs(d)
    if ds:
        big = [x for x in ds if abs(x[4]) >= NOISE[d]]
        out.append(f"<b>A · 기존 vs A · Qdrant</b>: {len(ds)}개 지표 중 우연 범위({NOISE[d]:.2f}) 이상 차이 <b>{len(big)}개</b>"
                   + (". 품질은 사실상 같습니다." if not big else f" ({', '.join(f'{MODE_LABEL[m]} {NAME[n]}' for m, n, *_ in big)})."))
    lat = {c: O[c]["latency"] for c in cells(d) if O[c]["latency"]}
    if lat:
        fast, slow = min(lat, key=lat.get), max(lat, key=lat.get)
        errs = sum(O[c]["errors"] for c in cells(d) if not runs[c].get("shared"))
        out.append(f"<b>운영</b>: 가장 빠른 {cell_name(fast)} {lat[fast]:.1f}초, 가장 느린 {cell_name(slow)} {lat[slow]:.1f}초. "
                   f"시스템 오류 {errs}건.")
    return out


def diffs(d):
    out = []
    for m in ("hybrid", "hybrid_rerank"):
        a, q = (d, "A-기존", m), (d, "A-Qdrant", m)
        if a in M and q in M:
            for n in (*RET, *JUDGE, *LLM):
                va, vq = M[a][n]["mean"], M[q][n]["mean"]
                if va is not None and vq is not None:
                    out.append((m, n, va, vq, vq - va))
    return out


# ---------------- PDF 부품 ----------------
base = ParagraphStyle("b", fontName="NG", fontSize=9.3, leading=14.3, textColor=colors.HexColor(INK), wordWrap="CJK")
small = ParagraphStyle("s", parent=base, fontSize=8.1, leading=12)
cellst = ParagraphStyle("c", parent=base, fontSize=7.4, leading=10.2)
cellb = ParagraphStyle("cb", parent=cellst, fontName="NGB")
muted = ParagraphStyle("m", parent=base, fontSize=8.4, leading=12.5, textColor=colors.HexColor(MUTED))
title = ParagraphStyle("t", parent=base, fontName="NGB", fontSize=20, leading=27)
h1 = ParagraphStyle("h1", parent=base, fontName="NGB", fontSize=14, leading=20, textColor=colors.HexColor("#2457a6"),
                    spaceBefore=8, spaceAfter=5, keepWithNext=1)
h2 = ParagraphStyle("h2", parent=base, fontName="NGB", fontSize=10.5, leading=15, spaceBefore=6, spaceAfter=3, keepWithNext=1)
W = 170 * mm


def P(t, s=base):
    return Paragraph(t, s)


def bullets(items, s=base):
    return [Paragraph(t, s, bulletText="•") for t in items]


def table(rows, widths, header=True, bold_first=True):
    data = [[Paragraph(str(v), cellb if (header and r == 0) or (bold_first and c == 0) else cellst)
             for c, v in enumerate(row)] for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd2d9")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("LEFTPADDING", (0, 0), (-1, -1), 3.2), ("RIGHTPADDING", (0, 0), (-1, -1), 3.2),
          ("TOPPADDING", (0, 0), (-1, -1), 2.3), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.3)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7")))
    t.setStyle(TableStyle(st))
    return t


def img(path, width=W):
    iw, ih = ImageReader(str(path)).getSize()
    return Image(str(path), width=width, height=width * ih / iw)


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("NG", 7.5)
    canvas.setFillColor(colors.HexColor(MUTED))
    canvas.drawString(20 * mm, 11 * mm, "retrieval_a / retrieval_b 비교 평가")
    canvas.drawRightString(190 * mm, 11 * mm, str(doc.page))
    canvas.restoreState()


# ---------------- 문서 ----------------
s = [P("retrieval_a / retrieval_b 비교 평가", title),
     P(f"난이도 {' · '.join(DONE)} · 분할 {', '.join(sorted({sp for k, r in runs.items() if k[0] in DONE for sp in r['splits']}))} · "
       f"수집 {escape(DATA['collected_at'])}", muted), Spacer(1, 8),
     P("1. 무엇을 비교했나", h1),
     table([["구성", "dense", "sparse", "점수 합치기", "리랭커"],
            ["A · 기존", "bge-m3", "bge-m3 sparse. 청크를 모두 받아 직접 계산", "전체 청크 min-max 가중합 0.7 / 0.3", "bge-reranker-v2-m3, 후보 50"],
            ["A · Qdrant", "bge-m3", "bge-m3 sparse. Qdrant m3 벡터로 후보 100개", "후보 안 min-max 가중합 0.7 / 0.3", "bge-reranker-v2-m3, 후보 50"],
            ["B", "bge-m3", "BM25 (Kiwi)", "RRF (k=60), 각 30개 후보", "bge-reranker-v2-m3, 후보 50"]],
           [22 * mm, 15 * mm, 58 * mm, 42 * mm, 33 * mm]),
     Spacer(1, 4),
     P("모든 구성이 같은 질문, 같은 답변 프롬프트와 생성 모델(gpt-6-luna, reasoning low), 같은 최종 문맥 수(k=5), "
       "같은 채점기(Langfuse 서버 evaluator)를 씁니다. naive는 sparse를 쓰지 않아 A · 기존과 A · Qdrant가 같으므로 한 번만 실행해 "
       "두 구성에 함께 표시했습니다(※).", small),
     Spacer(1, 3),
     table([["A · Qdrant 설정 변경 (실험 이후)", "이 보고서의 실험", "현재 hybrid.yaml", "advanced hybrid + rerank Hit@5 / Recall@5 / MRR@5 (검색 지표만)"],
            ["후보 수 candidate_k", "100", "500", "0.828 / 0.480 / 0.760 → 0.879 / 0.494 / 0.811"],
            ["가중치 dense : sparse", "0.7 : 0.3", "0.5 : 0.5", "basic · intermediate는 같거나 조금 좋아짐"]],
           [40 * mm, 26 * mm, 26 * mm, 78 * mm]),
     P("이 보고서의 답변 점수(Judge·RAGAS)는 실험 당시 설정으로 만든 답변 기준입니다. 현재 설정은 검색 지표로만 확인했습니다.", small),
     P("2. 지표 읽는 법", h1),
     table([["묶음", "지표", "뜻", "범위"],
            ["검색", "Hit@5 / Recall@5 / MRR@5", "참고 근거 문서(evidence_groups의 book_id)를 상위 5개 문맥에서 찾았나 · 몇 개 찾았나 · 몇 위에 처음 나왔나 (문서 단위, 발제 기준)", "0~1"],
            ["답변 Judge", "사실 / 조건 / 근거 / 판단", "평가 LLM이 참고 정답·채점 기준과 비교해 항목별 충족(1) 또는 미충족(0). 평균 = 충족률", "0~1"],
            ["답변 RAGAS", "Faithfulness 등 4개", "ragas 라이브러리로 코드에서 계산 (Faithfulness는 RAGAS 공식 예제: 질문·답변·실제 최종 문맥). 채점 모델 gpt-6-luna", "0~1"],
            ["운영", "응답 시간 / 오류 / 토큰", "질문부터 답변까지 시간(정상 요청 평균), 시스템 오류 수, 답변 생성 입력·출력 토큰", ""]],
           [20 * mm, 36 * mm, 100 * mm, 14 * mm]),
     Spacer(1, 3),
     P("평균은 채점된 문항만으로 계산하고, 해당 없음·평가 불가는 따로 셉니다. 난이도별 문항 수가 n일 때 "
       "<b>3문항(3/n) 미만 차이는 우연일 수 있습니다</b>(예: 60문항이면 0.05). 같은 문서를 찾은 구성끼리도 답변 점수가 이 정도 흔들렸습니다.", small)]

if len(DONE) > 1:
    cross_chart(FIG / "cross.png")
    s += [P("3. 난이도별 한눈에 보기 (hybrid + rerank)", h1), img(FIG / "cross.png"),
          P("서비스 기본 후보인 hybrid + rerank만 골라 난이도별로 나란히 놓았습니다. 방식별 전체 결과는 난이도 절에 있습니다.", small)]

sec = 4 if len(DONE) > 1 else 3
for d in DONE:
    metric_chart(d, RET, FIG / f"{d}_retrieval.png")
    for metric, ttl in (("hit", "Hit@k"), ("recall", "Recall@k"), ("mrr", "MRR@k")):
        k_chart(d, metric, ttl, FIG / f"{d}_{metric}_k.png")
    metric_chart(d, JUDGE, FIG / f"{d}_judge.png")
    metric_chart(d, LLM, FIG / f"{d}_llm.png")
    ops_chart(d, FIG / f"{d}_ops.png")
    qids, grid = heatmap(d, "judge_facts", FIG / f"{d}_heat_facts.png")
    summary = [["구성 · 방식", "Hit@5", "Recall@5", "MRR@5", "사실", "조건", "근거", "판단", "Faithful.", "응답(초)", "오류"]]
    for c in cells(d):
        summary.append([cell_name(c, True), *[fmt(M[c][n]["mean"]) for n in RET], *[fmt(M[c][n]["mean"], 2) for n in JUDGE],
                        fmt(M[c]["faithfulness"]["mean"], 2), fmt(O[c]["latency"], 1), str(O[c]["errors"])])
    s += [PageBreak(), P(f"{sec}. {d} ({N_Q[d]}문항)", h1), table(summary, [36 * mm] + [12.4 * mm] * 10), Spacer(1, 5),
          P("핵심 결과", h2), *bullets(findings(d)),
          P("검색 성능 (k=5)", h2), img(FIG / f"{d}_retrieval.png")]
    krows = [["구성 · 방식", *[f"Hit@{k}" for k in KS], *[f"Recall@{k}" for k in KS], *[f"MRR@{k}" for k in KS], "채점 / 제외"]]
    for c in cells(d):
        r = RK[c]
        krows.append([cell_name(c, True), *[fmt(r[f"hit@{k}"]["mean"]) for k in KS], *[fmt(r[f"recall@{k}"]["mean"]) for k in KS],
                      *[fmt(r[f"mrr@{k}"]["mean"]) for k in KS], f"{r['hit@5']['n']} / {r['hit@5']['skip']}"])
    s += [PageBreak(), P("검색 지표 전체 (k = 1, 3, 5)", h2),
          table(krows, [34 * mm] + [13 * mm] * 9 + [19 * mm]),
          P("최종 문맥(모델에 들어간 순서)의 상위 k개로 계산했습니다. Hit@k = 참고 문서를 하나라도 찾은 문항 비율, "
            "Recall@k = 찾은 참고 문서 수 / 전체 참고 문서 수, MRR@k = 참고 문서가 처음 나온 순위의 역수 평균. "
            "'제외'는 근거 문서가 없는 문항(해당 없음)과 서비스 오류 문항 수입니다.", small),
          img(FIG / f"{d}_hit_k.png"), img(FIG / f"{d}_recall_k.png"), img(FIG / f"{d}_mrr_k.png")]
    s += [PageBreak(), P("답변 품질 · Judge 4항목 (충족률)", h2), img(FIG / f"{d}_judge.png"),
          P("답변 품질 · RAGAS 4지표 (ragas 라이브러리)", h2),
          img(FIG / f"{d}_llm.png") if any(M[c][n]["mean"] is not None for c in cells(d) for n in LLM)
          else P("RAGAS 점수가 아직 없습니다.", small),
          P("운영 지표", h2), img(FIG / f"{d}_ops.png")]
    ds = diffs(d)
    if ds:
        rows = [["방식", "지표", "A · 기존", "A · Qdrant", "차이 (Q - 기존)", "판정"]]
        rows += [[MODE_LABEL[m], NAME[n], fmt(va), fmt(vq), f"{dv:+.3f}", "차이 있음" if abs(dv) >= NOISE[d] else "우연 범위"]
                 for m, n, va, vq, dv in ds]
        s += [PageBreak(), P("A · 기존 vs A · Qdrant 직접 비교", h2), table(rows, [26 * mm, 32 * mm, 24 * mm, 24 * mm, 30 * mm, 34 * mm])]
    orows = [["구성 · 방식", "요청", "오류", "오류율", "평균 응답(초)", "중앙값(초)", "입력 토큰", "출력 토큰", "생성 비용($)"]]
    for c in cells(d):
        o = O[c]
        orows.append([cell_name(c, True), o["n"], o["errors"], f"{o['errors'] / o['n'] * 100:.0f}%" if o["n"] else "-",
                      fmt(o["latency"], 1), fmt(o["latency_median"], 1), f"{o['tok_in']:,}", f"{o['tok_out']:,}", f"{o['cost']:.3f}"])
    s += [P("운영 지표 표", h2), table(orows, [36 * mm, 11 * mm, 11 * mm, 13 * mm, 19 * mm, 17 * mm, 22 * mm, 20 * mm, 21 * mm]),
          P("응답 시간은 평가 스크립트가 잰 질문 → 답변 시간(검색 + 리랭킹 + 생성)이고, 생성 비용은 Langfuse 가격표로 계산한 답변 생성 비용입니다. "
            "서버 evaluator 채점 비용은 포함하지 않습니다. A · 기존은 실행마다 첫 요청에서 청크를 모두 받아 오는 시간(advanced 약 7분)이 "
            "앞쪽 문항들의 응답 시간에 섞여 평균이 크게 나올 수 있어, 중앙값을 함께 봅니다.", small),
          P(f"문항별 사실(judge_facts) 지도", h2), img(FIG / f"{d}_heat_facts.png"),
          P("진한 파랑 = 충족(1), 연한 색 = 미충족(0), 빗금 = 해당 없음 또는 점수 없음. 열 전체가 연한 문항은 어떤 구성도 못 맞힌 문항이라 "
            "검색 방식보다 청킹·프롬프트·정답 해석 쪽을 봐야 합니다.", small)]
    hard = [qids[j] for j in range(len(qids)) if not np.all(np.isnan(grid[:, j])) and np.nanmax(grid[:, j]) == 0]
    if hard:
        s += [P(f"모든 구성이 사실 0점인 문항 ({len(hard)}개): {', '.join(hard)}", small)]
    sec += 1

s += [PageBreak(), P(f"{sec}. 한계", h1), *bullets([
    "질문은 공개된 골든 데이터셋(development·additional·expansion_validation)이라 비공개 테스트 성능으로 볼 수 없습니다.",
    "답변 점수는 LLM 채점이라 같은 답변도 실행마다 조금씩 달라질 수 있습니다. 대표 사례는 원문과 대조해야 합니다.",
    "검색 지표는 문서 단위입니다. 같은 문서의 다른 쪽을 찾아도 맞은 것으로 셉니다.",
    "A · 기존과 A · Qdrant의 hybrid는 정규화 범위(전체 vs 후보)가 달라 순위가 조금 다를 수 있습니다. A · Qdrant는 실험 이후 후보 500, 가중치 0.5 : 0.5로 바꿨습니다(1절 표).",
    "RAGAS는 실험 뒤에 저장된 답변·최종 문맥으로 따로 계산해 올렸습니다(답변은 다시 만들지 않음). 같은 팀 채점 코드(quality.py)를 씁니다.",
    "Judge는 Langfuse 서버 evaluator가 채점합니다. 해당 없음·평가 불가는 이유 첫머리 표시([해당 없음]·[평가 불가])로 읽어 평균에서 뺍니다(server_judge_summary.py 규칙).",
], small)]

doc = SimpleDocTemplate(str(OUT_PDF), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=16 * mm,
                        bottomMargin=18 * mm, title="retrieval_a / retrieval_b 비교 평가")
doc.build(s, onFirstPage=on_page, onLaterPages=on_page)
print("saved", OUT_PDF)
