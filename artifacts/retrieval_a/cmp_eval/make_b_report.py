"""retrieval_b(B 모델) 최종 문맥 수 k 종합 평가 PDF: 품질 · 토큰 · 속도 · 효율.

데이터: k_sweep.json(1단계 검색), collected.json(오전 k5 basic·intermediate), collected_k.json(k5 advanced·k8·k10),
collected_b_adv_k7_mine.json(advanced k7)
실행: uv run --no-project --with matplotlib --with reportlab --with numpy python artifacts/retrieval_a/cmp_eval/make_b_report.py
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
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
OUT = HERE / "retrieval_b_k_evaluation.pdf"
FIG = HERE / "figs" / "b"
FIG.mkdir(parents=True, exist_ok=True)

FONT, FONT_B = "C:/Windows/Fonts/NanumGothic.ttf", "C:/Windows/Fonts/NanumGothicBold.ttf"
font_manager.fontManager.addfont(FONT)
font_manager.fontManager.addfont(FONT_B)
plt.rcParams.update({"font.family": "NanumGothic", "axes.unicode_minus": False, "font.size": 9})
pdfmetrics.registerFont(TTFont("NG", FONT))
pdfmetrics.registerFont(TTFont("NGB", FONT_B))
pdfmetrics.registerFontFamily("NG", normal="NG", bold="NGB", italic="NG", boldItalic="NGB")

DIFFS = ("basic", "intermediate", "advanced")
KS = (5, 8, 10)
JUDGE = ("judge_facts", "judge_decision", "judge_evidence", "judge_conditions")
JLABEL = {"judge_facts": "사실", "judge_decision": "결론", "judge_evidence": "근거", "judge_conditions": "조건"}
INK, MUTED, GRID, SURFACE = "#1f2933", "#52606d", "#e4e7eb", "#fcfcfb"
BLUE, GREEN, ORANGE = "#2a78d6", "#1baf7a", "#eb6834"

# ---------- 데이터 ----------
SWEEP = json.loads((HERE / "k_sweep.json").read_text(encoding="utf-8"))
OLD = json.loads((HERE / "collected.json").read_text(encoding="utf-8"))["runs"]
NEW = json.loads((HERE / "collected_k.json").read_text(encoding="utf-8"))["runs"]
K7 = json.loads((HERE / "collected_b_adv_k7_mine.json").read_text(encoding="utf-8"))["runs"]

runs = {}
for r in OLD:  # k5 basic · intermediate: 오전 비교 실험
    if r["split"] == "development" and r["mode"] == "hybrid_rerank" and r["config"] == "B" and r["difficulty"] != "advanced":
        runs[(r["difficulty"], 5)] = r
for r in NEW:
    if r["config"] in ("B-k5", "B-k8", "B-k10"):
        k = int(r["config"].split("k")[1])
        if k == 5 and r["difficulty"] != "advanced":
            continue
        runs[(r["difficulty"], k)] = r
runs[("advanced", 7)] = K7[0]


def items(d, k):
    if d == "전체":
        return [i for x in DIFFS for i in runs[(x, k)]["items"]]
    return runs[(d, k)]["items"] if (d, k) in runs else []


def stats(its, k):
    judged = [i for i in its if any(n in i["scores"] for n in JUDGE)]
    ok = sum(1 for i in judged if all(i["scores"].get(n, 1.0) == 1.0 for n in JUDGE))

    def a(n):
        v = [i["scores"][n] for i in its if n in i["scores"]]
        return mean(v) if v else None

    tin = sum(i["tokens"]["input"] for i in its)
    tout = sum(i["tokens"]["output"] for i in its)
    lat = sorted((i["latency_ms"] or 0) / 1000 for i in its)
    return {
        "n": len(its), "ok": ok, "acc": ok / len(judged), **{n: a(n) for n in JUDGE},
        "faith": a("faithfulness"), "cr": a("context_recall"), "cp": a("context_precision"), "ar": a("answer_relevancy"),
        "hit": a(f"hit_at_{k}"), "rec": a(f"recall_at_{k}"), "mrr": a(f"mrr_at_{k}"),
        "tin": tin / len(its), "tout": tout / len(its), "tot": tin + tout,
        "p50": median(lat), "p90": lat[int(len(lat) * 0.9) - 1], "mean": mean(lat),
        "eff": ok / ((tin + tout) / 1e5),
        **{s: sum(1 for i in its if i["status"] == s) for s in ("answered", "needs_clarification", "abstained")},
    }


S = {(d, k): stats(items(d, k), k) for d in (*DIFFS, "전체") for k in KS}
S[("advanced", 7)] = stats(items("advanced", 7), 7)
SW = {d: next(x for x in SWEEP if x["difficulty"] == d and x["impl"] == "retrieval_b" and x["mode"] == "hybrid_rerank"
              and x["setting"] == "후보 30 · 리랭커 50") for d in DIFFS}
SW100 = {d: next(x for x in SWEEP if x["difficulty"] == d and x["impl"] == "retrieval_b" and x["mode"] == "hybrid_rerank"
                 and x["setting"] == "후보 100 · 리랭커 100") for d in DIFFS}


def f(v, d=3):
    return "-" if v is None else f"{v:.{d}f}"


# ---------- 그림 ----------
def style(ax, top=None):
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    if top:
        ax.set_ylim(0, top)


def fig_accuracy(path):
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 2.8), dpi=200, sharey=True)
    for ax, d in zip(axes, ("전체", *DIFFS), strict=True):
        ks = [k for k in (5, 7, 8, 10) if (d, k) in S]
        ys = [S[(d, k)]["acc"] for k in ks]
        ax.plot(ks, ys, color=GREEN, linewidth=2, marker="o", markersize=5, markeredgecolor="white", markeredgewidth=1.2, zorder=3)
        for k, y in zip(ks, ys, strict=True):
            ax.annotate(f"{y:.2f}", (k, y), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7, color=INK)
        ax.set_xticks(ks)
        ax.set_xlim(4, 11)
        ax.set_xlabel("k", fontsize=7.5, color=MUTED)
        ax.set_title(f"{d} · 정답률", fontsize=10, color=INK, loc="left", fontweight="bold")
        style(ax, 0.85)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def fig_tradeoff(path):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 3.0), dpi=200)
    ax = axes[0]
    xs = [S[("전체", k)]["tin"] for k in KS]
    ys = [S[("전체", k)]["acc"] for k in KS]
    ax.plot(xs, ys, color=GREEN, linewidth=2, marker="o", markersize=6, markeredgecolor="white", markeredgewidth=1.2, zorder=3)
    for k, x, y in zip(KS, xs, ys, strict=True):
        ax.annotate(f"k{k}  {y:.3f}", (x, y), xytext=(6, -10), textcoords="offset points", fontsize=7.5, color=INK)
    ax.set_xlabel("문항당 입력 토큰", fontsize=7.5, color=MUTED)
    ax.set_title("토큰을 더 쓸수록 정답률 (세 난이도 합계)", fontsize=10, color=INK, loc="left", fontweight="bold")
    style(ax)
    ax.set_ylim(0.35, 0.56)
    ax = axes[1]
    effs = [S[("전체", k)]["eff"] for k in KS]
    bars = ax.bar([f"k{k}" for k in KS], effs, color=BLUE, width=0.5, zorder=2)
    for b, v in zip(bars, effs, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.1f}", ha="center", fontsize=7.5, color=INK)
    ax.set_title("토큰 10만 개당 정답 수", fontsize=10, color=INK, loc="left", fontweight="bold")
    style(ax, 16)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def fig_stage1(path):
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 2.7), dpi=200, sharey=True)
    ks = (1, 3, 5, 8, 10, 15)
    for ax, d in zip(axes, DIFFS, strict=True):
        for key, color, label in (("hit", BLUE, "문서 Hit@k"), ("page_recall", ORANGE, "쪽 Recall@k")):
            ys = [SW[d][f"{key}@{k}"] for k in ks]
            ax.plot(ks, ys, color=color, linewidth=2, marker="o", markersize=4.5, markeredgecolor="white", markeredgewidth=1, label=label, zorder=3)
            ax.annotate(f"{ys[-1]:.2f}", (ks[-1], ys[-1]), xytext=(4, 0), textcoords="offset points", fontsize=7, color=INK, va="center")
        for k in KS:
            ax.axvline(k, color=GRID, linewidth=0.8, zorder=1)
        ax.set_xticks(ks)
        ax.set_xlim(0, 17)
        ax.set_xlabel("k", fontsize=7.5, color=MUTED)
        ax.set_title(d, fontsize=10, color=INK, loc="left", fontweight="bold")
        style(ax, 1.05)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(path, facecolor="white")
    plt.close(fig)


fig_accuracy(FIG / "accuracy.png")
fig_tradeoff(FIG / "tradeoff.png")
fig_stage1(FIG / "stage1.png")

# ---------- PDF ----------
base = ParagraphStyle("b", fontName="NG", fontSize=9.3, leading=14.3, textColor=colors.HexColor(INK), wordWrap="CJK")
small = ParagraphStyle("s", parent=base, fontSize=8.1, leading=12)
cellst = ParagraphStyle("c", parent=base, fontSize=7.6, leading=10.5)
cellb = ParagraphStyle("cb", parent=cellst, fontName="NGB")
muted = ParagraphStyle("m", parent=base, fontSize=8.4, leading=12.5, textColor=colors.HexColor(MUTED))
title = ParagraphStyle("t", parent=base, fontName="NGB", fontSize=20, leading=27)
h1 = ParagraphStyle("h1", parent=base, fontName="NGB", fontSize=14, leading=20, textColor=colors.HexColor("#2457a6"),
                    spaceBefore=8, spaceAfter=5, keepWithNext=1)
h2 = ParagraphStyle("h2", parent=base, fontName="NGB", fontSize=10.5, leading=15, spaceBefore=6, spaceAfter=3, keepWithNext=1)
W = 170 * mm


def P(t, s=base):
    return Paragraph(t, s)


def bullets(xs, s=base):
    return [Paragraph(t, s, bulletText="•") for t in xs]


def table(rows, widths, hl=()):
    data = [[Paragraph(str(v), cellb if r == 0 or c == 0 else cellst) for c, v in enumerate(row)] for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd2d9")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("LEFTPADDING", (0, 0), (-1, -1), 3.5), ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
          ("TOPPADDING", (0, 0), (-1, -1), 2.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6),
          ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7"))]
    for r in hl:
        st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#fff8e6")))
    t.setStyle(TableStyle(st))
    return t


def img(path):
    iw, ih = ImageReader(str(path)).getSize()
    return Image(str(path), width=W, height=W * ih / iw)


def note(text, label):
    t = Table([[Paragraph(f"<b>{label}</b>&nbsp;&nbsp;{text}", small)]], colWidths=[W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff8e6")),
                           ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor("#e0b84c")),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 5),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    return t


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("NG", 7.5)
    canvas.setFillColor(colors.HexColor(MUTED))
    canvas.drawString(20 * mm, 11 * mm, "retrieval_b · 최종 문맥 수 k 평가")
    canvas.drawRightString(190 * mm, 11 * mm, str(doc.page))
    canvas.restoreState()


def best_row(keys, getter, low=False):
    vals = [getter(k) for k in keys]
    nums = [v for v in vals if v is not None]
    best = (min if low else max)(nums)
    return [abs(v - best) < 1e-9 if v is not None else False for v in vals]


def cell(v, is_best, fmt):
    s = fmt(v)
    return f"<b>{s}</b>" if is_best else s


def scorecard():
    T = [S[("전체", k)] for k in KS]
    spec = [
        ("정답률", lambda s: s["acc"], lambda v: f"{v:.3f}", False, lambda s: f" ({s['ok']})"),
        ("Judge 사실 / 결론 / 근거 / 조건", None, None, False, None),
        ("Faithfulness", lambda s: s["faith"], f, False, None),
        ("Context Recall", lambda s: s["cr"], f, False, None),
        ("Context Precision", lambda s: s["cp"], f, False, None),
        ("Answer Relevancy", lambda s: s["ar"], f, False, None),
        ("검색 Hit@k / Recall@k", None, None, False, None),
        ("문항당 입력 토큰", lambda s: s["tin"], lambda v: f"{v:,.0f}", True, None),
        ("문항당 출력 토큰", lambda s: s["tout"], lambda v: f"{v:,.0f}", True, None),
        ("응답 시간 중앙값 / 상위 10%", None, None, False, None),
        ("토큰 10만 개당 정답 수", lambda s: s["eff"], lambda v: f"{v:.1f}", False, None),
        ("답함 / 되묻기 / 보류", None, None, False, None),
    ]
    rows = [["항목", *[f"k{k}" for k in KS]]]
    for name, get, fmt, low, extra in spec:
        if name.startswith("Judge"):
            rows.append([name, *[" / ".join(f"{s[n]:.2f}" for n in JUDGE) for s in T]])
        elif name.startswith("검색"):
            rows.append([name, *[f"{s['hit']:.2f} / {s['rec']:.2f}" for s in T]])
        elif name.startswith("응답"):
            rows.append([name, *[f"{s['p50']:.1f}초 / {s['p90']:.1f}초" for s in T]])
        elif name.startswith("답함"):
            rows.append([name, *[f"{s['answered']} / {s['needs_clarification']} / {s['abstained']}" for s in T]])
        else:
            bests = best_row(range(3), lambda j: get(T[j]), low)
            rows.append([name, *[cell(get(s), b, fmt) + (extra(s) if extra else "") for s, b in zip(T, bests, strict=True)]])
    return table(rows, [52 * mm, 39 * mm, 39 * mm, 40 * mm], hl=(1, 11))


def diff_table(d):
    ks = [k for k in (5, 7, 8, 10) if (d, k) in S]
    rows = [["k", "정답률", "Judge 사실·결론·근거·조건", "Faith.", "C.Recall", "C.Prec.", "Ans.Rel.", "Hit / Recall@k",
             "입력 토큰", "시간 중앙값", "10만 토큰당 정답", "답함·되묻기·보류"]]
    for k in ks:
        s = S[(d, k)]
        rows.append([f"k{k}", f"{s['acc']:.3f} ({s['ok']})", " · ".join(f"{s[n]:.2f}" for n in JUDGE), f(s["faith"]), f(s["cr"]),
                     f(s["cp"]), f(s["ar"]), f"{s['hit']:.2f} / {s['rec']:.2f}", f"{s['tin']:,.0f}", f"{s['p50']:.1f}초",
                     f"{s['eff']:.1f}", f"{s['answered']}·{s['needs_clarification']}·{s['abstained']}"])
    return table(rows, [9 * mm, 17 * mm, 25 * mm, 11 * mm, 13 * mm, 12 * mm, 12 * mm, 17 * mm, 13 * mm, 14 * mm, 14 * mm, 13 * mm])


T5, T8, T10 = (S[("전체", k)] for k in KS)
story = [P("retrieval_b · 최종 문맥 수 k 평가", title),
         P("B 모델(bge-m3 dense + BM25, RRF, 리랭커)에서 LLM에 넣는 최종 청크 수 k를 바꿔 품질 · 토큰 · 속도 · 효율을 비교했습니다.", muted),
         Spacer(1, 6),
         note(f"<b>k = 10.</b> 정답률 {T10['acc']:.3f}로 가장 높고(k5 {T5['acc']:.3f}), 토큰은 k5보다 {(T10['tin'] / T5['tin'] - 1) * 100:.0f}% 더 쓰지만 "
              f"토큰 10만 개당 정답 수는 {T10['eff']:.1f}로 k8({T8['eff']:.1f})과 같은 최고 수준입니다. "
              f"토큰을 아껴야 하면 k8(정답률 {T8['acc']:.3f}, 토큰 −{(1 - T8['tin'] / T10['tin']) * 100:.0f}%), "
              "난이도별로 나눌 수 있으면 basic · intermediate 10, advanced 8.", "결론"),
         Spacer(1, 4),
         P("B 모델 구성", h2),
         table([["단계", "설정"],
                ["dense 검색", "bge-m3, Qdrant kant_shipping_{난이도}_bge-m3, 후보 30개"],
                ["키워드 검색", "BM25 (Kiwi 형태소), 청크를 Qdrant에서 받아 메모리에서 계산, 후보 30개"],
                ["합치기", "RRF: 1/(60 + 순위)를 dense와 BM25에서 1:1로 더함 (가중치 없음)"],
                ["리랭커", "bge-reranker-v2-m3: RRF 상위 50개를 다시 매겨 최종 k개"],
                ["답변 · 채점", "답변 gpt-6-luna, RAGAS 채점 gpt-6-luna + bge-m3 임베딩, Judge는 Langfuse 서버 평가기"]],
               [30 * mm, 140 * mm]),
         P("종합 평가 (세 난이도 합계 90문항)", h1), scorecard(),
         P("굵은 글씨는 그 항목에서 가장 좋은 값(토큰은 가장 적은 값). 정답률 = 서버 Judge 4항목을 모두 통과한 문항 비율, 괄호는 문항 수.", muted)]

story += [PageBreak(), P("토큰 효율", h1), img(FIG / "tradeoff.png"),
          table([["", "k5", "k8", "k10"],
                 ["90문항 총 토큰 (입력 + 출력)", f"{T5['tot']:,}", f"{T8['tot']:,}", f"{T10['tot']:,}"],
                 ["토큰 10만 개당 정답 수", f"{T5['eff']:.1f}", f"<b>{T8['eff']:.1f}</b>", f"<b>{T10['eff']:.1f}</b>"],
                 ["정답 1개를 더 맞히는 데 든 추가 토큰", "(기준)", f"k5 대비 {(T8['tot'] - T5['tot']) / (T8['ok'] - T5['ok']):,.0f}",
                  f"k8 대비 {(T10['tot'] - T8['tot']) / (T10['ok'] - T8['ok']):,.0f}"],
                 ["문항 1,000개 처리 시 토큰 (추정)", f"약 {T5['tot'] / 90 * 1000 / 1e4:,.0f}만", f"약 {T8['tot'] / 90 * 1000 / 1e4:,.0f}만",
                  f"약 {T10['tot'] / 90 * 1000 / 1e4:,.0f}만"]],
                [62 * mm, 36 * mm, 36 * mm, 36 * mm])]
story += bullets([
    "<b>k5는 토큰이 가장 적지만 효율은 가장 낮습니다.</b> 정답을 놓치는 문항이 많아 토큰당 정답 수가 13.0입니다.",
    "<b>k8과 k10은 효율이 같습니다(13.5).</b> k10은 같은 효율로 정답을 4개 더 맞힙니다.",
    "출력 토큰은 k와 관계없이 문항당 230~250개로 거의 같습니다. 늘어나는 것은 문맥으로 들어가는 입력 토큰입니다.",
    "비용($)은 Langfuse 서버에 gpt-6-luna 가격이 등록되어 있지 않아 계산하지 않았습니다. 토큰 기준으로 k10은 k8보다 약 10% 비쌉니다.",
], small)

story += [PageBreak(), P("난이도별 결과", h1), img(FIG / "accuracy.png")]
for d in DIFFS:
    story += [P(d, h2), diff_table(d)]
story += [Spacer(1, 4),
          table([["난이도", "정답률 (k5 → k8 → k10)", "토큰 10만 개당 정답", "최적"],
                 ["basic", f"{S[('basic', 5)]['acc']:.2f} → {S[('basic', 8)]['acc']:.2f} → <b>{S[('basic', 10)]['acc']:.2f}</b>",
                  f"{S[('basic', 5)]['eff']:.1f} → {S[('basic', 8)]['eff']:.1f} → <b>{S[('basic', 10)]['eff']:.1f}</b>", "<b>k10</b>: 품질 · 효율 모두 최고"],
                 ["intermediate", f"{S[('intermediate', 5)]['acc']:.2f} → {S[('intermediate', 8)]['acc']:.2f} → <b>{S[('intermediate', 10)]['acc']:.2f}</b>",
                  f"<b>{S[('intermediate', 5)]['eff']:.1f}</b> → {S[('intermediate', 8)]['eff']:.1f} → {S[('intermediate', 10)]['eff']:.1f}", "품질은 k10, 효율은 k5"],
                 ["advanced", f"{S[('advanced', 5)]['acc']:.2f} → (k7 {S[('advanced', 7)]['acc']:.2f}) → <b>{S[('advanced', 8)]['acc']:.2f}</b> → {S[('advanced', 10)]['acc']:.2f}",
                  f"{S[('advanced', 5)]['eff']:.1f} → ({S[('advanced', 7)]['eff']:.1f}) → <b>{S[('advanced', 8)]['eff']:.1f}</b> → {S[('advanced', 10)]['eff']:.1f}",
                  "<b>k8</b>: k10과 정답 수가 같고 토큰은 적음"]],
                [24 * mm, 56 * mm, 50 * mm, 40 * mm])]

story += [PageBreak(), P("1단계: 검색만으로 본 k", h1),
          P("LLM 없이 검색 결과만 채점했습니다 (난이도별 정답 근거가 있는 전체 문항). 세로선은 2단계에서 답변까지 돌린 k = 5 · 8 · 10.", muted),
          img(FIG / "stage1.png"),
          table([["난이도", "구성", "k3", "k5", "k8", "k10", "k15", "검색 시간"],
                 *[[d, label, *[f"{x[f'hit@{k}']:.2f} / {x[f'page_recall@{k}']:.2f}" for k in (3, 5, 8, 10, 15)], f"{x['sec_mean']:.2f}초"]
                   for d in DIFFS for label, x in (("후보 30 · 리랭커 50 (현재)", SW[d]), ("후보 100 · 리랭커 100", SW100[d]))]],
                [22 * mm, 34 * mm, 17 * mm, 17 * mm, 17 * mm, 17 * mm, 17 * mm, 17 * mm], hl=(1, 3, 5)),
          P("값은 문서 Hit@k / 쪽 Recall@k.", muted)]
story += bullets([
    "문서 Hit은 k5에서 거의 꽉 찹니다. 쪽 Recall은 k를 늘릴수록 오르며 k15까지도 계속 오릅니다.",
    "<b>후보 수는 지금 값(30 · 50)이 가장 좋습니다.</b> 100 · 100은 점수가 같거나 낮고 검색 시간만 약 40% 늘어납니다.",
    "advanced는 정답이 후보 밖에 있는 것이 아니라, 회사 · 날짜만 다른 비슷한 문서와 순위 경쟁에서 밀리는 것이 원인이라 후보를 늘려도 좋아지지 않습니다.",
], small)

story += [P("읽을 때 주의할 점", h1)]
story += bullets([
    "난이도마다 30문항이라 1문항이 0.033입니다. 같은 advanced k5를 두 번 돌렸을 때 4문항과 5문항으로 1문항 차이가 났습니다. 판단은 세 난이도 합계(90문항)를 기준으로 했고, k5와 k10의 차이는 11문항, k8과 k10은 4문항입니다.",
    "k5의 basic · intermediate는 오전 비교 실험, advanced는 저녁에 다시 돌린 실험입니다. 오전 advanced k5(5/30)를 쓰면 합계 정답률은 0.411입니다.",
    "응답 시간 평균(8~10초)이 중앙값(5초)보다 큰 것은 advanced 몇 문항이 Qdrant 재시도로 30초 넘게 걸렸기 때문입니다. k와 관계없으며, retrieval_b에 Qdrant 재시도를 넣으면 줄어듭니다.",
    "advanced 정답률이 낮은 것(최대 0.23)은 k 문제가 아닙니다. 계산에 필요한 문서 3~4개 중 일부만 찾아 모델이 되묻기 · 보류로 답한 경우가 대부분입니다. 회사명 · 날짜 · 건 번호로 거르는 검색이 필요합니다.",
], small)

story += [P("실험 목록 (Langfuse)", h1),
          table([["난이도", "k", "실험 이름"],
                 *[[d, f"k{k}", runs[(d, k)]["experiment"]] for d in DIFFS for k in (5, 7, 8, 10) if (d, k) in runs]],
                [26 * mm, 12 * mm, 132 * mm]),
          P("Langfuse → Datasets → shipping-{난이도}-development-v1 → Experiments. 실험마다 accuracy_mean, judge_*_mean, RAGAS 평균, tokens_in_mean, latency_median이 실험 점수로 올라가 있습니다.", muted)]

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=16 * mm, bottomMargin=18 * mm,
                        title="retrieval_b 최종 문맥 수 k 평가")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print("saved", OUT)
