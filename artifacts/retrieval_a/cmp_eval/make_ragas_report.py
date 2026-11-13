"""collected.json → RAGAS 결과만 정리한 PDF (retrieval_a 기존 · Qdrant / retrieval_b, 세 난이도).

실행: uv run --no-project --with matplotlib --with reportlab --with numpy python artifacts/retrieval_a/cmp_eval/make_ragas_report.py
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

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
DATA = json.loads((HERE / "collected.json").read_text(encoding="utf-8"))
OUT = HERE / "ragas_results.pdf"
FIG = HERE / "figs" / "ragas"
FIG.mkdir(parents=True, exist_ok=True)

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
SHORT = {"A-기존": "A · 기존", "A-Qdrant": "A · Qdrant", "B": "B"}
COLOR = {"A-기존": "#2a78d6", "A-Qdrant": "#eb6834", "B": "#1baf7a"}  # 검증한 범주형 1~3번 (청록은 대비가 낮아 값을 직접 표시)
MODES = ("naive", "hybrid", "hybrid_rerank")
MODE_LABEL = {"naive": "naive", "hybrid": "hybrid", "hybrid_rerank": "hybrid + rerank"}
R = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
NAME = {"faithfulness": "Faithfulness", "answer_relevancy": "Answer Relevancy",
        "context_precision": "Context Precision", "context_recall": "Context Recall"}
STATUS = {"answered": "답변", "needs_clarification": "되묻기", "abstained": "보류"}
INK, MUTED, GRID, SURFACE = "#1f2933", "#52606d", "#e4e7eb", "#fcfcfb"

runs = {}
for r in DATA["runs"]:
    runs.setdefault((r["difficulty"], r["config"], r["mode"]), []).extend(r["items"])
for d in DIFFS:  # naive는 sparse를 안 써서 A-기존 = A-Qdrant (한 번만 실행)
    if (d, "A-Qdrant", "naive") in runs:
        runs.setdefault((d, "A-기존", "naive"), runs[(d, "A-Qdrant", "naive")])


def items(d, c, m):
    if d == "전체":
        return [i for x in DIFFS for i in runs.get((x, c, m), [])]
    return runs.get((d, c, m), [])


def avg(its, k):
    v = [i["scores"][k] for i in its if k in i["scores"]]
    return mean(v) if v else None


def counts(its, k):
    return sum(1 for i in its if k in i["scores"]), sum(1 for i in its if k not in i["scores"])


def f(v, d=3):
    return "-" if v is None else f"{v:.{d}f}"


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    ax.set_ylim(0, 1.14)


def chart(d, path):
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 2.9), dpi=200, sharey=True)
    for ax, k in zip(axes, R, strict=True):
        x = np.arange(len(MODES))
        for j, c in enumerate(CONFIGS):
            vals = [avg(items(d, c, m), k) for m in MODES]
            bs = ax.bar(x + (j - 1) * 0.26, [v or 0 for v in vals], 0.23, color=COLOR[c], label=LABEL[c], zorder=2)
            for b, v in zip(bs, vals, strict=True):
                if v is not None:
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=6.0, color=INK)
        ax.set_xticks(x, [MODE_LABEL[m] for m in MODES], fontsize=7.5)
        ax.set_title(NAME[k], fontsize=10, color=INK, loc="left", fontweight="bold")
        style_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def difficulty_chart(path, mode="hybrid_rerank"):
    """난이도별로 같은 방식(hybrid + rerank)의 4지표를 나란히: 난이도가 어려워질수록 어떻게 떨어지나."""
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 2.9), dpi=200, sharey=True)
    for ax, k in zip(axes, R, strict=True):
        x = np.arange(len(DIFFS))
        for j, c in enumerate(CONFIGS):
            vals = [avg(items(d, c, mode), k) for d in DIFFS]
            bs = ax.bar(x + (j - 1) * 0.26, [v or 0 for v in vals], 0.23, color=COLOR[c], label=LABEL[c], zorder=2)
            for b, v in zip(bs, vals, strict=True):
                if v is not None:
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=6.0, color=INK)
        ax.set_xticks(x, DIFFS, fontsize=7.5)
        ax.set_title(NAME[k], fontsize=10, color=INK, loc="left", fontweight="bold")
        style_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, facecolor="white")
    plt.close(fig)


# ---------------- PDF ----------------
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
    canvas.drawString(20 * mm, 11 * mm, "RAGAS 평가 결과")
    canvas.drawRightString(190 * mm, 11 * mm, str(doc.page))
    canvas.restoreState()


def metric_table(d):
    rows = [["방식", "구성", "n", *[NAME[k] for k in R]]]
    hl = []
    for m in ("hybrid_rerank", "hybrid", "naive"):
        for c in CONFIGS:
            its = items(d, c, m)
            if not its:
                continue
            best = {k: max(avg(items(d, x, m), k) or 0 for x in CONFIGS) for k in R}
            cells = [f"<b>{f(avg(its, k))}</b>" if abs((avg(its, k) or -1) - best[k]) < 1e-9 else f(avg(its, k)) for k in R]
            label = SHORT[c] + (" ※" if m == "naive" and c == "A-기존" else "")
            rows.append([MODE_LABEL[m], label, len(its), *cells])
            if m == "hybrid_rerank":
                hl.append(len(rows) - 1)
    return table(rows, [26 * mm, 26 * mm, 12 * mm, 26.5 * mm, 26.5 * mm, 26.5 * mm, 26.5 * mm], hl)


n_items = sum(len(r["items"]) for r in DATA["runs"])
scored = {k: sum(1 for r in DATA["runs"] for i in r["items"] if k in i["scores"]) for k in R}

s = [P("RAGAS 평가 결과", title),
     P(f"retrieval_a (기존 · Qdrant) / retrieval_b · 골든 데이터셋 192문항 · 답변 {n_items:,}건 · 수집 {DATA['collected_at'][:16].replace('T', ' ')}", muted),
     Spacer(1, 8),
     P("1. RAGAS 4지표", h1),
     table([["지표", "묻는 것", "계산", "정답 사용"],
            ["Faithfulness", "답변이 문맥에 있는 내용만 말했나", "답변을 짧은 주장들로 나누고, 각 주장이 실제 최종 문맥에서 확인되는 비율", "안 씀"],
            ["Answer Relevancy", "질문에 맞는 답인가", "답변으로 질문을 거꾸로 만들어 원래 질문과 얼마나 비슷한지 (임베딩 유사도)", "안 씀"],
            ["Context Precision", "쓸모 있는 문맥이 위쪽에 있나", "문맥마다 참고 정답에 도움이 되는지 판정하고, 위쪽일수록 큰 가중치", "씀"],
            ["Context Recall", "정답에 필요한 내용을 문맥이 담았나", "참고 정답을 주장들로 나누고, 각 주장이 문맥에서 확인되는 비율", "씀"]],
           [30 * mm, 42 * mm, 80 * mm, 18 * mm]),
     Spacer(1, 4),
     *bullets([
         "ragas 0.4.3 라이브러리로 코드에서 계산했습니다(팀 채점 코드 evaluation/answer/quality.py). Faithfulness는 RAGAS 공식 예제대로 "
         "질문 → user_input, 생성 답변 → response, 실제 최종 문맥 → retrieved_contexts.",
         "채점 모델 gpt-6-luna, Answer Relevancy의 유사도 계산은 bge-m3 임베딩. 세 구성에 같은 채점 모델·설정을 썼습니다.",
         f"채점 결과: 지표마다 {min(scored.values()):,}건 / {n_items:,}건 채점, 평가 불가 0건 (일시적 채점 호출 실패 221건은 다시 채점함).",
         "모두 0~1, 높을수록 좋습니다. 질문에 따라 1문항이 0.014~0.017이라 <b>0.03 안팎의 차이는 우연 범위</b>로 봅니다.",
     ], small),
     P("비교 구성", h2),
     table([["구성", "sparse", "점수 합치기", "리랭커"],
            ["A · 기존", "bge-m3 sparse, 청크 전체에 직접 계산", "전체 청크 min-max 0.7 : 0.3", "bge-reranker-v2-m3, 후보 50"],
            ["A · Qdrant", "bge-m3 sparse, Qdrant 후보 100개 (실험 당시)", "후보 안 min-max 0.7 : 0.3 (실험 당시)", "bge-reranker-v2-m3, 후보 50"],
            ["B", "BM25 (Kiwi), 후보 30개", "RRF (k = 60)", "bge-reranker-v2-m3, 후보 50"]],
           [24 * mm, 60 * mm, 50 * mm, 36 * mm]),
     P("세 구성 모두 같은 질문, 같은 답변 프롬프트와 생성 모델(gpt-6-luna), 같은 최종 문맥 수(k = 5)를 씁니다. "
       "naive는 sparse를 쓰지 않아 A · 기존과 A · Qdrant가 같은 실행입니다(※).", small),
     PageBreak(),
     P("2. 192문항 전체", h1), metric_table("전체"),
     P("노란 줄 = 서비스 기본 방식(hybrid + rerank). 굵은 글씨 = 같은 방식 안에서 가장 높은 값.", small)]
chart("전체", FIG / "all.png")
s += [img(FIG / "all.png")]

difficulty_chart(FIG / "by_difficulty.png")
s += [P("난이도별 (hybrid + rerank)", h2), img(FIG / "by_difficulty.png")]

for i, d in enumerate(DIFFS, start=3):
    chart(d, FIG / f"{d}.png")
    n = len(items(d, "B", "naive"))
    s += [PageBreak(), P(f"{i}. {d} ({n}문항)", h1), metric_table(d), Spacer(1, 4), img(FIG / f"{d}.png")]

# 답변 상태별
srows = [["지표", "구성", *[f"{STATUS[st]} 평균 (건수)" for st in STATUS]]]
for k in R:
    for c in CONFIGS:
        its = items("전체", c, "hybrid_rerank")
        cells = []
        for st in STATUS:
            v = [i["scores"][k] for i in its if i["status"] == st and k in i["scores"]]
            cells.append(f"{mean(v):.3f} ({len(v)})" if v else "-")
        srows.append([NAME[k], SHORT[c], *cells])
s += [PageBreak(), P("6. 답변 상태별 (192문항, hybrid + rerank)", h1),
      table(srows, [32 * mm, 24 * mm, 38 * mm, 38 * mm, 38 * mm]),
      P("되묻기와 보류는 사실 주장이 적고 질문에 직접 답하지 않아서 Answer Relevancy가 낮게 나옵니다. 그래서 이 지표는 상태별로 나눠 봐야 합니다.", small),
      P("7. 읽는 법", h1),
      *bullets([
          "<b>방식 차이가 구성 차이보다 큽니다.</b> naive → hybrid → hybrid + rerank로 갈수록 네 지표가 모두 오릅니다(192문항 Context Precision "
          f"{f(avg(items('전체', 'A-기존', 'naive'), 'context_precision'))} → 약 0.90, Context Recall {f(avg(items('전체', 'A-기존', 'naive'), 'context_recall'))} → 약 0.71). "
          "리랭커가 쓸모 있는 문맥을 위로 올리고 필요한 내용을 더 많이 담습니다.",
          "<b>같은 방식 안에서 A와 B의 차이는 대부분 0.01~0.03으로 우연 범위입니다.</b> naive는 A와 B가 같은 문서를 찾는데도 Faithfulness가 "
          f"{f(avg(items('전체', 'A-기존', 'naive'), 'faithfulness'))} 대 {f(avg(items('전체', 'B', 'naive'), 'faithfulness'))}로 달랐습니다. "
          "그만큼 LLM 답변 생성과 채점 자체가 흔들립니다.",
          "<b>advanced가 가장 낮습니다.</b> Context Recall이 0.5 안팎이라 정답에 필요한 내용의 절반쯤만 문맥에 들어갑니다. "
          "Answer Relevancy도 0.53~0.57로, 되묻기·보류가 많았습니다. 검색과 답변 모두 개선 여지가 가장 큰 난이도입니다.",
          "Faithfulness는 basic·intermediate에서 0.83~0.90으로, 답변이 대체로 문맥에 근거합니다. 정답 여부(사실·조건)는 RAGAS가 아니라 Judge 4항목으로 봅니다.",
      ], small),
      Spacer(1, 4),
      note("A · Qdrant는 이 실험 이후 후보 500, 가중치 0.5 : 0.5로 바꿨습니다(advanced hybrid + rerank 검색 Hit@5 0.828 → 0.879). "
           "이 보고서의 RAGAS는 실험 당시 설정(후보 100, 0.7 : 0.3)으로 만든 답변 기준입니다.", "참고")]

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=16 * mm, bottomMargin=18 * mm,
                        title="RAGAS 평가 결과")
doc.build(s, onFirstPage=on_page, onLaterPages=on_page)
print("saved", OUT)
