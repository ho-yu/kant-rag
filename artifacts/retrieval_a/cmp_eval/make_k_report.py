"""최적 k 실험 보고서 PDF: 1단계(검색만, k_sweep.json) + 2단계(전체 체인, collected.json의 k5 + collected_k.json의 k8·k10)
+ Langfuse에서 결과 확인하는 방법.

실행: uv run --no-project --with matplotlib --with reportlab --with numpy python artifacts/retrieval_a/cmp_eval/make_k_report.py
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
OUT = HERE / "k_experiment.pdf"
FIG = HERE / "figs" / "k"
FIG.mkdir(parents=True, exist_ok=True)

FONT, FONT_B = "C:/Windows/Fonts/NanumGothic.ttf", "C:/Windows/Fonts/NanumGothicBold.ttf"
font_manager.fontManager.addfont(FONT)
font_manager.fontManager.addfont(FONT_B)
plt.rcParams.update({"font.family": "NanumGothic", "axes.unicode_minus": False, "font.size": 9})
pdfmetrics.registerFont(TTFont("NG", FONT))
pdfmetrics.registerFont(TTFont("NGB", FONT_B))
pdfmetrics.registerFontFamily("NG", normal="NG", bold="NGB", italic="NG", boldItalic="NGB")

DIFFS = ("basic", "intermediate", "advanced")
KS2 = (5, 8, 10)
IMPLS = ("A", "B")
LABEL = {"A": "A · bge-m3 dense + sparse, min-max", "B": "B · bge-m3 dense + BM25, RRF"}
COLOR = {"A": "#eb6834", "B": "#1baf7a"}
JUDGE = ("judge_facts", "judge_decision", "judge_evidence", "judge_conditions")
R = ("faithfulness", "context_recall", "context_precision", "answer_relevancy")
NAME = {"faithfulness": "Faithfulness", "answer_relevancy": "Answer Relevancy",
        "context_precision": "Context Precision", "context_recall": "Context Recall", "accuracy": "정답률"}
INK, MUTED, GRID, SURFACE = "#1f2933", "#52606d", "#e4e7eb", "#fcfcfb"

# ---------- 데이터 ----------
SWEEP = json.loads((HERE / "k_sweep.json").read_text(encoding="utf-8"))
OLD = json.loads((HERE / "collected.json").read_text(encoding="utf-8"))["runs"]
NEW = json.loads((HERE / "collected_k.json").read_text(encoding="utf-8"))["runs"]

runs = {}  # (난이도, A|B, k) → run
for r in OLD:
    if r["split"] == "development" and r["mode"] == "hybrid_rerank" and r["config"] in ("A-Qdrant", "B"):
        runs[(r["difficulty"], "A" if r["config"] == "A-Qdrant" else "B", 5)] = r
for r in NEW:
    impl, k = r["config"].split("-k")
    runs[(r["difficulty"], impl, int(k))] = r


def items(d, impl, k):
    if d == "전체":
        return [i for x in DIFFS for i in items(x, impl, k)]
    r = runs.get((d, impl, k))
    return r["items"] if r else []


def avg(its, name):
    v = [i["scores"][name] for i in its if name in i["scores"]]
    return mean(v) if v else None


def accuracy(its):
    judged = [i for i in its if any(n in i["scores"] for n in JUDGE)]
    return mean(1.0 if all(i["scores"].get(n, 1.0) == 1.0 for n in JUDGE) else 0.0 for i in judged) if judged else None


def metric(its, name, k=None):
    if name == "accuracy":
        return accuracy(its)
    if name == "judge_mean":
        v = [avg(its, n) for n in JUDGE]
        v = [x for x in v if x is not None]
        return mean(v) if v else None
    if name in ("hit", "recall"):
        return avg(its, f"{name}_at_{k}")
    if name == "tok_in":
        return mean(i["tokens"]["input"] for i in its)
    if name == "lat_med":
        return median((i["latency_ms"] or 0) / 1000 for i in its)
    if name == "lat_mean":
        return mean((i["latency_ms"] or 0) / 1000 for i in its)
    return avg(its, name)


def f(v, d=3):
    return "-" if v is None else f"{v:.{d}f}"


def sweep_row(d, impl, mode, setting):
    impl_name = "retrieval_a" if impl == "A" else "retrieval_b"
    return next(x for x in SWEEP if x["difficulty"] == d and x["impl"] == impl_name and x["mode"] == mode and x["setting"] == setting)


SWEEP_RERANK = {"A": "qdrant후보 500 · 리랭커 50", "B": "후보 30 · 리랭커 50"}
KS1 = (1, 3, 5, 8, 10, 15)


# ---------- 그림 ----------
def style_axes(ax, top=1.0):
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    ax.set_ylim(0, top)


def line_panels(path, series_fn, xs, title_fn, panels, top=1.0, xlabel="k"):
    fig, axes = plt.subplots(1, len(panels), figsize=(12.4, 2.8), dpi=200, sharey=True)
    for ax, p in zip(axes, panels, strict=True):
        for impl in IMPLS:
            ys = [series_fn(p, impl, x) for x in xs]
            ax.plot(xs, [y if y is not None else float("nan") for y in ys], color=COLOR[impl], linewidth=2,
                    marker="o", markersize=5, markeredgecolor="white", markeredgewidth=1.2, label=LABEL[impl], zorder=3)
            last = next((y for y in reversed(ys) if y is not None), None)
            if last is not None:
                ax.annotate(f"{last:.2f}", (xs[-1], last), xytext=(4, 0), textcoords="offset points",
                            fontsize=7, color=INK, va="center")
        ax.set_xticks(xs)
        ax.set_xlim(xs[0] - 0.6, xs[-1] + 1.6)
        ax.set_xlabel(xlabel, fontsize=7.5, color=MUTED)
        ax.set_title(title_fn(p), fontsize=10, color=INK, loc="left", fontweight="bold")
        style_axes(ax, top)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(path, facecolor="white")
    plt.close(fig)


line_panels(FIG / "stage1_page_recall.png",
            lambda d, impl, k: sweep_row(d, impl, "hybrid_rerank", SWEEP_RERANK[impl])[f"page_recall@{k}"],
            KS1, lambda d: f"{d} · 쪽 Recall@k", DIFFS, top=0.85)
line_panels(FIG / "stage2_accuracy.png", lambda d, impl, k: metric(items(d, impl, k), "accuracy"),
            KS2, lambda d: f"{d} · 정답률", ("전체", *DIFFS), top=0.85)
line_panels(FIG / "stage2_faithfulness.png", lambda d, impl, k: metric(items(d, impl, k), "faithfulness"),
            KS2, lambda d: f"{d} · Faithfulness", ("전체", *DIFFS), top=1.0)

# ---------- PDF ----------
base = ParagraphStyle("b", fontName="NG", fontSize=9.3, leading=14.3, textColor=colors.HexColor(INK), wordWrap="CJK")
small = ParagraphStyle("s", parent=base, fontSize=8.1, leading=12)
cellst = ParagraphStyle("c", parent=base, fontSize=7.6, leading=10.5)
cellb = ParagraphStyle("cb", parent=cellst, fontName="NGB")
code = ParagraphStyle("code", parent=base, fontSize=7.6, leading=11, textColor=colors.HexColor("#243b53"))
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


def steps(xs, s=base):
    return [Paragraph(t, s, bulletText=f"{i}.") for i, t in enumerate(xs, 1)]


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


def codebox(lines):
    t = Table([[Paragraph("<br/>".join(lines), code)]], colWidths=[W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0f4f8")),
                           ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd2d9")),
                           ("LEFTPADDING", (0, 0), (-1, -1), 7), ("TOPPADDING", (0, 0), (-1, -1), 5),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
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
    canvas.drawString(20 * mm, 11 * mm, "최적 k 실험")
    canvas.drawRightString(190 * mm, 11 * mm, str(doc.page))
    canvas.restoreState()


def bold_best(vals):
    nums = [v for v in vals if v is not None]
    best = max(nums) if nums else None
    return [f"<b>{f(v)}</b>" if v is not None and best is not None and abs(v - best) < 1e-9 else f(v) for v in vals]


def stage2_table(d):
    rows = [["구성", "n", "정답률", "Judge 평균", *[NAME[k] for k in R]]]
    cols = []
    keys = [(impl, k) for impl in IMPLS for k in KS2]
    for name in ("accuracy", "judge_mean", *R):
        cols.append(bold_best([metric(items(d, impl, k), name) for impl, k in keys]))
    for j, (impl, k) in enumerate(keys):
        rows.append([f"{impl} · k{k}", len(items(d, impl, k)), *[c[j] for c in cols]])
    return table(rows, [22 * mm, 10 * mm, 18 * mm, 19 * mm, 25 * mm, 25 * mm, 26 * mm, 25 * mm])


def cost_table(d):
    rows = [["구성", "Hit@k", "Recall@k", "입력 토큰/문항", "출력 토큰/문항", "응답 시간 중앙값", "응답 시간 평균"]]
    for impl in IMPLS:
        base_tok = metric(items(d, impl, 5), "tok_in")
        for k in KS2:
            its = items(d, impl, k)
            tok = metric(its, "tok_in")
            grow = "" if k == 5 else f" (+{(tok / base_tok - 1) * 100:.0f}%)"
            rows.append([f"{impl} · k{k}", f(metric(its, "hit", k)), f(metric(its, "recall", k)), f"{tok:,.0f}{grow}",
                         f"{mean(i['tokens']['output'] for i in its):,.0f}", f"{metric(its, 'lat_med'):.1f}초",
                         f"{metric(its, 'lat_mean'):.1f}초"])
    return table(rows, [22 * mm, 18 * mm, 20 * mm, 32 * mm, 26 * mm, 26 * mm, 26 * mm])


def stage1_table(d):
    rows = [["구성", *[f"k={k}" for k in KS1], "검색 시간"]]
    for impl in IMPLS:
        for mode, setting in (("naive", "-"), ("hybrid", "qdrant후보 500" if impl == "A" else "후보 30"),
                              ("hybrid_rerank", SWEEP_RERANK[impl])):
            x = sweep_row(d, impl, mode, setting)
            rows.append([f"{impl} · {mode}", *[f"{x[f'hit@{k}']:.2f} / {x[f'page_recall@{k}']:.2f}" for k in KS1],
                         f"{x['sec_mean']:.2f}초"])
    return table(rows, [30 * mm, *[20 * mm] * 6, 20 * mm], hl=(3, 6))


story = [P("최적 k 실험", title),
         P("답변에 넣는 최종 문맥 수 k를 바꿔 retrieval_a · retrieval_b를 비교하고, Langfuse에서 결과를 확인하는 방법을 정리했습니다.", muted),
         Spacer(1, 6)]

tot = {(impl, k): items("전체", impl, k) for impl in IMPLS for k in KS2}
story += [P("한눈에 보기", h1),
          table([["구성", "정답률", "Faithfulness", "Context Recall", "Context Precision", "입력 토큰/문항"],
                 *[[f"{impl} · k{k}", *bold_best([metric(tot[(impl, kk)], "accuracy") for kk in KS2])[KS2.index(k):KS2.index(k) + 1],
                    f(metric(tot[(impl, k)], "faithfulness")), f(metric(tot[(impl, k)], "context_recall")),
                    f(metric(tot[(impl, k)], "context_precision")), f"{metric(tot[(impl, k)], 'tok_in'):,.0f}"]
                   for impl in IMPLS for k in KS2]],
                [28 * mm, 24 * mm, 28 * mm, 30 * mm, 30 * mm, 30 * mm], hl=(2, 6)),
          P("세 난이도 합계 (구성마다 90문항, hybrid + rerank). 정답률의 굵은 글씨는 같은 검색기 안에서 가장 높은 값.", muted),
          Spacer(1, 4)]
story += bullets([
    "<b>B는 k를 늘릴수록 좋아집니다.</b> 정답률 0.41 → 0.48 → 0.52 (90문항 중 10문항 더 맞힘), Faithfulness 0.825 → 0.846. 세 난이도 모두 k10이 k5보다 좋거나 같습니다.",
    "<b>A는 정답률이 거의 그대로</b>(0.42 → 0.42 → 0.44)이고, Faithfulness는 k8에서 가장 높습니다(0.859).",
    "k를 늘리면 Context Recall은 오르고 Context Precision은 내려갑니다. 필요한 내용이 더 들어오지만 쓸모없는 문맥도 함께 섞입니다.",
    "입력 토큰은 k10에서 약 +30%, 응답 시간 중앙값은 0.5~1초 늘었습니다.",
    "advanced는 k와 관계없이 정답률 0.13~0.23에 머뭅니다. k보다 질문 분해·되묻기 판정·리랭커 쪽의 문제입니다.",
])
story += [Spacer(1, 4), note("<b>B는 k=10, A는 k=8.</b> B는 정답률·Faithfulness가 함께 가장 높고, A는 정답률이 비슷한 가운데 Faithfulness가 가장 높고 토큰이 적습니다. "
                             "B는 k10까지 계속 오르고 있어 k15를 추가로 확인할 만합니다.", "추천")]

story += [P("실험 설계", h1)]
story += table([["", "1단계: 검색만", "2단계: 전체 체인"],
                ["목적", "후보 수·k를 넓게 바꿔 보고 2단계에서 돌릴 k를 고른다", "고른 k로 실제 답변을 만들어 정답률·RAGAS·토큰·시간을 본다"],
                ["k", "1, 3, 5, 8, 10, 15", "5 (오전 비교 실험 재사용), 8, 10"],
                ["대상", "naive · hybrid · hybrid + rerank, 후보 수·리랭커 후보 수 여러 개", "hybrid + rerank (A 후보 500·리랭커 50 / B 후보 30·리랭커 50)"],
                ["문항", "난이도별 정답 근거가 있는 전체 문항", "난이도별 development 30문항"],
                ["지표", "문서 Hit@k · Recall@k · MRR@k, 쪽 Hit@k · Recall@k, 검색 시간", "정답률, Judge 4항목, RAGAS 4지표, 검색 Hit@k · Recall@k, 토큰, 응답 시간"],
                ["LLM", "없음 (비용 0)", "답변·RAGAS 채점 gpt-6-luna, Judge는 Langfuse 서버 평가기"],
                ["결과 파일", "k_sweep.json", "Langfuse 실험 + collected_k.json"]],
               [22 * mm, 74 * mm, 74 * mm]),
story += [Spacer(1, 4)]
story += bullets([
    "<b>정답률</b> = 서버 Judge 4항목(사실 facts · 결론 decision · 근거 evidence · 조건 conditions)을 모두 통과한 문항의 비율.",
    "k 외의 설정(프롬프트, 답변 모델, 리랭커, 가중치 0.5/0.5, Qdrant 후보 수)은 모두 같게 두었습니다. 검색 판정 k도 넣은 문맥 수와 같게 맞췄습니다.",
], small)

story += [PageBreak(), P("1단계: 검색만으로 본 k", h1), img(FIG / "stage1_page_recall.png"),
          P("hybrid + rerank의 쪽 단위 Recall@k. 답변에 필요한 쪽(evidence_groups의 책·쪽)을 상위 k개 안에 얼마나 가져왔는지.", muted)]
for d in DIFFS:
    story += [P(f"{d} · 문서 Hit@k / 쪽 Recall@k", h2), stage1_table(d)]
story += [Spacer(1, 4)]
story += bullets([
    "<b>문서 Hit은 k=5에서 거의 꽉 찹니다.</b> 정답 문서는 이미 상위 5개 안에 있습니다.",
    "<b>쪽 Recall은 k를 늘릴수록 오릅니다.</b> 필요한 쪽이 여러 곳에 흩어져 있기 때문입니다. A advanced는 k8 이후 거의 그대로, B는 k15까지 오릅니다.",
    "k3 이하는 advanced 문서 Hit이 0.88 → 0.83으로 떨어져 제외, k15는 토큰이 k5의 약 3배라 2단계에서 일단 제외 → <b>2단계 후보 k = 5, 8, 10</b>.",
    "리랭커 후보 수 30 · 50 · 100은 차이가 거의 없고 100은 시간만 두 배라 50을 유지합니다.",
], small)

story += [PageBreak(), P("2단계: 전체 체인으로 본 k", h1), img(FIG / "stage2_accuracy.png"), img(FIG / "stage2_faithfulness.png")]
for d in DIFFS:
    story += [P(f"{d} · 답변 품질", h2), stage2_table(d)]
story += [PageBreak(), P("2단계: 검색 점수 · 토큰 · 응답 시간", h1)]
for d in DIFFS:
    story += [P(d, h2), cost_table(d)]
story += [Spacer(1, 4)]
story += bullets([
    "<b>basic</b>: 두 검색기 모두 k를 늘리면 정답률이 오릅니다 (A 0.47 → 0.60, B 0.50 → 0.67). A는 k가 커질수록 Judge '조건' 점수가 떨어졌습니다 (0.93 → 0.73).",
    "<b>intermediate</b>: B는 k10에서 정답률 0.67로 가장 높고, A는 k5(0.67)가 가장 높습니다. A의 Faithfulness는 k8부터 0.82 → 0.88로 올랐습니다.",
    "<b>advanced</b>: B는 k8부터 정답률 0.17 → 0.23, A는 0.13 그대로입니다. Context Recall은 둘 다 올랐지만 답변 상태 대부분이 되묻기·보류입니다.",
], small)
story += [Spacer(1, 4),
          note("30문항이라 1문항이 0.033입니다. 난이도 하나만 보면 차이 대부분이 1~4문항이라, 판단은 세 난이도 합계(90문항)를 기준으로 했습니다.", "읽을 때"),
          Spacer(1, 3),
          note("B advanced는 응답 시간 평균이 14~17초로 튀었습니다(중앙값 6초). 몇 문항이 Qdrant 재시도로 오래 걸린 것으로 보입니다. "
               "A의 k5 실험은 오전 설정이라 Qdrant 후보 수가 지금(500)과 달랐을 수 있습니다.", "주의")]

# ---------- Langfuse 확인 방법 ----------
story += [PageBreak(), P("Langfuse에서 확인하는 방법", h1),
          P("서버 Langfuse(4.76)에서는 예전의 Runs가 <b>Experiments</b>로 바뀌었습니다. 실험 하나 = 데이터셋 30문항을 한 구성으로 돌린 결과입니다.", base),
          P("실험 이름 읽는 법", h2),
          codebox(["b_hybrid_rerank_20261112-1711_basic-dev-k10-gpt6luna",
                   "└ b_ = retrieval_b (없으면 retrieval_a) · hybrid_rerank = 모드 · 20261112-1711 = 시작 시각",
                   "   · basic = 난이도 · dev = development 데이터 · k10 = 최종 문맥 수 · gpt6luna = 답변 LLM"]),
          Spacer(1, 3),
          table([["난이도", "k5 (기준)", "k8", "k10"],
                 *[[d, *["<br/>".join(runs[(d, impl, k)]["experiment"] for impl in IMPLS) for k in KS2]] for d in DIFFS]],
                [22 * mm, 50 * mm, 49 * mm, 49 * mm]),
          P("위가 A, 아래가 B. 오전 실험 중 _cmp_local · _cmp_qdrant · _cmp가 붙은 naive · hybrid 실험과 ragastest 등은 k 비교와 관계없습니다.", muted),
          P("1. 실험 목록에서 최종 점수 보기", h2)]
story += steps([
    "왼쪽 메뉴 <b>Datasets</b> → <b>shipping-basic-development-v1</b>(난이도별로 따로 있음) → 위쪽 <b>Experiments</b> 탭.",
    "검색창에 <b>dev-k</b>를 넣으면 이번 k 실험만 보입니다.",
    "표 오른쪽 위 <b>Columns</b>에서 보고 싶은 열을 켭니다 (아래 표).",
], small)
story += [Spacer(1, 3),
          table([["묶음", "켤 열", "뜻"],
                 ["실험 평균", "accuracy_mean", "정답률 (Judge 4항목 모두 통과한 문항 비율)"],
                 ["실험 평균", "judge_facts_mean · judge_decision_mean · judge_evidence_mean · judge_conditions_mean",
                  "Judge 항목별 평균 ([해당 없음]·[평가 불가]는 뺌)"],
                 ["실험 평균", "faithfulness_mean 등 RAGAS 4개", "RAGAS 평균"],
                 ["실험 평균", "tokens_in_mean · tokens_out_mean · latency_mean · latency_median", "문항당 토큰, 응답 시간(초)"],
                 ["Observation Scores", "Faithfulness · Context_precision · Context_recall · Answer_relevancy", "문항별 RAGAS (목록에서는 평균으로 보임)"],
                 ["Observation Scores", "Hit_at_k · Recall_at_k · Mrr_at_k", "문항별 검색 점수. k마다 이름이 달라 k5는 _at_5, k10은 _at_10에만 값"],
                 ["Eval (서버 평가기)", "Judge_facts 등", "문항별 Judge 점수"]],
                [30 * mm, 75 * mm, 65 * mm]),
          P("_state가 붙은 열(©)은 켜지 않아도 됩니다. 점수 대신 채점됨 · 해당 없음 · 평가 불가 상태만 들어 있습니다.", muted),
          P("2. 실험끼리 문항별로 비교하기", h2)]
story += steps([
    "실험 목록에서 비교할 실험(예: B k5 · k8 · k10)을 체크하고 <b>Compare</b>.",
    "첫 실험이 <b>Baseline</b>, 나머지가 <b>Comp</b>. 위쪽 SUMMARY에 지표별 평균과 차이, ↗ 오른 문항 수 · ↘ 내린 문항 수가 나옵니다.",
    "SUMMARY는 '이 페이지(20문항)' 기준입니다. 30문항 전체 평균은 실험 목록의 *_mean 열을 보세요.",
    "아래로 내리면 문항마다 두 실험의 답변과 점수를 나란히 볼 수 있습니다.",
], small)
story += [P("3. 문항 하나의 토큰 · 시간 · 검색 결과 보기", h2)]
story += steps([
    "Compare 화면이나 실험 문항 목록에서 문항을 누르면 trace 화면이 열립니다.",
    "맨 위에 전체 응답 시간과 토큰 합계, 왼쪽 단계 목록에 검색 → 답변 생성 순서가 나옵니다.",
    "LLM 호출(<b>ChatOpenAI</b>, generation)을 누르면 입력 · 출력 · 캐시 토큰이 나뉘어 보입니다.",
    "오른쪽 Scores 탭에 그 문항의 검색 점수 · RAGAS · Judge(설명 포함)가 모두 붙어 있습니다.",
], small)
story += [P("4. 여러 trace의 토큰을 한꺼번에 보기", h2)]
story += steps([
    "왼쪽 메뉴 <b>Tracing → Observations</b>.",
    "필터: Type = GENERATION, 필요하면 시간 범위나 이름 조건을 더합니다.",
    "Usage 열에 호출마다 입력 → 출력 토큰이 나옵니다.",
], small)
story += [Spacer(1, 4),
          note("Langfuse 서버에 gpt-6-luna 가격이 등록되어 있지 않아 비용(Total Cost)은 0으로 나옵니다. 토큰 수는 정상입니다. "
               "비용을 보려면 서버 관리자가 Settings → Models에 가격을 등록해야 합니다.", "비용")]

story += [Spacer(1, 10), P("실험 다시 돌리기 · 점수 올리기", h1),
          P("kant-rag 폴더, PowerShell 기준. 실험 하나(30문항)에 약 10분 걸립니다.", muted),
          P("실험 하나", h2),
          codebox(["$env:PYTHONIOENCODING = \"utf-8\"; $env:OPENAI_MODEL = \"gpt-6-luna\"",
                   ".venv/Scripts/python.exe -W ignore evaluation/langfuse_experiment.py `",
                   "  --impl retrieval_b --difficulty basic --split development `",
                   "  --modes hybrid_rerank --top-k 15 --concurrency 2 --tag basic-dev-k15-gpt6luna"]),
          Spacer(1, 3),
          table([["옵션", "뜻"],
                 ["--impl", "retrieval_a 또는 retrieval_b (B면 이름 앞에 b_)"],
                 ["--difficulty", "basic · intermediate · advanced"],
                 ["--top-k", "답변에 넣는 최종 문맥 수 k. 검색 점수도 이 k로 계산"],
                 ["--concurrency", "동시 처리 문항 수. GPU 8GB에서는 2 권장 (모델 3개가 약 7GB 사용)"],
                 ["--tag", "실험 이름 끝. {난이도}-dev-k{k}-gpt6luna 형식으로 맞춤"],
                 ["--no-ragas", "RAGAS 채점 없이 빠르게 (검색 점수 · 서버 Judge만)"]],
                [30 * mm, 140 * mm]),
          P("세 난이도 한 번에", h2),
          codebox(["foreach ($d in \"basic\",\"intermediate\",\"advanced\") {",
                   "  .venv/Scripts/python.exe -W ignore evaluation/langfuse_experiment.py --impl retrieval_b `",
                   "    --difficulty $d --split development --modes hybrid_rerank --top-k 15 --concurrency 2 `",
                   "    --tag \"$d-dev-k15-gpt6luna\"",
                   "}"]),
          P("실험 평균 점수 올리기 (정답률 · Judge 평균 · 토큰 · 응답 시간)", h2),
          P("검색 · RAGAS 평균은 실험이 끝날 때 자동으로 올라갑니다. 서버 Judge는 문항마다 따로 붙어 실험 평균이 없으므로 아래로 모아 올립니다. 다시 실행해도 덮어씁니다.", small),
          codebox([".venv/Scripts/python.exe -W ignore artifacts/retrieval_a/cmp_eval/collect_k.py",
                   ".venv/Scripts/python.exe -W ignore artifacts/retrieval_a/cmp_eval/upload_run_summary.py collected_k.json --judge"]),
          P("1단계 검색 실험", h2),
          codebox([".venv/Scripts/python.exe -W ignore artifacts/retrieval_a/cmp_eval/k_sweep.py retrieval_a advanced",
                   "# 검색기 · 난이도마다 따로 실행 권장 (GPU 메모리). 결과는 k_sweep.json에 이어서 기록"])]

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=16 * mm, bottomMargin=18 * mm,
                        title="최적 k 실험")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print("saved", OUT)
