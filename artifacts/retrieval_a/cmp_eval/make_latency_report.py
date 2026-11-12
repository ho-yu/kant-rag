"""latency.jsonl(로컬) + latency_server.jsonl(팀 서버) + diagnose_server.json → client 응답 시간 PDF.

실행: uv run --no-project --with matplotlib --with reportlab --with numpy python artifacts/retrieval_a/cmp_eval/make_latency_report.py
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
LAT = [json.loads(line) for line in (HERE / "latency.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
_SRV = HERE / "latency_server.jsonl"
SRV = [json.loads(line) for line in _SRV.read_text(encoding="utf-8").splitlines() if line.strip()] if _SRV.exists() else []
_DIAG = HERE / "diagnose_server.json"
DIAG = json.loads(_DIAG.read_text(encoding="utf-8")) if _DIAG.exists() else []
ACTION = {"answer": "답변", "clarify": "되묻기", "abstain": "보류"}
STATUS = {"answered": "답변", "needs_clarification": "되묻기", "abstained": "보류"}
OUT = HERE / "client_latency.pdf"
FIG = HERE / "figs" / "latency"
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
LABEL = {"A-기존": "A · 기존", "A-Qdrant": "A · Qdrant", "B": "B"}
MODES = ("naive", "hybrid", "hybrid_rerank")
MODE_LABEL = {"naive": "naive", "hybrid": "hybrid", "hybrid_rerank": "hybrid + rerank"}
ORDER_COLOR = {1: "#2a78d6", 2: "#eb6834"}  # 검증한 범주형 1·2번
ORDER_LABEL = {1: "첫 번째 질문", 2: "두 번째 질문"}
INK, MUTED, GRID, SURFACE = "#1f2933", "#52606d", "#e4e7eb", "#fcfcfb"
CAP = 60  # 축 상한(초). 넘는 막대는 잘라 그리고 실제 값을 적는다


def sec(c, d, m, o):
    return next((r["seconds"] for r in LAT if r["config"] == c and r["difficulty"] == d and r["mode"] == m and r["order"] == o), None)


def srv(d, m, o, key="seconds"):
    return next((r[key] for r in SRV if r["difficulty"] == d and r["mode"] == m and r["order"] == o), None)


def latency_chart(d, path):
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9), dpi=200, sharey=True)
    for ax, m in zip(axes, MODES, strict=True):
        x = np.arange(len(CONFIGS))
        for k, o in enumerate((1, 2)):
            vals = [sec(c, d, m, o) for c in CONFIGS]
            pos = x + (k - 0.5) * 0.36
            bs = ax.bar(pos, [min(v or 0, CAP) for v in vals], 0.33, color=ORDER_COLOR[o], label=ORDER_LABEL[o], zorder=2)
            for b, v in zip(bs, vals, strict=True):
                if v is None:
                    continue
                clipped = v > CAP
                ax.text(b.get_x() + b.get_width() / 2, min(v, CAP) + 1.2, f"{v:.0f}초" + (" ▲" if clipped else ""),
                        ha="center", va="bottom", fontsize=6.3, color=INK, fontweight="bold" if clipped else "normal")
        ax.set_xticks(x, [LABEL[c] for c in CONFIGS], fontsize=7.5)
        ax.set_title(MODE_LABEL[m], fontsize=10, color=INK, loc="left", fontweight="bold")
        ax.set_facecolor(SURFACE)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, length=0)
        ax.set_ylim(0, CAP * 1.15)
    axes[0].set_ylabel("초", color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
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


def bullets(items, s=base):
    return [Paragraph(t, s, bulletText="•") for t in items]


def table(rows, widths, bold_first=True, highlight_rows=()):
    data = [[v if hasattr(v, "wrap") else Paragraph(str(v), cellb if r == 0 or (bold_first and c == 0) else cellst)
             for c, v in enumerate(row)] for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd2d9")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("LEFTPADDING", (0, 0), (-1, -1), 3.5), ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
          ("TOPPADDING", (0, 0), (-1, -1), 2.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6),
          ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7"))]
    for r in highlight_rows:
        st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#fff8e6")))
    t.setStyle(TableStyle(st))
    return t


def img(path, width=W):
    iw, ih = ImageReader(str(path)).getSize()
    return Image(str(path), width=width, height=width * ih / iw)


def note(text, label="참고"):
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
    canvas.drawString(20 * mm, 11 * mm, "client 응답 시간")
    canvas.drawRightString(190 * mm, 11 * mm, str(doc.page))
    canvas.restoreState()


def escape_text(t):
    from xml.sax.saxutils import escape

    return escape(str(t))


def f(v, d=1):
    return "-" if v is None else f"{v:.{d}f}"


def fs(v):
    if v is None:
        return "-"
    return f"{v:.1f}" if v < 60 else f"{v:.0f} ({int(v // 60)}분 {int(v % 60)}초)"


s = [P("client 응답 시간", title),
     P(f"retrieval_a(기존 · Qdrant) / retrieval_b · 측정 {LAT[0]['at'][:16].replace('T', ' ')} ~ {LAT[-1]['at'][11:16]}", muted),
     Spacer(1, 8),
     P("1. 무엇을 쟀나", h1),
     *bullets([
         "<b>client와 같은 경로</b>: 사전 판단 LLM → 검색 → 답변 LLM → 사후 검사(규칙). 웹 화면·API가 쓰는 서비스 체인 그대로입니다.",
         "<b>측정 환경 ① 로컬</b> (2~4절): 내 PC가 서비스 체인을 직접 실행하고 임베딩·리랭커 모델을 내 PC GPU(노트북 RTX 5060)에 새로 불러옵니다. "
         "문서는 팀 Qdrant, LLM은 OpenAI. 서버를 막 켠 상태를 가정해 구성 × 난이도마다 새 프로세스에서 잽니다. 세 구성을 모두 잴 수 있습니다.",
         "<b>측정 환경 ② 팀 서버</b> (5절): 팀 서버 API(/query)로 서버에 이미 올라가 있는 모델·검색 자원을 그대로 씁니다. "
         "서버 구현은 retrieval_a(기존 방식) 하나라 A · 기존만 잴 수 있습니다.",
         "방식마다 naive → hybrid → hybrid + rerank 순서로 질문 두 개를 보냅니다. <b>첫 번째 질문</b>에는 그 방식을 처음 쓸 때의 준비(모델·청크 로딩)가 "
         "들어가고, <b>두 번째 질문</b>은 준비가 끝난 뒤 시간입니다.",
         "답변 모델 gpt-6-luna (서버와 같음). 질문은 각 난이도 development의 첫 두 문항. 조합마다 한 번씩 잰 값이라 LLM 속도에 따라 ±2초쯤 흔들릴 수 있습니다.",
     ], small),
     P("2. 로컬 측정 결과 (초): 첫 번째 / 두 번째 질문", h1)]
rows = [["난이도", "구성", "naive", "hybrid", "hybrid + rerank"]]
hl = []
for d in DIFFS:
    for c in CONFIGS:
        rows.append([d, LABEL[c], *[f"{fs(sec(c, d, m, 1))} / {f(sec(c, d, m, 2))}" for m in MODES]])
        if any((sec(c, d, m, 1) or 0) > 30 for m in MODES):
            hl.append(len(rows) - 1)
s += [table(rows, [24 * mm, 22 * mm, 34 * mm, 46 * mm, 44 * mm], highlight_rows=hl),
      P("노란 줄 = 첫 질문이 30초를 넘는 조합. 오류는 54번 중 0건입니다.", small),
      P("3. 무엇이 시간을 쓰나", h1),
      table([["상황", "늘어나는 시간", "원인"],
             ["어떤 방식이든 서버에서 처음 쓸 때", "+8~14초 (naive 첫 질문 15~21초)", "임베딩 모델·LLM 클라이언트 로딩"],
             ["A · 기존 hybrid를 처음 쓸 때", f"intermediate +{sec('A-기존', 'intermediate', 'hybrid', 1) - sec('A-기존', 'intermediate', 'hybrid', 2):.0f}초, "
              f"advanced +{(sec('A-기존', 'advanced', 'hybrid', 1) - sec('A-기존', 'advanced', 'hybrid', 2)) / 60:.1f}분", "청크와 벡터를 Qdrant에서 통째로 받음"],
             ["B hybrid를 처음 쓸 때", f"intermediate +{sec('B', 'intermediate', 'hybrid', 1) - sec('B', 'intermediate', 'hybrid', 2):.0f}초, "
              f"advanced +{(sec('B', 'advanced', 'hybrid', 1) - sec('B', 'advanced', 'hybrid', 2)) / 60:.1f}분", "BM25용 청크를 통째로 받음"],
             ["A · Qdrant hybrid를 처음 쓸 때", "거의 0초", "받아 올 것이 없음 (질문마다 Qdrant에 후보만 물음)"],
             ["두 번째 질문부터 (모든 구성)", "naive 6~9초, rerank 8~12초", "대부분 LLM 두 번(사전 판단 + 답변). 검색 자체는 1초 미만"]],
            [52 * mm, 50 * mm, 68 * mm]),
      Spacer(1, 4),
      note("hybrid + rerank의 첫 질문이 짧은 것은 측정 순서상 hybrid 준비가 이미 끝나 있었기 때문입니다. 사용자가 hybrid + rerank를 맨 처음 고르면 "
           "hybrid 첫 질문 시간이 그대로 더해집니다(예: advanced A · 기존 약 4분). 실제 팀 서버(main.py)는 시작할 때 세 난이도의 검색 자원을 미리 "
           "준비(워밍업)하므로 사용자는 대부분 두 번째 질문 시간을 겪고, 대신 서버 시작이 그만큼 늦어집니다.", "주의"),
      PageBreak(), P("4. 난이도별 차트 (로컬)", h1)]
for d in DIFFS:
    latency_chart(d, FIG / f"{d}.png")
    s += [P(d, h2), img(FIG / f"{d}.png")]
s += [P(f"축은 {CAP}초에서 잘랐습니다. ▲ 표시는 실제 값이 그보다 큰 막대입니다.", small)]
if SRV:
    srows = [["난이도", "방식", "질문 1 (서버)", "질문 2 (서버)", "질문 1 (로컬 A · 기존)", "질문 2 (로컬 A · 기존)"]]
    for d in DIFFS:
        for m in MODES:
            srows.append([d, MODE_LABEL[m], f(srv(d, m, 1)), f(srv(d, m, 2)), fs(sec("A-기존", d, m, 1)), f(sec("A-기존", d, m, 2))])
    s += [PageBreak(), P("5. 팀 서버 측정 (서버에 올라간 모델 사용, retrieval_a)", h1),
          P(f"측정 {SRV[0]['at'][11:16]} ~ {SRV[-1]['at'][11:16]} · {len(SRV)}번 중 오류 {sum(1 for r in SRV if r['error'])}번 · "
            "서버 처리 시간(elapsed)과 왕복 시간 차이 0.1초 이하", muted),
          table(srows, [24 * mm, 26 * mm, 26 * mm, 26 * mm, 34 * mm, 34 * mm]),
          Spacer(1, 4),
          *bullets([
              "서버는 시작할 때 세 난이도의 검색 자원을 미리 준비해 두어 <b>질문 1에도 로딩 대기가 없습니다</b>. "
              "로컬의 첫 번째 질문(15초 ~ 4분)은 내 PC에서 모델과 청크를 처음 불러오는 시간이 섞인 값입니다.",
              "서버에서 질문 1(약 4초)과 질문 2(6~10초)의 차이는 로딩이 아니라 <b>질문 내용</b> 때문입니다. 질문 2는 답이 길거나 "
              "되묻기로 처리된 경우가 많았습니다. 질문 2끼리 보면 서버와 로컬이 비슷해, 로컬의 두 번째 질문 시간이 실제 서버 체감과 가깝습니다.",
              "실제 사용자가 겪는 시간 ≈ 서버 측정값: 질문에 따라 약 3~10초이고, 대부분 LLM 두 번(사전 판단 + 답변)에 쓰입니다.",
          ], small)]

# ---------------- 두 모델 비교와 선정 ----------------
if DIAG:
    cellq = ParagraphStyle("cq", parent=cellst, fontSize=7.2, leading=10)
    qrows = [["난이도", "순서", "문항 ID", "질문 (측정에 넣은 원문)", "정답 파일의 기대 동작"]]
    for r in DIAG:
        qrows.append([r["difficulty"], f"질문 {r['order']}", r["question_id"], Paragraph(escape_text(r["question"]), cellq),
                      ACTION.get(r.get("expected_action"), r.get("expected_action") or "-")])
    drows = [["난이도", "질문", "묻는 내용", "하위 질문", "검색 횟수", "답변에 넣은 문맥", "답변 길이", "응답 상태", "시간(초)"]]
    for r in DIAG:
        drows.append([r["difficulty"], f"질문 {r['order']}", "한 가지" if r["n_sub_questions"] == 0 else "여러 가지",
                      f"{r['n_sub_questions']}개", f"{1 + r['n_sub_questions']}번", f"{r['n_contexts']}개", f"{r['answer_chars']}자",
                      STATUS.get(r["status"], r["status"]), f(r["seconds"])])
    ex = next((r for r in DIAG if r["difficulty"] == "basic" and r["order"] == 2), None)
    q1 = [r for r in DIAG if r["order"] == 1]
    q2 = [r for r in DIAG if r["order"] == 2]
    s += [PageBreak(), P("6. 서버에서 질문 2가 더 오래 걸리는 이유", h1),
          P("측정에 넣은 질문 (각 난이도 development의 첫 두 문항)", h2),
          table(qrows, [22 * mm, 14 * mm, 20 * mm, 92 * mm, 22 * mm]),
          Spacer(1, 4),
          P("질문 1 vs 질문 2 (서버, hybrid + rerank)", h2),
          P(f"원인 확인용으로 {DIAG[0]['at'][11:16]}에 같은 질문을 서버에 다시 보내 하위 질문·문맥·답변 길이를 기록했습니다(5절 측정과 별도 실행이라 시간은 조금 다릅니다).", muted),
          table(drows, [20 * mm, 14 * mm, 18 * mm, 16 * mm, 16 * mm, 24 * mm, 18 * mm, 18 * mm, 16 * mm]),
          Spacer(1, 4),
          note(f"질문 2는 모두 <b>여러 가지를 한 번에 묻는 복합 질문</b>이라 서비스의 사전 판단 LLM이 하위 질문 3개로 쪼갭니다. "
               f"그래서 검색을 4번 하고, 결과를 합쳐 답변에 문맥을 약 12개(질문 1은 5개) 넣고, 답도 약 4~7배 길게 씁니다. "
               f"평균 시간 질문 1 {mean(r['seconds'] for r in q1):.1f}초 → 질문 2 {mean(r['seconds'] for r in q2):.1f}초. 로딩과는 관계없습니다.", "결론"),
          P("시간이 늘어나는 곳", h2),
          table([["단계", "질문 2에서 하는 일", "늘어나는 시간 (대략)"],
                 ["① 사전 판단 LLM", "검색 질문과 하위 질문 3개를 직접 써야 해서 출력이 길어짐", "+1초 안팎"],
                 ["② 검색", "질문 4개를 각각 hybrid + rerank (리랭커도 4번)", "+1~2초"],
                 ["③ 답변 LLM", "문맥 약 12개(입력 약 2배)를 읽고 400~500자 답을 씀. LLM은 답이 길수록 오래 걸림", "+2~5초 (가장 큼)"]],
                [30 * mm, 100 * mm, 40 * mm])]
    if ex:
        s += [P("예: basic 질문 2를 사전 판단이 쪼갠 하위 질문", h2),
              *bullets([escape_text(x) for x in ex["sub_questions"]], small)]
    s += [Spacer(1, 3), *bullets([
        "되묻기(needs_clarification)도 무엇이 부족한지 설명을 써야 해서 답변과 비슷하게 시간이 걸립니다.",
        "평가 실험(langfuse_experiment.py)은 질문을 쪼개지 않아 복합 질문도 문맥 5개로 답합니다. 그래서 평가의 응답 시간(중앙값 약 4초)이 서비스보다 짧습니다.",
        "차이를 줄이려면 config/generation/rag.yaml의 max_sub_questions(3)나 max_docs(12)를 줄일 수 있지만, 복합 질문의 답이 덜 완전해질 수 있어 평가로 확인해야 합니다.",
        "실제 체감 시간: <b>단일 질문 약 4~5초, 복합 질문 약 8~14초</b>.",
    ], small)]

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=16 * mm, bottomMargin=18 * mm,
                        title="client 응답 시간")
doc.build(s, onFirstPage=on_page, onLaterPages=on_page)
print("saved", OUT)
