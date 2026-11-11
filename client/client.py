"""간단한 채팅 클라이언트 (Streamlit + OpenAI + RAG 연결 지점)

처리 흐름
    질문 입력 → decide_route(): 사전 판단 (generation/guard.py의 create_query_guard)
                 ├─ "block"  → 차단 안내문
                 ├─ "direct" → answer_directly(): OpenAI가 바로 답변
                 └─ "rag"    → answer_with_rag(): 팀 RAG 체인 (pipeline/chain.py의 create_chain)
                               이전 대화가 필요한 후속 질문은 검색용 질문(search_question)으로 다시 써서 넘긴다
              → check_answer(): 답변 사후 검사 (create_answer_guard) → 문제가 있으면 안내문으로 바꿈
              → 화면: 답변은 markdown, [출처] 아래는 접는 칸에 원문 그대로(st.text)

- 사이드바에서 난이도(basic/intermediate/advanced), 도메인, 검색 방식(naive/hybrid/hybrid+rerank)을 고른다.
- 이전 대화를 매번 함께 보내서 앞 내용을 기억하게 한다.
- 대화 기록은 화면 메모리에만 두고, "새 채팅"을 누르거나 난이도를 바꾸면 비운다.

실행: uv run streamlit run client/client.py
"""
import os
import re
import time

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAIError
from rag_service.generation.guard import (  # 차단 시 보여 줄 안내문
    BLOCKED_ANSWER,
    BLOCKED_QUESTION,
)

from rag_service.observability import (
    create_traced_openai_client,
    trace_request,
)
from langfuse import get_client as get_langfuse_client
from langfuse.langchain import CallbackHandler

load_dotenv()

# ───────────────────────── 1. 설정 (여기만 고치면 됨) ─────────────────────────
MODEL = os.getenv("OPENAI_MODEL", "")                    # .env에 OPENAI_MODEL=모델명
USE_ROUTER = True                                        # 사전 판단(차단·경로·질문 재작성). False면 항상 RAG + 질문 그대로
USE_ANSWER_GUARD = True                                  # 답변 사후 검사. False면 검사 없이 표시
MAX_HISTORY_TURNS = 5                                    # 함께 보낼 이전 대화 수 (질문+답변 1쌍 = 1턴)
HISTORY_ANSWER_CHARS = 300                               # 사전 판단에 넣을 이전 답변 길이 (결론이 검색어로 새지 않게 자름)

SOURCES_MARK = "\n\n\n[출처]\n"                          # generation/sources.py가 답변 뒤에 붙이는 구분자

# [변경] 난이도·도메인 선택지와 기본값
DIFFICULTIES = ["basic", "intermediate", "advanced"]
DOMAINS = ["finance", "manufacturing", "shipping"]
DEFAULT_DIFFICULTY = "basic"
DEFAULT_DOMAIN = "shipping"

# [변경] 검색 방식: 화면에 보이는 이름 → RAG 체인에 넘기는 값 (pipeline/chain.py의 retrieval_mode 키와 맞춤)
RETRIEVAL_MODES = {
    "naive": "naive",
    "hybrid": "hybrid",
    "hybrid+rerank": "hybrid_rerank",
}
DEFAULT_RETRIEVAL_MODE = "naive"

# page 확장: 찾은 문단·표 대신 그 쪽(page) 전체 원문을 LLM에 넣는다 (같은 파일+page는 한 번만).
# 사이드바 스위치의 처음 값. 켜면 답에 쓸 내용이 늘지만 토큰·응답 시간도 는다
DEFAULT_PAGE_CONTEXT = False

# [변경] 도메인별 설명 (판단 프롬프트에 쓰임). 도메인을 열 때 여기에 추가
DOMAIN_HINTS = {
    "shipping": "해운 예약·서류 제출·VGM·선하증권·마감일·담당 창구 등 업무 문서 내용",
}

SYSTEM_PROMPT = "너는 질문에 간결하고 정확하게 답하는 한국어 어시스턴트야. 모르는 것은 모른다고 답해."


# ───────────────────────── 2. 답변 처리 흐름 ─────────────────────────
@st.cache_resource
def get_client():
    return create_traced_openai_client()  # Langfuse 추적이 붙은 OpenAI 클라이언트 (OPENAI_API_KEY 자동 사용)

@st.cache_resource
def get_query_guard():
    from rag_service.generation.guard import create_query_guard
    return create_query_guard().with_config(
        callbacks=[CallbackHandler()], run_name="query-guard"
    )


@st.cache_resource
def get_answer_guard():
    from rag_service.generation.guard import create_answer_guard
    return create_answer_guard().with_config(
        callbacks=[CallbackHandler()], run_name="answer-guard"
    )


@st.cache_resource(show_spinner="RAG 준비 중 (난이도·도메인마다 처음 한 번)...")
def get_chain(domain, difficulty):
    """난이도·도메인 조합마다 체인을 한 번만 만든다 (리트리버 3개 생성·모델 로딩이 무거움)."""
    from rag_service.pipeline import create_chain
    return create_chain(domain=domain, difficulty=difficulty)


def get_answer(question, history, difficulty, domain, retrieval_mode, page_context=False):
    """전체 흐름: 사전 판단 → (차단 / 직접 답변 / RAG) → 답변 검사.
    {"answer", "sources", "route", "search_question"}를 돌려준다."""
    decision = decide_route(question, history, domain)
    if decision["action"] == "block":
        return {"answer": BLOCKED_QUESTION, "sources": [], "route": "block", "search_question": "", "sub_questions": []}
    if decision["action"] == "direct":
        result = answer_directly(question, history)
    else:
        result = answer_with_rag(
            decision["search_question"], decision["sub_questions"], history, difficulty, domain, retrieval_mode,
            page_context,
        )
    result = check_answer(question, result)
    result["route"] = decision["action"] if result.get("route") != "blocked_answer" else "blocked_answer"
    result["search_question"] = decision["search_question"]
    result["sub_questions"] = decision["sub_questions"]
    return result


def history_text(history):
    """사전 판단에 넣을 이전 대화 글. 사용자 질문은 그대로, 이전 답변은 앞부분만 (출처 목록은 뺌)."""
    lines = []
    for m in history:
        content = m["content"]
        if m["role"] == "assistant":
            content = content.split(SOURCES_MARK)[0][:HISTORY_ANSWER_CHARS]
        lines.append(f"{m['role']}: {content}")
    return "\n".join(lines) or "(없음)"


def decide_route(question, history, domain):
    """사전 판단: {"action": "rag"|"direct"|"block", "search_question": 검색용 질문}.
    판단을 끄거나 실패하면 RAG + 질문 그대로 (검사기 오류로 서비스가 멈추지 않게)."""
    fallback = {"action": "rag", "search_question": question, "sub_questions": []}
    if not USE_ROUTER:
        return fallback
    try:
        d = get_query_guard().invoke({
            "domain": DOMAIN_HINTS.get(domain, "업무 문서"),
            "history": history_text(history),
            "question": question,
        })
    except Exception:  # noqa: BLE001 — 판단 실패는 RAG로 처리
        return fallback
    search_question = (d.search_question or "").strip() or question
    sub_questions = [q.strip() for q in d.sub_questions if q.strip()][:3]
    return {"action": d.action, "search_question": search_question, "sub_questions": sub_questions}


def answer_with_rag(search_question, sub_questions, history, difficulty, domain, retrieval_mode, page_context=False):
    """RAG 체인을 호출한다. search_question은 사전 판단이 다시 쓴 검색용 질문 (필요 없으면 원래 질문).
    sub_questions는 여러 내용을 묻는 질문을 내용별로 나눈 검색 질문 (없으면 빈 목록 → 기존과 같은 검색).

    difficulty, domain으로 어떤 데이터(컬렉션)를 검색할지 정한다.
    retrieval_mode는 "naive" / "hybrid" / "hybrid_rerank" 중 하나 (RETRIEVAL_MODES 참고).
    page_context가 True면 찾은 청크 대신 그 쪽(page) 전체를 컨텍스트로 넣는다 (pipeline/chain.py의 to_pages).
    돌려줄 형식: {"answer": 답변 문자열(끝에 [출처] 포함), "sources": []}
    """
    result = get_chain(domain, difficulty).invoke(
        {
            "question": search_question,
            "sub_questions": sub_questions,
            "retrieval_mode": retrieval_mode,
            "page_context": page_context,
        }
    )
    return {"answer": result.answer, "sources": [], "status": result.status}


def check_answer(question, result):
    """답변 사후 검사. 문제가 있으면 안내문으로 바꾼다. 검사를 끄거나 실패하면 그대로 둔다."""
    if not USE_ANSWER_GUARD:
        return result
    answer_only = result["answer"].split(SOURCES_MARK)[0]  # 출처 원문은 빼고 모델이 쓴 답변만 검사
    try:
        check = get_answer_guard().invoke({"question": question, "answer": answer_only})
    except Exception:  # noqa: BLE001 — 검사 실패는 통과로 처리
        return result
    if check.verdict == "block":
        return {"answer": BLOCKED_ANSWER, "sources": [], "route": "blocked_answer"}
    return result


def answer_directly(question, history):
    """RAG 없이 OpenAI가 바로 답한다."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    messages.append({"role": "user", "content": question})
    response = get_client().chat.completions.create(model=MODEL, messages=messages, reasoning_effort="low")
    return {"answer": response.choices[0].message.content, "sources": []}


# ───────────────────────── 3. 화면 보조 함수 ─────────────────────────
def build_history():
    """화면에 쌓인 대화 중 최근 N턴만 골라 API 형식으로 만든다."""
    recent = st.session_state.messages[-MAX_HISTORY_TURNS * 2:]
    return [{"role": m["role"], "content": m["content"].split(SOURCES_MARK)[0]} for m in recent]  # 출처 목록은 빼고


def reset_chat():
    """대화 기록을 비운다. (새 채팅 버튼, 난이도 변경 시 사용)"""
    st.session_state.messages = []


def fix_bold(text):
    """**굵게**가 한글에 붙어 깨지는 것 막기. 예) **12시(한국 표준시)**입니다 → 별표가 그대로 보임.
    마크다운 규칙상 ** 안쪽 끝이 문장부호이고 바깥에 글자가 붙으면 굵게가 안 되므로,
    ** 안쪽 양 끝에 보이지 않는 문자(U+200B)를 넣어 준다. 글자는 그대로 보인다."""
    return re.sub(r"\*\*(?=\S)(.+?)(?<=\S)\*\*", lambda m: f"**\u200b{m.group(1)}\u200b**", text)


def strip_marks(text):
    """답변 본문의 [S1][S3] 같은 출처 번호 표시를 화면에서만 지운다 (출처 상자는 그대로)."""
    return re.sub(r"\s*\[S\d+\]", "", text)


def parse_sources(text):
    """generation/sources.py가 만든 [출처] 글을 출처별 dict 목록으로 나눈다.
    각 출처: [S1] 줄 + "키 : 값" 줄들, page_content는 마지막 항목이라 여러 줄(표 포함)일 수 있다."""
    blocks = []
    for chunk in re.split(r"\n\n(?=\[S\d+\]\n)", text.strip()):
        sid, _, rest = chunk.partition("\n")
        head, _, content = rest.partition("page_content : ")
        fields = dict(line.split(" : ", 1) for line in head.splitlines() if " : " in line)
        blocks.append({"id": sid.strip(), "content": content.strip(), **fields})
    return blocks


SOURCE_FIELDS = ["file_name", "page", "book_id", "chunk_id", "domain", "difficulty"]  # 출처에 한 줄씩 보여 줄 항목


def show_source(b):
    """출처 하나: [S1] → "항목 : 값"을 한 줄씩 → page_content(인용 상자)"""
    st.markdown(f"**{b['id']}**")
    st.markdown("  \n".join(f"{k} : {b[k]}" for k in SOURCE_FIELDS if b.get(k)))  # 줄 끝 공백 2개 = 줄바꿈
    st.markdown("page_content :")
    st.markdown("\n".join(f"> {line}" for line in b["content"].splitlines()))


def show_message(m):
    """메시지 하나를 화면에 그린다. (답변이면 출처, 사용 방식, 응답 시간도 함께)"""
    with st.chat_message(m["role"]):
        body, _, sources_text = m["content"].partition(SOURCES_MARK)
        st.markdown(fix_bold(strip_marks(body)))
        if sources_text:  # 답변 뒤 [출처] 글 → 출처마다 보기 좋게 나눠 보여 준다
            blocks = parse_sources(sources_text)
            with st.expander(f"출처 {len(blocks)}건"):
                for i, b in enumerate(blocks):
                    if i:
                        st.divider()
                    show_source(b)
        info = []
        if m.get("route") == "rag":
            info.append(f"RAG 사용 · {m['mode']}" if m.get("mode") else "RAG 사용")
            if m.get("page_context"):
                info.append("page 확장")
        elif m.get("route") == "direct":
            info.append("직접 답변")
        elif m.get("route") == "block":
            info.append("질문 차단")
        elif m.get("route") == "blocked_answer":
            info.append("답변 차단")
        if m.get("search_question") and m["search_question"] != m.get("question"):
            info.append(f"검색 질문: {m['search_question']}")
        if m.get("sub_questions"):
            info.append("나눠 찾기: " + " / ".join(m["sub_questions"]))
        if m.get("elapsed") is not None:
            info.append(f"응답 시간 {m['elapsed']:.1f}초")
        if info:
            st.caption(" · ".join(info))


# ───────────────────────── 4. 화면 구성 ─────────────────────────
st.set_page_config(page_title="Q&A", page_icon="💬")
st.title("💬 Q&A")

if not MODEL:
    st.error(".env 파일에 OPENAI_MODEL=사용할모델명 을 추가해주세요.")
    st.stop()

# 대화 기록은 화면 메모리에만 보관 (새로고침·새 채팅 시 사라짐)
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    # [변경] 난이도 선택: 바꾸면 다른 데이터를 쓰므로 대화를 비운다
    difficulty = st.radio(
        "난이도", DIFFICULTIES, index=DIFFICULTIES.index(DEFAULT_DIFFICULTY), on_change=reset_chat
    )
    # [변경] 도메인 선택: 지금은 shipping 고정. 열려면 disabled=True를 지우면 된다
    domain = st.radio(
        "도메인", DOMAINS, index=DOMAINS.index(DEFAULT_DOMAIN), disabled=True
    )
    # [변경] 검색 방식 선택 (기본 naive). 바꿔도 대화는 유지해서 같은 질문을 방식별로 비교할 수 있다
    mode_label = st.radio(
        "검색 방식", list(RETRIEVAL_MODES), index=list(RETRIEVAL_MODES).index(DEFAULT_RETRIEVAL_MODE)
    )
    retrieval_mode = RETRIEVAL_MODES[mode_label]
    # page 확장 켜고 끄기. 바꿔도 대화는 유지해서 같은 질문을 켜고 끈 상태로 비교할 수 있다
    page_context = st.toggle(
        "page 확장", value=DEFAULT_PAGE_CONTEXT,
        help="찾은 문단·표 대신 그 쪽(page) 전체를 LLM에 넣는다. 같은 파일+page는 한 번만",
    )
    if st.button("새 채팅", use_container_width=True):
        reset_chat()
        st.rerun()
    st.caption(f"모델: {MODEL}")

for m in st.session_state.messages:
    show_message(m)

if question := st.chat_input("질문을 입력하세요"):
    history = build_history()  # 새 질문을 넣기 전의 대화만 전달
    show_message({"role": "user", "content": question})

    with st.spinner("답변 생성 중..."):
        try:
            started = time.time()
            with trace_request(question) as trace:
                trace.update(metadata={  # tracing.py 대신 여기서 메타데이터 추가
                    "retrieval_mode": retrieval_mode,
                    "difficulty": difficulty,
                    "domain": domain,
                    "page_context": page_context,
                })
                result = get_answer(question, history, difficulty, domain, retrieval_mode, page_context)
                trace.update(output=result)
                request_id = trace.trace_id  # 요청 ID
            get_langfuse_client().flush()  # tracing.py 대신 여기서 전송
            elapsed = time.time() - started
        except OpenAIError as e:
            st.error(f"OpenAI 호출 실패: {e}")
            st.stop()  # 실패한 질문은 기록에 남기지 않는다
        except Exception as e:  # noqa: BLE001 — 검색(Qdrant·모델) 오류도 화면에 보여준다
            st.error(f"RAG 처리 실패: {type(e).__name__}: {e}")
            st.stop()

    reply = {
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "route": result["route"],
        "search_question": result.get("search_question", ""),
        "sub_questions": result.get("sub_questions", []),
        "question": question,
        "mode": mode_label,
        "page_context": page_context,
        "elapsed": elapsed,
    }
    show_message(reply)
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append(reply)
