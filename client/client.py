"""간단한 채팅 클라이언트 (Streamlit + OpenAI + RAG 연결 지점)

처리 흐름
    질문 입력 → decide_route(): RAG가 필요한지 판단
                 ├─ "rag"    → answer_with_rag(): RAG(LangChain) 체인 호출  ← 팀에서 연결
                 └─ "direct" → answer_directly(): OpenAI가 바로 답변

- 사이드바에서 난이도(basic/intermediate/advanced)와 도메인을 고른다.
- 이전 대화를 매번 함께 보내서 앞 내용을 기억하게 한다.
- 대화 기록은 화면 메모리에만 두고, "새 채팅"을 누르거나 난이도를 바꾸면 비운다.

실행: uv run streamlit run client/client.py
"""
import os
import time

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv()

# ───────────────────────── 1. 설정 (여기만 고치면 됨) ─────────────────────────
MODEL = os.getenv("OPENAI_MODEL", "")                    # .env에 OPENAI_MODEL=모델명
ROUTER_MODEL = os.getenv("OPENAI_ROUTER_MODEL", MODEL)   # 판단용 모델 (작은 모델로 바꿔도 됨)
USE_ROUTER = True                                        # False면 항상 RAG를 사용
MAX_HISTORY_TURNS = 5                                    # 함께 보낼 이전 대화 수 (질문+답변 1쌍 = 1턴)

# [변경] 난이도·도메인 선택지와 기본값
DIFFICULTIES = ["basic", "intermediate", "advanced"]
DOMAINS = ["finance", "manufacturing", "shipping"]
DEFAULT_DIFFICULTY = "basic"
DEFAULT_DOMAIN = "shipping"

# [변경] 도메인별 설명 (판단 프롬프트에 쓰임). 도메인을 열 때 여기에 추가
DOMAIN_HINTS = {
    "shipping": "해운 예약·서류 제출·VGM·선하증권·마감일·담당 창구 등 업무 문서 내용",
}

SYSTEM_PROMPT = "너는 질문에 간결하고 정확하게 답하는 한국어 어시스턴트야. 모르는 것은 모른다고 답해."
ROUTER_PROMPT = """사용자의 새 질문에 답하려면 업무 문서({domain_hint})를 찾아봐야 하는지 판단해.
- 문서를 찾아봐야 하면: RAG
- 인사, 잡담, 일반 상식처럼 문서와 무관하면: DIRECT
이전 대화가 있으면 후속 질문인지 보고 같은 기준으로 판단해. 확신이 없으면 RAG.
답은 RAG 또는 DIRECT 한 단어만 써."""


# ───────────────────────── 2. 답변 처리 흐름 ─────────────────────────
@st.cache_resource
def get_client():
    return OpenAI()  # OPENAI_API_KEY 환경변수를 자동으로 읽는다


def get_answer(question, history, difficulty, domain):  # [변경] difficulty, domain 추가
    """전체 흐름: 판단 → (RAG 또는 직접 답변). {"answer", "sources", "route"}를 돌려준다."""
    route = decide_route(question, history, domain)
    if route == "rag":
        result = answer_with_rag(question, history, difficulty, domain)
    else:
        result = answer_directly(question, history)
    result["route"] = route
    return result


def decide_route(question, history, domain):
    """RAG가 필요한지 판단해서 "rag" 또는 "direct"를 돌려준다. (애매하거나 실패하면 "rag")"""
    if not USE_ROUTER:
        return "rag"
    domain_hint = DOMAIN_HINTS.get(domain, "업무 문서")
    past = "\n".join(f"{m['role']}: {m['content'][:200]}" for m in history) or "(없음)"
    try:
        response = get_client().chat.completions.create(
            model=ROUTER_MODEL,
            messages=[
                {"role": "system", "content": ROUTER_PROMPT.format(domain_hint=domain_hint)},
                {"role": "user", "content": f"이전 대화:\n{past}\n\n새 질문: {question}"},
            ],
        )
    except OpenAIError:
        return "rag"
    text = (response.choices[0].message.content or "").strip().upper()
    return "direct" if text.startswith("DIRECT") else "rag"


def answer_with_rag(question, history, difficulty, domain):  # [변경] difficulty, domain 추가
    """RAG(LangChain) 체인을 호출한다.  ← 팀에서 만든 RAG를 여기에 연결

    difficulty, domain으로 어떤 데이터(컬렉션)를 검색할지 정한다.
    돌려줄 형식: {"answer": 답변 문자열, "sources": [{"book_id": "SB01", "page": 3}, ...]}
    """
    # TODO: 예) from rag_service.pipeline import run_rag
    #            return run_rag(question, history, difficulty=difficulty, domain=domain)
    return {
        "answer": f"(RAG 미연결 · {difficulty}/{domain}) answer_with_rag()에 RAG 체인 호출을 연결하세요.",
        "sources": [],
    }


def answer_directly(question, history):
    """RAG 없이 OpenAI가 바로 답한다."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    messages.append({"role": "user", "content": question})
    response = get_client().chat.completions.create(model=MODEL, messages=messages)
    return {"answer": response.choices[0].message.content, "sources": []}


# ───────────────────────── 3. 화면 보조 함수 ─────────────────────────
def build_history():
    """화면에 쌓인 대화 중 최근 N턴만 골라 API 형식으로 만든다."""
    recent = st.session_state.messages[-MAX_HISTORY_TURNS * 2:]
    return [{"role": m["role"], "content": m["content"]} for m in recent]


def reset_chat():
    """대화 기록을 비운다. (새 채팅 버튼, 난이도 변경 시 사용)"""
    st.session_state.messages = []


def show_message(m):
    """메시지 하나를 화면에 그린다. (답변이면 출처, 사용 방식, 응답 시간도 함께)"""
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander(f"출처 {len(m['sources'])}건"):
                for s in m["sources"]:
                    st.markdown(f"- **{s.get('book_id', '?')}** p.{s.get('page', '?')}")
        info = []
        if m.get("route"):
            info.append("RAG 사용" if m["route"] == "rag" else "직접 답변")
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
            result = get_answer(question, history, difficulty, domain)
            elapsed = time.time() - started
        except OpenAIError as e:
            st.error(f"OpenAI 호출 실패: {e}")
            st.stop()  # 실패한 질문은 기록에 남기지 않는다

    reply = {
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "route": result["route"],
        "elapsed": elapsed,
    }
    show_message(reply)
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append(reply)