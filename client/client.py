"""서버 API를 호출하는 Streamlit 화면 및 세션별 대화 기록 관리.

검색기와 모델의 로딩 없이 HTTP 요청으로 질문 처리 위임.
사용자 브라우저별 대화 상태와 서버에서 받은 출처의 화면 표시 담당.
"""

import os
import re

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# 기본 주소는 Streamlit 실행 컴퓨터의 백엔드이며 별도 서버 사용 시 RAG_API_URL 지정
API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.getenv("RAG_API_KEY", "").strip()
API_TIMEOUT = float(os.getenv("RAG_API_TIMEOUT", "600"))
MAX_HISTORY_TURNS = 5
DIFFICULTIES = ["basic", "intermediate", "advanced"]
RETRIEVAL_MODES = {
    "naive": "naive",
    "hybrid": "hybrid",
    "hybrid+rerank": "hybrid_rerank",
}
SOURCE_FIELDS = ["file_name", "page", "book_id", "chunk_id", "domain", "difficulty"]


def reset_chat():
    """새 채팅 또는 난이도 변경 시 현재 브라우저 세션의 대화 초기화."""
    st.session_state.messages = []


def build_history():
    """최근 대화의 본문만 전송 및 오류 안내와 출처 원문 제외."""
    messages = [m for m in st.session_state.messages if not m.get("error")]
    return [
        {"role": m["role"], "content": m["content"]}
        for m in messages[-MAX_HISTORY_TURNS * 2:]
    ]


def fix_bold(text):
    """한글과 인접한 굵은 글씨 표시의 마크다운 경계 보정."""
    return re.sub(
        r"\*\*(?=\S)(.+?)(?<=\S)\*\*",
        lambda m: f"**\u200b{m.group(1)}\u200b**", text,
    )


def show_message(message):
    """답변과 구조화된 출처 및 서버 요청 식별자 표시."""
    with st.chat_message(message["role"]):
        if message.get("error"):
            st.error(message["content"])
            return
        st.markdown(fix_bold(message["content"]))
        sources = message.get("sources", [])
        if sources:
            with st.expander(f"출처 {len(sources)}건"):
                for source in sources:
                    metadata = source["metadata"]
                    st.markdown(f"**[{source['id']}]**")
                    st.caption(" · ".join(
                        f"{key}: {metadata[key]}" for key in SOURCE_FIELDS if metadata.get(key)
                    ))
                    st.text(source["content"])
        if message.get("invalid_source_ids"):
            st.warning("일부 인용 번호에 대응하는 자료가 없어 출처 확인이 필요합니다.")
        if message.get("guard_errors"):
            st.warning("일부 검사 호출에 실패했습니다. 요청 ID로 서버 기록을 확인해주세요.")
        info = []
        if message.get("route"):
            info.append({
                "rag": "RAG 사용", "direct": "직접 답변",
                "block": "질문 차단", "blocked_answer": "답변 차단",
            }[message["route"]])
        if message.get("mode"):
            info.append(message["mode"])
        if message.get("page_context"):
            info.append("페이지 확장")
        if message.get("elapsed") is not None:
            info.append(f"서버 처리 {message['elapsed']:.1f}초")
        if message.get("request_id"):
            info.append(f"요청 ID {message['request_id']}")
        if info:
            st.caption(" · ".join(info))
        if message.get("search_question") and message["search_question"] != message.get("question"):
            st.caption(f"검색 질문: {message['search_question']}")
        if message.get("sub_questions"):
            st.caption("나눠 찾기: " + " / ".join(message["sub_questions"]))


def query_server(payload):
    """서버 API 호출 및 오류 안내 변환과 자동 재시도에 따른 중복 모델 호출 방지."""
    # 연결 대기 10초와 응답 대기 API_TIMEOUT초의 분리 및 자동 재전송 생략
    response = requests.post(
        f"{API_URL}/query", json=payload,
        headers={"X-API-Key": API_KEY}, timeout=(10, API_TIMEOUT),
    )
    if not response.ok:
        try:
            detail = response.json().get("detail", "서버 요청 실패")
        except ValueError:
            detail = "서버 응답 형식 확인 필요"
        if isinstance(detail, dict):
            detail = f"{detail.get('message', '서버 요청 실패')} / 요청 ID {detail.get('request_id', '-')}"
        elif isinstance(detail, list):
            detail = "질문 또는 대화 기록이 허용 범위를 초과했습니다. 새 채팅으로 다시 시도해주세요."
        raise ValueError(f"{response.status_code}: {detail}")
    return response.json()


st.set_page_config(page_title="Q&A", page_icon="💬")
st.title("💬 Q&A")
if not API_KEY:
    st.error("화면 실행 환경에 RAG_API_KEY를 설정해주세요.")
    st.stop()
if "messages" not in st.session_state:
    reset_chat()

with st.sidebar:
    difficulty = st.radio("난이도", DIFFICULTIES, on_change=reset_chat)
    st.caption("도메인: shipping")
    mode_label = st.radio("검색 방식", list(RETRIEVAL_MODES))
    page_context = st.toggle("페이지 확장", value=False)
    if st.button("새 채팅"):
        reset_chat()
        st.rerun()
    st.caption("검색과 모델 실행은 공용 백엔드 서버에서 처리")

for message in st.session_state.messages:
    show_message(message)

if question := st.chat_input("질문을 입력하세요", max_chars=10000):
    payload = {
        "question": question, "history": build_history(),
        "domain": "shipping", "difficulty": difficulty,
        "retrieval_mode": RETRIEVAL_MODES[mode_label], "page_context": page_context,
    }
    user_message = {"role": "user", "content": question}
    show_message(user_message)
    with st.spinner("서버에서 답변 생성 중..."):
        try:
            result = query_server(payload)
            reply = {
                **result, "role": "assistant", "content": result["answer"],
                "question": question, "mode": mode_label, "page_context": page_context,
            }
        except requests.Timeout:
            reply = {
                "role": "assistant", "error": True,
                "content": "서버 응답 대기 시간이 초과되었습니다. 서버가 계속 처리 중일 수 있으므로 실행 기록을 확인해주세요.",
            }
        except requests.RequestException:
            reply = {"role": "assistant", "error": True, "content": "백엔드 서버 연결을 확인해주세요."}
        except (ValueError, KeyError) as exc:
            reply = {"role": "assistant", "error": True, "content": str(exc)}
    show_message(reply)
    st.session_state.messages.extend([user_message, reply])
