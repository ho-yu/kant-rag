"""가온루트해운 업무 도우미. 검색/모델 없이 서버의 SSE 응답을 표시한다."""

import json
import os
import re
from html import escape
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

from rag_service.retrieval_a.naive import REPO_ROOT, collection_name, load_config

load_dotenv()

API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.getenv("RAG_API_KEY", "").strip()
API_TIMEOUT = float(os.getenv("RAG_API_TIMEOUT", "600"))
MAX_HISTORY_TURNS = 5
DATASETS = {level: f"Shipping · {level.title()}" for level in ("basic", "intermediate", "advanced")}
RETRIEVAL_MODES = {"naive": "기본 검색", "hybrid": "통합 검색", "hybrid_rerank": "통합 검색 + 재정렬"}
AVATARS = {"user": ":material/person:", "assistant": ":material/sailing:"}


def dataset_collection(difficulty, mode):
    """모델이나 DB에 연결하지 않고 검색기의 설정에서 컬렉션명을 읽는다."""
    config_name = "rerank" if mode == "hybrid_rerank" else mode
    config = load_config(REPO_ROOT / "config" / "retrieval_a" / f"{config_name}.yaml")
    if "hybrid_config" in config:
        config = load_config(REPO_ROOT / config["hybrid_config"])
    return collection_name(config, domain="shipping", difficulty=difficulty)


def reset_chat():
    st.session_state.messages = []


def build_history():
    messages = [m for m in st.session_state.messages if not m.get("error")]
    return [{"role": m["role"], "content": m["content"]} for m in messages[-MAX_HISTORY_TURNS * 2:]]


def answer_markdown(text):
    """원문은 보존하고 화면의 인용 표시·한글 굵은 글씨만 정리한다."""
    text = re.sub(r"\s*\[S\d+\]", "", text)
    text = re.sub(r"\[S\d*$", "", text)
    return re.sub(r"\*\*(?=\S)(.+?)(?<=\S)\*\*", lambda m: f"**\u200b{m.group(1)}\u200b**", text)


def show_processing_details(message):
    sources = message.get("sources", [])
    status = {
        "answered": "답변 완료", "needs_clarification": "추가 확인이 필요해요",
        "abstained": "자료 확인이 필요해요", "blocked": "안내",
    }.get(message.get("status"), "답변")
    info = [status]
    if sources:
        info.append(f"참고 문서 {len(sources)}건")
    if message.get("elapsed") is not None:
        info.append(f"{message['elapsed']:.1f}초")
    if message.get("mode"):
        info.append(RETRIEVAL_MODES.get(message["mode"], message["mode"]))
    st.markdown("**처리 상세**")
    st.caption(" · ".join(info))
    if message.get("search_question"):
        st.caption(f"검색 질문: {message['search_question']}")
    if message.get("sub_questions"):
        st.caption("나누어 찾은 질문\n\n" + "\n".join(f"- {q}" for q in message["sub_questions"]))
    if message.get("request_id"):
        st.caption(f"요청 ID: {message['request_id']}")
    st.divider()


def show_sources(message):
    sources = message.get("sources", [])
    if sources:
        with st.expander(f"참고 문서 확인 · {len(sources)}건", icon=":material/library_books:"):
            for i, source in enumerate(sources):
                meta = source.get("metadata", {})
                if i:
                    st.divider()
                title = meta.get("doc_title") or meta.get("file_name") or meta.get("book_id") or "참고 문서"
                st.markdown(f"**[{source['id']}] {title}**")
                location = [str(meta[k]) for k in ("file_name", "section_title") if meta.get(k)]
                if meta.get("page") is not None:
                    location.append(f"{meta['page']}페이지")
                st.caption(" · ".join(location))
                st.markdown("\n".join(f"> {line}" for line in source["content"].splitlines()))
    if message.get("invalid_source_ids"):
        st.warning("일부 인용 자료를 확인하지 못했어요. 참고 문서를 확인해주세요.")
    if message.get("guard_errors"):
        st.warning("답변 검사 일부가 완료되지 않았어요. 중요한 내용은 참고 문서와 함께 확인해주세요.")


def show_message(message):
    with st.chat_message(message["role"], avatar=AVATARS[message["role"]]):
        if message.get("error"):
            st.error(message["content"])
        elif message["role"] == "assistant":
            show_processing_details(message)
            st.markdown(answer_markdown(message["content"]))
            show_sources(message)
        else:
            st.markdown(message["content"])


def query_server(payload, on_event):
    """한 번의 요청으로 실제 SSE를 읽는다. 자동 재전송과 가짜 타이핑 효과는 없다."""
    with requests.post(
        f"{API_URL}/query", params={"stream": "true"}, json=payload,
        headers={"X-API-Key": API_KEY, "Accept": "text/event-stream"},
        timeout=(10, API_TIMEOUT), stream=True,
    ) as response:
        if not response.ok:
            try:
                detail = response.json().get("detail", "서버 요청에 실패했습니다.")
            except ValueError:
                detail = "서버 응답을 확인할 수 없습니다."
            if isinstance(detail, dict):
                detail = detail.get("message", "서버 요청에 실패했습니다.")
            elif isinstance(detail, list):
                detail = "입력 길이를 줄이거나 새 대화로 다시 시도해주세요."
            raise ValueError(f"{response.status_code}: {detail}")
        response.encoding = "utf-8"
        event = ""
        for line in response.iter_lines(chunk_size=1, decode_unicode=True):
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
                if event == "error":
                    raise ValueError(data["message"])
                if event == "done":
                    return data
                on_event(event, data)
    raise ValueError("답변 수신이 중단되었어요. 잠시 후 다시 시도해주세요.")


def show_intro():
    st.markdown('<div class="intro-label">무엇을 도와드릴까요?</div>', unsafe_allow_html=True)
    examples = [
        ("예약 · 장비", "예약할 때 어떤 정보가 필요한가요?", ":material/directions_boat:"),
        ("서류 · 마감", "SI와 VGM은 어디로 제출하나요?", ":material/description:"),
        ("비용 · 정산", "보관료와 장비반납지연료는 어떻게 다른가요?", ":material/receipt_long:"),
    ]
    selected = None
    for column, (label, question, icon) in zip(st.columns(3), examples, strict=True):
        with column, st.container(border=True):
            st.markdown(f"**{label}**")
            st.caption(question)
            if st.button("질문하기", key=label, icon=icon, width="stretch"):
                selected = question
    return selected


def main():
    st.set_page_config(page_title="가온루트해운 | 업무 도우미", page_icon="⚓", layout="wide")
    st.html(f"<style>{Path(__file__).with_name('style.css').read_text(encoding='utf-8')}</style>")
    if "messages" not in st.session_state:
        reset_chat()
    with st.sidebar:
        st.markdown('''<div class="sidebar-brand"><span class="brand-mark">G</span>
            <strong>가온루트해운</strong></div>''', unsafe_allow_html=True)
        if st.button("새 대화", icon=":material/add:", type="primary", width="stretch"):
            reset_chat()
            st.rerun()
        st.divider()
        difficulty = st.radio("데이터셋", list(DATASETS), format_func=DATASETS.get, on_change=reset_chat)
        dataset_info = st.empty()
        with st.expander("검색 설정", icon=":material/tune:"):
            mode = st.radio("검색 방식", list(RETRIEVAL_MODES), format_func=RETRIEVAL_MODES.get)
            page_context = st.toggle("페이지 전체 참고", value=False)
        dataset_info.markdown(f'''<div class="dataset-info">
            <div class="database-label"><span>VECTOR DB</span><strong>Qdrant</strong></div>
            <span>COLLECTION</span><code>{escape(dataset_collection(difficulty, mode))}</code>
            </div>''', unsafe_allow_html=True)
        st.markdown('''<div class="sidebar-footer"><span>서비스 제공</span>
            <strong>one token 해적단</strong></div>''', unsafe_allow_html=True)
    st.markdown('''<header class="page-header"><h1>가온루트해운 업무 도우미</h1>
        </header>''', unsafe_allow_html=True)
    if not API_KEY:
        st.error("서비스 연결 설정이 필요합니다. 관리자에게 문의해주세요.")
        st.stop()
    suggested = show_intro() if not st.session_state.messages else None
    for message in st.session_state.messages:
        show_message(message)
    question = st.chat_input("예약, 서류, 마감, 비용… 궁금한 업무를 질문하세요", max_chars=10000) or suggested
    if not question:
        return
    payload = {
        "question": question, "history": build_history(), "domain": "shipping",
        "difficulty": difficulty, "retrieval_mode": mode, "page_context": page_context,
    }
    user_message = {"role": "user", "content": question}
    show_message(user_message)
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        status, answer = st.empty(), st.empty()

        def on_event(event, data):
            if event == "status":
                status.markdown(
                    '<div class="processing-status" role="status">'
                    '<span class="loading-spinner" aria-hidden="true"></span>'
                    f'{escape(data["text"])}</div>', unsafe_allow_html=True,
                )
            elif event == "answer":
                answer.markdown(answer_markdown(data["text"]))

        on_event("status", {"text": "질문을 확인하고 있어요"})
        try:
            result = query_server(payload, on_event)
            reply = {
                **result, "role": "assistant", "content": result["answer"],
                "question": question, "mode": mode, "page_context": page_context,
            }
        except requests.Timeout:
            reply = {"role": "assistant", "error": True, "content": "응답 대기 시간이 초과되었어요. 잠시 후 다시 시도해주세요."}
        except requests.RequestException:
            reply = {"role": "assistant", "error": True, "content": "서버 연결이 끊겼어요. 잠시 후 다시 시도해주세요."}
        except (ValueError, KeyError) as exc:
            reply = {"role": "assistant", "error": True, "content": str(exc)}
        status.empty()
        answer.empty()
    st.session_state.messages.extend([user_message, reply])
    st.rerun()


if __name__ == "__main__":
    main()
