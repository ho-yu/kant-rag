"""LangChain 답변 이벤트를 SSE로 전달한다. 검사 모델의 출력은 노출하지 않는다."""

import json
import os

from fastapi.responses import StreamingResponse
from langchain_core.utils.json import parse_partial_json

from rag_service.generation.guard import check_answer_rules

# 규칙의 키/전화번호 등이 토큰 경계에서 잘려 먼저 노출되지 않도록 끝부분을 보류한다.
PREVIEW_TAIL = 128
STAGES = {
    "prepare_query_input": "질문을 확인하고 있어요",
    "retrieve_docs": "관련 문서를 찾고 있어요",
    "check_answer": "답변과 출처를 확인하고 있어요",
}


def encode_event(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def stream_answer(chain, body):
    """실제 생성 중인 본문과 최종 검사된 결과를 구분해 전달한다."""
    raw, shown, model_run = "", "", None
    blocked = False
    buffer_answer = (os.getenv("RAG_ANSWER_GUARD") or "rule").strip().lower() == "llm"
    yield "status", {"text": "질문을 확인하고 있어요"}
    async for event in chain.astream_events(body, version="v2"):
        kind, name = event["event"], event.get("name")
        if kind == "on_chain_start" and name in STAGES:
            yield "status", {"text": STAGES[name]}
        answer_format = event.get("metadata", {}).get("ui_answer")
        if kind == "on_chat_model_start" and answer_format:
            yield "status", {"text": "답변을 작성하고 있어요"}
        if kind == "on_chat_model_stream" and answer_format and not buffer_answer:
            if model_run != event["run_id"]:
                raw, model_run, blocked = "", event["run_id"], False
                if shown:
                    shown = ""
                    yield "answer", {"text": ""}
            raw += event["data"]["chunk"].text
            if blocked:
                continue
            try:
                draft = (parse_partial_json(raw) or {}).get("answer", "") if answer_format == "json" else raw
            except (ValueError, AttributeError):
                continue
            if not isinstance(draft, str):
                continue
            if check_answer_rules(draft):
                blocked = True
                shown = ""
                yield "answer", {"text": ""}
                continue
            preview = draft[:-PREVIEW_TAIL] if len(draft) > PREVIEW_TAIL else ""
            if preview != shown:
                shown = preview
                yield "answer", {"text": preview}
        if kind == "on_chain_end" and not event.get("parent_ids"):
            yield "done", event["data"]["output"]
            return
    raise RuntimeError("답변 스트림이 최종 결과 없이 종료되었습니다")


class QueryStreamResponse(StreamingResponse):
    """정상 종료뿐 아니라 연결 중단/전송 실패 때도 요청 처리 자리를 반환한다."""

    def __init__(self, content, slots, request_id):
        super().__init__(content, media_type="text/event-stream", headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "X-Request-ID": request_id,
        })
        self.slots = slots

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            try:
                await self.body_iterator.aclose()
            finally:
                self.slots.release()
