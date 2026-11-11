"""Windows 백엔드의 대표 실행 진입점 및 HTTP 요청과 서버 수명주기 관리."""

import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from threading import Lock
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.security import APIKeyHeader

from rag_service.contracts.query import QueryRequest, QueryResponse
from rag_service.observability.tracing import optional_trace, update_trace
from rag_service.pipeline import create_service_chain

logger = logging.getLogger(__name__)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 실행 위치의 환경 설정 로드 및 OpenAI 키와 별개인 서비스 접속 키 확인
    load_dotenv()
    if not os.getenv("RAG_API_KEY", "").strip():
        raise RuntimeError("서버 .env에 RAG_API_KEY를 설정해주세요")
    app.state.query_lock = Lock()
    # 서버 시작 시 처리 체인 한 번 구성 및 실제 RAG 요청 시 검색기 준비
    app.state.chain = create_service_chain()
    try:
        yield
    finally:
        # 요청별 전송 대기 대신 프로세스 종료 시 남은 기록 전송
        if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
            try:
                from langfuse import get_client

                get_client().shutdown()
            except Exception:
                logger.exception("Langfuse shutdown failed")


app = FastAPI(title="KANT RAG API", version="0.1.0", lifespan=lifespan)


def require_api_key(key: str | None = Depends(api_key_header)):
    """화면과 평가 요청의 공통 인증 키 확인."""
    expected = os.getenv("RAG_API_KEY", "").strip()
    if not expected:
        raise HTTPException(status_code=503, detail="API 키가 설정되지 않았습니다")
    if key is None or not secrets.compare_digest(key.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="API 키를 확인해주세요")


@app.get("/health")
def health():
    # 모델 다운로드나 Qdrant 조회를 발생시키지 않는 프로세스 상태 확인
    return {"status": "ok", "scope": "api_process"}


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
def query(body: QueryRequest, request: Request, response: Response):
    """서버 프로세스에서 한 요청씩 처리 및 추가 요청에 재시도 가능 상태 반환."""
    request_id = uuid4().hex
    # 대기열에 요청을 쌓지 않고 처리 중인 경우 즉시 429 응답
    if not request.app.state.query_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail={"message": "다른 질문을 처리 중입니다. 잠시 후 다시 시도해주세요", "request_id": request_id},
            headers={"Retry-After": "5"},
        )
    started = time.perf_counter()
    trace_id = None
    try:
        with optional_trace(body.question, request_id) as trace:
            trace_id = trace.trace_id if trace is not None else None
            update_trace(trace, metadata={
                "request_id": request_id,
                "domain": body.domain,
                "difficulty": body.difficulty,
                "retrieval_mode": body.retrieval_mode,
                "page_context": body.page_context,
            })
            result = request.app.state.chain.invoke(body)
            update_trace(trace, output=result)
        # 체인 결과의 응답 형식 검증 및 서버 로그와 Langfuse 조회용 식별자 추가
        payload = QueryResponse(
            **result, request_id=request_id, trace_id=trace_id,
            elapsed=round(time.perf_counter() - started, 3),
        )
        response.headers["X-Request-ID"] = request_id
        return payload
    except Exception:
        logger.exception("query failed request_id=%s trace_id=%s", request_id, trace_id)
        raise HTTPException(status_code=503, detail={
            "message": "질문 처리에 실패했습니다. 서버 실행 기록을 확인해주세요",
            "request_id": request_id, "trace_id": trace_id,
        }) from None
    finally:
        # 성공과 실패 모두 다음 질문을 받을 수 있도록 처리 잠금 해제
        request.app.state.query_lock.release()


if __name__ == "__main__":
    import uvicorn

    # 한 프로세스에서 검색 자원 재사용 및 외부 네트워크 요청 수신
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
