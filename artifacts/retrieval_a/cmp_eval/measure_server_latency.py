"""팀 서버 API(/query)로 응답 시간을 잰다. 서버에 이미 올라가 있는 모델·검색 자원을 그대로 쓴다 (서버 구현: retrieval_a).

measure_latency.py(로컬)와 같은 질문 두 개를 난이도·방식마다 보낸다.
왕복 시간(내 PC 기준)과 서버가 응답에 넣는 처리 시간(elapsed)을 함께 남긴다.
결과: latency_server.jsonl
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
OUT = HERE / "latency_server.jsonl"
load_dotenv(ROOT / ".env")
URL = os.environ["RAG_API_URL"].rstrip("/")
KEY = os.environ["RAG_API_KEY"]
TIMEOUT = float(os.getenv("RAG_API_TIMEOUT") or 600)

health = json.load(urllib.request.urlopen(URL + "/health", timeout=10))
print("health:", health, flush=True)
for d in ("basic", "intermediate", "advanced"):
    questions = [q["question"] for q in json.loads(
        (ROOT / "data" / d / "shipping" / "evaluation" / "development.json").read_text(encoding="utf-8-sig"))]
    for mode in ("naive", "hybrid", "hybrid_rerank"):
        for order, question in enumerate(questions[:2], start=1):
            body = {"question": question, "difficulty": d, "retrieval_mode": mode, "include_context": True}
            req = urllib.request.Request(URL + "/query", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "X-API-Key": KEY})
            t = time.perf_counter()
            error, res = None, {}
            try:
                res = json.load(urllib.request.urlopen(req, timeout=TIMEOUT))
            except Exception as e:  # noqa: BLE001
                error = f"{type(e).__name__}: {e}"
            rtt = time.perf_counter() - t
            elapsed = res.get("elapsed")
            if isinstance(elapsed, dict):
                elapsed = elapsed.get("total") or elapsed.get("seconds")
            row = {"config": "서버 (retrieval_a)", "difficulty": d, "mode": mode, "order": order, "seconds": round(rtt, 2),
                   "server_elapsed": elapsed, "route": res.get("route"), "status": res.get("status"),
                   "n_contexts": len(res.get("contexts") or []), "request_id": res.get("request_id"), "error": error,
                   "at": datetime.now().isoformat(timespec="seconds")}
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{d} {mode} #{order}: 왕복 {rtt:.1f}s · 서버 {elapsed} ({row['route']}/{row['status']}){' ' + error if error else ''}", flush=True)
