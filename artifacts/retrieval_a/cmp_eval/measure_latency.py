"""client와 같은 서비스 체인(사전 판단 → 검색 → 답변 → 사후 검사)으로 첫 번째·두 번째 질문 응답 시간을 잰다.

새 프로세스 하나에서 한 구성·한 난이도를 잰다 (서버를 막 켠 상태와 같게).
방식마다 naive → hybrid → hybrid_rerank 순서로 질문 두 개를 보낸다.
첫 번째 질문에는 그 방식을 처음 쓸 때의 준비(모델·청크 로딩)가 들어가고, 두 번째는 준비가 끝난 뒤의 시간이다.

사용: python measure_latency.py <A-기존|A-Qdrant|B> <basic|intermediate|advanced>  → latency.jsonl에 한 줄씩 추가
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

CONFIG, DIFFICULTY = sys.argv[1], sys.argv[2]
HERE = Path(__file__).parent
ROOT = HERE.parents[2]
OUT = HERE / "latency.jsonl"

os.environ["OPENAI_MODEL"] = "gpt-6-luna"
if CONFIG.startswith("A-"):
    os.environ["RETRIEVAL_A_SPARSE_SOURCE"] = "local" if CONFIG == "A-기존" else "qdrant"

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from rag_service.contracts.query import QueryRequest  # noqa: E402
from rag_service.pipeline import chain as chain_mod  # noqa: E402

if CONFIG == "B":
    _cache = {}

    def _b_resources(domain, difficulty):
        """서비스 체인이 retrieval_b 검색기를 쓰도록 측정할 때만 바꿔 끼운다."""
        if (domain, difficulty) not in _cache:
            rag = chain_mod.create_chain(domain=domain, difficulty=difficulty, impl="retrieval_b",
                                         include_context=True, trace=False)
            _cache[(domain, difficulty)] = (rag, None)
        return _cache[(domain, difficulty)]

    chain_mod._cached_rag_resources = _b_resources

questions = [q["question"] for q in json.loads(
    (ROOT / "data" / DIFFICULTY / "shipping" / "evaluation" / "development.json").read_text(encoding="utf-8-sig"))]
service = chain_mod.create_service_chain()

for mode in ("naive", "hybrid", "hybrid_rerank"):
    for order, question in enumerate(questions[:2], start=1):
        req = QueryRequest(question=question, difficulty=DIFFICULTY, retrieval_mode=mode, include_context=True)
        t = time.perf_counter()
        error = None
        try:
            result = service.invoke(req)
            route, status = result.get("route"), result.get("status")
        except Exception as e:  # noqa: BLE001
            route, status, error = None, "error", f"{type(e).__name__}: {e}"
        sec = time.perf_counter() - t
        row = {"config": CONFIG, "difficulty": DIFFICULTY, "mode": mode, "order": order, "seconds": round(sec, 2),
               "route": route, "status": status, "error": error, "at": datetime.now().isoformat(timespec="seconds")}
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{CONFIG} {DIFFICULTY} {mode} #{order}: {sec:.1f}s ({route}/{status}){' ' + error if error else ''}", flush=True)
