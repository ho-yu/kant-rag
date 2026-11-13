"""최적 k 2단계 실험(--tag <난이도>-dev-k<k>-gpt6luna)을 Langfuse에서 모은다 → collected_k.json
수집 방식은 collect.py와 같다 (문항 점수·Judge 규칙·토큰). k5 기준은 collected.json의 hybrid_rerank를 쓴다.
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("collect", Path(__file__).with_name("collect.py"))
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

C.OUT = Path(__file__).with_name("collected_k.json")
C.CONFIGS = {}
for k in (5, 7, 8, 10, 15):
    C.CONFIGS[f"A-k{k}"] = rf"^(hybrid_rerank)_\d{{8}}-\d{{4}}_[a-z]+-dev-k{k}-[a-z0-9]+$"
    C.CONFIGS[f"B-k{k}"] = rf"^b_(hybrid_rerank)_\d{{8}}-\d{{4}}_[a-z]+-dev-k{k}-[a-z0-9]+$"

if __name__ == "__main__":
    C.main()
