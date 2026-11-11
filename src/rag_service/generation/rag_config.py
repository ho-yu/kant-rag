"""RAG 답변 체인 설정(config/generation/rag.yaml) 읽기. 파일이나 값이 없으면 기본값 사용."""

from functools import lru_cache
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "generation" / "rag.yaml"
DEFAULTS = {"query_guard": {"max_sub_questions": 3}, "context": {"max_docs": 8}}


@lru_cache
def load_rag_config() -> dict:
    """기본값 위에 rag.yaml의 값을 덮어쓴 설정."""
    data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) if CONFIG_PATH.exists() else None
    return {section: {**values, **((data or {}).get(section) or {})} for section, values in DEFAULTS.items()}


def max_sub_questions() -> int:
    """질문 사전 판단에서 나눠 찾는 검색 질문 최대 개수."""
    return int(load_rag_config()["query_guard"]["max_sub_questions"])


def max_docs() -> int:
    """나눠 찾은 검색 결과를 합쳐 LLM에 넘기는 최대 자료 수."""
    return int(load_rag_config()["context"]["max_docs"])
