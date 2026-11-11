"""retrieval_b 설정과 입력 검증.

값은 성격에 따라 두 곳에서 정한다.
- 요청마다 달라지는 값 (domain, difficulty): build_*_retriever 인자로 받고 validate_scope로 검사한다.
  chain.py가 create_chain(domain, difficulty)로 받은 값을 그대로 넘겨 준다.
- 서버가 정하는 튜닝 값 (k, levels, candidate_k, rrf_k, rerank_n, 리랭커 max_length·batch_size):
  Settings 한 곳에 모으고, 환경변수(.env 포함)로 코드를 고치지 않고 바꾼다.
  chain.py는 .invoke(질문)만 부르므로 이 값들은 요청마다 바꿀 수 없다.

우선순위: build_*_retriever에 직접 넘긴 인자 > 환경변수 > Settings 기본값
모델 이름은 예전처럼 EMBEDDING_MODEL, RERANKER_MODEL 환경변수로 바꾼다.

환경변수 (비어 있으면 기본값)
    RETRIEVAL_B_K                    최종으로 돌려줄 청크 수            (기본 5)
    RETRIEVAL_B_LEVELS               검색할 청크 레벨, 쉼표로 구분       (기본 para,table)
    RETRIEVAL_B_CANDIDATE_K          Hybrid가 Dense·BM25에서 각각 뽑는 후보 수 (기본 30)
    RETRIEVAL_B_RRF_K                RRF 상수                         (기본 60)
    RETRIEVAL_B_RERANK_N             리랭커에 넘길 Hybrid 후보 수       (기본 50)
    RETRIEVAL_B_RERANK_MAX_LENGTH    리랭커 입력 최대 토큰 수           (기본 1024)
    RETRIEVAL_B_RERANK_BATCH_SIZE    리랭커 배치 크기                  (기본 16)

[2026-11-10 신설] 이유: 튜닝 상수가 여러 파일에 하드코딩돼 있고 입력 검증이 없었음 → 효과: 설정값을 한 곳에서 환경변수로 바꾸고 잘못된 값은 시작 시 바로 오류.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import get_args

from dotenv import load_dotenv

from rag_service.contracts.chunk import Level
from rag_service.retrieval_b.fusion import RRF_K
from rag_service.storage.corpus import repo_root

DOMAINS = ("shipping", "finance", "manufacturing")
DIFFICULTIES = ("basic", "intermediate", "advanced")
LEVELS: tuple[str, ...] = get_args(Level)

ENV_PREFIX = "RETRIEVAL_B_"

# 정수 설정 이름 → (환경변수 이름, 허용 최솟값)
_INT_FIELDS = {
    "k": ("K", 1),
    "candidate_k": ("CANDIDATE_K", 1),
    "rrf_k": ("RRF_K", 0),
    "rerank_n": ("RERANK_N", 1),
    "rerank_max_length": ("RERANK_MAX_LENGTH", 1),
    "rerank_batch_size": ("RERANK_BATCH_SIZE", 1),
}


def check_int(name: str, value: object, minimum: int = 1) -> int:
    """정수이고 minimum 이상인지 본다. bool(True=1)은 정수로 치지 않는다."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name}는 정수여야 합니다: {value!r}")
    if value < minimum:
        raise ValueError(f"{name}는 {minimum} 이상이어야 합니다: {value}")
    return value


def check_levels(levels: Iterable[str]) -> tuple[str, ...]:
    """공백·대소문자를 정리하고 중복을 뺀다 (순서는 유지). 모르는 레벨이나 빈 목록은 오류."""
    if isinstance(levels, str):
        raise TypeError(f"levels는 문자열 하나가 아니라 목록이어야 합니다: {levels!r}")
    out = tuple(dict.fromkeys(str(x).strip().lower() for x in levels))
    if not out:
        raise ValueError("levels가 비어 있습니다.")
    unknown = [x for x in out if x not in LEVELS]
    if unknown:
        raise ValueError(f"levels는 {LEVELS} 중에서 골라야 합니다: {unknown}")
    return out


def validate_scope(domain: str, difficulty: str) -> tuple[str, str]:
    """클라이언트가 고른 domain·difficulty를 검사한다. " Basic " 같은 입력은 "basic"으로 맞춘다."""
    if not isinstance(domain, str) or not isinstance(difficulty, str):
        raise TypeError(
            f"domain과 difficulty는 문자열이어야 합니다: {domain!r}, {difficulty!r}"
        )
    domain, difficulty = domain.strip().lower(), difficulty.strip().lower()
    if domain not in DOMAINS:
        raise ValueError(f"domain은 {DOMAINS} 중 하나여야 합니다: {domain!r}")
    if difficulty not in DIFFICULTIES:
        raise ValueError(
            f"difficulty는 {DIFFICULTIES} 중 하나여야 합니다: {difficulty!r}"
        )
    return domain, difficulty


@dataclass(frozen=True)
class Settings:
    """서버가 정하는 검색 튜닝 값. 만들 때 값을 검사하므로 잘못된 설정은 바로 오류가 난다."""

    k: int = 5
    levels: tuple[str, ...] = ("para", "table")
    candidate_k: int = 30
    rrf_k: int = RRF_K
    rerank_n: int = 50
    rerank_max_length: int = 1024
    rerank_batch_size: int = 16

    def __post_init__(self) -> None:
        for name, (_, minimum) in _INT_FIELDS.items():
            check_int(name, getattr(self, name), minimum)
        object.__setattr__(self, "levels", check_levels(self.levels))
        if self.rerank_n < self.k:
            raise ValueError(
                f"rerank_n({self.rerank_n})은 k({self.k}) 이상이어야 합니다."
            )


def load_settings(env: Mapping[str, str]) -> Settings:
    """환경변수 묶음에서 Settings를 만든다. 비어 있는 값은 기본값을 쓴다."""
    values: dict[str, object] = {}
    for name, (suffix, _) in _INT_FIELDS.items():
        var = ENV_PREFIX + suffix
        raw = (env.get(var) or "").strip()
        if not raw:
            continue
        try:
            values[name] = int(raw)
        except ValueError:
            raise ValueError(f"환경변수 {var}는 정수여야 합니다: {raw!r}") from None
    raw = (env.get(ENV_PREFIX + "LEVELS") or "").strip()
    if raw:
        values["levels"] = tuple(x for x in raw.split(",") if x.strip())
    try:
        return Settings(**values)
    except ValueError as error:
        raise ValueError(
            f"retrieval_b 환경변수 설정이 잘못되었습니다: {error}"
        ) from error


# 결과 이름에 붙일 짧은 이름. 배치 크기는 점수를 바꾸지 않아서 뺀다.
TAG_NAMES = {
    "k": "k",
    "levels": "lv",
    "candidate_k": "cand",
    "rrf_k": "rrf",
    "rerank_n": "n",
    "rerank_max_length": "len",
}


def settings_tag(settings: Settings) -> str:
    """기본값과 다른 튜닝 값만 이어 붙인 이름 꼬리 (예: _k3_n100, _lvpage). 모두 기본값이면 ""."""
    default = Settings()
    parts = []
    for name, short in TAG_NAMES.items():
        value = getattr(settings, name)
        if value != getattr(default, name):
            text = "-".join(value) if isinstance(value, tuple) else str(value)
            parts.append(f"_{short}{text}")
    return "".join(parts)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """.env와 환경변수를 읽어 프로세스당 한 번만 만든다. 값을 바꿨으면 get_settings.cache_clear()."""
    load_dotenv(repo_root() / ".env")
    return load_settings(os.environ)
