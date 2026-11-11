"""retrieval_b 설정과 입력 검증.

값은 성격에 따라 두 곳에서 정한다.
- 요청마다 달라지는 값 (domain, difficulty): build_*_retriever 인자로 받고 validate_scope로 검사한다.
  chain.py가 create_chain(domain, difficulty)로 받은 값을 그대로 넘겨 준다.
- 서버가 정하는 튜닝 값 (k, levels, candidate_k, rrf_k, rerank_n, 리랭커 max_length·batch_size):
  기본값은 Settings 한 곳에만 둔다. .env와 환경변수는 읽지 않는다.
  팀 기본값을 바꾸려면 Settings의 숫자를 고친다 (팀 결정).

개인 실험은 평가 스크립트의 명령줄 옵션으로 바꾼다 (add_settings_args, settings_from_args).
    --k  --levels  --cand  --rrf  --rerank-n  --rerank-len
    예: uv run python evaluation/answer/run_answers.py --rerank-n 20

우선순위: build_*_retriever에 직접 넘긴 인자(명령줄 옵션) > Settings 기본값
모델 이름은 예전처럼 EMBEDDING_MODEL, RERANKER_MODEL 환경변수로 바꾼다.

[2026-11-10 신설] 이유: 튜닝 상수가 여러 파일에 하드코딩돼 있고 입력 검증이 없었음 → 효과: 설정값을 한 곳에 모으고 잘못된 값은 시작 시 바로 오류.
[2026-11-11 리팩토링] 이유: 개인 실험 값(RETRIEVAL_B_*)이 팀 공용 .env에 섞임 → 효과: .env에는 키·주소·모델 이름만 두고, 실험 값은 명령줄 옵션으로 넘김.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from typing import get_args

from rag_service.contracts.chunk import Level
from rag_service.retrieval_b.fusion import RRF_K

DOMAINS = ("shipping", "finance", "manufacturing")
DIFFICULTIES = ("basic", "intermediate", "advanced")
LEVELS: tuple[str, ...] = get_args(Level)

# 정수 설정 이름 → 허용 최솟값
_INT_FIELDS = {
    "k": 1,
    "candidate_k": 1,
    "rrf_k": 0,
    "rerank_n": 1,
    "rerank_max_length": 1,
    "rerank_batch_size": 1,
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
        for name, minimum in _INT_FIELDS.items():
            check_int(name, getattr(self, name), minimum)
        object.__setattr__(self, "levels", check_levels(self.levels))
        if self.rerank_n < self.k:
            raise ValueError(
                f"rerank_n({self.rerank_n})은 k({self.k}) 이상이어야 합니다."
            )


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


# ---------- 명령줄 옵션 ----------

# 설정 이름 → (명령줄 옵션, 도움말). 배치 크기는 점수를 바꾸지 않아서 옵션을 두지 않는다.
CLI_OPTIONS = {
    "k": ("--k", "최종으로 돌려줄 청크 수"),
    "levels": ("--levels", f"검색할 청크 레벨, {' '.join(LEVELS)} 중 여러 개"),
    "candidate_k": ("--cand", "Hybrid가 Dense·BM25에서 각각 뽑는 후보 수"),
    "rrf_k": ("--rrf", "RRF 상수, 0 이상"),
    "rerank_n": ("--rerank-n", "리랭커에 넘길 Hybrid 후보 수, k 이상"),
    "rerank_max_length": ("--rerank-len", "리랭커 입력 최대 토큰 수"),
}


def add_settings_args(
    parser: argparse.ArgumentParser, *, skip: Iterable[str] = ()
) -> None:
    """검색 튜닝 옵션을 parser에 붙인다. 주지 않은 옵션은 None으로 남아 Settings 기본값을 쓴다.

    skip: 스크립트가 따로 다루는 설정 이름 (예: --ks를 쓰는 비교 스크립트는 skip=("k",)).
    """
    skip = set(skip)
    unknown = skip - CLI_OPTIONS.keys()
    if unknown:
        raise ValueError(
            f"skip은 {tuple(CLI_OPTIONS)} 중에서 골라야 합니다: {sorted(unknown)}"
        )
    default = Settings()
    group = parser.add_argument_group(
        "retrieval_b 튜닝 값 (주지 않으면 settings.py의 Settings 기본값)"
    )
    for name, (flag, text) in CLI_OPTIONS.items():
        if name in skip:
            continue
        value = getattr(default, name)
        if name == "levels":
            group.add_argument(
                flag,
                dest=name,
                nargs="+",
                metavar="LEVEL",
                help=f"{text} (기본 {' '.join(value)})",
            )
        else:
            group.add_argument(
                flag, dest=name, type=int, metavar="N", help=f"{text} (기본 {value})"
            )


def settings_from_args(args: argparse.Namespace) -> Settings:
    """명령줄에서 준 값만 기본 Settings 위에 덮어쓴다. 잘못된 값이면 이유를 보여 주고 끝낸다."""
    changes = {
        name: getattr(args, name)
        for name in CLI_OPTIONS
        if getattr(args, name, None) is not None
    }
    try:
        return dataclasses.replace(get_settings(), **changes)
    except (TypeError, ValueError) as error:
        raise SystemExit(f"retrieval_b 튜닝 값이 잘못되었습니다: {error}") from None


def retriever_kwargs(settings: Settings, mode: str) -> dict[str, object]:
    """settings 값을 build_*_retriever 인자로 바꾼다. 그 방식이 쓰지 않는 값은 넣지 않는다.

    mode: naive, bm25, hybrid, rerank (run_answers의 hybrid_rerank도 rerank로 본다)
    """
    kwargs: dict[str, object] = {"k": settings.k, "levels": list(settings.levels)}
    if mode in ("hybrid", "rerank", "hybrid_rerank"):
        kwargs.update(candidate_k=settings.candidate_k, rrf_k=settings.rrf_k)
    if mode in ("rerank", "hybrid_rerank"):
        kwargs.update(rerank_n=settings.rerank_n, max_length=settings.rerank_max_length)
    return kwargs


def describe_settings(settings: Settings) -> str:
    """실행 첫 줄에 찍을 설정 요약. 기본값과 다른 값에는 * 를 붙인다."""
    default = Settings()
    parts = []
    for name, (flag, _) in CLI_OPTIONS.items():
        value = getattr(settings, name)
        text = ",".join(value) if isinstance(value, tuple) else str(value)
        mark = "*" if value != getattr(default, name) else ""
        parts.append(f"{flag.lstrip('-')}={text}{mark}")
    return " ".join(parts)


# 예전에 쓰던 환경변수 앞머리. 남아 있으면 무시된다고 한 번 알려 준다.
_OLD_ENV_PREFIX = "RETRIEVAL_B_"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """팀 기본 튜닝 값. .env와 환경변수는 읽지 않는다 (실험 값은 명령줄 옵션으로)."""
    old = sorted(
        name
        for name, value in os.environ.items()
        if name.startswith(_OLD_ENV_PREFIX) and value.strip()
    )
    if old:
        warnings.warn(
            f"{', '.join(old)}는 더 이상 읽지 않습니다. "
            "실험 값은 명령줄 옵션(--k, --rerank-n 등)으로 넘기세요.",
            stacklevel=2,
        )
    return Settings()
