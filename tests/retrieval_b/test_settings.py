"""settings.py 테스트: 환경변수 → Settings, domain·difficulty·levels 검사.

실행 (레포 맨 위에서)
    uv run pytest tests/retrieval_b/test_settings.py -q

[2026-11-10 신설] 이유: 환경변수로 설정을 바꾸는 기능이 새로 생김 → 효과: 잘못된 환경변수 값이 어떤 변수 문제인지 알려주는 오류로 막히는지 확인됨.
"""

from __future__ import annotations

import pytest

from rag_service.retrieval_b import settings as settings_mod
from rag_service.retrieval_b.settings import (
    Settings,
    check_levels,
    load_settings,
    validate_scope,
)


def test_empty_env_gives_defaults():
    assert load_settings({}) == Settings()


def test_env_overrides_each_value():
    s = load_settings(
        {
            "RETRIEVAL_B_K": "7",
            "RETRIEVAL_B_LEVELS": " Para , table,para ",
            "RETRIEVAL_B_CANDIDATE_K": "40",
            "RETRIEVAL_B_RRF_K": "0",
            "RETRIEVAL_B_RERANK_N": "80",
            "RETRIEVAL_B_RERANK_MAX_LENGTH": "512",
            "RETRIEVAL_B_RERANK_BATCH_SIZE": "8",
        }
    )
    assert s == Settings(
        k=7,
        levels=("para", "table"),
        candidate_k=40,
        rrf_k=0,
        rerank_n=80,
        rerank_max_length=512,
        rerank_batch_size=8,
    )


def test_blank_env_values_are_ignored():
    assert (
        load_settings({"RETRIEVAL_B_K": "  ", "RETRIEVAL_B_LEVELS": ""}) == Settings()
    )


@pytest.mark.parametrize(
    "env, word",
    [
        ({"RETRIEVAL_B_K": "five"}, "RETRIEVAL_B_K"),
        ({"RETRIEVAL_B_K": "0"}, "k"),
        ({"RETRIEVAL_B_RRF_K": "-1"}, "rrf_k"),
        ({"RETRIEVAL_B_LEVELS": "para,chapter"}, "chapter"),
        ({"RETRIEVAL_B_LEVELS": ",,"}, "levels"),
        ({"RETRIEVAL_B_K": "60"}, "rerank_n"),  # 기본 rerank_n 50 < k 60
    ],
)
def test_bad_env_values_say_what_is_wrong(env, word):
    with pytest.raises(ValueError, match=word):
        load_settings(env)


def test_settings_are_frozen():
    with pytest.raises(AttributeError):
        Settings().k = 3  # type: ignore[misc]


def test_get_settings_reads_environment_once(monkeypatch):
    monkeypatch.setattr(settings_mod, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("RETRIEVAL_B_K", "9")
    settings_mod.get_settings.cache_clear()
    try:
        assert settings_mod.get_settings().k == 9
        monkeypatch.setenv("RETRIEVAL_B_K", "3")
        assert settings_mod.get_settings().k == 9  # 프로세스당 한 번만 읽는다
        settings_mod.get_settings.cache_clear()
        assert settings_mod.get_settings().k == 3
    finally:
        settings_mod.get_settings.cache_clear()


@pytest.mark.parametrize(
    "domain, difficulty, expected",
    [
        ("shipping", "basic", ("shipping", "basic")),
        (" Finance ", "ADVANCED", ("finance", "advanced")),
        ("manufacturing", "intermediate", ("manufacturing", "intermediate")),
    ],
)
def test_validate_scope_normalizes(domain, difficulty, expected):
    assert validate_scope(domain, difficulty) == expected


@pytest.mark.parametrize(
    "domain, difficulty, error",
    [
        ("shiping", "basic", ValueError),
        ("shipping", "hard", ValueError),
        ("", "basic", ValueError),
        (None, "basic", TypeError),
        ("shipping", 1, TypeError),
    ],
)
def test_validate_scope_rejects(domain, difficulty, error):
    with pytest.raises(error):
        validate_scope(domain, difficulty)


def test_check_levels_keeps_order_and_drops_duplicates():
    assert check_levels(["table", "PARA", "table"]) == ("table", "para")
