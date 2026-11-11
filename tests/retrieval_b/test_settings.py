"""settings.py 테스트: 명령줄 옵션 → Settings, domain·difficulty·levels 검사.

실행 (레포 맨 위에서)
    uv run pytest tests/retrieval_b/test_settings.py -q

[2026-11-10 신설] 이유: 환경변수로 설정을 바꾸는 기능이 새로 생김 → 효과: 잘못된 환경변수 값이 어떤 변수 문제인지 알려주는 오류로 막히는지 확인됨.
[2026-11-11 리팩토링] 이유: 실험 값을 .env 대신 명령줄 옵션으로 받게 바뀜 → 효과: 옵션별 반영, 잘못된 값의 종료 메시지, 환경변수 무시를 확인함.
"""

from __future__ import annotations

import argparse
import re

import pytest

from rag_service.retrieval_b import settings as settings_mod
from rag_service.retrieval_b.settings import (
    Settings,
    add_settings_args,
    check_levels,
    describe_settings,
    retriever_kwargs,
    settings_from_args,
    settings_tag,
    validate_scope,
)


def parse(argv: list[str], **kwargs) -> Settings:
    parser = argparse.ArgumentParser()
    add_settings_args(parser, **kwargs)
    return settings_from_args(parser.parse_args(argv))


@pytest.fixture
def fresh_settings():
    settings_mod.get_settings.cache_clear()
    yield
    settings_mod.get_settings.cache_clear()


def test_no_options_give_defaults(fresh_settings):
    assert parse([]) == Settings()


def test_options_override_each_value(fresh_settings):
    argv = "--k 7 --levels Para table para --cand 40 --rrf 0 --rerank-n 80 --rerank-len 512"
    s = parse(argv.split())
    assert s == Settings(
        k=7,
        levels=("para", "table"),
        candidate_k=40,
        rrf_k=0,
        rerank_n=80,
        rerank_max_length=512,
    )
    assert settings_tag(s) == "_k7_cand40_rrf0_n80_len512"


@pytest.mark.parametrize(
    "argv, word",
    [
        ("--k 0", "k는 1 이상"),
        ("--rrf -1", "rrf_k는 0 이상"),
        ("--levels para chapter", "chapter"),
        ("--k 9 --rerank-n 5", r"rerank_n\(5\)은 k\(9\) 이상이어야 합니다"),
        ("--k 60", "rerank_n"),  # 기본 rerank_n 50 < k 60
    ],
)
def test_bad_options_stop_with_reason(fresh_settings, argv, word):
    with pytest.raises(
        SystemExit, match="retrieval_b 튜닝 값이 잘못되었습니다"
    ) as info:
        parse(argv.split())
    assert re.search(word, str(info.value))  # 어떤 값이 문제인지 알려 준다


def test_non_integer_option_is_argparse_error(fresh_settings, capsys):
    with pytest.raises(SystemExit) as info:
        parse(["--k", "five"])
    assert info.value.code == 2
    assert "--k" in capsys.readouterr().err


def test_skip_leaves_option_out(fresh_settings):
    parser = argparse.ArgumentParser()
    add_settings_args(parser, skip=("k",))
    assert "--k" not in parser.format_help()
    assert "--rerank-n" in parser.format_help()
    assert settings_from_args(parser.parse_args(["--rerank-n", "20"])) == Settings(
        rerank_n=20
    )
    with pytest.raises(ValueError, match="skip"):
        add_settings_args(argparse.ArgumentParser(), skip=("kk",))


def test_get_settings_ignores_old_environment(fresh_settings, monkeypatch):
    monkeypatch.setenv("RETRIEVAL_B_K", "9")
    monkeypatch.setenv("RETRIEVAL_B_LEVELS", "")
    with pytest.warns(UserWarning, match="RETRIEVAL_B_K는 더 이상 읽지 않습니다"):
        assert settings_mod.get_settings() == Settings()
    assert parse([]) == Settings()  # 명령줄 옵션이 없으면 그대로 기본값


def test_retriever_kwargs_only_pass_what_each_mode_uses():
    s = Settings(k=3, levels=("page",), candidate_k=40, rrf_k=20, rerank_n=20)
    base = {"k": 3, "levels": ["page"]}
    hybrid = {**base, "candidate_k": 40, "rrf_k": 20}
    rerank = {**hybrid, "rerank_n": 20, "max_length": 1024}
    assert retriever_kwargs(s, "naive") == base
    assert retriever_kwargs(s, "bm25") == base
    assert retriever_kwargs(s, "hybrid") == hybrid
    assert retriever_kwargs(s, "rerank") == rerank
    assert retriever_kwargs(s, "hybrid_rerank") == rerank


def test_describe_settings_marks_changed_values():
    assert describe_settings(Settings()) == (
        "k=5 levels=para,table cand=30 rrf=60 rerank-n=50 rerank-len=1024"
    )
    assert "rerank-n=20*" in describe_settings(Settings(rerank_n=20))
    assert "k=5 " in describe_settings(Settings(rerank_n=20))


def test_settings_are_frozen():
    with pytest.raises(AttributeError):
        Settings().k = 3  # type: ignore[misc]


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


def test_settings_tag_only_lists_changed_values():
    assert settings_tag(Settings()) == ""
    assert settings_tag(Settings(k=3)) == "_k3"
    assert settings_tag(Settings(rerank_n=100, candidate_k=50)) == "_cand50_n100"
    assert settings_tag(Settings(levels=("page",))) == "_lvpage"
    assert (
        settings_tag(Settings(rerank_batch_size=4)) == ""
    )  # 점수를 안 바꾸는 값은 뺀다
