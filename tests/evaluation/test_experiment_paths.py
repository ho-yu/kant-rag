"""실험 이름 규칙·결과 폴더(evaluation/experiment_paths.py) 테스트. 비교판 파일 없이 돈다.

uv run pytest tests/evaluation/test_experiment_paths.py -q
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PATHS_PATH = Path(__file__).resolve().parents[2] / "evaluation" / "experiment_paths.py"


@pytest.fixture(scope="module")
def paths():
    spec = importlib.util.spec_from_file_location("experiment_paths", PATHS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["page_k", "A1", "rrf20-k", "x" * 64])
def test_check_exp_accepts_valid_names(paths, name):
    assert paths.check_exp(name) == name


@pytest.mark.parametrize(
    "name", ["../x", "a/b", "a\\b", ".hidden", "a.b", "_x", "-x", "x" * 65, "한글"]
)
def test_check_exp_rejects_bad_names(paths, name):
    with pytest.raises(ValueError, match="실험 이름"):
        paths.check_exp(name)


def test_results_dir_and_banner(paths, tmp_path):
    assert paths.check_exp(None) is None and paths.check_exp("") is None
    assert paths.results_dir(tmp_path) == tmp_path / "results"
    assert (
        paths.results_dir(tmp_path, "page_k")
        == tmp_path / "results" / "experiments" / "page_k"
    )
    assert paths.board_path(tmp_path, "page_k").name == "experiment_board.html"
    assert paths.exp_banner(None) == "[실험] 기본 (results/)"
    assert paths.exp_banner("page_k") == "[실험] page_k (results/experiments/page_k)"


def test_refresh_board_is_silent_noop_without_board_file(paths, tmp_path, capsys):
    """비교판(로컬 도구)이 없는 레포: 경고·오류 없이 None."""
    assert paths.refresh_board(tmp_path, None) is None
    assert paths.refresh_board(tmp_path, "page_k") is None
    assert capsys.readouterr().out == ""
    assert not (tmp_path / "results").exists()  # 아무것도 만들지 않는다


def test_refresh_board_calls_board_refresh_when_present(paths, tmp_path, capsys):
    script = tmp_path / paths.BOARD_SCRIPT
    script.parent.mkdir(parents=True)
    script.write_text(
        "def refresh(root, exp=None):\n    print('refreshed', exp)\n    return root / 'x.html'\n",
        encoding="utf-8",
    )
    assert paths.refresh_board(tmp_path, "page_k") == tmp_path / "x.html"
    assert capsys.readouterr().out == "refreshed page_k\n"


def test_refresh_board_warns_only_on_real_failure(paths, tmp_path, capsys):
    script = tmp_path / paths.BOARD_SCRIPT
    script.parent.mkdir(parents=True)
    script.write_text(
        "def refresh(root, exp=None):\n    raise OSError('disk')\n", encoding="utf-8"
    )
    assert paths.refresh_board(tmp_path) is None
    assert capsys.readouterr().out.strip() == "[경고] 비교판 갱신 실패: OSError: disk"
