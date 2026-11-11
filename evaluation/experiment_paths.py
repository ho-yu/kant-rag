"""실험 이름 규칙과 결과 폴더 경로. 모든 평가 스크립트가 이 파일 하나를 같이 쓴다 (표준 라이브러리만 씀).

  --exp 없음  → results/
  --exp 이름  → results/experiments/<이름>/ (안에 comparison/·diagnose/·answer/ 같은 하위 폴더)
  이름은 영문·숫자로 시작하고 영문·숫자·_·-만 (최대 64자).

실험 비교판(evaluation/reporting/experiment_board.py)은 개인 로컬 도구라 레포에 없을 수 있다.
refresh_board()는 그 파일이 있을 때만 비교판을 다시 만들고, 없으면 아무것도 하지 않는다.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

EXPERIMENTS_DIR = Path("results") / "experiments"
EXP_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
BOARD_HTML = "experiment_board.html"
BOARD_SCRIPT = Path("evaluation") / "reporting" / "experiment_board.py"


def check_exp(name: str | None) -> str | None:
    """실험 이름 검사. 없으면 None(기본 실험). 경로 구분자·점 등 규칙 밖 글자는 오류."""
    if name is None or name == "":
        return None
    if not EXP_PATTERN.match(name):
        raise ValueError(
            f"실험 이름은 영문·숫자로 시작하고 영문·숫자·_·-만 쓸 수 있어요 (최대 64자): {name!r}"
        )
    return name


def results_dir(root: Path, exp: str | None = None) -> Path:
    """결과 최상위 폴더. 기본은 results/, 실험이면 results/experiments/<이름>/."""
    exp = check_exp(exp)
    return root / "results" if exp is None else root / EXPERIMENTS_DIR / exp


def board_path(root: Path, exp: str | None = None) -> Path:
    return results_dir(root, exp) / BOARD_HTML


def exp_banner(exp: str | None) -> str:
    """실험 스크립트가 시작할 때 찍는 한 줄."""
    exp = check_exp(exp)
    return (
        f"[실험] {exp} (results/experiments/{exp})" if exp else "[실험] 기본 (results/)"
    )


def load_module(path: Path, name: str) -> ModuleType:
    """파일 경로로 모듈을 불러온다 (evaluation/은 패키지가 아니라서)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def refresh_board(root: Path, exp: str | None = None) -> Path | None:
    """비교판 스크립트가 있으면 그 실험의 비교판을 다시 만든다.

    파일이 없으면 조용히 None. 있는데 실패하면 경고 한 줄만 찍고 None (실행 결과는 그대로 둔다).
    """
    path = root / BOARD_SCRIPT
    if not path.is_file():
        return None
    try:
        return load_module(path, "experiment_board").refresh(root, exp)
    except Exception as e:  # noqa: BLE001 — 비교판 문제로 실행 결과를 버리지 않는다
        print(f"[경고] 비교판 갱신 실패: {type(e).__name__}: {e}")
        return None
