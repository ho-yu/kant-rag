"""Langfuse 점수 설정(score config) 등록: 평가 점수 이름마다 형식과 범위를 서버에 등록한다.

등록하면
- langfuse_experiment.py가 점수를 보낼 때 config_id를 붙여, 서버가 형식·범위를 검증하고 설정 기준으로 표시한다
- 사람이 Langfuse 화면에서 점수를 고칠 때(발제 "자동 채점 오류 수정")도 같은 형식을 따른다

등록하는 점수 (langfuse_experiment.py의 to_evaluations와 같은 이름·형식)
    NUMERIC 0~1   faithfulness, answer_relevancy, context_precision, context_recall   (RAGAS)
                  judge_facts, judge_conditions, judge_evidence, judge_decision       (Judge, 값은 0 또는 1)
                  hit_at_5, recall_at_5, mrr_at_5                                      (검색)
    CATEGORICAL   <위 이름>_state = na | unavailable                                   (해당 없음·평가 불가)

이미 같은 이름의 설정이 있으면 건너뛴다 (여러 번 실행해도 안전). 서버 API로는 지울 수 없고 보관(archive)만 된다.

실행 (레포 맨 위에서, .env에 LANGFUSE_* 필요)
    uv run python evaluation/register_score_configs.py --dry-run    # 무엇을 만들지 보기만
    uv run python evaluation/register_score_configs.py
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
K = 5

RAGAS = {
    "faithfulness": "RAGAS: 답변의 주장이 실제 최종 문맥에서 확인되는 비율",
    "answer_relevancy": "RAGAS: 답변이 질문에 맞는 정도 (답변으로 만든 질문과 원래 질문의 유사도)",
    "context_precision": "RAGAS: 참고 정답에 도움이 되는 문맥이 위쪽에 있는 정도",
    "context_recall": "RAGAS: 참고 정답의 주장이 문맥에서 확인되는 비율",
}
JUDGE = {
    "judge_facts": "Judge 사실: 필요한 사실과 수치를 정확히 제시 (1 충족 / 0 미충족)",
    "judge_conditions": "Judge 조건: 회사·모델·기간·단위·정정 범위가 맞음 (1 / 0)",
    "judge_evidence": "Judge 근거: 중요한 주장에 실제 문맥 근거가 있고 인용이 뒷받침 (1 / 0)",
    "judge_decision": "Judge 판단: 답변·추가 확인·보류의 선택과 이유가 맞음 (1 / 0)",
}
RETRIEVAL = {
    f"hit_at_{K}": f"검색: 상위 {K}개 문맥에서 참고 문서를 하나라도 찾으면 1",
    f"recall_at_{K}": f"검색: 찾은 참고 문서 수 / 전체 참고 문서 수 (상위 {K}개, 문서 단위)",
    f"mrr_at_{K}": f"검색: 참고 문서가 처음 나온 순위의 역수 (상위 {K}개, 없으면 0)",
}
STATE_CATEGORIES = (("na", 0), ("unavailable", 1))


def wanted_configs() -> list[dict[str, Any]]:
    """등록할 설정 목록 (이름 순서 고정)."""
    out = []
    for group in (RAGAS, JUDGE, RETRIEVAL):
        for name, desc in group.items():
            out.append({"name": name, "data_type": "NUMERIC", "min_value": 0.0, "max_value": 1.0, "description": desc})
    for group in (RAGAS, JUDGE, RETRIEVAL):
        for name in group:
            out.append(
                {
                    "name": f"{name}_state",
                    "data_type": "CATEGORICAL",
                    "categories": [{"label": label, "value": value} for label, value in STATE_CATEGORIES],
                    "description": f"{name}를 채점하지 못한 이유: na(해당 없음) / unavailable(평가 불가). comment에 사유",
                }
            )
    return out


def existing_configs(langfuse: Any) -> dict[str, Any]:
    """서버의 보관되지 않은 설정 {이름: 설정}. 여러 쪽에 걸쳐 있으면 모두 읽는다."""
    out: dict[str, Any] = {}
    page = 1
    while True:
        res = langfuse.api.score_configs.get(page=page, limit=100)
        for c in res.data:
            if not getattr(c, "is_archived", False):
                out.setdefault(c.name, c)
        total_pages = getattr(getattr(res, "meta", None), "total_pages", 1) or 1
        if page >= total_pages:
            return out
        page += 1


def config_ids(langfuse: Any) -> dict[str, str]:
    """{점수 이름: config_id}. 서버에 연결하지 못하면 빈 dict (점수는 설정 없이 그대로 보낸다)."""
    try:
        return {name: c.id for name, c in existing_configs(langfuse).items()}
    except Exception as e:  # noqa: BLE001 — 설정 조회 실패로 실험을 멈추지 않는다
        print(f"[점수 설정] 조회 실패, config_id 없이 보냄: {type(e).__name__}: {e}")
        return {}


def register(langfuse: Any, configs: Iterable[dict[str, Any]], *, dry_run: bool = False) -> dict[str, list[str]]:
    """없는 설정만 만든다. 같은 이름이 있는데 형식이 다르면 만들지 않고 알려 준다."""
    from langfuse.api import ConfigCategory

    have = existing_configs(langfuse)
    result: dict[str, list[str]] = {"created": [], "skipped": [], "conflict": []}
    for cfg in configs:
        old = have.get(cfg["name"])
        if old is not None:
            same = str(getattr(old, "data_type", "")).upper().endswith(cfg["data_type"])
            result["skipped" if same else "conflict"].append(cfg["name"])
            continue
        result["created"].append(cfg["name"])
        if dry_run:
            continue
        kwargs = {k: v for k, v in cfg.items() if k != "categories"}
        if "categories" in cfg:
            kwargs["categories"] = [ConfigCategory(**c) for c in cfg["categories"]]
        langfuse.api.score_configs.create(**kwargs)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    p = argparse.ArgumentParser(description="Langfuse 점수 설정 등록 (평가 점수 이름별 형식·범위)")
    p.add_argument("--dry-run", action="store_true", help="만들 설정만 보여 주고 서버는 바꾸지 않는다")
    args = p.parse_args(argv)

    from langfuse import get_client

    langfuse = get_client()
    if not langfuse.auth_check():
        raise SystemExit("Langfuse 연결 설정(.env의 LANGFUSE_*)을 확인해 주세요.")
    result = register(langfuse, wanted_configs(), dry_run=args.dry_run)
    verb = "만들 예정" if args.dry_run else "만듦"
    print(f"[점수 설정] {verb} {len(result['created'])}개: {', '.join(result['created']) or '-'}")
    print(f"[점수 설정] 이미 있음 {len(result['skipped'])}개")
    if result["conflict"]:
        print(f"[점수 설정] 같은 이름인데 형식이 다름 (그대로 둠): {', '.join(result['conflict'])}")


if __name__ == "__main__":
    main()
