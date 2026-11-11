"""evaluation/register_score_configs.py 테스트 (Langfuse 점수 설정 등록과 점수 연결)."""

from __future__ import annotations

import importlib.util
import re
from types import SimpleNamespace

from rag_service.storage.corpus import repo_root


def _load(rel: str, name: str):
    path = repo_root() / "evaluation" / rel
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rsc = _load("register_score_configs.py", "register_score_configs")
lx = _load("langfuse_experiment.py", "langfuse_experiment")
rs = lx.retrieval_scores
quality = lx.quality


class FakeConfigs:
    def __init__(self, existing=()):
        self.items = [SimpleNamespace(id=f"id-{n}", name=n, data_type=t, is_archived=a) for n, t, a in existing]
        self.created = []

    def get(self, page=1, limit=100):
        return SimpleNamespace(data=self.items, meta=SimpleNamespace(total_pages=1))

    def create(self, **kw):
        self.created.append(kw)


def fake_langfuse(existing=()):
    return SimpleNamespace(api=SimpleNamespace(score_configs=FakeConfigs(existing)))


def test_wanted_configs_cover_every_uploaded_score_name():
    names = [c["name"] for c in rsc.wanted_configs()]
    metrics = [*quality.ALL_NAMES, *(rs.metric_name(m, rs.K) for m in rs.METRICS)]
    assert names == [*metrics, *(f"{m}_state" for m in metrics)]
    assert all(len(n) <= 35 and re.fullmatch(r"[A-Za-z0-9_ .()-]+", n) for n in names)


def test_numeric_range_and_state_categories():
    by_name = {c["name"]: c for c in rsc.wanted_configs()}
    assert by_name["judge_facts"]["data_type"] == "NUMERIC"  # 코드가 0/1 숫자로 보낸다
    assert (by_name["hit_at_5"]["min_value"], by_name["hit_at_5"]["max_value"]) == (0.0, 1.0)
    labels = [c["label"] for c in by_name["faithfulness_state"]["categories"]]
    assert labels == ["na", "unavailable"]


def test_register_creates_only_missing_and_reports_conflict():
    lf = fake_langfuse([("faithfulness", "NUMERIC", False), ("hit_at_5", "CATEGORICAL", False), ("mrr_at_5", "NUMERIC", True)])
    result = rsc.register(lf, rsc.wanted_configs())
    assert result["skipped"] == ["faithfulness"]
    assert result["conflict"] == ["hit_at_5"]
    assert "mrr_at_5" in result["created"]  # 보관된 설정은 없는 것으로 본다
    created = {c["name"]: c for c in lf.api.score_configs.created}
    assert len(created) == 20 and "faithfulness" not in created
    assert created["recall_at_5_state"]["categories"][0].label == "na"


def test_dry_run_creates_nothing():
    lf = fake_langfuse()
    result = rsc.register(lf, rsc.wanted_configs(), dry_run=True)
    assert len(result["created"]) == 22 and lf.api.score_configs.created == []


def test_config_ids_and_failure_falls_back_to_empty():
    lf = fake_langfuse([("hit_at_5", "NUMERIC", False)])
    assert rsc.config_ids(lf) == {"hit_at_5": "id-hit_at_5"}
    assert rsc.config_ids(SimpleNamespace()) == {}


def test_to_evaluations_attaches_config_ids():
    scores = [quality.score("hit_at_5", 1.0, "scored"), quality.score("recall_at_5", None, "na", "no_evidence", "x")]
    ids = {"hit_at_5": "c1", "recall_at_5_state": "c2"}
    ev = lx.to_evaluations(scores, status="answered", config_ids=ids)
    assert [(e.name, e.config_id) for e in ev] == [("hit_at_5", "c1"), ("recall_at_5_state", "c2")]
    assert all(e.config_id is None for e in lx.to_evaluations(scores, status="answered"))
