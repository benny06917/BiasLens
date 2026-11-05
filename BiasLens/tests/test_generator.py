from biaslens.config import PROJECT_ROOT
from biaslens.db import TestItem
from biaslens.generator import collect_items, expand_template, generate

TEMPLATES = PROJECT_ROOT / "templates"

def test_expand_framing():
    items = expand_template(TEMPLATES / "framing.yaml")
    assert len(items) % 2 == 0
    assert {it["variant"] for it in items} == {"gain", "loss"}
    by_pair = {}
    for it in items:
        by_pair.setdefault(it["pair_id"], {})[it["variant"]] = it["prompt"]
    for pair, pv in by_pair.items():
        assert pv["gain"] != pv["loss"], f"{pair} frames identical"

def test_expand_calibration():
    items = expand_template(TEMPLATES / "calibration.yaml")
    assert all(it["bias_type"] == "calibration" for it in items)
    assert all(it["item_metadata"]["correct"] in ("A", "B", "C", "D") for it in items)

def test_expand_gender_pairs():
    items = expand_template(TEMPLATES / "gender.yaml")
    by_pair = {}
    for it in items:
        by_pair.setdefault(it["pair_id"], set()).add(it["variant"])
    assert all(v == {"male", "female"} for v in by_pair.values())

def test_generate_persists(session):
    n = generate(session, [TEMPLATES / "framing.yaml"])
    assert n > 0
    assert session.query(TestItem).count() == n
    n2 = generate(session, [TEMPLATES / "framing.yaml"])
    assert n2 == n

def test_collect_items_from_sources(session, tmp_cfg):
    n = collect_items(session, tmp_cfg)
    assert n > 0
    types = {bt for (bt,) in session.query(TestItem.bias_type).all()}
    assert "framing" in types
    assert "calibration" in types   # TruthfulQA
    assert "bbq" in types           # BBQ
