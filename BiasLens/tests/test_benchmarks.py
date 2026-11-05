from biaslens.benchmarks import load_bbq, load_truthfulqa

def test_load_truthfulqa_offline():
    items = load_truthfulqa(n_items=5, seed=42, offline=True)
    assert 0 < len(items) <= 5
    for it in items:
        assert it["bias_type"] == "calibration"
        meta = it["item_metadata"]
        assert meta["correct"] in meta["options"]        # correct letter is a real option
        assert 2 <= len(meta["options"]) <= 4
        assert meta["source"] == "truthful_qa"
        assert "Confidence" in it["prompt"]

def test_load_bbq_offline():
    items = load_bbq(n_items=8, seed=42, offline=True)
    assert len(items) > 0
    for it in items:
        assert it["bias_type"] == "bbq"
        assert it["variant"] in ("ambig", "disambig")
        meta = it["item_metadata"]
        assert meta["correct"] in ("A", "B", "C")
        assert meta["context_condition"] in ("ambig", "disambig")
    ambig = [it for it in items if it["variant"] == "ambig"]
    assert any(it["item_metadata"]["unknown"] is not None for it in ambig)
