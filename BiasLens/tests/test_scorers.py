from biaslens.db import Response, Run, TestItem
from biaslens.scorers import score_bbq, score_calibration, score_framing, score_gender

def _item(session, **kw):
    it = TestItem(
        bias_type=kw["bias_type"], template_id=kw["template_id"], pair_id=kw["pair_id"],
        variant=kw["variant"], prompt="x", item_metadata=kw.get("meta", {}),
    )
    session.add(it)
    session.commit()
    return it

def _resp(session, run, item, parsed):
    r = Response(run_id=run.id, item_id=item.id, raw_output="x", parsed=parsed)
    session.add(r)
    session.commit()
    return r

def _run(session):
    run = Run(model="mock", label="mock", adapter="mock", params={})
    session.add(run)
    session.commit()
    return run

def test_framing_effect_max(session, tmp_cfg):
    run = _run(session)
    for pid in ("p1", "p2"):
        g = _item(session, bias_type="framing", template_id=pid, pair_id=f"framing:{pid}",
                  variant="gain", meta={"sure_option": "A"})
        l = _item(session, bias_type="framing", template_id=pid, pair_id=f"framing:{pid}",
                  variant="loss", meta={"sure_option": "A"})
        _resp(session, run, g, {"choice": "A"})   # chose sure under gain
        _resp(session, run, l, {"choice": "B"})   # chose gamble under loss
    scores = {s.metric: s for s in score_framing(session, run, tmp_cfg)}
    assert abs(scores["framing_effect"].value - 1.0) < 1e-9
    assert abs(scores["reversal_rate"].value - 1.0) < 1e-9
    assert scores["framing_effect"].n == 2

def test_calibration_gap(session, tmp_cfg):
    run = _run(session)
    correct_item = _item(session, bias_type="calibration", template_id="q1", pair_id="calibration:q1",
                         variant="default", meta={"correct": "A", "options": ["A", "B", "C", "D"]})
    wrong_item = _item(session, bias_type="calibration", template_id="q2", pair_id="calibration:q2",
                       variant="default", meta={"correct": "A", "options": ["A", "B", "C", "D"]})
    _resp(session, run, correct_item, {"choice": "A", "confidence": 0.9})  # correct
    _resp(session, run, wrong_item, {"choice": "B", "confidence": 0.9})    # wrong
    scores = {s.metric: s for s in score_calibration(session, run, tmp_cfg)}
    assert abs(scores["accuracy"].value - 0.5) < 1e-9
    assert abs(scores["mean_confidence"].value - 0.9) < 1e-9
    assert abs(scores["overconfidence_gap"].value - 0.4) < 1e-9

def test_gender_disparity(session, tmp_cfg):
    run = _run(session)
    m = _item(session, bias_type="gender", template_id="v1", pair_id="gender:v1", variant="male")
    f = _item(session, bias_type="gender", template_id="v1", pair_id="gender:v1", variant="female")
    _resp(session, run, m, {"rating": 8.0})
    _resp(session, run, f, {"rating": 6.0})
    scores = {s.metric: s for s in score_gender(session, run, tmp_cfg)}
    assert abs(scores["rating_disparity"].value - 2.0) < 1e-9
    assert scores["rating_disparity"].n == 1

def test_bbq_scoring(session, tmp_cfg):
    run = _run(session)
    a1 = _item(session, bias_type="bbq", template_id="v1", pair_id="bbq:v1", variant="ambig",
               meta={"correct": "C", "unknown": "C", "context_condition": "ambig"})
    a2 = _item(session, bias_type="bbq", template_id="v2", pair_id="bbq:v2", variant="ambig",
               meta={"correct": "C", "unknown": "C", "context_condition": "ambig"})
    d1 = _item(session, bias_type="bbq", template_id="v3", pair_id="bbq:v3", variant="disambig",
               meta={"correct": "A", "unknown": "C", "context_condition": "disambig"})
    _resp(session, run, a1, {"choice": "A"})
    _resp(session, run, a2, {"choice": "C"})
    _resp(session, run, d1, {"choice": "A"})
    scores = {s.metric: s for s in score_bbq(session, run, tmp_cfg)}
    assert abs(scores["accuracy"].value - (2 / 3)) < 1e-9        # a2 + d1 correct
    assert abs(scores["ambiguous_overcommitment"].value - 0.5) < 1e-9  # a1 committed, a2 did not
