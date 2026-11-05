from pathlib import Path

from biaslens.db import Response, Score, get_sessionmaker
from biaslens.generator import collect_items
from biaslens.report import build_report, latest_runs
from biaslens.runner import run_all
from biaslens.scorers import score_run

def test_pipeline_offline(tmp_cfg, tmp_path):
    Session = get_sessionmaker(tmp_cfg.db_url)
    session = Session()

    n_items = collect_items(session, tmp_cfg)
    assert n_items > 0

    runs = run_all(session, tmp_cfg)
    assert len(runs) == len(tmp_cfg.models)

    total_expected = n_items * len(tmp_cfg.models) * tmp_cfg.n_reps
    assert session.query(Response).count() == total_expected
    assert session.query(Response).filter(Response.error.isnot(None)).count() == 0

    for run in latest_runs(session):
        score_run(session, run, tmp_cfg)

    bias_types = {s.bias_type for s in session.query(Score).all()}
    assert {"framing", "calibration", "bbq"}.issubset(bias_types)

    metrics = {s.metric for s in session.query(Score).all()}
    assert {"framing_effect", "overconfidence_gap", "ambiguous_overcommitment"}.issubset(metrics)

    report = build_report(session, tmp_path / "out")
    assert Path(report["_md_path"]).exists()
    assert Path(report["_json_path"]).exists()
    assert report["models"]
