from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import func

from .db import Response, Run, Score

def latest_runs(session) -> list[Run]:
    """Most recent run per model."""
    sub = (
        session.query(Run.model, func.max(Run.id).label("max_id"))
        .group_by(Run.model)
        .subquery()
    )
    return session.query(Run).join(sub, Run.id == sub.c.max_id).all()

PRIMARY = {
    "framing": ("framing_effect", "Framing effect", 0),
    "calibration": ("overconfidence_gap", "Overconfidence gap", 1),
    "bbq": ("ambiguous_overcommitment", "BBQ over-commitment", 2),
    "gender": ("rating_disparity", "Gender disparity", 3),
}

def _fmt(value: float, ci_low, ci_high) -> str:
    base = f"{value:+.3f}"
    if ci_low is None or ci_high is None:
        return base
    return f"{base} [{ci_low:+.3f}, {ci_high:+.3f}]"

def build_report(session, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = latest_runs(session)

    report: dict = {"generated_at": datetime.now(timezone.utc).isoformat(), "models": {}}
    for run in runs:
        scores = session.query(Score).filter(Score.run_id == run.id).all()
        n_resp = session.query(func.count(Response.id)).filter(Response.run_id == run.id).scalar()
        n_err = (
            session.query(func.count(Response.id))
            .filter(Response.run_id == run.id, Response.error.isnot(None))
            .scalar()
        )
        report["models"][run.label] = {
            "model_id": run.model,
            "adapter": run.adapter,
            "responses": int(n_resp or 0),
            "errors": int(n_err or 0),
            "scores": [
                {"bias_type": s.bias_type, "metric": s.metric, "value": s.value,
                 "ci_low": s.ci_low, "ci_high": s.ci_high, "n": s.n}
                for s in scores
            ],
        }

    present = sorted({s["bias_type"] for m in report["models"].values() for s in m["scores"]
                      if s["bias_type"] in PRIMARY},
                     key=lambda b: PRIMARY[b][2])

    lines = ["# BiasLens — Bias Report", "", f"_Generated {report['generated_at']}_", ""]
    lines.append("## Headline bias metrics\n")
    headers = ["Model"] + [PRIMARY[b][1] for b in present]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for label, m in report["models"].items():
        primary_vals = {b: "—" for b in present}
        for s in m["scores"]:
            metric = PRIMARY.get(s["bias_type"], (None,))[0]
            if s["metric"] == metric:
                primary_vals[s["bias_type"]] = _fmt(s["value"], s["ci_low"], s["ci_high"])
        lines.append("| " + " | ".join([label] + [primary_vals[b] for b in present]) + " |")
    lines.append("")
    lines.append("All headline metrics are 0 when unbiased. CI = 95% bootstrap interval; a value whose "
                 "interval excludes 0 is a statistically detectable bias for that model.\n")

    lines.append("## Full metrics by model\n")
    for label, m in report["models"].items():
        lines.append(f"### {label}  `{m['model_id']}`")
        lines.append(f"- responses: {m['responses']}  |  errors: {m['errors']}  |  adapter: {m['adapter']}")
        for s in m["scores"]:
            ci = "" if s["ci_low"] is None else f"  [{s['ci_low']:+.3f}, {s['ci_high']:+.3f}]"
            lines.append(f"  - {s['bias_type']}.{s['metric']}: {s['value']:+.4f}{ci}  (n={s['n']})")
        lines.append("")

    md_path = out_dir / "report.md"
    json_path = out_dir / "report.json"
    md_path.write_text("\n".join(lines))
    json_path.write_text(json.dumps(report, indent=2))
    report["_md_path"] = str(md_path)
    report["_json_path"] = str(json_path)
    return report
