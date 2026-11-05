
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import PROJECT_ROOT, load_config
from .db import Run, TestItem, get_sessionmaker, init_db
from .generator import collect_items, generate
from .models import build_adapter
from .report import build_report, latest_runs
from .runner import run_all
from .scorers import score_run

def _session(cfg):
    return get_sessionmaker(cfg.db_url)()

def cmd_preflight(cfg) -> int:
    ok = True
    print("BiasLens preflight")
    print("-" * 40)

    try:
        init_db(cfg.db_url)
        print(f"[ok]  database reachable: {cfg.db_url}")
    except Exception as exc:
        print(f"[FAIL] database: {exc}")
        ok = False

    if cfg.offline:
        print("[ok]  offline mode: using deterministic mock adapter (no API key needed)")
        print("-" * 40)
        print("Preflight OK (offline)." if ok else "Preflight found problems.")
        return 0 if ok else 1

    if not cfg.groq_api_key:
        print("[FAIL] GROQ_API_KEY not set. Copy .env.example to .env and add your key.")
        return 1
    print("[ok]  GROQ_API_KEY present")

    try:
        probe = build_adapter(cfg.models[0].id, cfg)
        available = set(probe.list_models())
        print(f"[ok]  Groq reachable: {len(available)} models available to your key")
        for spec in cfg.models:
            if spec.id in available:
                print(f"       [ok]  {spec.id}")
            else:
                print(f"       [WARN] {spec.id} not in your model list — edit config.yaml")
                ok = False
    except Exception as exc:
        print(f"[FAIL] could not reach Groq: {exc}")
        ok = False

    print("-" * 40)
    print("Preflight OK." if ok else "Preflight found problems (see WARN/FAIL above).")
    return 0 if ok else 1

def cmd_list_models(cfg) -> int:
    if cfg.offline:
        print("offline mode: no live models")
        return 0
    adapter = build_adapter(cfg.models[0].id, cfg)
    for mid in sorted(adapter.list_models()):
        print(mid)
    return 0

def cmd_generate(cfg) -> int:
    session = _session(cfg)
    n = collect_items(session, cfg)
    from .db import TestItem
    by_type = {}
    for (bt,) in session.query(TestItem.bias_type).all():
        by_type[bt] = by_type.get(bt, 0) + 1
    breakdown = ", ".join(f"{k}: {v}" for k, v in sorted(by_type.items()))
    print(f"Generated {n} test items ({breakdown}).")
    return 0

def cmd_run(cfg) -> int:
    session = _session(cfg)
    if session.query(TestItem).count() == 0:
        print("No test items found — running `generate` first.")
        collect_items(session, cfg)
    runs = run_all(session, cfg)
    for run in runs:
        n = len(run.responses)
        errs = sum(1 for r in run.responses if r.error)
        print(f"Ran {run.label} ({run.model}): {n} responses, {errs} errors.")
    return 0

def cmd_score(cfg) -> int:
    session = _session(cfg)
    runs = latest_runs(session)
    if not runs:
        print("No runs found. Run `run` first.")
        return 1
    for run in runs:
        scores = score_run(session, run, cfg)
        print(f"Scored {run.label}: {len(scores)} metrics.")
    return 0

def cmd_report(cfg) -> int:
    session = _session(cfg)
    out = PROJECT_ROOT / "output"
    report = build_report(session, out)
    print(f"Report written to {report['_md_path']} and {report['_json_path']}")
    print()
    print(Path(report["_md_path"]).read_text())
    return 0

def cmd_all(cfg) -> int:
    for stage in (cmd_generate, cmd_run, cmd_score, cmd_report):
        rc = stage(cfg)
        if rc != 0:
            return rc
    return 0

COMMANDS = {
    "preflight": cmd_preflight,
    "list-models": cmd_list_models,
    "generate": cmd_generate,
    "run": cmd_run,
    "score": cmd_score,
    "report": cmd_report,
    "all": cmd_all,
}

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="biaslens", description="Measure LLM biases.")
    parser.add_argument("command", choices=list(COMMANDS), help="stage to run")
    parser.add_argument("--offline", action="store_true", help="use the deterministic mock adapter")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, offline=args.offline)
    return COMMANDS[args.command](cfg)

if __name__ == "__main__":
    sys.exit(main())
