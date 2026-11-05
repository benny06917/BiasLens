from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import Config
from .db import Response, Run, TestItem
from .models import BaseAdapter, build_adapter
from .parsing import parse_choice, parse_confidence, parse_mcq, parse_rating

def _parse_for(item: TestItem, text: str) -> dict:
    if item.bias_type == "framing":
        return {"choice": parse_choice(text, ("A", "B"))}
    if item.bias_type == "calibration":
        letters = tuple(item.item_metadata.get("options", ["A", "B", "C", "D"]))
        return {"choice": parse_mcq(text, letters), "confidence": parse_confidence(text)}
    if item.bias_type == "gender":
        return {"rating": parse_rating(text, 1.0, 10.0)}
    if item.bias_type == "bbq":
        return {"choice": parse_choice(text, ("A", "B", "C"))}
    return {}

def _run_one(adapter: BaseAdapter, item: TestItem, rep: int) -> dict:
    comp = adapter.complete(item.prompt, system=item.system_prompt)
    parsed = _parse_for(item, comp.text) if not comp.error else {}
    return {
        "item_id": item.id,
        "rep": rep,
        "raw_output": comp.text,
        "parsed": parsed,
        "prompt_tokens": comp.prompt_tokens,
        "completion_tokens": comp.completion_tokens,
        "latency_s": comp.latency_s,
        "error": comp.error,
    }

def run_model(session, cfg: Config, model_id: str, label: str, progress=None) -> Run:

    adapter = build_adapter(model_id, cfg)
    run = Run(
        model=model_id,
        label=label,
        adapter=adapter.name,
        params={
            "temperature": cfg.temperature,
            "n_reps": cfg.n_reps,
            "max_tokens": cfg.max_tokens,
        },
    )
    session.add(run)
    session.commit()

    items = session.query(TestItem).all()
    tasks = [(item, rep) for item in items for rep in range(cfg.n_reps)]
    total = len(tasks)

    if progress is None:
        print(f"  [{label}] running {total} calls (concurrency={cfg.concurrency})...", flush=True)
    results: list[dict] = []
    done = 0
    errors = 0
    with ThreadPoolExecutor(max_workers=max(1, cfg.concurrency)) as pool:
        futures = {pool.submit(_run_one, adapter, item, rep): (item, rep) for item, rep in tasks}
        for fut in as_completed(futures):
            item, rep = futures[fut]
            try:
                r = fut.result()
            except Exception as exc:  # never let one item abort the whole run
                r = {
                    "item_id": item.id, "rep": rep, "raw_output": None, "parsed": {},
                    "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            if r.get("error"):
                errors += 1
            results.append(r)
            done += 1
            if progress is not None:
                progress(label, done, total, errors)
            elif done % 10 == 0 or done == total:
                print(f"  [{label}] {done}/{total} done ({errors} errors)", flush=True)

    for r in results:
        session.add(Response(run_id=run.id, **r))
    session.commit()
    return run

def run_all(session, cfg: Config, progress=None) -> list[Run]:
    runs = []
    for spec in cfg.models:
        runs.append(run_model(session, cfg, spec.id, spec.label, progress=progress))
    return runs