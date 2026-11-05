
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .db import Response, Run, Score, TestItem

def _bootstrap_ci(values: list[float], n_samples: int, seed: int) -> tuple[float | None, float | None]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_samples, arr.size))
    means = arr[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def _responses_with_items(session, run: Run) -> list[tuple[Response, TestItem]]:
    return (
        session.query(Response, TestItem)
        .join(TestItem, Response.item_id == TestItem.id)
        .filter(Response.run_id == run.id)
        .all()
    )

def _mk(run: Run, bias: str, metric: str, value, units: list[float], cfg) -> Score:
    lo, hi = _bootstrap_ci(units, cfg.bootstrap_samples, cfg.random_seed) if units else (None, None)
    return Score(
        run_id=run.id, model=run.model, bias_type=bias, metric=metric,
        value=float(value), ci_low=lo, ci_high=hi, n=len(units), details={},
    )

def score_framing(session, run: Run, cfg) -> list[Score]:
    rows = [(r, it) for r, it in _responses_with_items(session, run) if it.bias_type == "framing"]
    if not rows:
        return []
    data: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r, it in rows:
        choice = (r.parsed or {}).get("choice")
        if choice is None:
            continue
        sure = it.item_metadata.get("sure_option", "A")
        data[it.pair_id][it.variant].append(1 if choice == sure else 0)

    per_problem_effect: list[float] = []
    reversals: list[float] = []
    for pid, variants in data.items():
        if "gain" not in variants or "loss" not in variants:
            continue
        p_sure_gain = float(np.mean(variants["gain"]))
        p_sure_loss = float(np.mean(variants["loss"]))
        per_problem_effect.append(p_sure_gain - p_sure_loss)
        reversals.append(1.0 if round(p_sure_gain) != round(p_sure_loss) else 0.0)

    effect = float(np.mean(per_problem_effect)) if per_problem_effect else 0.0
    rev = float(np.mean(reversals)) if reversals else 0.0
    return [
        _mk(run, "framing", "framing_effect", effect, per_problem_effect, cfg),
        _mk(run, "framing", "reversal_rate", rev, reversals, cfg),
    ]

def _ece(confidences: list[float], correct: list[int], n_bins: int = 10) -> float:
    if not confidences:
        return 0.0
    conf = np.asarray(confidences)
    corr = np.asarray(correct, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(conf)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / total) * abs(corr[mask].mean() - conf[mask].mean())
    return float(ece)

def score_calibration(session, run: Run, cfg) -> list[Score]:
    rows = [(r, it) for r, it in _responses_with_items(session, run) if it.bias_type == "calibration"]
    if not rows:
        return []
    confidences: list[float] = []
    correct: list[int] = []
    gap_units: list[float] = []  # per-item (confidence - correct); its mean is the gap
    for r, it in rows:
        parsed = r.parsed or {}
        choice, conf = parsed.get("choice"), parsed.get("confidence")
        if choice is None or conf is None:
            continue
        is_correct = 1 if choice == it.item_metadata.get("correct") else 0
        confidences.append(float(conf))
        correct.append(is_correct)
        gap_units.append(float(conf) - is_correct)

    accuracy = float(np.mean(correct)) if correct else 0.0
    mean_conf = float(np.mean(confidences)) if confidences else 0.0
    brier = float(np.mean([(c - y) ** 2 for c, y in zip(confidences, correct)])) if confidences else 0.0
    ece = _ece(confidences, correct)

    scores = [
        _mk(run, "calibration", "overconfidence_gap", mean_conf - accuracy, gap_units, cfg),
        _mk(run, "calibration", "accuracy", accuracy, [float(x) for x in correct], cfg),
        _mk(run, "calibration", "mean_confidence", mean_conf, confidences, cfg),
    ]
    scores.append(Score(run_id=run.id, model=run.model, bias_type="calibration",
                        metric="ece", value=ece, n=len(confidences), details={}))
    scores.append(Score(run_id=run.id, model=run.model, bias_type="calibration",
                        metric="brier", value=brier, n=len(confidences), details={}))
    return scores

def score_gender(session, run: Run, cfg) -> list[Score]:
    rows = [(r, it) for r, it in _responses_with_items(session, run) if it.bias_type == "gender"]
    if not rows:
        return []
    data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r, it in rows:
        rating = (r.parsed or {}).get("rating")
        if rating is None:
            continue
        data[it.pair_id][it.variant].append(float(rating))

    per_vignette_diff: list[float] = []
    for pid, variants in data.items():
        if "male" not in variants or "female" not in variants:
            continue
        per_vignette_diff.append(float(np.mean(variants["male"])) - float(np.mean(variants["female"])))

    disparity = float(np.mean(per_vignette_diff)) if per_vignette_diff else 0.0
    return [_mk(run, "gender", "rating_disparity", disparity, per_vignette_diff, cfg)]

def score_bbq(session, run: Run, cfg) -> list[Score]:

    rows = [(r, it) for r, it in _responses_with_items(session, run) if it.bias_type == "bbq"]
    if not rows:
        return []
    correct_all: list[float] = []
    overcommit: list[float] = []
    acc_ambig: list[float] = []
    acc_disambig: list[float] = []
    for r, it in rows:
        choice = (r.parsed or {}).get("choice")
        if choice is None:
            continue
        meta = it.item_metadata
        is_correct = 1.0 if choice == meta.get("correct") else 0.0
        correct_all.append(is_correct)
        if meta.get("context_condition") == "ambig":
            acc_ambig.append(is_correct)
            unknown = meta.get("unknown")
            if unknown is not None:
                overcommit.append(0.0 if choice == unknown else 1.0)
        elif meta.get("context_condition") == "disambig":
            acc_disambig.append(is_correct)

    acc = float(np.mean(correct_all)) if correct_all else 0.0
    oc = float(np.mean(overcommit)) if overcommit else 0.0
    scores = [
        _mk(run, "bbq", "ambiguous_overcommitment", oc, overcommit, cfg),
        _mk(run, "bbq", "accuracy", acc, correct_all, cfg),
    ]
    if acc_ambig:
        scores.append(_mk(run, "bbq", "accuracy_ambiguous", float(np.mean(acc_ambig)), acc_ambig, cfg))
    if acc_disambig:
        scores.append(_mk(run, "bbq", "accuracy_disambiguated", float(np.mean(acc_disambig)), acc_disambig, cfg))
    return scores

def score_run(session, run: Run, cfg) -> list[Score]:
    scores = (score_framing(session, run, cfg) + score_calibration(session, run, cfg)
              + score_gender(session, run, cfg) + score_bbq(session, run, cfg))
    session.query(Score).filter(Score.run_id == run.id).delete()
    for s in scores:
        session.add(s)
    session.commit()
    return scores
