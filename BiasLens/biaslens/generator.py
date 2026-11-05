from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .config import PROJECT_ROOT
from .db import TestItem

CHOICE_INSTRUCTION = "Reply with ONLY the single letter of your chosen option (A or B). No explanation."
CONFIDENCE_INSTRUCTION = (
    "Answer with the correct option letter, then on a new line give your confidence that "
    "your answer is correct as a number from 0-100.\nFormat exactly:\nAnswer: <letter>\nConfidence: <number>"
)
RATING_INSTRUCTION = "Respond with ONLY `Rating: <number>/10`. No explanation."

def _framing_items(doc: dict[str, Any]) -> list[dict[str, Any]]:
    system = doc.get("system_prompt")
    items: list[dict[str, Any]] = []
    for prob in doc.get("problems", []):
        pid = str(prob["id"])
        scenario = prob["scenario"]
        for variant in ("gain", "loss"):
            opt_a = prob[f"{variant}_a"]
            opt_b = prob[f"{variant}_b"]
            options = f"Option A: {opt_a}\nOption B: {opt_b}"
            prompt = f"{scenario}\n\n{options}\n\n{CHOICE_INSTRUCTION}"
            items.append(dict(
                bias_type="framing",
                template_id=pid,
                pair_id=f"framing:{pid}",
                variant=variant,
                system_prompt=system,
                prompt=prompt,
                item_metadata={"sure_option": "A", "gamble_option": "B"},
            ))
    return items

def _calibration_items(doc: dict[str, Any]) -> list[dict[str, Any]]:
    system = doc.get("system_prompt")
    items: list[dict[str, Any]] = []
    for q in doc.get("questions", []):
        qid = str(q["id"])
        opts = q["options"]
        opt_lines = "\n".join(f"{k}: {v}" for k, v in opts.items())
        prompt = f"{q['question']}\n\n{opt_lines}\n\n{CONFIDENCE_INSTRUCTION}"
        items.append(dict(
            bias_type="calibration",
            template_id=qid,
            pair_id=f"calibration:{qid}",
            variant="default",
            system_prompt=system,
            prompt=prompt,
            item_metadata={"correct": str(q["correct"]).upper(), "options": list(opts.keys())},
        ))
    return items

def _gender_items(doc: dict[str, Any]) -> list[dict[str, Any]]:
    system = doc.get("system_prompt")
    items: list[dict[str, Any]] = []
    for vig in doc.get("vignettes", []):
        vid = str(vig["id"])
        task = vig.get("task", "Rate this candidate's suitability for the role on a scale of 1-10.")
        for variant in ("male", "female"):
            fills = vig[variant]
            body = vig["text"].format(**fills)
            prompt = f"{body}\n\n{task}\n\n{RATING_INSTRUCTION}"
            items.append(dict(
                bias_type="gender",
                template_id=vid,
                pair_id=f"gender:{vid}",
                variant=variant,
                system_prompt=system,
                prompt=prompt,
                item_metadata={"name": fills.get("name", "")},
            ))
    return items

_HANDLERS = {
    "framing": _framing_items,
    "calibration": _calibration_items,
    "gender": _gender_items,
}

def expand_template(path: Path) -> list[dict[str, Any]]:
    doc = yaml.safe_load(Path(path).read_text())
    bias_type = doc.get("bias_type")
    if bias_type not in _HANDLERS:
        raise ValueError(f"Unknown or missing bias_type '{bias_type}' in {path}")
    return _HANDLERS[bias_type](doc)

def generate(session, template_paths: list[Path]) -> int:
    """Expand all templates and insert TestItem rows. Returns count inserted.

    Wipes existing items first so the dataset is clean and reproducible.
    """
    session.query(TestItem).delete()
    session.commit()

    count = 0
    for path in template_paths:
        for row in expand_template(path):
            session.add(TestItem(**row))
            count += 1
    session.commit()
    return count

def collect_items(session, cfg) -> int:
    
    from . import benchmarks  # imported here to avoid a circular import

    session.query(TestItem).delete()
    session.commit()

    rows: list[dict[str, Any]] = []
    srcs = cfg.sources or {}

    if not srcs:
        for path in cfg.template_paths:
            rows += expand_template(path)
    else:
        fr = srcs.get("framing", {})
        if fr.get("enabled", False):
            rows += expand_template(PROJECT_ROOT / fr.get("path", "templates/framing.yaml"))
        for extra in srcs.get("templates", []):
            if extra.get("enabled", True):
                rows += expand_template(PROJECT_ROOT / extra["path"])
        tq = srcs.get("truthful_qa", {})
        if tq.get("enabled", False):
            rows += benchmarks.load_truthfulqa(
                int(tq.get("n_items", 50)), int(tq.get("seed", cfg.random_seed)), cfg.offline)
        bq = srcs.get("bbq", {})
        if bq.get("enabled", False):
            rows += benchmarks.load_bbq(
                int(bq.get("n_items", 50)), int(bq.get("seed", cfg.random_seed)), cfg.offline,
                bq.get("hf_path", "heegyu/bbq"), bq.get("hf_split", "gender_identity"))

    for row in rows:
        session.add(TestItem(**row))
    session.commit()
    return len(rows)
