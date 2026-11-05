
from __future__ import annotations

import pandas as pd
import streamlit as st
from sqlalchemy import func

from biaslens.config import PROJECT_ROOT, ModelSpec, load_config
from biaslens.db import Response, Score, TestItem, get_sessionmaker
from biaslens.generator import collect_items
from biaslens.report import build_report, latest_runs
from biaslens.runner import run_model
from biaslens.scorers import score_run

st.set_page_config(page_title="BiasLens", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;800&display=swap');
html, body, [class*="css"] { font-family:'Inter',sans-serif; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility:hidden; }
.block-container { padding-top:2.2rem; max-width:1250px; }
.hero-title { font-size:2.7rem; font-weight:800; line-height:1.05;
  background:linear-gradient(90deg,#a78bfa,#22d3ee); -webkit-background-clip:text;
  -webkit-text-fill-color:transparent; margin:0; }
.hero-sub { color:#94a3b8; font-size:1.05rem; margin:.3rem 0 1.4rem; }
.card { background:linear-gradient(180deg,#161d31,#121829); border:1px solid #233052;
  border-radius:16px; padding:1.1rem 1.2rem; height:100%; }
.card-title { font-weight:700; font-size:1.02rem; color:#f1f5f9; }
.card-sub { color:#7f8aa3; font-size:.78rem; margin:.1rem 0 .7rem; line-height:1.3; }
.row { display:flex; justify-content:space-between; align-items:center;
  padding:.55rem 0; border-top:1px solid #1e2740; }
.row-model { color:#cbd5e1; font-weight:600; font-size:.9rem; }
.row-val { color:#e6e9f2; font-variant-numeric:tabular-nums; font-size:.92rem; font-weight:600; }
.row-ci { color:#64748b; font-size:.72rem; font-variant-numeric:tabular-nums; }
.badge { font-size:.66rem; font-weight:700; padding:.16rem .55rem; border-radius:999px;
  margin-left:.5rem; white-space:nowrap; }
.badge-red { background:rgba(244,63,94,.16); color:#fb7185; }
.badge-green { background:rgba(16,185,129,.16); color:#34d399; }
.badge-grey { background:rgba(148,163,184,.14); color:#94a3b8; }
.section { font-size:1.15rem; font-weight:700; color:#f1f5f9; margin:1.4rem 0 .5rem; }
.prompt-box { background:#0f1526; border:1px solid #233052; border-radius:12px; padding:.85rem 1rem;
  font-size:.82rem; color:#cbd5e1; white-space:pre-wrap; max-height:260px; overflow:auto; line-height:1.4; }
.frame-tag { font-size:.72rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase;
  color:#a78bfa; margin-bottom:.4rem; }
.answer-chip { display:inline-block; background:rgba(34,211,238,.14); color:#67e8f9;
  border-radius:8px; padding:.25rem .6rem; font-weight:700; font-size:.85rem; margin-top:.5rem; }
.model-pill { background:#0f1526; border:1px solid #233052; border-radius:8px; padding:.35rem .6rem;
  color:#cbd5e1; font-size:.82rem; font-variant-numeric:tabular-nums; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero-title">BiasLens</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Measuring framing, overconfidence &amp; gender bias across LLMs — '
            'live.</div>', unsafe_allow_html=True)

base_cfg = load_config()

st.sidebar.title("Settings")
key_ok = bool(base_cfg.groq_api_key)
st.sidebar.markdown(f"**API key:** {'🟢 detected' if key_ok else '🔴 not in .env'}")
offline = st.sidebar.toggle("Offline mode (mock model)", value=not key_ok,
                            help="Run with a fake model + sample data. No API key needed.")

st.sidebar.subheader("Models")
if "models" not in st.session_state:
    st.session_state.models = [m.id for m in base_cfg.models] or ["openai/gpt-oss-20b"]

remove_idx = None
for idx, mid in enumerate(st.session_state.models):
    c1, c2 = st.sidebar.columns([5, 1])
    c1.markdown(f'<div class="model-pill">{mid}</div>', unsafe_allow_html=True)
    if c2.button("✕", key=f"rm_{idx}", help="Remove"):
        remove_idx = idx
if remove_idx is not None:
    st.session_state.models.pop(remove_idx)
    st.rerun()

add_col1, add_col2 = st.sidebar.columns([5, 2])
new_model = add_col1.text_input("Add model id", key="new_model_input", label_visibility="collapsed",
                                placeholder="owner/model-id")
if add_col2.button("Add", use_container_width=True):
    if new_model.strip() and new_model.strip() not in st.session_state.models:
        st.session_state.models.append(new_model.strip())
        st.rerun()

st.sidebar.subheader("Test sources")
src = base_cfg.sources or {}
use_framing = st.sidebar.checkbox("Framing set", value=True)
tqa_n = st.sidebar.slider("TruthfulQA cases (overconfidence)", 0, 200,
                          int(src.get("truthful_qa", {}).get("n_items", 40)))
bbq_n = st.sidebar.slider("BBQ cases (gender / social)", 0, 200,
                          int(src.get("bbq", {}).get("n_items", 40)))

st.sidebar.subheader("Sampling")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, float(base_cfg.temperature), 0.1)
concurrency = st.sidebar.slider("Concurrency", 1, 8, int(base_cfg.concurrency))

with st.sidebar.expander("Advanced"):
    n_reps = st.number_input("Repetitions per item", 1, 5, int(base_cfg.n_reps))
    max_tokens = st.number_input("Max tokens", 32, 2048, int(base_cfg.max_tokens), step=32)
    bootstrap = st.number_input("Bootstrap samples", 200, 10000, int(base_cfg.bootstrap_samples), step=200)
    seed = st.number_input("Random seed", 0, 99999, int(base_cfg.random_seed))
    bbq_path = st.text_input("BBQ dataset", value=str(src.get("bbq", {}).get("hf_path", "Elfsong/BBQ")))
    bbq_split = st.text_input("BBQ split", value=str(src.get("bbq", {}).get("hf_split", "gender_identity")))
    tqa_path = st.text_input("TruthfulQA dataset", value=str(src.get("truthful_qa", {}).get("hf_path", "truthful_qa")))

def effective_cfg():
    cfg = load_config(offline=offline)
    cfg.temperature = temperature
    cfg.concurrency = concurrency
    cfg.n_reps = int(n_reps)
    cfg.max_tokens = int(max_tokens)
    cfg.bootstrap_samples = int(bootstrap)
    cfg.random_seed = int(seed)
    cfg.models = [ModelSpec(id=i, label=i.split("/")[-1]) for i in st.session_state.models]
    cfg.sources = {
        "framing": {"enabled": use_framing, "path": "templates/framing.yaml"},
        "truthful_qa": {"enabled": tqa_n > 0, "n_items": tqa_n, "hf_path": tqa_path},
        "bbq": {"enabled": bbq_n > 0, "n_items": bbq_n, "hf_path": bbq_path, "hf_split": bbq_split},
    }
    return cfg

PRIMARY_UI = [
    ("framing", "framing_effect", "Framing effect",
     "Does the answer flip when a choice is reworded as a gain vs a loss?"),
    ("calibration", "overconfidence_gap", "Overconfidence",
     "Stated confidence minus actual accuracy. Positive = overconfident."),
    ("bbq", "ambiguous_overcommitment", "Gender bias (BBQ)",
     "Guesses a gendered answer when the context gives no basis for one."),
]

def classify(sc):
    if sc is None or sc.ci_low is None:
        return "badge-grey", "n/a"
    if sc.ci_low > 0 or sc.ci_high < 0:
        return "badge-red", "Detected"
    return "badge-green", "None"

if "view" not in st.session_state:
    st.session_state.view = None

run_clicked = st.button("Run full pipeline", type="primary", use_container_width=True,
                        disabled=not (offline or key_ok))
if not offline and not key_ok:
    st.info("No GROQ_API_KEY in `.env`. Add it, or turn on **Offline mode** in the sidebar.")

if run_clicked:
    cfg = effective_cfg()
    if not cfg.models:
        st.error("Add at least one model in the sidebar.")
        st.stop()
    session = get_sessionmaker(cfg.db_url)()
    with st.status("Running pipeline…", expanded=True) as status:
        st.write("Preparing test cases…")
        n = collect_items(session, cfg)
        total_calls = n * len(cfg.models) * cfg.n_reps
        bar = st.progress(0.0)
        line = st.empty()
        tracker = {"base": 0}

        def cb(label, done, total, errors):
            g = tracker["base"] + done
            bar.progress(min(1.0, g / max(1, total_calls)))
            line.markdown(f"**{label}** — {done}/{total} · {errors} errors   ·   overall {g}/{total_calls}")

        for spec in cfg.models:
            run = run_model(session, cfg, spec.id, spec.label, progress=cb)
            score_run(session, run, cfg)
            tracker["base"] += n * cfg.n_reps
            st.write(f"Done: {spec.label}")
        build_report(session, PROJECT_ROOT / "output")
        status.update(label="Pipeline complete", state="complete")
    st.success("Run complete. Open a panel below to explore the results.")

session = get_sessionmaker(base_cfg.db_url)()
runs = latest_runs(session)
if not runs:
    st.info("No results yet. Set it up in the sidebar and click **Run full pipeline**.")
    st.stop()

scores_by_run = {
    r.id: {(s.bias_type, s.metric): s for s in session.query(Score).filter(Score.run_id == r.id).all()}
    for r in runs
}

def toggle(view_name):
    st.session_state.view = None if st.session_state.view == view_name else view_name

st.markdown('<div class="section">Explore</div>', unsafe_allow_html=True)
b1, b2, b3 = st.columns(3)
b1.button("Findings", use_container_width=True, on_click=toggle, args=("findings",))
b2.button("Full metrics", use_container_width=True, on_click=toggle, args=("metrics",))
b3.button("Test cases", use_container_width=True, on_click=toggle, args=("cases",))
view = st.session_state.view

if view == "findings":
    n_sig = sum(1 for r in runs for bt, m, *_ in PRIMARY_UI
                if classify(scores_by_run[r.id].get((bt, m)))[1] == "Detected")
    s1, s2 = st.columns(2)
    s1.metric("Models compared", len(runs))
    s2.metric("Biases detected", n_sig)
    cols = st.columns(3)
    for col, (bt, metric, title, sub) in zip(cols, PRIMARY_UI):
        html = f'<div class="card"><div class="card-title">{title}</div><div class="card-sub">{sub}</div>'
        for r in runs:
            sc = scores_by_run[r.id].get((bt, metric))
            bcls, btxt = classify(sc)
            val = f"{sc.value:+.3f}" if sc else "—"
            ci = (f'<div class="row-ci">95% CI [{sc.ci_low:+.3f}, {sc.ci_high:+.3f}]</div>'
                  if sc and sc.ci_low is not None else "")
            html += (f'<div class="row"><span class="row-model">{r.label}</span>'
                     f'<span><span class="row-val">{val}</span>'
                     f'<span class="badge {bcls}">{btxt}</span>{ci}</span></div>')
        html += "</div>"
        col.markdown(html, unsafe_allow_html=True)
    st.caption("Red = statistically detectable (95% bootstrap CI excludes 0). Green = none detected.")

if view == "metrics":
    rows = []
    for r in runs:
        sc = scores_by_run[r.id]

        def v(bt, m, _sc=sc):
            s = _sc.get((bt, m))
            return s.value if s else None
        rows.append({"Model": r.label,
                     "Framing": v("framing", "framing_effect"),
                     "Overconfidence": v("calibration", "overconfidence_gap"),
                     "BBQ overcommit": v("bbq", "ambiguous_overcommitment"),
                     "Accuracy": v("calibration", "accuracy"),
                     "ECE": v("calibration", "ece")})
    st.dataframe(pd.DataFrame(rows).set_index("Model").style.format("{:+.3f}", na_rep="—"),
                 use_container_width=True)
    report_md = PROJECT_ROOT / "output" / "report.md"
    if report_md.exists():
        st.download_button("Download report.md", report_md.read_text(), file_name="report.md")

if view == "cases":
    model_label = st.radio("Model", [r.label for r in runs], horizontal=True)
    run = next(r for r in runs if r.label == model_label)
    pair_rows = session.query(TestItem.pair_id).filter(TestItem.bias_type == "framing").distinct().all()
    names = [p[0].split(":", 1)[-1] for p in pair_rows]
    if "case" not in st.session_state or st.session_state.get("case") not in names:
        st.session_state.case = names[0] if names else None

    st.caption("Pick a scenario:")
    per_row = 6
    for start in range(0, len(names), per_row):
        chunk = names[start:start + per_row]
        chip_cols = st.columns(per_row)
        for i, name in enumerate(chunk):
            chip_cols[i].button(name, key=f"chip_{name}", use_container_width=True,
                                on_click=lambda n=name: st.session_state.update(case=n))

    chosen = st.session_state.case
    if chosen:
        pair_id = f"framing:{chosen}"
        cols = st.columns(2)
        for col, variant in zip(cols, ("gain", "loss")):
            item = (session.query(TestItem)
                    .filter(TestItem.pair_id == pair_id, TestItem.variant == variant).first())
            if not item:
                continue
            resp = (session.query(Response)
                    .filter(Response.run_id == run.id, Response.item_id == item.id).first())
            ans = resp.raw_output if resp else "n/a"
            with col:
                st.markdown(f'<div class="frame-tag">{variant} frame</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="prompt-box">{item.prompt}</div>', unsafe_allow_html=True)
                st.markdown(f'<span class="answer-chip">Model chose: {ans}</span>', unsafe_allow_html=True)