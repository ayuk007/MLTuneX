"""
mltunex.ui.app  —  MLTuneX Streamlit UI
────────────────────────────────────────
Launch via:
    mltunex ui                  (after pip install)
    python -m mltunex.ui        (from source)
"""
from __future__ import annotations

import io
import json
import logging
import os
import sys
import tempfile
import threading
import time
import warnings
from queue import Empty, Queue
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

# ── Silence ALL terminal output before anything else ─────────────────────────
# LightGBM writes "[LightGBM] [Warning]…" to C-level stdout; setting the env
# var before the library is imported is the only reliable suppressor.
os.environ.setdefault("LIGHTGBM_VERBOSITY", "-1")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.filterwarnings("ignore")

# Route every Python logger away from the terminal
logging.basicConfig(handlers=[logging.NullHandler()])
for _noisy in ("lightgbm", "xgboost", "catboost", "optuna", "sklearn",
               "urllib3", "httpx", "httpcore", "langchain"):
    _lg = logging.getLogger(_noisy)
    _lg.setLevel(logging.CRITICAL)
    _lg.propagate = False

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="MLTuneX",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Header ── */
.mltunex-header {
  background: linear-gradient(135deg, #1e3a8a 0%, #7c3aed 100%);
  padding: 1.4rem 2rem; border-radius: 12px;
  margin-bottom: 1.2rem; color: white;
}
.mltunex-header h1 { color: white !important; margin: 0; font-size: 2rem; }
.mltunex-header p  { color: #cbd5e1; margin: 0.25rem 0 0; font-size: 0.95rem; }

/* ── Sidebar — carefully scoped so inputs stay readable ── */
section[data-testid="stSidebar"] { background: #0f172a !important; }
/* Labels only (not input text) */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown h3 {
  color: #94a3b8 !important;
}
/* Input backgrounds: white bg, dark text so typing is visible */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea {
  background-color: #1e293b !important;
  color: #f1f5f9 !important;
  border: 1px solid #475569 !important;
}
section[data-testid="stSidebar"] input::placeholder { color: #64748b !important; }
/* Selectbox text */
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
  background-color: #1e293b !important;
  color: #f1f5f9 !important;
  border-color: #475569 !important;
}
/* Slider track */
section[data-testid="stSidebar"] .stSlider > div { color: #94a3b8 !important; }
/* Toggle label */
section[data-testid="stSidebar"] .stToggle span { color: #cbd5e1 !important; }
/* Section headings in sidebar */
section[data-testid="stSidebar"] h3 { color: #e2e8f0 !important; font-size: 0.9rem; }
/* Number input */
section[data-testid="stSidebar"] .stNumberInput input { color: #f1f5f9 !important; }
/* Run button */
section[data-testid="stSidebar"] .stButton > button {
  background: linear-gradient(90deg,#4f46e5,#7c3aed) !important;
  color: white !important; font-weight: 700; border: none;
}

/* ── Event / stage cards ── */
.stage-card {
  background: #1e293b; border: 1px solid #334155;
  border-left: 4px solid #6366f1; border-radius: 8px;
  padding: 0.65rem 1.1rem; margin: 0.3rem 0;
  color: #e2e8f0; font-size: 0.88rem;
}
.stage-card.done  { border-left-color: #22c55e; }
.stage-card.error { border-left-color: #ef4444; }
.stage-card.run   { border-left-color: #f59e0b; }

/* ── Metric pill ── */
.metric-pill {
  display: inline-block; background: #1e3a5f; color: #93c5fd;
  border-radius: 20px; padding: 2px 10px; font-size: 0.78rem;
  margin: 1px 2px; font-family: monospace;
}

/* ── Section titles ── */
.section-title {
  font-size: 1.1rem; font-weight: 700; color: #6366f1;
  border-bottom: 2px solid #334155; padding-bottom: 0.35rem;
  margin: 1.4rem 0 0.7rem;
}

/* ── Result banner ── */
.result-banner {
  background: linear-gradient(90deg,#14532d,#166534);
  border: 1px solid #22c55e; border-radius: 10px;
  padding: 1.1rem 1.5rem; color: #dcfce7; margin: 0.8rem 0;
}

/* ── Progress bar wrapper ── */
.prog-wrap { margin: 0.5rem 0 1rem; }

/* Hide Streamlit chrome */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="mltunex-header">
  <h1>🤖 MLTuneX</h1>
  <p>Automated Machine Learning Fine-Tuning System</p>
</div>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
_DEFAULTS: Dict[str, Any] = {
    "running":       False,
    "finished":      False,
    "error":         None,
    "events":        [],
    "model_rows":    [],
    "eval_df":       None,
    "top_models_df": None,
    "best_model":    None,
    "best_params":   {},
    "best_score":    0.0,
    "trial_history": [],
    "profile":       None,
    "log_dir":       None,
    "queue":         None,
    "stage_text":    [],   # list of (css_class, html) for live card rendering
    "n_models_total": 0,
    "n_models_done":  0,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v
if st.session_state.queue is None:
    st.session_state.queue = Queue()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📁 Data")
    uploaded   = st.file_uploader("Upload dataset", type=["csv","xlsx","parquet"],
                                  help="CSV, Excel, or Parquet file.")
    target_col = st.text_input("Target column", placeholder="e.g. Survived")
    task_type  = st.selectbox("Task type", ["classification","regression"])

    st.markdown("### 🏋️ Training")
    test_size  = st.slider("Test split", 0.1, 0.4, 0.2, 0.05)
    preprocess = st.toggle("Adaptive preprocessing", value=True)
    parallel   = st.toggle("Parallel training", value=False)
    n_jobs     = st.number_input("Workers (-1 = all CPUs)", min_value=-1, value=-1, step=1)

    st.markdown("### 🔬 Tuning")
    tune_models = st.toggle("AI hyperparameter tuning", value=True)
    n_trials    = st.number_input("Optuna trials", min_value=5, max_value=200, value=25,
                                  disabled=not tune_models)

    # ── AI / API key section (hidden when tuning disabled) ─────────────
    if tune_models:
        st.markdown("### 🤖 AI Advisor")
        llm_provider = st.selectbox("Provider", ["Groq", "OpenAI"])
        llm_model_defaults = {
            "Groq":   "qwen/qwen3-32b",
            "OpenAI": "gpt-4o",
        }
        llm_model  = st.text_input("Model name",
                                   value=llm_model_defaults[llm_provider])
        api_key    = st.text_input(
            f"{llm_provider} API Key",
            type="password",
            placeholder="Paste your API key here…",
            help="Key is used only for this session and never stored.",
        )
    else:
        llm_provider = "Groq"
        llm_model    = ""
        api_key      = ""

    st.markdown("### 🎯 Model Selection")
    selector  = st.selectbox("Strategy", ["topk","stability","generalization"])
    top_k     = st.number_input("Top-K candidates", min_value=1, max_value=10, value=3)
    profiling = st.selectbox("Profiling depth", ["extended","basic"])

    st.markdown("### 💾 Output")
    log_dir     = st.text_input("Log directory",     value="logs/")
    exp_name    = st.text_input("Experiment name",   value="exp")
    results_dir = st.text_input("Results directory", value="results/")
    models_dir  = st.text_input("Models directory",  value="models/")

    st.markdown("---")
    run_btn = st.button("🚀 Run MLTuneX", use_container_width=True,
                        type="primary", disabled=st.session_state.running)


# ── Validation helper ─────────────────────────────────────────────────────────
def _validate_inputs() -> Optional[str]:
    if not uploaded:
        return "Please upload a dataset file."
    if not target_col.strip():
        return "Please enter the target column name."
    if tune_models and not api_key.strip():
        return f"Please enter your {llm_provider} API key."
    return None


# ── Background worker ─────────────────────────────────────────────────────────
def _run_pipeline_thread(cfg_dict: dict, queue: Queue,
                         tmp_path: str, api_key: str,
                         llm_provider: str) -> None:
    """
    Runs the full MLTuneX pipeline in a daemon thread.

    stdout/stderr are redirected to devnull at the very top — before any
    MLTuneX code is imported or constructed — so that even __init__ prints
    with Unicode characters never reach the Windows console/colorama.
    """
    # ── Redirect FIRST, before any MLTuneX imports ────────────────────
    # This must happen before ExperimentLogger.__init__ which prints
    # a Unicode arrow that crashes cp1252 on Windows via colorama.
    _devnull = open(os.devnull, "w", encoding="utf-8")
    _old_stdout, _old_stderr = sys.stdout, sys.stderr
    sys.stdout = _devnull
    sys.stderr = _devnull

    # Also redirect at the C-level fd for LightGBM/XGBoost native libs
    _devnull_fd = None
    _saved_fds: dict = {}
    try:
        _devnull_fd = os.open(os.devnull, os.O_WRONLY)
        for _fd in (1, 2):
            try:
                _saved_fds[_fd] = os.dup(_fd)
                os.dup2(_devnull_fd, _fd)
            except Exception:
                pass
    except Exception:
        pass

    # Suppress C-level LightGBM / XGBoost output via env vars
    os.environ["LIGHTGBM_VERBOSITY"] = "-1"
    os.environ["VERBOSITY"] = "0"

    # Set API key as env var so LangChain clients pick it up automatically
    if api_key:
        key_env = "GROQ_API_KEY" if llm_provider.lower() == "groq" else "OPENAI_API_KEY"
        os.environ[key_env] = api_key

    try:
        from mltunex.orchestrator import MLTuneXOrchestrator, OrchestratorConfig
        import pandas as _pd

        ext = tmp_path.rsplit(".", 1)[-1].lower()
        if ext == "csv":
            df = _pd.read_csv(tmp_path)
        elif ext in ("xlsx", "xls"):
            df = _pd.read_excel(tmp_path)
        else:
            df = _pd.read_parquet(tmp_path)
        cfg_dict["source"] = df

        cfg = OrchestratorConfig(**cfg_dict)
        orc = MLTuneXOrchestrator(cfg)
        orc.set_progress_callback(lambda s, p: queue.put((s, p)))
        orc.run()
        queue.put(("__done__", {}))

    except Exception as exc:
        import traceback
        queue.put(("__error__", {"msg": str(exc), "tb": traceback.format_exc()}))

    finally:
        # Restore C-level fds first
        for _fd, _saved in _saved_fds.items():
            try:
                os.dup2(_saved, _fd)
                os.close(_saved)
            except Exception:
                pass
        if _devnull_fd is not None:
            try:
                os.close(_devnull_fd)
            except Exception:
                pass
        # Restore Python-level streams
        sys.stdout = _old_stdout
        sys.stderr = _old_stderr
        try:
            _devnull.close()
        except Exception:
            pass


# ── Launch ────────────────────────────────────────────────────────────────────
if run_btn:
    err_msg = _validate_inputs()
    if err_msg:
        st.error(f"⚠️ {err_msg}")
    else:
        # Reset all state
        for k, v in _DEFAULTS.items():
            st.session_state[k] = ([] if isinstance(v, list)
                                   else {} if isinstance(v, dict)
                                   else v)
        st.session_state.queue    = Queue()
        st.session_state.running  = True
        st.session_state.log_dir  = log_dir

        suffix = "." + uploaded.name.rsplit(".", 1)[-1]
        tmp    = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded.read())
        tmp.close()

        llm_str = f"{llm_provider}:{llm_model}" if tune_models else "Groq:none"

        cfg_dict = dict(
            source                    = tmp.name,
            target_column             = target_col.strip(),
            task_type                 = task_type,
            model_provider_model_name = llm_str,
            result_csv_path           = results_dir,
            model_dir_path            = models_dir,
            log_dir                   = log_dir,
            experiment_name           = exp_name,
            test_size                 = float(test_size),
            preprocess                = bool(preprocess),
            parallel_training         = bool(parallel),
            n_jobs                    = int(n_jobs),
            tune_models               = bool(tune_models),
            n_trials                  = int(n_trials),
            profiling_strategy        = profiling,
            selection_strategy        = selector,
            top_k                     = int(top_k),
        )

        threading.Thread(
            target=_run_pipeline_thread,
            args=(cfg_dict, st.session_state.queue, tmp.name,
                  api_key, llm_provider),
            daemon=True,
        ).start()
        st.rerun()


# ── Queue drain — runs on every rerun, processes ALL pending events ───────────
_new_events_this_cycle = 0
if st.session_state.running or not st.session_state.queue.empty():
    q = st.session_state.queue
    while True:
        try:
            stage, payload = q.get_nowait()
        except Empty:
            break

        _new_events_this_cycle += 1

        if stage == "__done__":
            st.session_state.running  = False
            st.session_state.finished = True

        elif stage == "__error__":
            st.session_state.running = False
            st.session_state.error   = payload

        elif stage == "n_models_total":
            st.session_state.n_models_total = payload.get("total", 0)

        elif stage == "data_loaded":
            st.session_state.stage_text.append(
                ("done", f"🗂 <b>Data loaded</b> — "
                         f"{payload.get('rows','?')} rows × "
                         f"{payload.get('cols','?')} columns"))

        elif stage == "profiled":
            st.session_state.profile = payload.get("profile")
            st.session_state.stage_text.append(
                ("done", "🔍 <b>Data profiled</b>"))

        elif stage == "model_done":
            st.session_state.model_rows.append(payload)
            st.session_state.n_models_done += 1
            m = payload.get("metrics", {})
            pills = "  ".join(
                f'<span class="metric-pill">{k}: {v:.4f}</span>'
                for k, v in m.items() if isinstance(v, float)
            )
            st.session_state.stage_text.append(
                ("done", f"✅ <b>{payload.get('model','?')}</b> "
                         f"— {payload.get('elapsed',0):.1f}s  {pills}"))

        elif stage == "metrics_ready":
            st.session_state.eval_df = payload.get("eval_df")
            st.session_state.stage_text.append(
                ("done", "📊 <b>Model metrics saved</b>"))

        elif stage == "selection_done":
            top = payload.get("top_models")
            st.session_state.top_models_df = top
            names = (top["Model"].tolist()
                     if top is not None and "Model" in top.columns else [])
            st.session_state.stage_text.append(
                ("done", f"🎯 <b>Models selected</b> — {', '.join(names)}"))

        elif stage == "tuning_done":
            st.session_state.best_model    = payload.get("best_model")
            st.session_state.best_params   = payload.get("best_params", {})
            st.session_state.best_score    = payload.get("best_score", 0.0)
            st.session_state.trial_history = payload.get("trial_history", [])
            st.session_state.stage_text.append(
                ("done", f"🏆 <b>Tuning complete</b> — best: "
                         f"{payload.get('best_model','?')} "
                         f"({payload.get('best_score',0):.4f})"))

        st.session_state.events.append((stage, payload))


# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🏃 Live Progress",
    "📊 Data Profile",
    "📈 Model Results",
    "🎯 Selection",
    "🔬 Tuning",
    "📄 Reports",
])


# ─── TAB 0 : Live Progress ────────────────────────────────────────────────────
with tabs[0]:

    # ── Status banner ─────────────────────────────────────────────────
    status_slot = st.empty()
    if st.session_state.running:
        status_slot.info("⏳ Pipeline is running — updates appear below in real time…")
    elif st.session_state.error:
        err = st.session_state.error
        status_slot.error(f"**Pipeline failed:** {err.get('msg','Unknown error')}")
        with st.expander("Full traceback"):
            st.code(err.get("tb",""), language="python")
    elif st.session_state.finished:
        if st.session_state.best_model:
            status_slot.markdown(f"""
<div class="result-banner">
  <h3 style="margin:0">✅ Pipeline completed!</h3>
  <span>Best model: <b>{st.session_state.best_model}</b>
        — score: <b>{st.session_state.best_score:.4f}</b></span>
</div>""", unsafe_allow_html=True)
        else:
            status_slot.markdown("""
<div class="result-banner">
  <h3 style="margin:0">✅ Pipeline completed!</h3>
</div>""", unsafe_allow_html=True)
    elif not st.session_state.events:
        status_slot.markdown(
            "👈 **Configure your run in the sidebar and click 🚀 Run MLTuneX.**")

    # ── Progress bar (model training) ─────────────────────────────────
    total = st.session_state.n_models_total
    done  = st.session_state.n_models_done
    if total > 0 or done > 0:
        pct = done / max(total, done)
        st.markdown(f'<div class="prog-wrap">', unsafe_allow_html=True)
        st.progress(pct, text=f"Models trained: {done} / {total or '?'}")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Live event cards ──────────────────────────────────────────────
    if st.session_state.stage_text:
        st.markdown('<div class="section-title">Pipeline Events</div>',
                    unsafe_allow_html=True)
        cards_html = "\n".join(
            f'<div class="stage-card {cls}">{html}</div>'
            for cls, html in st.session_state.stage_text
        )
        st.markdown(cards_html, unsafe_allow_html=True)

    # ── Live training table ───────────────────────────────────────────
    if st.session_state.model_rows:
        st.markdown('<div class="section-title">Training Results</div>',
                    unsafe_allow_html=True)
        rows_data = [
            {"Model": r["model"],
             **{k: round(v, 4) for k, v in r.get("metrics",{}).items()
                if isinstance(v, float)},
             "⏱ Time (s)": round(r.get("elapsed", 0), 2)}
            for r in st.session_state.model_rows
        ]
        st.dataframe(pd.DataFrame(rows_data),
                     use_container_width=True, hide_index=True)

    # ── Completion metrics ────────────────────────────────────────────
    if st.session_state.finished and st.session_state.best_model:
        st.markdown('<div class="section-title">Final Results</div>',
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("🏆 Best Model", st.session_state.best_model)
        c2.metric("🎯 Best Score", f"{st.session_state.best_score:.4f}")
        c3.metric("🧪 Trials",     len(st.session_state.trial_history))


# ─── TAB 1 : Data Profile ─────────────────────────────────────────────────────
with tabs[1]:
    profile = st.session_state.profile
    if not profile:
        st.info("Profile appears here after the pipeline starts.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🗂 Rows",      profile.get("num_rows","—"))
        c2.metric("🔢 Features",  profile.get("num_features","—"))
        c3.metric("🎯 Target",    profile.get("target_column","—"))
        c4.metric("⚖️ Imbalance", profile.get("imbalance_ratio","—"))

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Numeric features**")
            feats = profile.get("numeric_features",[])
            st.caption(", ".join(f"`{f}`" for f in feats) if feats else "—")
        with col2:
            st.markdown("**Categorical features**")
            feats = profile.get("categorical_features",[])
            st.caption(", ".join(f"`{f}`" for f in feats) if feats else "—")

        missing = {k:v for k,v in profile.get("missing_counts",{}).items() if v>0}
        if missing:
            st.markdown('<div class="section-title">Missing Data</div>',
                        unsafe_allow_html=True)
            pct_map = profile.get("missing_pct",{})
            miss_df = pd.DataFrame([
                {"Column": k, "Missing": v, "Pct (%)": pct_map.get(k,0)}
                for k,v in missing.items()
            ])
            st.dataframe(miss_df, use_container_width=True, hide_index=True)
        else:
            st.success("✅ No missing values.")

        if "skewness" in profile:
            st.markdown('<div class="section-title">Top Skewed Features</div>',
                        unsafe_allow_html=True)
            items = sorted(profile["skewness"].items(),
                           key=lambda x: abs(x[1]), reverse=True)[:12]
            skew_df = pd.DataFrame(items, columns=["Feature","Skewness"])
            st.bar_chart(skew_df.set_index("Feature"))


# ─── TAB 2 : Model Results ────────────────────────────────────────────────────
with tabs[2]:
    eval_df = st.session_state.eval_df
    if eval_df is None or (hasattr(eval_df,"empty") and eval_df.empty):
        st.info("Model results appear here after training completes.")
    else:
        primary = ("Accuracy" if "Accuracy" in eval_df.columns
                   else "R2"  if "R2"       in eval_df.columns
                   else eval_df.columns[1] if len(eval_df.columns)>1 else None)
        df_show = (eval_df.sort_values(primary, ascending=False)
                   if primary and primary in eval_df.columns else eval_df)

        st.dataframe(df_show, use_container_width=True, hide_index=True)

        if primary and "Model" in df_show.columns:
            st.markdown(f"**{primary} — all models**")
            st.bar_chart(df_show[["Model", primary]].set_index("Model"))


# ─── TAB 3 : Selection ───────────────────────────────────────────────────────
with tabs[3]:
    top_df = st.session_state.top_models_df
    if top_df is None:
        st.info("Selection results appear here after the pipeline runs.")
    else:
        if tune_models:
            st.success(f"✅ **{len(top_df)}** model(s) selected for hyperparameter optimisation.")
        else:
            st.success(
                f"✅ **{len(top_df)}** top model(s) saved "
                f"(AI tuning was disabled — these are your final models).")
        st.dataframe(top_df, use_container_width=True, hide_index=True)


# ─── TAB 4 : Tuning ──────────────────────────────────────────────────────────
with tabs[4]:
    if not tune_models:
        st.info("AI hyperparameter tuning was disabled for this run.")
    elif not st.session_state.best_model:
        st.info("Tuning results appear here after optimisation completes.")
    else:
        st.markdown(f"""
<div class="result-banner">
  <b>🏆 Best Model:</b> {st.session_state.best_model} &nbsp;|&nbsp;
  <b>🎯 Score:</b> {st.session_state.best_score:.5f}
</div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Optimal Hyperparameters**")
            params_df = pd.DataFrame(
                list(st.session_state.best_params.items()),
                columns=["Parameter","Value"],
            )
            st.dataframe(params_df, use_container_width=True, hide_index=True)

        with col2:
            if st.session_state.trial_history:
                st.markdown("**Score Progression**")
                scores_df = pd.DataFrame([
                    {"Trial": t["trial"], "Score": t.get("score",0)}
                    for t in st.session_state.trial_history
                ]).set_index("Trial")
                st.line_chart(scores_df)

        if st.session_state.trial_history:
            st.markdown('<div class="section-title">All Trials</div>',
                        unsafe_allow_html=True)
            th_df = pd.DataFrame([
                {"Rank":  rank,
                 "Trial": t["trial"],
                 "Model": t.get("model","?"),
                 "Score": round(t.get("score",0), 5),
                 "Params": json.dumps(t.get("params",{}), default=str)}
                for rank, t in enumerate(
                    sorted(st.session_state.trial_history,
                           key=lambda x: x.get("score",0), reverse=True), 1)
            ])
            st.dataframe(th_df, use_container_width=True, hide_index=True)


# ─── TAB 5 : Reports ─────────────────────────────────────────────────────────
with tabs[5]:
    log_dir_val = st.session_state.log_dir or log_dir
    if not st.session_state.finished and not st.session_state.events:
        st.info("Reports appear here after the pipeline runs.")
    elif not log_dir_val or not os.path.isdir(log_dir_val):
        st.warning(f"Log directory not found: `{log_dir_val}`")
    else:
        exp_dirs = sorted(
            [d for d in os.listdir(log_dir_val)
             if os.path.isdir(os.path.join(log_dir_val, d))],
            reverse=True,
        )
        if not exp_dirs:
            st.info("No experiment directories found yet.")
        else:
            chosen   = st.selectbox("Experiment run", exp_dirs)
            exp_path = os.path.join(log_dir_val, chosen)

            r_tabs = st.tabs([
                "🔍 Preprocessing", "📊 Metrics",
                "🎯 Selection", "🔬 Tuning", "📋 Log"
            ])

            def _render_md(path: str) -> None:
                if os.path.exists(path):
                    txt = open(path, encoding="utf-8").read()
                    txt = txt.replace("<!-- generated by MLTuneX -->","").strip()
                    st.markdown(txt)
                    st.download_button("⬇️ Download", data=txt,
                                       file_name=os.path.basename(path),
                                       mime="text/markdown")
                else:
                    st.info(f"`{os.path.basename(path)}` not yet generated.")

            with r_tabs[0]:
                _render_md(os.path.join(exp_path, "preprocessing_report.md"))

            with r_tabs[1]:
                xlsx = os.path.join(exp_path, "model_metrics.xlsx")
                csv  = os.path.join(exp_path, "model_metrics.csv")
                if os.path.exists(xlsx):
                    with open(xlsx,"rb") as f:
                        st.download_button(
                            "⬇️ Download model_metrics.xlsx", data=f.read(),
                            file_name="model_metrics.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
                    try:
                        st.dataframe(pd.read_excel(xlsx, sheet_name="Ranked"),
                                     use_container_width=True, hide_index=True)
                    except Exception:
                        pass
                elif os.path.exists(csv):
                    st.dataframe(pd.read_csv(csv),
                                 use_container_width=True, hide_index=True)
                else:
                    st.info("Model metrics not yet generated.")

            with r_tabs[2]:
                _render_md(os.path.join(exp_path, "selection_report.md"))

            with r_tabs[3]:
                _render_md(os.path.join(exp_path, "tuning_report.md"))

            with r_tabs[4]:
                log_path = os.path.join(exp_path, "pipeline.log")
                if os.path.exists(log_path):
                    st.code(open(log_path, encoding="utf-8").read(), language="text")
                else:
                    st.info("Pipeline log not yet generated.")


# ── Auto-rerun while pipeline is running ─────────────────────────────────────
# Only trigger a rerun when there are new events OR the pipeline is active.
# This gives incremental, progressive updates rather than a single batch
# at the end.
if st.session_state.running or _new_events_this_cycle > 0:
    time.sleep(0.6)
    st.rerun()
