"""
mltunex.ui.app  —  MLTuneX Streamlit UI
Launch: mltunex ui  |  python -m mltunex.ui
"""
from __future__ import annotations
import io, json, logging, os, sys, tempfile, threading, time, warnings
from queue import Empty, Queue
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

os.environ.setdefault("LIGHTGBM_VERBOSITY", "-1")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.filterwarnings("ignore")
logging.basicConfig(handlers=[logging.NullHandler()])
for _n in ("lightgbm","xgboost","catboost","optuna","sklearn","urllib3","httpx","langchain"):
    _l = logging.getLogger(_n); _l.setLevel(logging.CRITICAL); _l.propagate = False

st.set_page_config(
    page_title="MLTuneX",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Design system
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html,body,[class*="css"]{font-family:'Inter',-apple-system,sans-serif;}

/* ── Global background ── */
.stApp{background:#070d1a;}
.stApp > header{background:transparent!important;}
[data-testid="stAppViewContainer"]{background:#070d1a!important;}
[data-testid="stVerticalBlock"]{background:transparent!important;}
.block-container{background:#070d1a!important;padding-top:1.2rem!important;}
.main .block-container{
  background:#070d1a!important;max-width:100%!important;
  padding:1.2rem 2rem 2rem!important;}

/* ── Hide Streamlit chrome ── */
#MainMenu,footer,[data-testid="stToolbar"],
[data-testid="stHeader"],[data-testid="stDecoration"],
section[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"]{display:none!important;}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"]{
  background:#0a0f1e!important;border-bottom:1px solid #141d2e!important;
  gap:0!important;padding:0 .5rem!important;}
.stTabs [data-baseweb="tab"]{
  background:transparent!important;color:#475569!important;
  font-size:.82rem!important;font-weight:600!important;padding:.65rem 1.2rem!important;
  border-bottom:2px solid transparent!important;border-radius:0!important;
  letter-spacing:.02em!important;}
.stTabs [aria-selected="true"]{color:#818cf8!important;border-bottom-color:#6366f1!important;}
.stTabs [data-baseweb="tab-panel"]{padding:1.4rem 0 0!important;}

/* ── Metric cards ── */
[data-testid="stMetric"]{
  background:#0a0f1e;border:1px solid #141d2e;border-radius:10px;padding:1rem 1.2rem;}
[data-testid="stMetricLabel"]{
  color:#475569!important;font-size:.7rem!important;font-weight:600!important;
  text-transform:uppercase;letter-spacing:.07em;}
[data-testid="stMetricValue"]{
  color:#e2e8f0!important;font-size:1.55rem!important;font-weight:700!important;}

/* ── Inputs (used in Configure tab) ── */
.stTextInput input, .stNumberInput input, .stTextArea textarea{
  background:#0f1623!important;color:#e2e8f0!important;
  border:1px solid #1e2d42!important;border-radius:6px!important;}
.stTextInput input:focus, .stNumberInput input:focus{
  border-color:#6366f1!important;box-shadow:0 0 0 2px rgba(99,102,241,.2)!important;}
.stTextInput input::placeholder{color:#334155!important;}
.stTextInput label,.stNumberInput label,.stSelectbox label,
.stSlider label,.stFileUploader label,.stToggle label{
  color:#64748b!important;font-size:.82rem!important;}
.stSelectbox div[data-baseweb="select"]>div{
  background:#0f1623!important;color:#e2e8f0!important;border:1px solid #1e2d42!important;}
.stToggle span{color:#94a3b8!important;}
[data-testid="stToggle"] input:checked + div{background:#6366f1!important;}
[data-baseweb="slider"] [role="slider"]{background:#6366f1!important;border-color:#6366f1!important;}
.stSlider [data-baseweb="slider"] > div:first-child > div:first-child{background:#1e293b!important;}
.stSlider [data-baseweb="slider"] > div:first-child > div:nth-child(2){
  background:linear-gradient(90deg,#6366f1,#8b5cf6)!important;}
[data-baseweb="popover"] ul{background:#0f1623!important;border:1px solid #1e2d42!important;}
[data-baseweb="popover"] li{color:#e2e8f0!important;}
[data-baseweb="popover"] li:hover{background:#1e293b!important;}
[data-testid="stNumberInput"] button{
  background:#161d2e!important;color:#6366f1!important;border-color:#1e2d42!important;}

/* ── File uploader ── */
[data-testid="stFileUploader"]{
  background:#0f1623!important;border:1px dashed #2d3f55!important;border-radius:8px!important;}
[data-testid="stFileUploaderDropzone"]{background:#0f1623!important;border:none!important;padding:.6rem!important;}
[data-testid="stFileUploaderDropzone"] button{
  background:#161d2e!important;color:#818cf8!important;
  border:1px solid #2d3f55!important;border-radius:6px!important;font-size:.8rem!important;font-weight:600!important;}
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span{color:#334155!important;font-size:.75rem!important;}

/* ── Dataframe ── */
[data-testid="stDataFrame"]{border:1px solid #141d2e!important;border-radius:10px!important;overflow:hidden!important;}

/* ── Alerts ── */
.stAlert{border-radius:8px!important;}

/* ── Progress bar ── */
.stProgress>div>div{background:linear-gradient(90deg,#6366f1,#8b5cf6)!important;border-radius:4px!important;}

/* ── Run button ── */
.run-btn>button{
  background:linear-gradient(135deg,#4f46e5,#7c3aed)!important;
  color:#ffffff!important;-webkit-text-fill-color:#ffffff!important;
  font-weight:700!important;font-size:.95rem!important;
  border:none!important;border-radius:10px!important;
  box-shadow:0 4px 20px rgba(99,102,241,.45)!important;
  transition:all .2s!important;}
.run-btn>button p{color:#ffffff!important;-webkit-text-fill-color:#ffffff!important;}
.run-btn>button:hover{box-shadow:0 6px 28px rgba(99,102,241,.65)!important;transform:translateY(-1px)!important;}
.run-btn>button:disabled{opacity:.45!important;transform:none!important;}

/* ── Page header ── */
.page-header{
  background:linear-gradient(135deg,#0b1220 0%,#160529 50%,#0b1220 100%);
  border:1px solid #141d2e;border-radius:12px;padding:1.1rem 1.6rem;
  margin-bottom:1rem;position:relative;overflow:hidden;
  display:flex;align-items:center;gap:1.2rem;}
.page-header::before{
  content:'';position:absolute;inset:0;
  background:radial-gradient(ellipse at 10% 50%,rgba(99,102,241,.2) 0%,transparent 50%),
             radial-gradient(ellipse at 90% 50%,rgba(139,92,246,.12) 0%,transparent 50%);
  pointer-events:none;}
.page-header .ph-text{position:relative;}
.page-header h1{color:#e2e8f0!important;font-size:1.45rem!important;font-weight:800!important;margin:0!important;letter-spacing:-.03em;}
.page-header .sub{color:#475569;font-size:.78rem;margin:.15rem 0 0;}
.page-header .badges{margin-top:.5rem;display:flex;gap:.35rem;flex-wrap:wrap;}
.badge{display:inline-block;background:rgba(99,102,241,.12);color:#818cf8;
  border:1px solid rgba(99,102,241,.25);border-radius:20px;
  padding:.12rem .6rem;font-size:.67rem;font-weight:600;letter-spacing:.02em;}

/* ── Config tab card ── */
.cfg-card{
  background:#0a0f1e;border:1px solid #141d2e;border-radius:12px;
  padding:1.4rem 1.6rem;margin-bottom:.8rem;}
.cfg-card-title{
  color:#818cf8;font-size:.72rem;font-weight:700;letter-spacing:.1em;
  text-transform:uppercase;margin-bottom:1rem;
  display:flex;align-items:center;gap:.5rem;}
.cfg-card-title::before{
  content:'';display:inline-block;width:3px;height:14px;
  background:linear-gradient(180deg,#6366f1,#8b5cf6);border-radius:2px;}

/* ── Section heading (used in Run/Results etc) ── */
.sec-head{
  color:#94a3b8;font-size:.78rem;font-weight:700;letter-spacing:.1em;
  text-transform:uppercase;border-bottom:1px solid #141d2e;
  padding-bottom:.4rem;margin:1.4rem 0 .8rem;}

/* ── Event log ── */
.event-log{display:flex;flex-direction:column;gap:.3rem;margin:.4rem 0;}
.ev-card{
  display:flex;align-items:flex-start;gap:.75rem;background:#0a0f1e;
  border:1px solid #141d2e;border-left:3px solid #1e293b;border-radius:8px;
  padding:.5rem .85rem;font-size:.82rem;color:#64748b;}
.ev-card.done{border-left-color:#22c55e;}
.ev-card.error{border-left-color:#ef4444;}
.ev-card .ev-t{color:#334155;font-family:'JetBrains Mono',monospace;
  font-size:.7rem;white-space:nowrap;padding-top:1px;min-width:42px;}
.ev-card .ev-b b{color:#cbd5e1;font-weight:600;}
.ev-card code{background:#0f1623;color:#818cf8;border-radius:4px;
  padding:1px 5px;font-family:'JetBrains Mono',monospace;font-size:.73rem;}

/* ── Banners ── */
.banner{border-radius:10px;padding:.9rem 1.3rem;margin:.6rem 0;display:flex;align-items:center;gap:.8rem;}
.banner.ok{background:linear-gradient(90deg,#052e16,#053d20);border:1px solid #065f46;color:#d1fae5;}
.banner.run{background:linear-gradient(90deg,#0f172a,#1e1b4b);border:1px solid #312e81;color:#c7d2fe;}
.banner.err{background:linear-gradient(90deg,#1c0505,#200808);border:1px solid #7f1d1d;color:#fecaca;}
.banner .bi{font-size:1.3rem;}
.banner .bt h3{margin:0;font-size:.95rem;font-weight:700;}
.banner .bt p{margin:.1rem 0 0;font-size:.8rem;opacity:.75;}

/* ── Artefact grid ── */
.art-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:.7rem;margin:.7rem 0;}
.art-card{background:#0a0f1e;border:1px solid #141d2e;border-radius:10px;padding:.9rem 1rem;}
.art-card .at{color:#475569;font-size:.67rem;font-weight:700;text-transform:uppercase;letter-spacing:.09em;margin-bottom:.35rem;}
.art-card .ap{color:#818cf8;font-family:'JetBrains Mono',monospace;font-size:.71rem;word-break:break-all;}

/* ── Strategy card ── */
.strat-card{background:#0a0f1e;border:1px solid #141d2e;border-radius:10px;padding:1rem 1.2rem;margin-bottom:.8rem;}
.strat-card .sn{color:#818cf8;font-weight:700;font-size:.9rem;}
.strat-card .sd{color:#475569;font-size:.8rem;margin-top:.2rem;}
.strat-card .sl{color:#334155;font-size:.67rem;font-weight:700;text-transform:uppercase;letter-spacing:.09em;margin-bottom:.3rem;}

/* ── Markdown report container (light theme) ── */
.md-report{
  background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
  padding:1.5rem 2rem;color:#1e293b;line-height:1.7;}
.md-report h1,.md-report h2,.md-report h3{color:#0f172a;margin-top:1.4rem;margin-bottom:.5rem;}
.md-report h1{font-size:1.4rem;border-bottom:2px solid #6366f1;padding-bottom:.4rem;}
.md-report h2{font-size:1.1rem;border-bottom:1px solid #e2e8f0;padding-bottom:.3rem;}
.md-report h3{font-size:.95rem;color:#4f46e5;}
.md-report table{width:100%;border-collapse:collapse;font-size:.85rem;margin:1rem 0;}
.md-report th{background:#6366f1;color:#fff;padding:.45rem .75rem;text-align:left;}
.md-report td{padding:.4rem .75rem;border-bottom:1px solid #e2e8f0;color:#334155;}
.md-report tr:hover td{background:#f1f5f9;}
.md-report code{background:#ede9fe;color:#4f46e5;border-radius:4px;padding:1px 5px;font-family:'JetBrains Mono',monospace;font-size:.82rem;}
.md-report blockquote{border-left:3px solid #6366f1;margin:0;padding:.4rem 1rem;background:#f1f5f9;color:#475569;font-style:italic;}
.md-report hr{border:none;border-top:1px solid #e2e8f0;margin:1.2rem 0;}
.md-report strong{color:#0f172a;}

/* ── Scrollbar ── */
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:#070d1a;}
::-webkit-scrollbar-thumb{background:#1e293b;border-radius:3px;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Page header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <div style="font-size:2rem;position:relative">🤖</div>
  <div class="ph-text">
    <h1>MLTuneX</h1>
    <p class="sub">Automated Machine Learning Fine-Tuning System</p>
    <div class="badges">
      <span class="badge">AutoML</span>
      <span class="badge">AI-Guided Tuning</span>
      <span class="badge">Optuna</span>
      <span class="badge">sklearn · XGBoost · LightGBM</span>
      <span class="badge">v0.2.0</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS: Dict[str, Any] = {
    "running": False, "finished": False, "error": None,
    "events": [], "model_rows": [], "stage_events": [],
    "eval_df": None, "top_models_df": None,
    "best_model": None, "best_params": {}, "best_score": 0.0,
    "trial_history": [], "profile": None,
    "log_dir": None, "queue": None,
    "n_models_total": 0, "n_models_done": 0,
    "saved_models": [], "pipeline_path": None, "run_log_dir": None,
    "start_time": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v
if st.session_state.queue is None:
    st.session_state.queue = Queue()

# ─────────────────────────────────────────────────────────────────────────────
# Tabs  (Configure is first — it's the entry point)
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "  Configure  ",
    "  Run  ",
    "  Profile  ",
    "  Results  ",
    "  Selection  ",
    "  Tuning  ",
    "  Reports  ",
])

# ─────────────────────────────────────────────────────────────────────────────
# Background thread
# ─────────────────────────────────────────────────────────────────────────────

def _run_thread(cfg: dict, queue: Queue, tmp: str, key: str, prov: str) -> None:
    _null = open(os.devnull, "w", encoding="utf-8")
    _so, _se = sys.stdout, sys.stderr
    sys.stdout = _null; sys.stderr = _null
    _nfd = None; _sfds: dict = {}
    try:
        _nfd = os.open(os.devnull, os.O_WRONLY)
        for _fd in (1, 2):
            try: _sfds[_fd] = os.dup(_fd); os.dup2(_nfd, _fd)
            except Exception: pass
    except Exception: pass
    os.environ["LIGHTGBM_VERBOSITY"] = "-1"
    if key:
        os.environ["GROQ_API_KEY" if prov.lower()=="groq" else "OPENAI_API_KEY"] = key
    try:
        from mltunex.orchestrator import MLTuneXOrchestrator, OrchestratorConfig
        import pandas as _pd
        ext = tmp.rsplit(".",1)[-1].lower()
        df  = (_pd.read_csv(tmp) if ext=="csv"
               else _pd.read_excel(tmp) if ext in ("xlsx","xls")
               else _pd.read_parquet(tmp))
        cfg["source"] = df
        orc = MLTuneXOrchestrator(OrchestratorConfig(**cfg))
        orc.set_progress_callback(lambda s, p: queue.put((s, p)))
        orc.run()
        queue.put(("__done__", {}))
    except Exception as exc:
        import traceback
        queue.put(("__error__", {"msg": str(exc), "tb": traceback.format_exc()}))
    finally:
        for _fd, _s in _sfds.items():
            try: os.dup2(_s, _fd); os.close(_s)
            except Exception: pass
        if _nfd is not None:
            try: os.close(_nfd)
            except Exception: pass
        sys.stdout = _so; sys.stderr = _se
        try: _null.close()
        except Exception: pass



# ═══ TAB 0 — Configure ════════════════════════════════════════════════════════
with tabs[0]:

    # ── Data Source ──────────────────────────────────────────────────────────
    st.markdown('<div class="cfg-card"><div class="cfg-card-title">Data Source</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Dataset file", type=["csv","xlsx","parquet"],
        label_visibility="collapsed",
        help="Accepts .csv, .xlsx, .parquet",
    )
    if not uploaded:
        st.caption("Accepts .csv · .xlsx · .parquet")

    _cols: list = []
    if uploaded:
        try:
            _ext = uploaded.name.rsplit(".",1)[-1].lower()
            uploaded.seek(0)
            _raw = uploaded.read()
            if _ext == "csv":
                _cols = pd.read_csv(io.BytesIO(_raw), nrows=0).columns.tolist()
            elif _ext in ("xlsx","xls"):
                _cols = pd.read_excel(io.BytesIO(_raw), nrows=0).columns.tolist()
            else:
                _cols = pd.read_parquet(io.BytesIO(_raw)).columns.tolist()
            uploaded.seek(0)
        except Exception:
            _cols = []

    dc1, dc2 = st.columns(2)
    with dc1:
        if _cols:
            target_col = st.selectbox("Target column", _cols,
                                      help="Column you want to predict.")
        else:
            target_col = st.text_input("Target column",
                                       placeholder="e.g.  Survived, price, label")
    with dc2:
        task_type = st.selectbox("Task type", ["classification","regression"])
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Training ─────────────────────────────────────────────────────────────
    st.markdown('<div class="cfg-card"><div class="cfg-card-title">Training</div>', unsafe_allow_html=True)
    tc1, tc2, tc3 = st.columns([2,1,1])
    with tc1:
        test_size = st.slider("Test split", 0.1, 0.4, 0.2, 0.05,
                              help="Fraction of data held out for evaluation.")
    with tc2:
        preprocess = st.toggle("Preprocessing", value=True,
                               help="Auto-build preprocessing pipeline from data profile.")
    with tc3:
        parallel = st.toggle("Parallel training", value=False,
                             help="Train all models simultaneously using multiprocessing.")
    n_jobs = st.number_input("Workers  (-1 = all CPUs)", min_value=-1, value=-1, step=1,
                             disabled=not parallel)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Tuning ───────────────────────────────────────────────────────────────
    st.markdown('<div class="cfg-card"><div class="cfg-card-title">Hyperparameter Tuning</div>', unsafe_allow_html=True)
    tune_models = st.toggle("AI-guided hyperparameter tuning", value=True)
    n_trials    = st.slider("Optuna trials", 5, 200, 25, 5, disabled=not tune_models)

    if tune_models:
        ai1, ai2 = st.columns(2)
        with ai1:
            llm_provider = st.selectbox("AI provider", ["Groq","OpenAI"])
            llm_model    = st.text_input(
                "Model name",
                value={"Groq":"qwen/qwen3-32b","OpenAI":"gpt-4o"}[llm_provider],
            )
        with ai2:
            api_key = st.text_input(
                f"{llm_provider} API key", type="password",
                placeholder="Paste key — never stored on disk",
            )
            st.caption("Used only for this session. Never written to disk.")
    else:
        llm_provider = "Groq"; llm_model = ""; api_key = ""
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Model Selection ───────────────────────────────────────────────────────
    st.markdown('<div class="cfg-card"><div class="cfg-card-title">Model Selection</div>', unsafe_allow_html=True)
    ms1, ms2, ms3 = st.columns(3)
    with ms1:
        selector  = st.selectbox(
            "Strategy",
            ["topk","stability","generalization"],
            help=(
                "topk: rank by primary metric  |  "
                "stability: penalise high variance  |  "
                "generalization: penalise train/test gap"
            ),
        )
    with ms2:
        top_k     = st.number_input("Top-K candidates", min_value=1, max_value=10, value=3,
                                    help="Models forwarded to the optimiser.")
    with ms3:
        profiling = st.selectbox("Profiling depth", ["extended","basic"],
                                 help="extended: full skew/kurtosis/correlation stats.")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Output ────────────────────────────────────────────────────────────────
    st.markdown('<div class="cfg-card"><div class="cfg-card-title">Output Paths</div>', unsafe_allow_html=True)
    op1, op2, op3, op4 = st.columns(4)
    with op1: log_dir     = st.text_input("Log directory", value="logs/")
    with op2: exp_name    = st.text_input("Run tag",       value="exp")
    with op3: results_dir = st.text_input("Results dir",   value="results/")
    with op4: models_dir  = st.text_input("Models dir",    value="models/")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Run button ────────────────────────────────────────────────────────────
    st.markdown("")
    rc1, rc2, rc3 = st.columns([1,2,1])
    with rc2:
        st.markdown('<div class="run-btn">', unsafe_allow_html=True)
        run_btn = st.button(
            "▶  Run Pipeline" if not st.session_state.running else "⏳  Running…",
            use_container_width=True,
            type="primary",
            disabled=st.session_state.running,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Validation feedback directly in the Configure tab
    if run_btn:
        _errs = []
        if not uploaded:               _errs.append("Upload a dataset file.")
        if not target_col.strip():     _errs.append("Specify the target column.")
        if tune_models and not api_key.strip():
            _errs.append(f"Paste your {llm_provider} API key.")
        if _errs:
            for _e in _errs:
                st.error(f"⚠  {_e}")
        else:
            for k, v in _DEFAULTS.items():
                st.session_state[k] = ([] if isinstance(v, list)
                                       else {} if isinstance(v, dict) else v)
            st.session_state.queue      = Queue()
            st.session_state.running    = True
            st.session_state.log_dir    = log_dir
            st.session_state.start_time = time.time()
            suf = "." + uploaded.name.rsplit(".",1)[-1]
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suf)
            tmp.write(uploaded.read()); tmp.close()
            cfg_dict = dict(
                source=tmp.name, target_column=str(target_col).strip(),
                task_type=task_type,
                model_provider_model_name=f"{llm_provider}:{llm_model}" if tune_models else "Groq:none",
                result_csv_path=results_dir, model_dir_path=models_dir,
                log_dir=log_dir, experiment_name=exp_name,
                test_size=float(test_size), preprocess=bool(preprocess),
                parallel_training=bool(parallel), n_jobs=int(n_jobs),
                tune_models=bool(tune_models), n_trials=int(n_trials),
                profiling_strategy=profiling, selection_strategy=selector, top_k=int(top_k),
            )
            threading.Thread(
                target=_run_thread,
                args=(cfg_dict, st.session_state.queue, tmp.name, api_key, llm_provider),
                daemon=True,
            ).start()
            st.rerun()

    if st.session_state.running:
        st.info("⏳ Pipeline is running — switch to the **Run** tab to see live progress.")
    elif st.session_state.finished:
        st.success("✓ Pipeline completed. Switch to the **Run** tab for results.")


# ─────────────────────────────────────────────────────────────────────────────
# Queue drain
# ─────────────────────────────────────────────────────────────────────────────
_new = 0
if st.session_state.running or not st.session_state.queue.empty():
    q = st.session_state.queue
    while True:
        try: stage, payload = q.get_nowait()
        except Empty: break
        _new += 1
        el = (f"{time.time()-st.session_state.start_time:.1f}s"
              if st.session_state.start_time else "")

        if stage == "__done__":
            st.session_state.running = False; st.session_state.finished = True
        elif stage == "__error__":
            st.session_state.running = False; st.session_state.error = payload
        elif stage == "n_models_total":
            st.session_state.n_models_total = payload.get("total", 0)
        elif stage == "data_loaded":
            st.session_state.stage_events.append((el, "done",
                f"<b>Data loaded</b> &nbsp;&middot;&nbsp; "
                f"{payload.get('rows','?')} rows &times; {payload.get('cols','?')} cols"))
        elif stage == "profiled":
            st.session_state.profile = payload.get("profile")
            st.session_state.stage_events.append((el, "done", "<b>Dataset profiled</b>"))
        elif stage == "model_done":
            st.session_state.model_rows.append(payload)
            st.session_state.n_models_done += 1
            # No entry in stage_events — Training Results table below handles this
        elif stage == "metrics_ready":
            st.session_state.eval_df = payload.get("eval_df")
            st.session_state.stage_events.append((el, "done", "<b>Metrics saved</b>"))
        elif stage == "selection_done":
            top = payload.get("top_models")
            st.session_state.top_models_df = top
            names = (top["Model"].tolist()
                     if top is not None and "Model" in top.columns else [])
            st.session_state.stage_events.append((el, "done",
                f"<b>Selection complete</b> &nbsp;&middot;&nbsp; "
                f"<code>{', '.join(names)}</code>"))
        elif stage == "tuning_done":
            st.session_state.best_model    = payload.get("best_model")
            st.session_state.best_params   = payload.get("best_params", {})
            st.session_state.best_score    = payload.get("best_score", 0.0)
            st.session_state.trial_history = payload.get("trial_history", [])
            st.session_state.stage_events.append((el, "done",
                f"<b>Tuning complete</b> &nbsp;&middot;&nbsp; "
                f"<code>{payload.get('best_model','?')}</code> "
                f"score <code>{payload.get('best_score',0):.4f}</code>"))
        elif stage == "model_saved":
            st.session_state.stage_events.append((el, "done",
                f"<b>Model saved</b> &nbsp;<code>{payload.get('path','?')}</code>"))
        elif stage == "pipeline_saved":
            st.session_state.pipeline_path = payload.get("path")
            st.session_state.stage_events.append((el, "done",
                f"<b>Pipeline saved</b> &nbsp;<code>{payload.get('path','?')}</code>"))
        elif stage == "run_summary":
            st.session_state.saved_models  = payload.get("saved_models", [])
            st.session_state.pipeline_path = payload.get("pipeline_path")
            st.session_state.run_log_dir   = payload.get("log_dir")
        st.session_state.events.append((stage, payload))

# ═══ TAB 1 — Run ══════════════════════════════════════════════════════════════
with tabs[1]:
    if st.session_state.running:
        el = (f"{time.time()-st.session_state.start_time:.0f}s"
              if st.session_state.start_time else "")
        st.markdown(
            f'<div class="banner run"><div class="bi">⏳</div>'
            f'<div class="bt"><h3>Running {el}</h3>'
            f'<p>Events stream below as each stage completes.</p></div></div>',
            unsafe_allow_html=True,
        )
    elif st.session_state.error:
        err = st.session_state.error
        st.markdown(
            f'<div class="banner err"><div class="bi">&#10007;</div>'
            f'<div class="bt"><h3>Pipeline failed</h3>'
            f'<p>{err.get("msg","Unknown error")}</p></div></div>',
            unsafe_allow_html=True,
        )
        with st.expander("Full traceback"):
            st.code(err.get("tb",""), language="python")
    elif st.session_state.finished:
        bm  = st.session_state.best_model
        el  = (f"{time.time()-st.session_state.start_time:.0f}s"
               if st.session_state.start_time else "")
        sub = (f"Best model: <code>{bm}</code> &nbsp;&middot;&nbsp; "
               f"score <code>{st.session_state.best_score:.4f}</code>"
               if bm else "Top models saved successfully.")
        st.markdown(
            f'<div class="banner ok"><div class="bi">&#10003;</div>'
            f'<div class="bt"><h3>Completed '
            f'<span style="font-weight:400;font-size:.82rem">{el}</span></h3>'
            f'<p>{sub}</p></div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="text-align:center;padding:3rem 1rem;">'
            '<div style="font-size:2.5rem;margin-bottom:.8rem">🤖</div>'
            '<div style="color:#475569;font-size:.95rem;font-weight:600">'
            'Go to <b>Configure</b> to set up and launch your run</div>'
            '<div style="color:#334155;font-size:.82rem;margin-top:.35rem">'
            'Upload data &nbsp;&middot;&nbsp; Set options &nbsp;&middot;&nbsp; '
            'Click Run Pipeline</div></div>',
            unsafe_allow_html=True,
        )

    total = st.session_state.n_models_total
    done  = st.session_state.n_models_done
    if total > 0 or done > 0:
        st.progress(done / max(total, done, 1),
                    text=f"Training  {done} / {total or '?'}  models")

    if st.session_state.stage_events:
        st.markdown('<div class="sec-head">Event Log</div>', unsafe_allow_html=True)
        cards = "".join(
            f'<div class="ev-card {cls}"><span class="ev-t">{ts}</span>'
            f'<span class="ev-b">{body}</span></div>'
            for ts, cls, body in st.session_state.stage_events
        )
        st.markdown(f'<div class="event-log">{cards}</div>', unsafe_allow_html=True)

    if st.session_state.model_rows:
        st.markdown('<div class="sec-head">Training Results</div>', unsafe_allow_html=True)
        rows = [
            {"Model": r["model"],
             **{k: round(v,4) for k,v in r.get("metrics",{}).items() if isinstance(v,float)},
             "Time (s)": round(r.get("elapsed",0),2)}
            for r in st.session_state.model_rows
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if st.session_state.finished and st.session_state.best_model:
        st.markdown('<div class="sec-head">Final Result</div>', unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Best Model",   st.session_state.best_model)
        c2.metric("Best Score",   f"{st.session_state.best_score:.4f}")
        c3.metric("Trials Run",   len(st.session_state.trial_history))
        c4.metric("Models Saved", len(st.session_state.saved_models))

    if st.session_state.finished and (
        st.session_state.saved_models or st.session_state.pipeline_path
    ):
        st.markdown('<div class="sec-head">Saved Artefacts</div>', unsafe_allow_html=True)
        html = '<div class="art-grid">'
        for p in st.session_state.saved_models:
            html += (f'<div class="art-card"><div class="at">Model &middot; .joblib</div>'
                     f'<div class="ap">{p}</div></div>')
        if st.session_state.pipeline_path:
            html += (f'<div class="art-card"><div class="at">Preprocessing Pipeline &middot; .joblib</div>'
                     f'<div class="ap">{st.session_state.pipeline_path}</div></div>')
        if st.session_state.run_log_dir:
            html += (f'<div class="art-card"><div class="at">Logs &amp; Reports</div>'
                     f'<div class="ap">{st.session_state.run_log_dir}</div></div>')
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)
        st.code(
            "import joblib\n"
            "pipeline = joblib.load('models/preprocessing_pipeline.joblib')\n"
            "model    = joblib.load('models/<ModelName>.joblib')\n"
            "preds    = model.predict(pipeline.transform(X_new))",
            language="python",
        )

# ═══ TAB 2 — Profile ══════════════════════════════════════════════════════════
with tabs[2]:
    profile = st.session_state.profile
    if not profile:
        st.info("Data profile appears here once profiling completes.")
    else:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Rows",      profile.get("num_rows","—"))
        c2.metric("Features",  profile.get("num_features","—"))
        c3.metric("Target",    profile.get("target_column","—"))
        c4.metric("Imbalance", str(profile.get("imbalance_ratio","—")))
        st.markdown('<div class="sec-head">Feature Types</div>', unsafe_allow_html=True)
        f1,f2 = st.columns(2)
        with f1:
            st.markdown("**Numeric**")
            nf = profile.get("numeric_features",[])
            st.caption(", ".join(f"`{x}`" for x in nf) if nf else "_none_")
        with f2:
            st.markdown("**Categorical**")
            cf = profile.get("categorical_features",[])
            st.caption(", ".join(f"`{x}`" for x in cf) if cf else "_none_")
        missing = {k:v for k,v in profile.get("missing_counts",{}).items() if v>0}
        st.markdown('<div class="sec-head">Missing Data</div>', unsafe_allow_html=True)
        if missing:
            pct = profile.get("missing_pct",{})
            st.dataframe(pd.DataFrame([
                {"Column":k,"Count":v,"Pct (%)":pct.get(k,0),
                 "Severity":"High" if pct.get(k,0)>30 else("Medium" if pct.get(k,0)>5 else "Low")}
                for k,v in missing.items()
            ]), use_container_width=True, hide_index=True)
        else:
            st.success("No missing values detected.")
        if "skewness" in profile:
            st.markdown('<div class="sec-head">Feature Skewness (top 12)</div>', unsafe_allow_html=True)
            items = sorted(profile["skewness"].items(), key=lambda x: abs(x[1]), reverse=True)[:12]
            st.bar_chart(pd.DataFrame(items, columns=["Feature","Skewness"]).set_index("Feature"))
        if "target_distribution" in profile:
            td = profile["target_distribution"]
            if td and len(td) <= 25:
                st.markdown('<div class="sec-head">Target Distribution</div>', unsafe_allow_html=True)
                st.bar_chart(pd.DataFrame(list(td.items()), columns=["Class","Proportion"]).set_index("Class"))

# ═══ TAB 3 — Results ══════════════════════════════════════════════════════════
with tabs[3]:
    eval_df = st.session_state.eval_df
    if eval_df is None or (hasattr(eval_df,"empty") and eval_df.empty):
        st.info("Evaluation results appear here after training completes.")
    else:
        num_cols = eval_df.select_dtypes("number").columns.tolist()
        primary  = ("Accuracy" if "Accuracy" in eval_df.columns
                    else "R2" if "R2" in eval_df.columns
                    else (num_cols[0] if num_cols else None))
        df_show  = eval_df.sort_values(primary, ascending=False) if primary else eval_df
        if not df_show.empty:
            best = df_show.iloc[0]
            s1,s2,s3 = st.columns(3)
            s1.metric("Best Model",       best["Model"])
            s2.metric(f"Best {primary}",  f"{best[primary]:.4f}" if primary else "—")
            s3.metric("Models Evaluated", len(df_show))
        st.markdown('<div class="sec-head">Full Leaderboard</div>', unsafe_allow_html=True)
        st.dataframe(df_show, use_container_width=True, hide_index=True)
        if primary and "Model" in df_show.columns:
            st.markdown(f'<div class="sec-head">{primary} Comparison</div>', unsafe_allow_html=True)
            st.bar_chart(df_show[["Model",primary]].set_index("Model"))

# ═══ TAB 4 — Selection ════════════════════════════════════════════════════════
with tabs[4]:
    top_df = st.session_state.top_models_df
    if top_df is None:
        st.info("Selection results appear here after the pipeline runs.")
    else:
        meta = {
            "topk":           ("Top-K by Metric",      "Ranks all models by the primary metric."),
            "stability":      ("Stability-Aware",       "Penalises models with high cross-metric variance."),
            "generalization": ("Generalisation-Aware",  "Penalises large train/test performance gaps."),
        }
        sn, sd = meta.get(selector, ("Custom",""))
        st.markdown(
            f'<div class="strat-card"><div class="sl">Strategy</div>'
            f'<div class="sn">{sn}</div><div class="sd">{sd}</div></div>',
            unsafe_allow_html=True,
        )
        verb = "selected for hyperparameter tuning" if tune_models else "saved as final models"
        st.success(f"**{len(top_df)}** model(s) {verb}.")
        st.dataframe(top_df, use_container_width=True, hide_index=True)

# ═══ TAB 5 — Tuning ═══════════════════════════════════════════════════════════
with tabs[5]:
    if not tune_models:
        st.info("AI hyperparameter tuning was disabled for this run.")
    elif not st.session_state.best_model:
        st.info("Tuning results appear here after optimisation completes.")
    else:
        t1,t2,t3 = st.columns(3)
        t1.metric("Best Model",   st.session_state.best_model)
        t2.metric("Best Score",   f"{st.session_state.best_score:.5f}")
        t3.metric("Total Trials", len(st.session_state.trial_history))
        tc1,tc2 = st.columns(2)
        with tc1:
            st.markdown('<div class="sec-head">Optimal Parameters</div>', unsafe_allow_html=True)
            if st.session_state.best_params:
                st.dataframe(
                    pd.DataFrame(list(st.session_state.best_params.items()),
                                 columns=["Parameter","Value"]),
                    use_container_width=True, hide_index=True,
                )
        with tc2:
            if st.session_state.trial_history:
                st.markdown('<div class="sec-head">Score Progression</div>', unsafe_allow_html=True)
                st.line_chart(pd.DataFrame([
                    {"Trial": t["trial"], "Score": t.get("score",0)}
                    for t in st.session_state.trial_history
                ]).set_index("Trial"))
        if st.session_state.trial_history:
            st.markdown('<div class="sec-head">Trial History</div>', unsafe_allow_html=True)
            th = sorted(st.session_state.trial_history, key=lambda x: x.get("score",0), reverse=True)
            st.dataframe(pd.DataFrame([{
                "Rank": i, "Trial": t["trial"], "Model": t.get("model","?"),
                "Score": round(t.get("score",0),5),
                "Params": json.dumps(t.get("params",{}), default=str),
            } for i,t in enumerate(th,1)]), use_container_width=True, hide_index=True)

# ═══ TAB 6 — Reports ══════════════════════════════════════════════════════════
with tabs[6]:
    ldir = st.session_state.log_dir or "logs/"
    if not st.session_state.finished and not st.session_state.events:
        st.info("Reports are written here after the pipeline finishes.")
    elif not ldir or not os.path.isdir(ldir):
        st.warning(f"Log directory not found: `{ldir}`")
    else:
        exp_dirs = sorted(
            [d for d in os.listdir(ldir) if os.path.isdir(os.path.join(ldir,d))],
            reverse=True,
        )
        if not exp_dirs:
            st.info("No experiment directories found yet.")
        else:
            chosen   = st.selectbox("Experiment run", exp_dirs, label_visibility="collapsed")
            exp_path = os.path.join(ldir, chosen)
            r_tabs   = st.tabs(["Preprocessing","Metrics","Selection","Tuning","Log"])

            def _md(fp: str) -> None:
                import re as _re
                if os.path.exists(fp):
                    raw = open(fp, encoding="utf-8").read()
                    raw = raw.replace("<!-- generated by MLTuneX -->","").strip()
                    txt = _re.sub(r'^---\n.*?\n---\n', '', raw, count=1, flags=_re.DOTALL).strip()
                    st.markdown(f'<div class="md-report">{txt}</div>', unsafe_allow_html=True)
                    st.divider()
                    st.download_button("Download Markdown", data=raw,
                                       file_name=os.path.basename(fp), mime="text/markdown")
                else:
                    st.info(f"`{os.path.basename(fp)}` not yet generated.")

            with r_tabs[0]: _md(os.path.join(exp_path,"preprocessing_report.md"))
            with r_tabs[1]:
                xlsx = os.path.join(exp_path,"model_metrics.xlsx")
                csv  = os.path.join(exp_path,"model_metrics.csv")
                if os.path.exists(xlsx):
                    with open(xlsx,"rb") as f:
                        st.download_button("Download Excel", data=f.read(),
                                           file_name="model_metrics.xlsx",
                                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                    try:
                        st.dataframe(pd.read_excel(xlsx, sheet_name="Ranked"),
                                     use_container_width=True, hide_index=True)
                    except Exception: pass
                elif os.path.exists(csv):
                    st.dataframe(pd.read_csv(csv), use_container_width=True, hide_index=True)
                else:
                    st.info("Not yet generated.")
            with r_tabs[2]: _md(os.path.join(exp_path,"selection_report.md"))
            with r_tabs[3]: _md(os.path.join(exp_path,"tuning_report.md"))
            with r_tabs[4]:
                lp = os.path.join(exp_path,"pipeline.log")
                if os.path.exists(lp):
                    st.code(open(lp,encoding="utf-8").read(), language="text")
                else:
                    st.info("Not yet generated.")

# ─────────────────────────────────────────────────────────────────────────────
# Auto-rerun while pipeline is active
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.running or _new > 0:
    time.sleep(0.5)
    st.rerun()