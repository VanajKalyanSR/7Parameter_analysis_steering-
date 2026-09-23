"""
EOL (End-of-Line) Quality Analysis Dashboard
=============================================
- Loads EOL test report data (any number of parameters, each with Min/Max/Val/Result)
- Trains a classifier to predict overall Result (OK / NOK)
- Runs IQR-based outlier analysis per parameter
- Uses Groq LLM to narrate patterns/trends and root-cause hints
"""

import io
import json
import html as html_lib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EOL Quality Analytics",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

PARAMETERS = [
    "L-R_Balance_1",
    "In_Torque_1-L",
    "In_Torque_1-R",
    "In_Torque_2-L",
    "In_Torque_2-R",
    "Hysteresis_1-L",
    "Hysteresis_1-R",
]  # a known/expected set used only as a naming hint — NOT a hard requirement; see
   # detect_available_parameters() below, which discovers whatever parameters are
   # actually present in the uploaded file, whether there are fewer or more than 7.

BASE_COLS = ["SAMPLE", "Result", "Model"]

# ---------------------------------------------------------------------------
# GROQ API KEY  -- put your key here
# ---------------------------------------------------------------------------
GROQ_API_KEY = "PASTE_YOUR_GROQ_API_KEY_HERE"
GROQ_MODEL = "openai/gpt-oss-120b"   # llama-3.3-70b-versatile was decommissioned by Groq (Aug 2026); this is Groq's recommended replacement

# ---------------------------------------------------------------------------
# STYLES
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #6b7280;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem 1.2rem;
    }
    .ok-badge {
        background: #dcfce7; color: #166534;
        padding: 2px 10px; border-radius: 999px; font-weight: 600;
    }
    .nok-badge {
        background: #fee2e2; color: #991b1b;
        padding: 2px 10px; border-radius: 999px; font-weight: 600;
    }
    section[data-testid="stSidebar"] { background-color: #111827; }
    section[data-testid="stSidebar"] * { color: #f3f4f6 !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def find_param_columns(df, param):
    """Return dict of actual column names in df for a given parameter, tolerant of naming variants."""
    cols = {}
    for suffix in ["Result", "Max", "Val", "Min"]:
        candidates = [c for c in df.columns if c.lower() == f"{param}_{suffix}".lower()]
        cols[suffix] = candidates[0] if candidates else None
    return cols


def detect_available_parameters(df):
    """Auto-discover every parameter present in the uploaded file, however many there are.

    Works for datasets with fewer than 7, exactly 7, or more than 7 parameters: any group
    of columns following the "<ParamName>_Result / _Max / _Val / _Min" naming convention
    is picked up automatically, regardless of the parameter's name. The known PARAMETERS
    list above is just a preferred ordering hint for the common case — it is not required
    for a parameter to be detected, and unrecognized/extra parameter names work fine too.
    """
    suffix_pattern = ["Result", "Max", "Val", "Min"]
    discovered = {}
    for col in df.columns:
        for suffix in suffix_pattern:
            marker = f"_{suffix}"
            if col.lower().endswith(marker.lower()):
                base = col[: -len(marker)]
                discovered.setdefault(base, set()).add(suffix)
                break

    # keep only groups that at least have a Val column (the minimum needed for any analysis)
    valid_params = [base for base, suffixes in discovered.items() if "Val" in suffixes]

    # order: known PARAMETERS first (in their canonical order), then any extra/unexpected
    # parameters found in the file, in the order they first appear as columns
    ordered = [p for p in PARAMETERS if p in valid_params]
    ordered += [p for p in valid_params if p not in ordered]
    return ordered


def compute_iqr_bounds(df, param_cols, k=1.5):
    """IQR analysis for each parameter's Val column."""
    rows = []
    for p, cols in param_cols.items():
        val_col = cols["Val"]
        if val_col is None or val_col not in df.columns:
            continue
        series = pd.to_numeric(df[val_col], errors="coerce").dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        n_outliers = ((series < lower) | (series > upper)).sum()
        rows.append({
            "Parameter": p,
            "Q1": round(q1, 4),
            "Q3": round(q3, 4),
            "IQR": round(iqr, 4),
            "Lower Bound": round(lower, 4),
            "Upper Bound": round(upper, 4),
            "Min Observed": round(series.min(), 4),
            "Max Observed": round(series.max(), 4),
            "Outlier Count": int(n_outliers),
            "Outlier %": round(100 * n_outliers / len(series), 2),
        })
    return pd.DataFrame(rows)


def build_data_context(df_raw, param_cols, iqr_k, trained_model=None, feature_cols=None):
    """Build a single text summary of the dataset (counts, IQR stats, fail breakdowns,
    feature importances) reused by both the auto-insight report and the custom Q&A box."""
    result_upper = df_raw["Result"].astype(str).str.strip().str.upper()
    n_ok = int((result_upper.isin(["OK", "PASS", "1"])).sum())
    n_nok = len(df_raw) - n_ok

    iqr_df = compute_iqr_bounds(df_raw, param_cols, k=iqr_k)

    fail_counts = {}
    for p, cols in param_cols.items():
        rc = cols["Result"]
        if rc and rc in df_raw.columns:
            v = df_raw[rc].astype(str).str.strip().str.upper()
            fail_counts[p] = int((~v.isin(["OK", "PASS", "1"])).sum())

    half = len(df_raw) // 2
    first_half_fail = (~result_upper.iloc[:half].isin(["OK", "PASS", "1"])).mean() * 100 if half > 0 else None
    second_half_fail = (~result_upper.iloc[half:].isin(["OK", "PASS", "1"])).mean() * 100 if half > 0 else None

    feat_imp_text = ""
    if trained_model is not None and feature_cols is not None:
        imp_raw = pd.DataFrame({"Feature": feature_cols, "Importance": trained_model.feature_importances_})
        imp_raw["Parameter"] = imp_raw["Feature"].apply(
            lambda f: next((p for p in param_cols if f.startswith(p + "_")), f)
        )
        imp = imp_raw.groupby("Parameter", as_index=False)["Importance"].sum().sort_values(
            "Importance", ascending=False)
        imp["Contribution %"] = (100 * imp["Importance"] / imp["Importance"].sum()).round(1)
        feat_imp_text = imp[["Parameter", "Contribution %"]].to_string(index=False)

    model_breakdown = ""
    if "Model" in df_raw.columns:
        tmp = df_raw.copy()
        tmp["_fail"] = (~result_upper.isin(["OK", "PASS", "1"])).astype(int)
        by_model = tmp.groupby("Model")["_fail"].agg(["count", "sum"]).reset_index()
        by_model["fail_rate_%"] = (100 * by_model["sum"] / by_model["count"]).round(2)
        model_breakdown = by_model.to_string(index=False)

    context = f"""
OVERALL: {len(df_raw)} samples, {n_ok} OK, {n_nok} NOK ({100*n_nok/len(df_raw):.2f}% fail rate).

FAIL RATE TREND (first half vs second half of dataset, in row order):
First half fail rate: {first_half_fail:.2f}% | Second half fail rate: {second_half_fail:.2f}%

PER-PARAMETER FAIL COUNTS (out of spec occurrences):
{json.dumps(fail_counts, indent=2)}

IQR BOUNDS PER PARAMETER (based on measured Val):
{iqr_df.to_string(index=False)}

{"PARAMETER CONTRIBUTION TO NOK PREDICTIONS (from trained classifier):" if feat_imp_text else ""}
{feat_imp_text}

{"FAIL RATE BY MODEL:" if model_breakdown else ""}
{model_breakdown}
"""
    return context, iqr_df, n_ok, n_nok


def build_feature_matrix(df, param_cols):
    """Build numeric feature matrix from Val columns + margin-to-spec-limit features."""
    feats = pd.DataFrame(index=df.index)
    for p, cols in param_cols.items():
        val_col, min_col, max_col = cols["Val"], cols["Min"], cols["Max"]
        if val_col is None or val_col not in df.columns:
            continue
        val = pd.to_numeric(df[val_col], errors="coerce")
        feats[f"{p}_Val"] = val
        if min_col and max_col and min_col in df.columns and max_col in df.columns:
            mn = pd.to_numeric(df[min_col], errors="coerce")
            mx = pd.to_numeric(df[max_col], errors="coerce")
            span = (mx - mn).replace(0, np.nan)
            # normalized position within spec window; <0 or >1 means out of spec
            feats[f"{p}_NormPos"] = (val - mn) / span
            # margin to nearer limit (negative = out of spec)
            feats[f"{p}_MarginLow"] = val - mn
            feats[f"{p}_MarginHigh"] = mx - val
    return feats


def parse_result_col(series):
    """Map OK/NOK-like strings to binary 1/0 (1 = NOK / fail) for modeling clarity."""
    s = series.astype(str).str.strip().str.upper()
    mapping = {"OK": 0, "NOK": 1, "NG": 1, "FAIL": 1, "PASS": 0, "1": 1, "0": 0}
    return s.map(mapping)


def failing_parameters_per_row(df, param_cols):
    """For each row, list which parameters have Result != OK."""
    fail_lists = []
    for idx, row in df.iterrows():
        fails = []
        for p, cols in param_cols.items():
            rc = cols["Result"]
            if rc and rc in df.columns:
                val = str(row[rc]).strip().upper()
                if val not in ("OK", "PASS", "1", "1.0"):
                    fails.append(p)
        fail_lists.append(fails)
    return fail_lists


def call_groq(prompt, api_key, model=GROQ_MODEL, temperature=0.4, max_tokens=900):
    """Call Groq's OpenAI-compatible chat completions endpoint."""
    import requests
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": (
                "You are a manufacturing quality engineer analyzing End-Of-Line (EOL) "
                "test data for a component (torque/hysteresis/balance testing). "
                "Be precise, quantitative, and reference the actual numbers given to you. "
                "Structure your answer with short headers. Avoid generic filler."
            )},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


class InsightParseError(Exception):
    """Raised when the LLM's response couldn't be parsed as JSON; carries the raw text."""
    def __init__(self, raw_text):
        super().__init__("Could not parse LLM response as JSON")
        self.raw_text = raw_text


def _extract_json(text):
    """Best-effort extraction of a JSON object from LLM text: strips code fences and
    trims to the outermost {...} span, so minor formatting deviations don't break parsing."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1] if t.count("```") >= 2 else t.strip("`")
        t = t.lstrip("json").lstrip("JSON").strip()
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end != -1 and end > start:
        t = t[start:end + 1]
    return json.loads(t)


def call_groq_json(prompt, api_key, model=GROQ_MODEL, temperature=0.3, max_tokens=1100):
    """Call Groq and get back a parsed JSON object for structured, styled rendering.

    Tries strict `response_format: json_object` first. Some Groq-hosted models (notably
    openai/gpt-oss-*) intermittently reject or mishandle that mode with a 400, so on
    failure this retries as a plain call and parses the text leniently instead.
    """
    import requests

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    system_msg = (
        "You are a manufacturing quality engineer analyzing End-Of-Line (EOL) test data. "
        "Respond with ONLY a single valid JSON object — no markdown code fences, no preamble, "
        "no commentary before or after it. Every text field must reference actual numbers from "
        "the data provided."
    )

    def _post(use_json_mode):
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if use_json_mode:
            payload["response_format"] = {"type": "json_object"}
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    try:
        content = _post(use_json_mode=True)
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 400:
            content = _post(use_json_mode=False)  # fall back to plain-text mode
        else:
            raise

    try:
        return _extract_json(content)
    except (json.JSONDecodeError, ValueError):
        raise InsightParseError(content)


# ---------------------------------------------------------------------------
# SIDEBAR - DATA LOAD
# ---------------------------------------------------------------------------
st.sidebar.title("🔧 EOL Analytics")
st.sidebar.markdown("Upload your EOL report (CSV or Excel).")

uploaded_file = st.sidebar.file_uploader("Dataset", type=["csv", "xlsx", "xls"])

api_key_input = st.sidebar.text_input(
    "Groq API Key (optional override)",
    value="" if GROQ_API_KEY.startswith("PASTE") else GROQ_API_KEY,
    type="password",
    help="If left blank, the key hardcoded in app.py is used.",
)
effective_api_key = api_key_input.strip() or GROQ_API_KEY

groq_model_input = st.sidebar.text_input(
    "Groq model name",
    value=GROQ_MODEL,
    help="e.g. openai/gpt-oss-120b, openai/gpt-oss-20b, qwen/qwen3.6-27b. "
         "Older models like llama-3.3-70b-versatile were decommissioned by Groq.",
)
effective_model = groq_model_input.strip() or GROQ_MODEL

st.sidebar.markdown("---")
iqr_k = st.sidebar.slider("IQR multiplier (k)", 1.0, 3.0, 1.5, 0.1)
test_size = st.sidebar.slider("Test set size (%)", 10, 40, 20, 5) / 100
n_estimators = st.sidebar.slider("Random Forest trees", 50, 500, 200, 50)

st.markdown('<div class="main-header">EOL Quality Analytics Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">OK/NOK classification · IQR outlier analysis · LLM-generated defect insight</div>', unsafe_allow_html=True)

if uploaded_file is None:
    st.info("👈 Upload a CSV/Excel EOL report to begin. Expected columns: SAMPLE, Result, Model, "
            "and for each parameter: `<param>_Result`, `<param>_Max`, `<param>_Val`, `<param>_Min`. "
            "Any number of parameters is supported — the app auto-detects however many are present.")
    with st.expander("Commonly seen parameter names (not required — any name works)"):
        st.code("\n".join(PARAMETERS))
    st.stop()

# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------
try:
    if uploaded_file.name.lower().endswith(".csv"):
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

df_raw.columns = [str(c).strip() for c in df_raw.columns]

param_cols = {p: find_param_columns(df_raw, p) for p in detect_available_parameters(df_raw)}

if not param_cols:
    st.error("No recognizable parameter columns found. Check column naming against the expected list in the sidebar info box.")
    st.stop()

if "Result" not in df_raw.columns:
    st.error("No overall 'Result' column found in the dataset.")
    st.stop()

tabs = st.tabs(["📊 Overview", "📈 IQR Analysis", "🤖 OK/NOK Model", "🔮 Predict New Sample", "🧠 AI Insight (Groq)"])

# ---------------------------------------------------------------------------
# TAB 1: OVERVIEW
# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    total = len(df_raw)
    result_upper = df_raw["Result"].astype(str).str.strip().str.upper()
    n_ok = (result_upper.isin(["OK", "PASS", "1"])).sum()
    n_nok = total - n_ok
    c1.metric("Total Samples", total)
    c2.metric("OK", n_ok, delta=f"{100*n_ok/total:.1f}% of total", delta_color="off")
    c3.metric("NOK", n_nok, delta=f"{100*n_nok/total:.1f}% fail rate", delta_color="inverse")
    c4.metric("Parameters Detected", len(param_cols))

    st.markdown("#### Preview")
    st.dataframe(df_raw.head(20), use_container_width=True)

    st.markdown("#### Result Distribution")
    fig = px.pie(names=["OK", "NOK"], values=[n_ok, n_nok],
                 color=["OK", "NOK"], color_discrete_map={"OK": "#22c55e", "NOK": "#ef4444"},
                 hole=0.5)
    fig.update_traces(textinfo="percent+label+value", texttemplate="%{label}<br>%{value} (%{percent})")
    st.plotly_chart(fig, use_container_width=True)

    if "Model" in df_raw.columns:
        st.markdown("#### Fail Rate by Model")
        tmp = df_raw.copy()
        tmp["_fail"] = (~result_upper.isin(["OK", "PASS", "1"])).astype(int)
        by_model = tmp.groupby("Model")["_fail"].agg(["count", "sum"]).reset_index()
        by_model["fail_rate_%"] = (100 * by_model["sum"] / by_model["count"]).round(2)
        by_model["share_of_total_%"] = (100 * by_model["count"] / total).round(2)
        by_model.columns = ["Model", "Total", "NOK Count", "Fail Rate %", "Share of All Samples %"]
        st.dataframe(by_model.sort_values("Fail Rate %", ascending=False), use_container_width=True)
        fig2 = px.bar(by_model, x="Model", y="Fail Rate %", color="Fail Rate %",
                      color_continuous_scale="Reds", text="Fail Rate %")
        fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Which parameter fails most often?")
    fail_counts = {}
    for p, cols in param_cols.items():
        rc = cols["Result"]
        if rc and rc in df_raw.columns:
            v = df_raw[rc].astype(str).str.strip().str.upper()
            fail_counts[p] = int((~v.isin(["OK", "PASS", "1"])).sum())
    fc_df = pd.DataFrame(list(fail_counts.items()), columns=["Parameter", "Fail Count"])
    fc_df["Fail %"] = (100 * fc_df["Fail Count"] / total).round(2)
    fc_df = fc_df.sort_values("Fail Count", ascending=False)
    st.dataframe(fc_df, use_container_width=True)
    fig3 = px.bar(fc_df, x="Parameter", y="Fail %", color="Fail %", color_continuous_scale="OrRd",
                  text="Fail Count", hover_data={"Fail Count": True, "Fail %": ":.2f"})
    fig3.update_traces(texttemplate="%{text} fails", textposition="outside")
    fig3.update_layout(yaxis_title="Fail % of Total Samples")
    st.plotly_chart(fig3, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 2: IQR ANALYSIS
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("IQR Outlier Analysis (per parameter, based on Val)")
    iqr_df = compute_iqr_bounds(df_raw, param_cols, k=iqr_k)
    st.dataframe(iqr_df, use_container_width=True)

    sns.set_style("whitegrid")

    st.markdown("#### Boxplots — Statistical (IQR) View vs Spec Range View")
    st.caption("Left: measured value distribution with statistical IQR bounds (Q1 − 1.5·IQR / Q3 + 1.5·IQR). "
               "Right: the same values against the fixed spec Min/Max — points outside the spec band are outliers.")
    for p, cols in param_cols.items():
        val_col = cols["Val"]
        min_col, max_col = cols["Min"], cols["Max"]
        if val_col is None:
            continue

        series = pd.to_numeric(df_raw[val_col], errors="coerce").dropna()
        row = iqr_df[iqr_df["Parameter"] == p]

        c1, c2 = st.columns(2)

        # ---- LEFT: original IQR statistical boxplot ----
        with c1:
            fig, ax = plt.subplots(figsize=(4.5, 3.6))
            sns.boxplot(y=series, ax=ax, color="#6366f1", width=0.35, fliersize=4)
            sns.stripplot(y=series, ax=ax, color="#312e81", size=3, alpha=0.35, jitter=0.15)
            if not row.empty:
                ax.axhline(row["Lower Bound"].values[0], color="red", linestyle="--", linewidth=1,
                           label="IQR Lower Bound")
                ax.axhline(row["Upper Bound"].values[0], color="red", linestyle="--", linewidth=1,
                           label="IQR Upper Bound")
                ax.legend(fontsize=7, loc="best")
            ax.set_title(f"{p} — IQR View", fontsize=10)
            ax.set_ylabel("Value", fontsize=9)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # ---- RIGHT: spec Min/Max range band with in/out-of-spec points ----
        spec_min = spec_max = None
        if min_col and min_col in df_raw.columns:
            mn_series = pd.to_numeric(df_raw[min_col], errors="coerce").dropna()
            if not mn_series.empty:
                spec_min = mn_series.mode().iloc[0]  # constant across rows -> mode is safe
        if max_col and max_col in df_raw.columns:
            mx_series = pd.to_numeric(df_raw[max_col], errors="coerce").dropna()
            if not mx_series.empty:
                spec_max = mx_series.mode().iloc[0]

        if spec_min is not None and spec_max is not None:
            status = np.where((series < spec_min) | (series > spec_max), "Out of Spec", "In Spec")
        else:
            status = np.array(["In Spec"] * len(series))

        plot_df = pd.DataFrame({"Parameter": p, "Value": series.values, "Status": status})

        with c2:
            fig2, ax2 = plt.subplots(figsize=(4.5, 3.6))
            if spec_min is not None and spec_max is not None:
                ax2.axhspan(spec_min, spec_max, color="#bbf7d0", alpha=0.4, zorder=0, label="Spec Range (Min–Max)")
                ax2.axhline(spec_min, color="#059669", linestyle="--", linewidth=1.2)
                ax2.axhline(spec_max, color="#059669", linestyle="--", linewidth=1.2)

            sns.boxplot(y=series, ax=ax2, color="#a5b4fc", width=0.3, fliersize=0, zorder=1)
            sns.stripplot(
                data=plot_df, y="Value", hue="Status", ax=ax2,
                palette={"In Spec": "#1e3a8a", "Out of Spec": "#dc2626"},
                size=4, alpha=0.7, jitter=0.15, zorder=2,
            )
            n_out = int((status == "Out of Spec").sum())
            ax2.set_title(f"{p} — Spec Range View (out-of-spec={n_out})", fontsize=10)
            ax2.set_ylabel("Value", fontsize=9)
            ax2.legend(fontsize=7, loc="best")
            fig2.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

        if spec_min is not None and spec_max is not None:
            out_pct = 100 * n_out / len(series) if len(series) else 0
            st.caption(f"**{p}** — Spec: Min = {spec_min}, Max = {spec_max} · {n_out} of {len(series)} "
                       f"values fall outside the spec range ({out_pct:.2f}%).")
        st.markdown("---")

    st.markdown("#### Value Trend Across Samples")
    tc1, tc2 = st.columns([1, 1])
    with tc1:
        sel_param = st.selectbox("Choose parameter to trend", list(param_cols.keys()))
    with tc2:
        bound_view = st.selectbox("Reference bounds to show", ["IQR Bounds (statistical)", "Spec Min/Max (actual)"])
    vc = param_cols[sel_param]["Val"]
    if vc:
        trend_df = pd.DataFrame({
            "Sample": range(1, len(df_raw) + 1),
            "Value": pd.to_numeric(df_raw[vc], errors="coerce"),
        })
        fig4 = px.line(trend_df, x="Sample", y="Value", markers=True)
        if bound_view.startswith("IQR"):
            row = iqr_df[iqr_df["Parameter"] == sel_param]
            if not row.empty:
                fig4.add_hline(y=row["Lower Bound"].values[0], line_dash="dot", line_color="red",
                               annotation_text="IQR Lower Bound")
                fig4.add_hline(y=row["Upper Bound"].values[0], line_dash="dot", line_color="red",
                               annotation_text="IQR Upper Bound")
                n_breach = int(((trend_df["Value"] < row["Lower Bound"].values[0]) |
                               (trend_df["Value"] > row["Upper Bound"].values[0])).sum())
                st.caption(f"{n_breach} of {len(trend_df)} samples fall outside the IQR bounds "
                          f"({100*n_breach/len(trend_df):.2f}%).")
        else:
            mn_col, mx_col = param_cols[sel_param]["Min"], param_cols[sel_param]["Max"]
            spec_min = spec_max = None
            if mn_col and mn_col in df_raw.columns:
                mn_s = pd.to_numeric(df_raw[mn_col], errors="coerce").dropna()
                if not mn_s.empty:
                    spec_min = mn_s.mode().iloc[0]
            if mx_col and mx_col in df_raw.columns:
                mx_s = pd.to_numeric(df_raw[mx_col], errors="coerce").dropna()
                if not mx_s.empty:
                    spec_max = mx_s.mode().iloc[0]
            if spec_min is not None and spec_max is not None:
                fig4.add_hline(y=spec_min, line_dash="dot", line_color="#059669", annotation_text="Spec Min")
                fig4.add_hline(y=spec_max, line_dash="dot", line_color="#059669", annotation_text="Spec Max")
                n_breach = int(((trend_df["Value"] < spec_min) | (trend_df["Value"] > spec_max)).sum())
                st.caption(f"{n_breach} of {len(trend_df)} samples fall outside the spec range "
                          f"({100*n_breach/len(trend_df):.2f}%).")
            else:
                st.info("No Min/Max spec columns found for this parameter.")
        st.plotly_chart(fig4, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 3: MODEL TRAINING
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("OK / NOK Classification Model (Random Forest)")

    y_raw = parse_result_col(df_raw["Result"])
    X = build_feature_matrix(df_raw, param_cols)

    valid_mask = X.notna().all(axis=1) & y_raw.notna()
    X_clean = X[valid_mask]
    y_clean = y_raw[valid_mask].astype(int)

    st.write(f"Usable rows after cleaning: **{len(X_clean)}** / {len(df_raw)}")

    if len(X_clean) < 20 or y_clean.nunique() < 2:
        st.warning("Not enough clean, class-balanced data to train reliably (need both OK and NOK rows, "
                    "and complete Min/Max/Val values). Showing what's available anyway.")
    else:
        Xtr, Xte, ytr, yte = train_test_split(
            X_clean, y_clean, test_size=test_size, random_state=42, stratify=y_clean
        )
        clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=None, random_state=42,
            class_weight="balanced", n_jobs=-1,
        )
        clf.fit(Xtr, ytr)
        ypred = clf.predict(Xte)

        st.session_state["trained_model"] = clf
        st.session_state["feature_cols"] = list(X_clean.columns)
        st.session_state["param_cols"] = param_cols

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{100*accuracy_score(yte, ypred):.1f}%")
        c2.metric("Precision (NOK)", f"{100*precision_score(yte, ypred, zero_division=0):.1f}%")
        c3.metric("Recall (NOK)", f"{100*recall_score(yte, ypred, zero_division=0):.1f}%")
        c4.metric("F1 (NOK)", f"{100*f1_score(yte, ypred, zero_division=0):.1f}%")

        st.markdown("#### Confusion Matrix")
        st.caption("Each cell shows the sample count and its share of the test set.")
        cm = confusion_matrix(yte, ypred)
        cm_pct = 100 * cm / cm.sum()
        cm_text = np.array([[f"{cm[i,j]}<br>({cm_pct[i,j]:.1f}%)" for j in range(cm.shape[1])]
                            for i in range(cm.shape[0])])
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm, x=["Pred OK", "Pred NOK"], y=["Actual OK", "Actual NOK"],
            text=cm_text, texttemplate="%{text}", colorscale="Blues",
            showscale=False,
        ))
        fig_cm.update_layout(height=350)
        st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown("#### Classification Report")
        report = classification_report(yte, ypred, target_names=["OK", "NOK"], output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()
        for col in ["precision", "recall", "f1-score"]:
            if col in report_df.columns:
                report_df[col] = (report_df[col] * 100).round(1).astype(str) + "%"
        st.dataframe(report_df, use_container_width=True)

        st.markdown("#### Which Parameter Drives NOK the Most?")
        st.caption("Shown as each parameter's overall contribution to the model's predictions.")
        # aggregate the underlying technical features (Val / NormPos / MarginLow / MarginHigh)
        # back up to their parent parameter, so the UI only shows parameter names
        imp_raw = pd.DataFrame({"Feature": X_clean.columns, "Importance": clf.feature_importances_})
        imp_raw["Parameter"] = imp_raw["Feature"].apply(
            lambda f: next((p for p in param_cols if f.startswith(p + "_")), f)
        )
        imp = imp_raw.groupby("Parameter", as_index=False)["Importance"].sum().sort_values(
            "Importance", ascending=False)
        imp["Contribution %"] = (100 * imp["Importance"] / imp["Importance"].sum()).round(1)
        st.dataframe(imp[["Parameter", "Contribution %"]], use_container_width=True)
        fig_imp = px.bar(imp, x="Contribution %", y="Parameter", orientation="h", color="Contribution %",
                          color_continuous_scale="Viridis", text="Contribution %")
        fig_imp.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_imp.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_imp, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 4: PREDICT NEW SAMPLE
# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Predict OK / NOK for a New Sample")
    if "trained_model" not in st.session_state:
        st.info("Train the model in the 'OK/NOK Model' tab first.")
    else:
        clf = st.session_state["trained_model"]
        feature_cols = st.session_state["feature_cols"]
        pc = st.session_state["param_cols"]

        st.markdown("Enter the **Val** (and Min/Max if you want normalized-position features) for each parameter:")
        input_vals = {}
        cols_ui = st.columns(2)
        i = 0
        for p, cols in pc.items():
            with cols_ui[i % 2]:
                st.markdown(f"**{p}**")
                v = st.number_input(f"{p} - Val", key=f"val_{p}", value=0.0, format="%.4f")
                mn = st.number_input(f"{p} - Min", key=f"min_{p}", value=0.0, format="%.4f")
                mx = st.number_input(f"{p} - Max", key=f"max_{p}", value=0.0, format="%.4f")
                input_vals[p] = (v, mn, mx)
            i += 1

        if st.button("Predict", type="primary"):
            row = {}
            for p, (v, mn, mx) in input_vals.items():
                row[f"{p}_Val"] = v
                span = (mx - mn) if (mx - mn) != 0 else np.nan
                row[f"{p}_NormPos"] = (v - mn) / span if span == span else np.nan
                row[f"{p}_MarginLow"] = v - mn
                row[f"{p}_MarginHigh"] = mx - v
            x_new = pd.DataFrame([row])
            x_new = x_new.reindex(columns=feature_cols, fill_value=np.nan)
            if x_new.isna().any(axis=1).iloc[0]:
                st.warning("Some features are missing/NaN (e.g. Min=Max gives divide-by-zero); prediction may be less reliable.")
                x_new = x_new.fillna(0)
            pred = clf.predict(x_new)[0]
            proba = clf.predict_proba(x_new)[0]
            label = "NOK" if pred == 1 else "OK"
            badge_class = "nok-badge" if pred == 1 else "ok-badge"
            st.markdown(f"### Result: <span class='{badge_class}'>{label}</span>", unsafe_allow_html=True)
            st.write(f"Confidence — OK: {proba[0]:.2%} | NOK: {proba[1]:.2%}")

# ---------------------------------------------------------------------------
# TAB 5: AI INSIGHT (GROQ)
# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("AI-Generated Pattern & Root-Cause Insight")
    st.caption("Uses Groq LLM to interpret the IQR stats, fail-rate trends, and feature importances.")

    if effective_api_key.startswith("PASTE") or not effective_api_key:
        st.warning("No Groq API key set. Paste it in the sidebar, or hardcode it in `GROQ_API_KEY` in app.py.")

    trained_model = st.session_state.get("trained_model")
    feature_cols = st.session_state.get("feature_cols")

    if st.button("🧠 Generate Insight Report", type="primary"):
        context, iqr_df, n_ok, n_nok = build_data_context(
            df_raw, param_cols, iqr_k, trained_model, feature_cols)

        prompt = f"""
Analyze this End-Of-Line (EOL) test dataset for a mechanical/electromechanical component.

{context}

Respond with ONLY a JSON object matching exactly this schema (no extra keys, no markdown fences):
{{
  "trend_direction": "improving" | "declining" | "stable",
  "verdict": "1-2 sentence overall verdict, citing the first-half vs second-half fail rate numbers",
  "major_parameters": [
      {{"parameter": "<name>", "detail": "why it's a major driver, citing its fail count / IQR bounds / contribution %"}}
  ],
  "root_cause": "2-4 sentence likely physical root cause (motor winding, gear mesh wear, sensor calibration drift, assembly misalignment, etc.) tied to the specific failing parameter(s)",
  "model_notes": "1-2 sentences on fail rate differences by Model, or 'No significant model-to-model difference observed.' if none",
  "recommended_actions": ["action 1", "action 2", "action 3"]
}}

List 1-3 items in major_parameters, ranked by contribution. Reference the actual numbers given above in every text field.
"""
        try:
            with st.spinner("Calling Groq LLM..."):
                insight_data = call_groq_json(prompt, effective_api_key, model=effective_model)
            st.session_state["last_insight_data"] = insight_data
            st.session_state["last_insight_meta"] = {
                "n_ok": int(n_ok), "n_nok": int(n_nok), "total": len(df_raw),
            }
        except InsightParseError as e:
            st.warning("The model didn't return valid JSON — showing raw response instead.")
            st.text(e.raw_text)
            st.session_state["last_insight"] = e.raw_text
        except Exception as e:
            st.error(f"Groq API call failed: {e}")
            st.caption("A 404 usually means the model name is invalid or was decommissioned, and a 400 "
                       "often means the model rejected structured-JSON mode — this app now falls back "
                       "automatically for that case. Check https://console.groq.com/docs/models for "
                       "currently supported models, or try a different model in the sidebar. Also verify your API key.")

    # ---------------------------------------------------------------------
    # RENDER: styled "LaTeX paper" report (rendered as isolated HTML,
    # not st.markdown, so the LLM's text can never break the layout)
    # ---------------------------------------------------------------------
    if "last_insight_data" in st.session_state:
        d = st.session_state["last_insight_data"]
        meta = st.session_state.get("last_insight_meta", {})

        def esc(v):
            return html_lib.escape(str(v)) if v is not None else ""

        trend = str(d.get("trend_direction", "stable")).lower()
        badge_bg, badge_fg, badge_border = {
            "improving": ("#dcfce7", "#166534", "#166534"),
            "declining": ("#fee2e2", "#991b1b", "#991b1b"),
        }.get(trend, ("#fef9c3", "#854d0e", "#854d0e"))
        trend_symbol = {
            "improving": "\u2193 Fail rate decreasing",
            "declining": "\u2191 Fail rate increasing",
        }.get(trend, "\u2192 Fail rate stable")

        params_rows = "".join(
            f"<tr><td><b>{esc(mp.get('parameter',''))}</b></td><td>{esc(mp.get('detail',''))}</td></tr>"
            for mp in d.get("major_parameters", [])
        ) or '<tr><td colspan="2">No dominant parameter identified.</td></tr>'

        actions_html = "".join(
            f'<div class="action-item"><span class="action-num">{i+1}</span>{esc(a)}</div>'
            for i, a in enumerate(d.get("recommended_actions", []))
        ) or "<p>No specific actions returned.</p>"

        full_html = f"""
        <html>
        <head>
        <style>
            body {{ margin: 0; padding: 0; }}
            .paper {{
                font-family: Georgia, "Times New Roman", serif;
                background: #ffffff;
                border: 1px solid #d1d5db;
                border-radius: 4px;
                padding: 2.2rem 2.6rem;
                color: #111827;
                line-height: 1.65;
                box-sizing: border-box;
            }}
            .paper-title {{ font-size: 1.5rem; text-align: center; margin: 0 0 0.1rem 0; font-weight: 700; }}
            .paper-subtitle {{ text-align: center; color: #6b7280; font-size: 0.85rem; margin-bottom: 1.6rem; font-style: italic; }}
            .section-num {{ font-weight: 700; color: #1f2937; border-bottom: 1px solid #9ca3af;
                             display: block; margin-top: 1.4rem; margin-bottom: 0.5rem; font-size: 1.05rem; }}
            .verdict-badge {{ display: inline-block; padding: 4px 16px; border-radius: 3px; font-weight: 700;
                               font-family: Georgia, serif; font-size: 0.95rem; letter-spacing: 0.02em;
                               background: {badge_bg}; color: {badge_fg}; border: 1px solid {badge_border}; }}
            .root-cause-box {{ background: #f9fafb; border-left: 4px solid #4338ca; padding: 0.9rem 1.2rem;
                                margin: 0.6rem 0; font-style: italic; }}
            table.param-table {{ width: 100%; border-collapse: collapse; margin: 0.6rem 0 1rem 0; font-size: 0.92rem; }}
            table.param-table th {{ background: #1f2937; color: #f9fafb; text-align: left; padding: 6px 10px; }}
            table.param-table td {{ border-bottom: 1px solid #e5e7eb; padding: 6px 10px; }}
            .action-item {{ margin: 4px 0; padding-left: 0.2rem; }}
            .action-num {{ display: inline-block; width: 22px; height: 22px; border-radius: 50%;
                            background: #4338ca; color: white; text-align: center; line-height: 22px;
                            font-size: 0.75rem; margin-right: 8px; font-family: Arial, sans-serif; }}
            p {{ margin: 0.4rem 0 0.8rem 0; }}
        </style>
        </head>
        <body>
            <div class="paper">
                <h1 class="paper-title">EOL Quality Insight Report</h1>
                <div class="paper-subtitle">Automatically generated from {esc(meta.get('total','-'))} samples
                ({esc(meta.get('n_ok','-'))} OK / {esc(meta.get('n_nok','-'))} NOK)</div>

                <span class="section-num">1. Overall Verdict</span>
                <span class="verdict-badge">{esc(trend.upper())} &nbsp;&middot;&nbsp; {esc(trend_symbol)}</span>
                <p>{esc(d.get('verdict',''))}</p>

                <span class="section-num">2. Major Contributing Parameters</span>
                <table class="param-table">
                    <tr><th>Parameter</th><th>Why it matters</th></tr>
                    {params_rows}
                </table>

                <span class="section-num">3. Likely Physical Root Cause</span>
                <div class="root-cause-box">{esc(d.get('root_cause',''))}</div>

                <span class="section-num">4. Model-Specific Notes</span>
                <p>{esc(d.get('model_notes',''))}</p>

                <span class="section-num">5. Recommended Actions</span>
                {actions_html}
            </div>
        </body>
        </html>
        """

        # height scales roughly with content so the iframe doesn't clip or leave dead space
        est_height = 480 + 60 * len(d.get("major_parameters", [])) + 40 * len(d.get("recommended_actions", []))
        components.html(full_html, height=min(est_height, 1400), scrolling=True)

        # A small formal (LaTeX-rendered) statement of the IQR rule used throughout the analysis
        st.markdown("&nbsp;")
        st.caption("Statistical rule underlying the IQR bounds referenced above:")
        st.latex(r"\text{Lower Bound} = Q_1 - k \cdot IQR \qquad \text{Upper Bound} = Q_3 + k \cdot IQR "
                 r"\qquad \text{where } IQR = Q_3 - Q_1,\ k = " + f"{iqr_k}")

        st.download_button(
            "Download Insight Report (.json)",
            data=json.dumps(d, indent=2),
            file_name="eol_ai_insight_report.json",
        )
    elif "last_insight" in st.session_state:
        st.download_button(
            "Download Insight Report (.txt)",
            data=st.session_state["last_insight"],
            file_name="eol_ai_insight_report.txt",
        )

    st.markdown("---")
    st.markdown("#### 💬 Ask Your Own Question About This Data")
    st.caption("The assistant only answers using the uploaded dataset's numbers — it will decline anything "
               "unrelated to this EOL data rather than guess.")
    user_question = st.text_area(
        "Your question",
        placeholder="e.g. Which model number has the worst hysteresis performance? "
                    "Is the fail rate getting worse over time? What should we check first on the line?",
        height=90,
    )
    if st.button("Ask", type="secondary"):
        if not user_question.strip():
            st.warning("Type a question first.")
        elif effective_api_key.startswith("PASTE") or not effective_api_key:
            st.warning("No Groq API key set.")
        else:
            context, iqr_df, n_ok, n_nok = build_data_context(
                df_raw, param_cols, iqr_k, trained_model, feature_cols)

            qa_prompt = f"""
Here is the full statistical summary of the uploaded End-Of-Line (EOL) test dataset:

{context}

USER QUESTION: {user_question.strip()}

Answer ONLY using the data summary above. If the question cannot be answered from this data (e.g. it asks
about something not present in the summary, or is unrelated to this EOL dataset/manufacturing quality
context), say clearly that you cannot answer it from the uploaded data rather than guessing or making
anything up. Do not invent numbers, parameters, or facts that are not in the summary above. Be concise
and cite the specific numbers you're using.
"""
            try:
                with st.spinner("Thinking..."):
                    answer = call_groq(
                        qa_prompt, effective_api_key, model=effective_model,
                        temperature=1.0,  # as high as reasonably possible while staying coherent
                    )
                st.markdown(answer)
            except Exception as e:
                st.error(f"Groq API call failed: {e}")
                st.caption("Check that your API key is valid and `requests` can reach api.groq.com.")
