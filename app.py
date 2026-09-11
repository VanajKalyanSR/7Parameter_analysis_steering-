"""
EOL (End-of-Line) Quality Analysis Dashboard
=============================================
- Loads EOL test report data (7 parameters, each with Min/Max/Val/Result)
- Trains a classifier to predict overall Result (OK / NOK)
- Runs IQR-based outlier analysis per parameter
- Uses Groq LLM to narrate patterns/trends and root-cause hints
"""

import io
import json
import numpy as np
import pandas as pd
import streamlit as st
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
]

BASE_COLS = ["SAMPLE", "Result", "Model"]

# ---------------------------------------------------------------------------
# GROQ API KEY  -- put your key here
# ---------------------------------------------------------------------------
GROQ_API_KEY = ""
GROQ_MODEL = "openai/gpt-oss-120b"   # change if you prefer another Groq-hosted model

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
    available = []
    for p in PARAMETERS:
        cols = find_param_columns(df, p)
        if cols["Val"] is not None:
            available.append(p)
    return available


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

st.sidebar.markdown("---")
iqr_k = st.sidebar.slider("IQR multiplier (k)", 1.0, 3.0, 1.5, 0.1)
test_size = st.sidebar.slider("Test set size (%)", 10, 40, 20, 5) / 100
n_estimators = st.sidebar.slider("Random Forest trees", 50, 500, 200, 50)

st.markdown('<div class="main-header">EOL Quality Analytics Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">OK/NOK classification · IQR outlier analysis · LLM-generated defect insight</div>', unsafe_allow_html=True)

if uploaded_file is None:
    st.info("👈 Upload a CSV/Excel EOL report to begin. Expected columns: SAMPLE, Result, Model, "
            "and for each of the 7 parameters: `<param>_Result`, `<param>_Max`, `<param>_Val`, `<param>_Min`.")
    with st.expander("Expected parameter names"):
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
    c2.metric("OK", n_ok)
    c3.metric("NOK", n_nok, delta=f"{100*n_nok/total:.1f}% fail rate", delta_color="inverse")
    c4.metric("Parameters Detected", len(param_cols))

    st.markdown("#### Preview")
    st.dataframe(df_raw.head(20), use_container_width=True)

    st.markdown("#### Result Distribution")
    fig = px.pie(names=["OK", "NOK"], values=[n_ok, n_nok],
                 color=["OK", "NOK"], color_discrete_map={"OK": "#22c55e", "NOK": "#ef4444"},
                 hole=0.5)
    st.plotly_chart(fig, use_container_width=True)

    if "Model" in df_raw.columns:
        st.markdown("#### Fail Rate by Model")
        tmp = df_raw.copy()
        tmp["_fail"] = (~result_upper.isin(["OK", "PASS", "1"])).astype(int)
        by_model = tmp.groupby("Model")["_fail"].agg(["count", "sum"]).reset_index()
        by_model["fail_rate_%"] = (100 * by_model["sum"] / by_model["count"]).round(2)
        by_model.columns = ["Model", "Total", "NOK Count", "Fail Rate %"]
        st.dataframe(by_model.sort_values("Fail Rate %", ascending=False), use_container_width=True)
        fig2 = px.bar(by_model, x="Model", y="Fail Rate %", color="Fail Rate %",
                      color_continuous_scale="Reds")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Which parameter fails most often?")
    fail_counts = {}
    for p, cols in param_cols.items():
        rc = cols["Result"]
        if rc and rc in df_raw.columns:
            v = df_raw[rc].astype(str).str.strip().str.upper()
            fail_counts[p] = int((~v.isin(["OK", "PASS", "1"])).sum())
    fc_df = pd.DataFrame(list(fail_counts.items()), columns=["Parameter", "Fail Count"]).sort_values(
        "Fail Count", ascending=False)
    st.dataframe(fc_df, use_container_width=True)
    fig3 = px.bar(fc_df, x="Parameter", y="Fail Count", color="Fail Count", color_continuous_scale="OrRd")
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
            st.caption(f"**{p}** — Spec: Min = {spec_min}, Max = {spec_max} · {n_out} of {len(series)} "
                       f"values fall outside the spec range.")
        st.markdown("---")

    st.markdown("#### Value Trend Across Samples")
    sel_param = st.selectbox("Choose parameter to trend", list(param_cols.keys()))
    vc = param_cols[sel_param]["Val"]
    if vc:
        trend_df = pd.DataFrame({
            "Sample": range(1, len(df_raw) + 1),
            "Value": pd.to_numeric(df_raw[vc], errors="coerce"),
        })
        fig4 = px.line(trend_df, x="Sample", y="Value", markers=True)
        row = iqr_df[iqr_df["Parameter"] == sel_param]
        if not row.empty:
            fig4.add_hline(y=row["Lower Bound"].values[0], line_dash="dot", line_color="red")
            fig4.add_hline(y=row["Upper Bound"].values[0], line_dash="dot", line_color="red")
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
        c1.metric("Accuracy", f"{accuracy_score(yte, ypred):.3f}")
        c2.metric("Precision (NOK)", f"{precision_score(yte, ypred, zero_division=0):.3f}")
        c3.metric("Recall (NOK)", f"{recall_score(yte, ypred, zero_division=0):.3f}")
        c4.metric("F1 (NOK)", f"{f1_score(yte, ypred, zero_division=0):.3f}")

        cm = confusion_matrix(yte, ypred)
        fig_cm = px.imshow(cm, text_auto=True, x=["Pred OK", "Pred NOK"], y=["Actual OK", "Actual NOK"],
                            color_continuous_scale="Blues")
        st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown("#### Classification Report")
        report = classification_report(yte, ypred, target_names=["OK", "NOK"], output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

        st.markdown("#### Feature Importance (which parameter drives NOK the most)")
        imp = pd.DataFrame({
            "Feature": X_clean.columns,
            "Importance": clf.feature_importances_,
        }).sort_values("Importance", ascending=False)
        st.dataframe(imp, use_container_width=True)
        fig_imp = px.bar(imp.head(15), x="Importance", y="Feature", orientation="h", color="Importance",
                          color_continuous_scale="Viridis")
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

    if st.button("🧠 Generate Insight Report", type="primary"):
        iqr_df = compute_iqr_bounds(df_raw, param_cols, k=iqr_k)
        result_upper = df_raw["Result"].astype(str).str.strip().str.upper()
        n_ok = (result_upper.isin(["OK", "PASS", "1"])).sum()
        n_nok = len(df_raw) - n_ok

        fail_counts = {}
        for p, cols in param_cols.items():
            rc = cols["Result"]
            if rc and rc in df_raw.columns:
                v = df_raw[rc].astype(str).str.strip().str.upper()
                fail_counts[p] = int((~v.isin(["OK", "PASS", "1"])).sum())

        # trend: split data into halves (chronological, as-uploaded order) and compare fail rate
        half = len(df_raw) // 2
        first_half_fail = (~result_upper.iloc[:half].isin(["OK", "PASS", "1"])).mean() * 100 if half > 0 else None
        second_half_fail = (~result_upper.iloc[half:].isin(["OK", "PASS", "1"])).mean() * 100 if half > 0 else None

        feat_imp_text = ""
        if "trained_model" in st.session_state:
            clf = st.session_state["trained_model"]
            feature_cols = st.session_state["feature_cols"]
            imp = pd.DataFrame({"Feature": feature_cols, "Importance": clf.feature_importances_}) \
                    .sort_values("Importance", ascending=False).head(10)
            feat_imp_text = imp.to_string(index=False)

        model_breakdown = ""
        if "Model" in df_raw.columns:
            tmp = df_raw.copy()
            tmp["_fail"] = (~result_upper.isin(["OK", "PASS", "1"])).astype(int)
            by_model = tmp.groupby("Model")["_fail"].agg(["count", "sum"]).reset_index()
            by_model["fail_rate_%"] = (100 * by_model["sum"] / by_model["count"]).round(2)
            model_breakdown = by_model.to_string(index=False)

        prompt = f"""
Analyze this End-Of-Line (EOL) test dataset for a mechanical/electromechanical component.

OVERALL: {len(df_raw)} samples, {n_ok} OK, {n_nok} NOK ({100*n_nok/len(df_raw):.2f}% fail rate).

FAIL RATE TREND (first half vs second half of dataset, in row order):
First half fail rate: {first_half_fail:.2f}% | Second half fail rate: {second_half_fail:.2f}%

PER-PARAMETER FAIL COUNTS (out of spec occurrences):
{json.dumps(fail_counts, indent=2)}

IQR BOUNDS PER PARAMETER (based on measured Val):
{iqr_df.to_string(index=False)}

{"FEATURE IMPORTANCE FROM TRAINED CLASSIFIER (top drivers of NOK):" if feat_imp_text else ""}
{feat_imp_text}

{"FAIL RATE BY MODEL:" if model_breakdown else ""}
{model_breakdown}

Please provide:
1. **Overall Verdict** — is quality trending better, worse, or stable across the dataset?
2. **Major Contributing Parameter(s)** — which parameter(s) drive most NOKs, and by how much (cite numbers)?
3. **Likely Physical Root Cause** — for a torque/hysteresis/balance EOL test, what mechanical/electrical causes
   typically produce this kind of failure pattern (e.g. motor winding issue, gear mesh wear, sensor calibration drift,
   assembly misalignment)? Tie this to which specific parameter(s) are failing.
4. **Model-specific notes** — if fail rate differs meaningfully by Model, call it out.
5. **Recommended Action** — 2-3 concrete next steps for the quality/process engineering team.

Be specific and reference the actual numbers above. Keep it under 400 words.
"""
        try:
            with st.spinner("Calling Groq LLM..."):
                insight = call_groq(prompt, effective_api_key)
            st.markdown(insight)
            st.session_state["last_insight"] = insight
        except Exception as e:
            st.error(f"Groq API call failed: {e}")
            st.caption("Check that your API key is valid and `requests` can reach api.groq.com.")

    if "last_insight" in st.session_state:
        st.download_button(
            "Download Insight Report (.txt)",
            data=st.session_state["last_insight"],
            file_name="eol_ai_insight_report.txt",
        )
