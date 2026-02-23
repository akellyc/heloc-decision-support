import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="HELOC Credit Risk Dashboard", layout="wide")

st.markdown("## 🏦 Simon Bank of Rochester")
st.markdown("### Credit Risk Pre-Screening Dashboard")
st.caption(
    "Initial screening tool: predicts whether an application should be sent to a loan officer for review. "
    "If denied, provides a clear explanation and actionable improvement tips."
)
st.markdown("---")

MODEL_PATH = "model.joblib"

# -----------------------------
# Load model (pipeline)
# -----------------------------
if not os.path.exists(MODEL_PATH):
    st.error(
        "model.joblib not found. Please upload your trained sklearn pipeline (preprocessing + RandomForestClassifier) "
        "to the same folder as app.py."
    )
    st.stop()

pipeline = joblib.load(MODEL_PATH)

# Try to infer expected feature names (best case)
expected_features = getattr(pipeline, "feature_names_in_", None)
if expected_features is not None:
    expected_features = list(expected_features)

# -----------------------------
# Dictionary
# -----------------------------

DICT_PATH = "heloc_data_dictionary.xlsx"

@st.cache_data
def load_dictionary(path: str) -> dict:
    import pandas as pd
    if not os.path.exists(path):
        return {}
    d = pd.read_excel(path)
    return dict(zip(d["Variable Names"], d["Description"]))

DESC = load_dictionary(DICT_PATH)

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("🧾 Applicant Inputs")
threshold = st.sidebar.slider(
    "Decision threshold (probability to send to loan officer)",
    min_value=0.05, max_value=0.95, value=0.70, step=0.05
)

st.sidebar.divider()
st.sidebar.caption("Tip: The bank can tune this threshold to balance risk and operational workload.")

# ----
# Default UI config (EDIT THESE to match your dataset meaningfully)
# If expected_features exists, make sure all of them appear here (or the app will warn you).
# ----

FEATURE_META = {
    # -----------------------------
    # Credit Quality / Payment behavior
    # -----------------------------
    "ExternalRiskEstimate": {
        "group": "1) Credit quality & risk score",
        "label": "External Risk Estimate (overall risk score)",
        "type": "int", "min": 0, "max": 100, "default": 70,
    },
    "PercentTradesNeverDelq": {
        "group": "1) Credit quality & risk score",
        "label": "% Trades Never Delinquent",
        "type": "int", "min": 0, "max": 100, "default": 85,
    },
    "MSinceMostRecentDelq": {
        "group": "1) Credit quality & risk score",
        "label": "Months Since Most Recent Delinquency",
        "type": "int", "min": -9, "max": 400, "default": 60,
    },
    "MaxDelq2PublicRecLast12M": {
        "group": "1) Credit quality & risk score",
        "label": "Worst delinquency / public record (last 12M) — severity code",
        "type": "int", "min": 0, "max": 9, "default": 1,
    },
    "MaxDelqEver": {
        "group": "1) Credit quality & risk score",
        "label": "Worst delinquency ever — severity code",
        "type": "int", "min": 0, "max": 9, "default": 2,
    },
    "NumTrades60Ever2DerogPubRec": {
        "group": "1) Credit quality & risk score",
        "label": "# Trades 60+ days delinquent or derog/public record (ever)",
        "type": "int", "min": 0, "max": 50, "default": 0,
    },

    # -----------------------------
    # Credit History / Depth
    # -----------------------------
    "MSinceOldestTradeOpen": {
        "group": "2) Credit history & depth",
        "label": "Months Since Oldest Trade Open (credit history length)",
        "type": "int", "min": -9, "max": 1000, "default": 200,
    },
    "MSinceMostRecentTradeOpen": {
        "group": "2) Credit history & depth",
        "label": "Months Since Most Recent Trade Open",
        "type": "int", "min": -9, "max": 400, "default": 20,
    },
    "AverageMInFile": {
        "group": "2) Credit history & depth",
        "label": "Average Months in File (average account age)",
        "type": "int", "min": -9, "max": 400, "default": 60,
    },
    "NumTotalTrades": {
        "group": "2) Credit history & depth",
        "label": "# Total Trades (all accounts)",
        "type": "int", "min": 0, "max": 150, "default": 30,
    },
    "NumSatisfactoryTrades": {
        "group": "2) Credit history & depth",
        "label": "# Satisfactory Trades (in good standing)",
        "type": "int", "min": 0, "max": 100, "default": 20,
    },

    # -----------------------------
    # Utilization / Balances
    # -----------------------------
    "NetFractionRevolvingBurden": {
        "group": "3) Utilization & balances",
        "label": "Net Fraction Revolving Burden (revolving utilization proxy)",
        "type": "float", "min": 0.0, "max": 1.0, "default": 0.40,
    },
    "NetFractionInstallBurden": {
        "group": "3) Utilization & balances",
        "label": "Net Fraction Installment Burden (installment burden proxy)",
        "type": "float", "min": 0.0, "max": 1.0, "default": 0.30,
    },
    "PercentTradesWBalance": {
        "group": "3) Utilization & balances",
        "label": "% Trades With Balance",
        "type": "int", "min": 0, "max": 100, "default": 60,
    },
    "NumBank2NatlTradesWHighUtilization": {
        "group": "3) Utilization & balances",
        "label": "# Bank/National Trades With High Utilization",
        "type": "int", "min": 0, "max": 20, "default": 1,
    },
    "NumRevolvingTradesWBalance": {
        "group": "3) Utilization & balances",
        "label": "# Revolving Trades With Balance",
        "type": "int", "min": 0, "max": 50, "default": 5,
    },
    "NumInstallTradesWBalance": {
        "group": "3) Utilization & balances",
        "label": "# Installment Trades With Balance",
        "type": "int", "min": 0, "max": 50, "default": 3,
    },
    "PercentInstallTrades": {
        "group": "3) Utilization & balances",
        "label": "% Installment Trades",
        "type": "int", "min": 0, "max": 100, "default": 40,
    },

    # -----------------------------
    # Recent behavior / Credit seeking
    # -----------------------------
    "NumTradesOpeninLast12M": {
        "group": "4) Recent activity & credit seeking",
        "label": "# Trades Opened in Last 12 Months",
        "type": "int", "min": 0, "max": 50, "default": 5,
    },
    "NumInqLast6Mexcl7days": {
        "group": "4) Recent activity & credit seeking",
        "label": "# Inquiries in Last 6 Months (excluding last 7 days)",
        "type": "int", "min": 0, "max": 20, "default": 1,
    },
    "MSinceMostRecentInqexcl7days": {
        "group": "4) Recent activity & credit seeking",
        "label": "Months Since Most Recent Inquiry (excluding last 7 days)",
        "type": "int", "min": -9, "max": 400, "default": 10,
    },
}

def build_help_text(feature_name: str) -> str:
    """Tooltip text shown in the (?) icon for each input."""
    definition = DESC.get(feature_name, "").strip()
    if not definition:
        definition = "No description available."

    meta = FEATURE_META.get(feature_name, {})
    fmin = meta.get("min", None)
    fmax = meta.get("max", None)
    fdef = meta.get("default", None)

    range_txt = ""
    if fmin is not None and fmax is not None:
        range_txt = f"\n\nRange: {fmin} to {fmax}"
    if fdef is not None:
        range_txt += f"\nDefault: {fdef}"

    return (
        f"Technical name: {feature_name}\n\n"
        f"Definition: {definition}"
        f"{range_txt}"
    )

expected_features = getattr(pipeline, "feature_names_in_", None)

feature_list_for_ui = (
    list(expected_features)
    if expected_features is not None
    else list(FEATURE_META.keys())
)

def render_control(feature: str, meta: dict):

    # 👇 usa la nueva función completa
    help_text = build_help_text(feature)

    # 👇 solo nombre humano (sin technical name visible)
    label = meta["label"]

    if meta["type"] == "int":
        return st.sidebar.number_input(
            label,
            min_value=int(meta["min"]),
            max_value=int(meta["max"]),
            value=int(meta["default"]),
            step=1,
            help=help_text
        )
    else:
        return st.sidebar.number_input(
            label,
            min_value=float(meta["min"]),
            max_value=float(meta["max"]),
            value=float(meta["default"]),
            step=0.01,
            format="%.2f",
            help=help_text
        )

# Orden por grupos
groups = {}
for f in feature_list_for_ui:
    if f in FEATURE_META:
        g = FEATURE_META[f]["group"]
        groups.setdefault(g, []).append(f)

inputs = {}
for g in ["1) Credit quality & risk score", "2) Credit history & depth",
          "3) Utilization & balances", "4) Recent activity & credit seeking"]:
    if g in groups:
        st.sidebar.subheader(g)
        for f in groups[g]:
            inputs[f] = render_control(f, FEATURE_META[f])
        st.sidebar.divider()

X_input = pd.DataFrame([inputs])

# -----------------------------
# LOCAL DRIVER FUNCTIONS (PRO VERSION)
# -----------------------------
def build_baseline_row_from_applicant(X_input, feature_list):
    """
    Baseline = median-like synthetic profile based on applicant.
    Slightly perturbs values so deltas are not artificially zero.
    """
    base = {}
    x0 = X_input.iloc[0]

    for f in feature_list:
        val = x0[f]

        if isinstance(val, (int, float)):
            # move 20% toward neutral midpoint
            base[f] = val * 0.8
        else:
            base[f] = val

    return pd.DataFrame([base])

def compute_local_drivers(pipeline, X_input, feature_list, baseline_row, top_k=5):

    X0 = X_input[feature_list].copy()
    base_prob = float(pipeline.predict_proba(X0)[0, 1])

    rows = []
    for f in feature_list:
        X_pert = X0.copy()
        X_pert.loc[:, f] = baseline_row.loc[0, f]
        pert_prob = float(pipeline.predict_proba(X_pert)[0, 1])

        delta = base_prob - pert_prob
        rows.append({"feature": f, "delta_prob": delta})

    drivers = pd.DataFrame(rows)
    drivers["abs_delta"] = drivers["delta_prob"].abs()
    drivers = drivers.sort_values("abs_delta", ascending=False).head(top_k).copy()

    drivers["direction"] = drivers["delta_prob"].apply(
        lambda x: "↑ helps approval" if x >= 0 else "↓ hurts approval"
    )

    return base_prob, drivers

# -----------------------------
# Predict
# -----------------------------
if not hasattr(pipeline, "predict_proba"):
    st.error("Loaded pipeline does not support predict_proba(). Please ensure it's a classifier pipeline.")
    st.stop()

# --- Ensure columns match training schema (same names + same order) ---
expected = getattr(pipeline, "feature_names_in_", None)

if expected is not None:
    # Add any missing expected cols (shouldn't happen if UI is complete)
    for col in expected:
        if col not in X_input.columns:
            X_input[col] = np.nan

    # Keep ONLY expected cols and in the right order
    X_input = X_input.loc[:, list(expected)]

prob_positive = float(pipeline.predict_proba(X_input)[0, 1])
decision = "Positive" if prob_positive >= threshold else "Negative"

baseline_row = build_baseline_row_from_applicant(X_input, feature_list_for_ui)
base_prob, local_drivers = compute_local_drivers(
    pipeline=pipeline,
    X_input=X_input,
    feature_list=feature_list_for_ui,
    baseline_row=baseline_row,
    top_k=5
)

# -----------------------------
# Layout: Results + Explanation
# -----------------------------
left, right = st.columns([1.05, 0.95], gap="large")

with left:

    st.subheader("1) Credit decision summary")

    # Top row metrics
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Model Approval Probability", f"{prob_positive:.2%}")

    with col2:
        st.metric("Policy Threshold", f"{threshold:.0%}")

    approval_buffer = prob_positive - threshold

    if prob_positive >= threshold:

        if approval_buffer >= 0.15:
            level = "Low Risk"
            icon = "🟢"
        else:
            level = "Moderate Risk"
            icon = "🟡"

        st.success(
            f"{icon} **Approved for Review**\n\n"
            f"Risk Level: **{level}**\n\n"
            f"Approval buffer: **+{approval_buffer:.2%} above threshold**"
        )

    else:

        st.error(
            f"🔴 **Not Approved (Initial Screening)**\n\n"
            f"Shortfall: **{approval_buffer:.2%} below threshold**"
        )
        
    st.caption(
    "This probability reflects model-based screening only. "
    "Final underwriting may incorporate income verification, "
    "collateral assessment, policy overrides, and additional documentation."
    )

st.subheader("2) Applicant snapshot")
st.dataframe(X_input, use_container_width=True)

# -----------------------------
# Glossary Section
# -----------------------------
with st.expander("📘 Glossary: What do these inputs mean?"):
    glossary_rows = []
    for f in feature_list_for_ui:
        if f in FEATURE_META:
            glossary_rows.append({
                "Feature (technical name)": f,
                "Plain-English name": FEATURE_META[f]["label"],
                "Category": FEATURE_META[f]["group"],
                "Definition (data dictionary)": DESC.get(f, "")
            })
    st.dataframe(pd.DataFrame(glossary_rows), use_container_width=True)

st.subheader("3) What drove this decision? (Local)")
st.caption("Top factors that most influenced THIS applicant (relative to baseline).")

st.info(
    "How to read this chart:\n"
    "- This is a **local explanation**: it shows what influenced THIS applicant's approval probability.\n"
    "- Bars to the **right (green)** increase approval probability.\n"
    "- Bars to the **left (red)** decrease approval probability.\n"
    "- Values are shown in **basis points (bps)** where 100 bps = 1 percentage point.\n\n"
    "Example: +500 bps means the feature increased approval probability by 5.0 percentage points "
    "relative to the model baseline."
)

def pretty_name(f):
    if f in FEATURE_META:
        return FEATURE_META[f]["label"]   
    return f

local_display = local_drivers.copy()
local_display["Feature"] = local_display["feature"].apply(pretty_name)
local_display["Impact (basis points)"] = local_display["delta_prob"].map(lambda x: f"{x*10000:+.1f} bps")
local_display["Direction"] = local_display["direction"]

st.dataframe(
local_display[["Feature", "Impact (basis points)", "Direction"]],
use_container_width=True,
hide_index=True
)

# ===== LOCAL DRIVER BAR CHART (from formatted table) =====

import matplotlib.pyplot as plt
import re

# 1️⃣ Copiamos la tabla que ya estás mostrando
plot_df = local_display.copy()

# 2️⃣ Convertimos "Impact (basis points)" de string a número
def extract_bps(value):
    # Extrae número tipo "+501.4 bps"
    match = re.search(r"[-+]?\d+\.?\d*", str(value))
    return float(match.group()) if match else 0.0

plot_df["impact_bps"] = plot_df["Impact (basis points)"].apply(extract_bps)

# 3️⃣ Ordenamos para que negativos queden abajo y positivos arriba
plot_df = plot_df.sort_values("impact_bps")

# 4️⃣ Colores según signo
colors = [
    "#2e7d32" if v > 0 else "#c62828" if v < 0 else "#9e9e9e"
    for v in plot_df["impact_bps"]
]

# 5️⃣ Crear gráfico horizontal
fig, ax = plt.subplots(figsize=(10, 5))

plot_df["pretty"] = plot_df["feature"].apply(pretty_name)

bars = ax.barh(
    plot_df["pretty"],   
    plot_df["impact_bps"],
    color=colors
)

ax.axvline(0, linewidth=1)  # línea central en 0
ax.set_xlabel("Impact on approval probability (basis points)")
ax.set_title("Top Local Drivers — Applicant vs Baseline")

# 6️⃣ Agregar labels al final de cada barra
for bar, val in zip(bars, plot_df["impact_bps"]):
    y = bar.get_y() + bar.get_height() / 2
    x = bar.get_width()

    ha = "left" if val >= 0 else "right"
    offset = 8 if val >= 0 else -8

    ax.annotate(
        f"{val:+.1f} bps",
        xy=(x, y),
        xytext=(offset, 0),
        textcoords="offset points",
        va="center",
        ha=ha,
        fontsize=10
    )

plt.tight_layout()
st.pyplot(fig, use_container_width=True)


st.subheader("4) Model drivers (Global)")
st.caption(
    "These are the model’s overall top drivers across the dataset (not personalized). "
    "Useful for auditors and technical stakeholders."
)
st.info(
    "How to read this chart:\n"
    "- This is **global** importance: it summarizes what the model relied on most **across the full dataset** (not this specific applicant).\n"
    "- A larger bar means the model used that variable more often / more strongly when making predictions.\n"
    "- It does **not** say whether the feature increases or decreases approval—only how influential it is overall.\n\n"
)

# Attempt to access the RandomForest inside the pipeline
rf = None
if hasattr(pipeline, "named_steps") and "classifier" in pipeline.named_steps:
    rf = pipeline.named_steps["classifier"]

if rf is None or not hasattr(rf, "feature_importances_"):
    st.info(
        "Feature importance not available. Ensure the pipeline ends with RandomForestClassifier "
        "and is accessible via pipeline.named_steps['classifier']."
    )
else:
    importances = rf.feature_importances_

    if expected_features is not None:
        feat_names = expected_features
    else:
        feat_names = [f"feature_{i}" for i in range(len(importances))]

    fi = pd.DataFrame({
        "feature": feat_names,
        "importance": importances
    }).sort_values("importance", ascending=False).head(10)

    def pretty_name_global(f):
        if f in FEATURE_META:
            return FEATURE_META[f]["label"]
        return f

    fi["pretty"] = fi["feature"].apply(pretty_name_global)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.barh(fi["pretty"][::-1], fi["importance"][::-1])

    ax.set_xlabel("Relative importance (higher = more influence in the model)")
    ax.set_title("Top 10 model drivers (global)")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# ============================================================
# 5) Decision Explanation & Next Steps (Personalized) — FINAL REPORT STYLE
# (Pega este bloque donde va tu punto 5)
# Requiere que ya existan:
#   - prob_positive (float 0-1)
#   - threshold (float 0-1)
#   - decision (str: "Positive" / "Negative")  (o lo ajustas abajo)
#   - plot_df  -> dataframe del punto 3 con columnas: ["Feature", "impact_bps"] (o similar)
#   - global_importance_df -> df del punto 4 con columnas: ["Feature", "Importance"]
#   - FEATURE_META dict (para pretty labels)
# ============================================================

st.subheader("5) Decision explanation & Next steps")

# ---------- helpers ----------
def pretty_name(feature: str) -> str:
    if feature in FEATURE_META:
        return FEATURE_META[feature]["label"]
    return feature

def fmt_pp(x: float) -> str:
    # percentage points
    return f"{x*100:.2f} pp"

def fmt_pct(x: float) -> str:
    return f"{x*100:.1f}%"

# ---------- decision + buffer ----------
# Si ya tienes decision definido en tu código, comenta la línea de abajo.
decision = "Positive" if prob_positive >= threshold else "Negative"

buffer_pp = prob_positive - threshold  # in probability units
buffer_bps = buffer_pp * 10000         # basis points

# Banner corto, no repetitivo
if decision == "Positive":
    st.success("✅ Approved for Review — Your application passes initial screening.")
else:
    st.error("❌ Not Approved (Initial Screening) — The application does not meet the current screening policy.")

st.caption(
    f"Model approval probability: {fmt_pct(prob_positive)}  |  "
    f"Policy threshold: {fmt_pct(threshold)}  |  "
    f"Buffer: {fmt_pp(buffer_pp)} ({buffer_bps:+.0f} bps)"
)

st.divider()

# ---------- Build LOCAL summary (from plot_df) ----------
# Esperamos plot_df con columnas: Feature (tech name o ya pretty) y impact_bps
local_df = plot_df.copy()

# Asegurar que Feature sea technical name para mapear a pretty.
# Si tu plot_df ya tiene pretty names, igual funciona (no rompe).
if "Feature" not in local_df.columns:
    # por si tu df usa otra columna, ajusta aquí
    local_df = local_df.rename(columns={"feature": "Feature", "impact": "impact_bps"})

local_df["pretty"] = local_df["Feature"].apply(pretty_name)

# impactos positivos/negativos
pos_local = local_df[local_df["impact_bps"] > 0].sort_values("impact_bps", ascending=False).head(2)
neg_local = local_df[local_df["impact_bps"] < 0].sort_values("impact_bps", ascending=True).head(2)

# ---------- Build GLOBAL summary (from global_importance_df) ----------
global_df = fi.copy()
if "Feature" not in global_df.columns:
    global_df = global_df.rename(columns={"feature": "Feature", "importance": "Importance"})

global_df["pretty"] = global_df["Feature"].apply(pretty_name)
top_global = global_df.sort_values("Importance", ascending=False).head(3)["pretty"].tolist()

# ---------- Final report tabs (short, human, no repetition) ----------
tab_local, tab_global, tab_next = st.tabs(["🔎 Local (this applicant)", "🌍 Global (model drivers)", "➡️ Next steps"])

with tab_local:
    st.markdown("### Local conclusion (personalized)")
    if decision == "Positive":
        st.markdown(
            f"**Good news:** the applicant is **above policy by {fmt_pp(buffer_pp)}**. "
            "The model sees a risk profile that clears initial screening."
        )
    else:
        st.markdown(
            f"**Key takeaway:** the applicant is **below policy by {fmt_pp(abs(buffer_pp))}**. "
            "The model flagged a risk profile that does not clear initial screening today."
        )

    # 1–2 lines: what helped / hurt (no tablas)
    local_lines = []
    if len(pos_local) > 0:
        pos_text = ", ".join([f"**{r.pretty}**" for r in pos_local.itertuples()])
        local_lines.append(f"**What helped approval:** {pos_text}.")
    if len(neg_local) > 0:
        neg_text = ", ".join([f"**{r.pretty}**" for r in neg_local.itertuples()])
        local_lines.append(f"**What hurt approval:** {neg_text}.")

    if local_lines:
        for line in local_lines:
            st.write(line)
    else:
        st.write("No strong local drivers were detected for this input (impacts close to baseline).")

    st.caption("Local = drivers for *this* applicant vs the model baseline.")

with tab_global:
    st.markdown("### Global context (not personalized)")
    if top_global:
        st.write(
            "Across the full dataset, the model relies most on: "
            + ", ".join([f"**{x}**" for x in top_global])
            + "."
        )
    st.caption(
        "Global = overall feature importance across the dataset. "
        "Useful for governance/audit; not a reason by itself for an individual applicant."
    )

with tab_next:
    st.markdown("### Recommended next steps")
    if decision == "Positive":
        st.write(
            "🎉 **Congratulations!** Your application passed initial screening. "
            "Next, a loan officer will review documentation and policy requirements (income verification, collateral details, and any required checks)."
        )
        st.markdown("**What you can do now:**")
        st.markdown(
            "- Prepare recent income documentation and any supporting files.\n"
            "- Be ready to clarify outstanding balances or recent credit activity if requested.\n"
            "- The final decision may still change after full underwriting review."
        )
    else:
        st.write(
            "The application did not pass initial screening today. "
            "If you plan to reapply, focus on the items that most reduced approval probability."
        )

        # Convert top negatives into action-style tips (short + safe)
        if len(neg_local) > 0:
            st.markdown("**Priority focus areas (based on your profile):**")
            for r in neg_local.itertuples():
                # Simple human tip templates (no over-promising)
                if "balance" in r.pretty.lower() or "utilization" in r.pretty.lower():
                    st.markdown(f"- **{r.pretty}:** aim to reduce revolving balances/utilization over time before reapplying.")
                elif "inquir" in r.pretty.lower() or "inquiries" in r.pretty.lower():
                    st.markdown(f"- **{r.pretty}:** limit new credit inquiries for a period to show stability.")
                elif "trade" in r.pretty.lower() or "opened" in r.pretty.lower():
                    st.markdown(f"- **{r.pretty}:** avoid opening multiple new accounts in a short window.")
                else:
                    st.markdown(f"- **{r.pretty}:** consider improving this metric before reapplying.")
        else:
            st.markdown("- Consider waiting a few months and reapplying after your profile changes meaningfully.")

        st.caption(
            "Note: This is model-based initial screening guidance only. "
            "Final credit decisions require full underwriting review."
        )
