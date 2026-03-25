code = '''import streamlit as st
import pandas as pd
from models import ALL_CASES, get_case
from forensics import run_full_autopsy, find_actual_failures
from visualisations import (chart_shap_waterfall, chart_confidence_gauge, 
                            chart_neighbour_comparison, chart_feature_profile, chart_latent_space)

st.set_page_config(layout="wide", page_title="AI Autopsy v2.0", page_icon="🔬")

# --- CSS ENGINE ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    .stApp { background-color: #0E1117; }
    * { font-family: 'JetBrains Mono', monospace !important; }
    .forensic-card {
        background: rgba(23, 28, 38, 0.8);
        border: 1px solid #2d343f;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        backdrop-filter: blur(12px);
    }
    .pulse {
        display: inline-block; width: 8px; height: 8px; border-radius: 50%;
        background: #FF4B4B; box-shadow: 0 0 10px #FF4B4B;
        animation: pulse-red 2s infinite; margin-right: 8px;
    }
    @keyframes pulse-red {
        0% { transform: scale(0.95); opacity: 0.7; }
        70% { transform: scale(1.2); opacity: 1; }
        100% { transform: scale(0.95); opacity: 0.7; }
    }
</style>
""", unsafe_allow_html=True)

# --- STATE & DATA ---
if "current_case_name" not in st.session_state:
    st.session_state.current_case_name = list(ALL_CASES.keys())[0]
case = get_case(st.session_state.current_case_name)
wrong_indices = find_actual_failures(case)
if "sample_idx" not in st.session_state or st.session_state.sample_idx not in wrong_indices:
    st.session_state.sample_idx = wrong_indices[0]

autopsy = run_full_autopsy(case, st.session_state.sample_idx)
sev = autopsy["severity"]

# --- SIDEBAR ---
with st.sidebar:
    st.title("🕹️ CONTROL")
    st.selectbox("DATA DOMAIN", list(ALL_CASES.keys()), key="current_case_name")
    st.selectbox("FAILURE ID", wrong_indices, key="sample_idx")
    st.markdown("---")
    st.subheader("🛠️ WHAT-IF ENGINE")
    top_feat = list(autopsy["shap"].keys())[0]
    st.slider(f"Simulate {top_feat}", -2.0, 2.0, float(autopsy["shap"][top_feat]))
    if st.button("RUN SIMULATION"): st.toast("Recalculating...")

# --- MAIN UI ---
st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
        <h2>🔬 AUTOPSY ENGINE <span style="color:#4f46e5; font-size:16px;">v2.0</span></h2>
        <div style="text-align: right;">
            <span class="pulse"></span><span style="color:#FF4B4B; font-weight:bold; font-size:12px;">ANOMALY DETECTED</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# TIER 1: VITALS
v1, v2, v3 = st.columns(3)
v1.markdown(f'<div class="forensic-card"><small>PREDICTION</small><br><b>{case["class_names"][sev["predicted"]]}</b></div>', unsafe_allow_html=True)
v2.markdown(f'<div class="forensic-card"><small>TRUTH</small><br><b style="color:#1D9E75;">{case["class_names"][sev["true_label"]]}</b></div>', unsafe_allow_html=True)
v3.markdown(f'<div class="forensic-card"><small>CONFIDENCE</small><br><b>{sev["score"]:.1%}</b></div>', unsafe_allow_html=True)

# TIER 2: ANALYSIS
c1, c2 = st.columns([2, 1])
with c1:
    st.markdown('<div class="forensic-card"><b>FEATURE ATTRIBUTION</b>', unsafe_allow_html=True)
    st.plotly_chart(chart_shap_waterfall(autopsy["shap"], case["name"]), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="forensic-card"><b>LATENT DRIFT</b>', unsafe_allow_html=True)
    st.plotly_chart(chart_latent_space(autopsy, case), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# TIER 3: EVIDENCE
st.markdown('<div class="forensic-card"><b>🗄️ EVIDENCE LOCKER</b>', unsafe_allow_html=True)
st.dataframe(pd.DataFrame([autopsy["sample"]], columns=case['feature_names']), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)
'''
with open("app.py", "w") as f:
    f.write(code)
