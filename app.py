import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
import shap

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(layout="wide", page_title="AI Autopsy Pro", page_icon="🔬", initial_sidebar_state="expanded")

# --- 2. ADVANCED CSS ENGINE ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    
    /* Global Theme */
    .stApp { background-color: #0b0f19; color: #e2e8f0; }
/* Apply custom font but protect Streamlit's built-in icons */
html, body, div, p, span, h1, h2, h3, h4, h5, h6 { 
    font-family: 'JetBrains Mono', monospace; 
}
.material-symbols-rounded, .material-symbols-outlined, .stIcon { 
    font-family: 'Material Symbols Rounded', sans-serif !important; 
}    
/* Hide default Streamlit clutter */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Keep the header transparent but leave the sidebar button visible */
    [data-testid="stHeader"] {background-color: transparent;}    
    /* Metric Cards - Perfect Alignment using Flexbox */
    .metric-container {
        display: flex;
        justify-content: space-between;
        gap: 20px;
        margin-bottom: 30px;
    }
    .metric-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 20px;
        flex: 1;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        border-top: 3px solid #3b82f6;
    }
    .metric-card.alert { border-top: 3px solid #ef4444; }
    .metric-card.safe { border-top: 3px solid #10b981; }
    
    .metric-label { color: #9ca3af; font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 8px;}
    .metric-value { font-size: 26px; font-weight: bold; color: #f3f4f6; }
    
    /* Headers and Dividers */
    .section-header {
        font-size: 14px;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 2px;
        border-bottom: 1px solid #1f2937;
        padding-bottom: 8px;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    /* Pulse Animation */
    .pulse {
        display: inline-block; width: 10px; height: 10px; border-radius: 50%;
        background: #ef4444; box-shadow: 0 0 12px #ef4444;
        animation: pulse-red 2s infinite; margin-right: 10px;
    }
    @keyframes pulse-red {
        0% { transform: scale(0.9); opacity: 0.6; }
        70% { transform: scale(1.3); opacity: 1; }
        100% { transform: scale(0.9); opacity: 0.6; }
    }
</style>
""", unsafe_allow_html=True)

# --- 3. THE FORENSIC ENGINE ---
@st.cache_resource
def initialize_engine():
    np.random.seed(42)
    X = pd.DataFrame({
        'Income_Level': np.random.rand(500),
        'Credit_Stability': np.random.rand(500),
        'Debt_To_Income': np.random.rand(500),
        'Risk_Factor_Z': np.random.rand(500)
    })
    y = ((X['Income_Level'] + X['Credit_Stability'] > 1.0) & (X['Risk_Factor_Z'] < 0.85)).astype(int)
    
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42).fit(X, y)
    explainer = shap.TreeExplainer(model)
    
    preds = model.predict(X)
    failure_indices = np.where(preds != y)[0]
    return X, y, model, explainer, failure_indices

X, y, model, explainer, failure_indices = initialize_engine()

# --- 4. SIDEBAR ---
with st.sidebar:
    st.markdown("<h3 style='color: #f3f4f6;'>⚙️ SYSTEM CONTROL</h3>", unsafe_allow_html=True)
    st.markdown("---")
    sample_id = st.selectbox("🎯 TARGET ANOMALY ID", failure_indices)
    
    st.markdown("<br><br><h4 style='color: #9ca3af; font-size: 12px;'>SIMULATION PARAMETERS</h4>", unsafe_allow_html=True)
    sim_val = st.slider("Risk_Factor_Z Override", 0.0, 1.0, float(X.iloc[sample_id]['Risk_Factor_Z']))
    
    if st.button("EXECUTE DIAGNOSTIC", use_container_width=True):
        st.toast("Diagnostic complete.", icon="✅")

# --- 5. DATA LOGIC ---
instance = X.iloc[[sample_id]]
actual_label = y[sample_id]
pred_label = model.predict(instance)[0]
prob = model.predict_proba(instance)[0][pred_label]

shap_values = explainer.shap_values(instance)
if isinstance(shap_values, list):
    current_shap = shap_values[pred_label][0]
else:
    current_shap = shap_values[0, :, pred_label]

# --- 6. MAIN UI ---

# Top Navigation / Header
st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; background: #111827; padding: 15px 25px; border-radius: 8px; border: 1px solid #1f2937;">
        <div>
            <h1 style="margin:0; font-size:22px; color: #f3f4f6;">🔬 AI FORENSIC AUTOPSY <span style="color:#3b82f6;">v4.0</span></h1>
            <p style="color:#6b7280; font-size:12px; margin:0; margin-top: 4px;">SECURE CONNECTION // LATENT INVESTIGATION: AX-{sample_id}</p>
        </div>
        <div style="text-align: right; background: rgba(239, 68, 68, 0.1); padding: 8px 16px; border-radius: 6px; border: 1px solid #ef4444;">
            <span class="pulse"></span><span style="color:#ef4444; font-weight:bold; font-size:13px; letter-spacing: 1px;">ANOMALY DETECTED</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# Tier 1: Unified Metric Cards
st.markdown(f"""
    <div class="metric-container">
        <div class="metric-card">
            <div class="metric-label">Model Prediction</div>
            <div class="metric-value" style="color: #3b82f6;">{"Approved" if pred_label == 1 else "Rejected"}</div>
        </div>
        <div class="metric-card safe">
            <div class="metric-label">Ground Truth</div>
            <div class="metric-value" style="color: #10b981;">{"Approved" if actual_label == 1 else "Rejected"}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Confidence Score</div>
            <div class="metric-value">{prob:.1%}</div>
        </div>
        <div class="metric-card alert">
            <div class="metric-label">System Status</div>
            <div class="metric-value" style="color: #ef4444;">CRITICAL</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Tier 2: Charts Area
col1, col2 = st.columns([1.5, 1])

with col1:
    st.markdown('<div class="section-header">📊 Feature Attribution (SHAP Vectors)</div>', unsafe_allow_html=True)
    
    # Perfectly sized SHAP chart
    fig_shap = go.Figure(go.Bar(
        x=current_shap, y=X.columns, orientation='h',
        marker=dict(
            color=['#ef4444' if v < 0 else '#3b82f6' for v in current_shap],
            line=dict(width=0)
        ),
        width=0.6 # Slimmer bars
    ))
    fig_shap.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)', 
        font=dict(color='#9ca3af', family='JetBrains Mono'),
        margin=dict(l=0, r=20, t=10, b=20),
        height=320,
        xaxis=dict(showgrid=True, gridcolor='#1f2937', zerolinecolor='#4b5563'),
        yaxis=dict(showgrid=False)
    )
    st.plotly_chart(fig_shap, use_container_width=True, config={'displayModeBar': False})

with col2:
    st.markdown('<div class="section-header">🌐 Latent Drift Radar</div>', unsafe_allow_html=True)
    
    # Perfectly sized Radar chart
    categories = list(X.columns)
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=[1]*4, theta=categories, fill='toself', name='Baseline', 
        line=dict(color='#10b981', width=1), fillcolor='rgba(16, 185, 129, 0.1)'
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=instance.values[0]*1.5, theta=categories, fill='toself', name='Anomaly', 
        line=dict(color='#3b82f6', width=2), fillcolor='rgba(59, 130, 246, 0.3)'
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=False),
            angularaxis=dict(color="#9ca3af", gridcolor="#1f2937", linecolor="#1f2937")
        ),
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='JetBrains Mono', size=10),
        showlegend=False,
        margin=dict(l=40, r=40, t=20, b=20),
        height=320
    )
    st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})

# Tier 3: Evidence Locker
st.markdown('<div class="section-header">🗄️ Raw Evidence Matrix</div>', unsafe_allow_html=True)
st.dataframe(
    instance.style.background_gradient(cmap='ocean', axis=1).format("{:.4f}"), 
    use_container_width=True,
    hide_index=True
)
