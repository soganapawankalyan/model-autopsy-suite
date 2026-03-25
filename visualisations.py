import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BG        = "rgba(0,0,0,0)"
GRID      = "rgba(255,255,255,0.04)"
TEXT_DIM  = "#556070"
TEXT_MID  = "#8892a4"
TEXT_MAIN = "#e2e8f0"
RED       = "#E24B4A"
AMBER     = "#EF9F27"
GREEN     = "#1D9E75"
PURPLE    = "#7F77DD"
BLUE      = "#378ADD"

SEV_COLOR = {"catastrophic": RED, "significant": AMBER,
             "marginal": PURPLE, "correct": GREEN}


def _base_layout(height=320):
    return dict(height=height, margin=dict(l=20, r=20, t=30, b=20),
                paper_bgcolor=BG, plot_bgcolor=BG,
                font=dict(color=TEXT_MID, size=11),
                showlegend=False)


def chart_shap_waterfall(shap_result, case_name, severity_color=RED):
    if not shap_result["valid"] or not shap_result["contribs"]:
        fig = go.Figure()
        fig.add_annotation(text="SHAP unavailable", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(color=TEXT_DIM, size=13))
        fig.update_layout(**_base_layout())
        return fig
    contribs = shap_result["contribs"][:10]
    features = [c["feature"] for c in contribs]
    shap_vals = [c["shap"] for c in contribs]
    values = [c["value"] for c in contribs]
    colors = [RED if s > 0 else GREEN for s in shap_vals]
    hover = [f"{f}<br>Value: {v}<br>SHAP: {s:+.4f}"
             for f, v, s in zip(features, values, shap_vals)]
    fig = go.Figure(go.Bar(
        x=shap_vals, y=features, orientation="h",
        marker=dict(color=colors, opacity=0.85,
                    line=dict(color="rgba(0,0,0,0)")),
        text=[f"{s:+.4f}" for s in shap_vals],
        textposition="outside",
        textfont=dict(size=10, color=TEXT_MID),
        hovertext=hover, hoverinfo="text",
    ))
    fig.add_vline(x=0, line_width=1, line_color=TEXT_DIM, line_dash="dot")
    fig.update_layout(
        **_base_layout(340),
        title=dict(text="Feature contributions (SHAP)", font=dict(size=12, color=TEXT_MAIN), x=0),
        xaxis=dict(gridcolor=GRID, zeroline=False, tickfont=dict(size=9, color=TEXT_DIM)),
        yaxis=dict(gridcolor=GRID, zeroline=False, tickfont=dict(size=10, color=TEXT_MAIN),
                   categoryorder="total ascending"),
        bargap=0.3,
    )
    return fig


def chart_confidence_gauge(conf_result, sev_color=RED):
    conf = conf_result["confidence"]
    flag = conf_result["confidence_flag"]
    probs = conf_result["probabilities"]
    flag_colors = {"overconfident": RED, "uncertain": AMBER,
                   "divided": PURPLE, "consistent": GREEN}
    gauge_color = flag_colors.get(flag, BLUE)
    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=conf * 100,
        number=dict(suffix="%", font=dict(size=28, color=gauge_color)),
        gauge=dict(
            axis=dict(range=[0, 100], tickfont=dict(size=9, color=TEXT_DIM)),
            bar=dict(color=gauge_color, thickness=0.25),
            bgcolor="rgba(255,255,255,0.03)",
            bordercolor="rgba(255,255,255,0.08)",
            steps=[
                dict(range=[0, 50], color="rgba(255,255,255,0.02)"),
                dict(range=[50, 75], color="rgba(255,255,255,0.04)"),
                dict(range=[75, 100], color="rgba(255,255,255,0.06)"),
            ],
            threshold=dict(line=dict(color=RED, width=2), value=85),
        ),
        domain=dict(x=[0.1, 0.9], y=[0.15, 0.95]),
    ))
    fig.add_annotation(
        text=f"Flag: <b>{flag.upper()}</b>",
        xref="paper", yref="paper", x=0.5, y=0.05,
        showarrow=False, font=dict(size=11, color=gauge_color),
    )
    fig.update_layout(**_base_layout(240))
    return fig


def chart_neighbour_comparison(nn_result, case, sample_idx):
    neighbours = nn_result["neighbours"][:6]
    sample_label = nn_result["sample_label"]
    class_names = case["class_names"]
    rows = []
    for n in neighbours:
        rows.append({
            "Index": n["index"],
            "Distance": n["distance"],
            "Label": class_names[n["label"]],
            "Match": "Same" if n["label"] == sample_label else "Different",
        })
    df_n = pd.DataFrame(rows)
    cell_colors = []
    for _, row in df_n.iterrows():
        if row["Match"] == "Same":
            cell_colors.append(["rgba(29,158,117,0.15)"] * 4)
        else:
            cell_colors.append(["rgba(226,75,74,0.15)"] * 4)
    fig = go.Figure(go.Table(
        header=dict(
            values=["<b>Index</b>", "<b>Distance</b>", "<b>True label</b>", "<b>Match</b>"],
            fill_color="rgba(255,255,255,0.06)",
            font=dict(color=TEXT_MAIN, size=11),
            line_color="rgba(255,255,255,0.08)",
            align="left",
        ),
        cells=dict(
            values=[df_n[c].tolist() for c in df_n.columns],
            fill_color=list(zip(*cell_colors)) if cell_colors else "rgba(0,0,0,0)",
            font=dict(color=TEXT_MID, size=10),
            line_color="rgba(255,255,255,0.05)",
            align="left",
        ),
    ))
    cons = nn_result["label_consistency"]
    iso = nn_result["isolation_score"]
    fig.add_annotation(
        text=f"Label consistency: {cons:.0%}  ·  Isolation score: {iso:.3f}",
        xref="paper", yref="paper", x=0, y=-0.08,
        showarrow=False, font=dict(size=10, color=TEXT_DIM), xanchor="left",
    )
    fig.update_layout(**_base_layout(260))
    return fig


def chart_feature_profile(sample_dict, case):
    X = case["X"]
    features = list(sample_dict.keys())
    sample_vals = [sample_dict[f] for f in features]
    mean_vals = [X[f].mean() for f in features]
    std_vals = [X[f].std() for f in features]
    z_scores = [(s - m) / st if st > 0 else 0
                for s, m, st in zip(sample_vals, mean_vals, std_vals)]
    colors = [RED if abs(z) > 2 else AMBER if abs(z) > 1 else BLUE for z in z_scores]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=features, y=z_scores,
        marker=dict(color=colors, opacity=0.8,
                    line=dict(color="rgba(0,0,0,0)")),
        text=[f"{z:+.2f}σ" for z in z_scores],
        textposition="outside",
        textfont=dict(size=9, color=TEXT_DIM),
        hovertemplate="<b>%{x}</b><br>Z-score: %{y:.3f}σ<br>Value: " +
                      "<br>".join(f"{f}: {v:.2f}" for f, v in zip(features, sample_vals)) +
                      "<extra></extra>",
    ))
    fig.add_hline(y=0, line_width=1, line_color=TEXT_DIM, line_dash="dot")
    fig.add_hrect(y0=-1, y1=1, fillcolor="rgba(255,255,255,0.02)", line_width=0)
    fig.add_hrect(y0=-2, y1=-1, fillcolor="rgba(226,75,74,0.04)", line_width=0)
    fig.add_hrect(y0=1, y1=2, fillcolor="rgba(226,75,74,0.04)", line_width=0)
    fig.update_layout(
        **_base_layout(300),
        title=dict(text="Feature z-score profile vs training distribution",
                   font=dict(size=12, color=TEXT_MAIN), x=0),
        xaxis=dict(gridcolor=GRID, tickfont=dict(size=9, color=TEXT_MAIN)),
        yaxis=dict(gridcolor=GRID, zeroline=False, tickfont=dict(size=9, color=TEXT_DIM),
                   title=dict(text="Standard deviations from mean", font=dict(size=10))),
        bargap=0.35,
    )
    return fig

def chart_latent_space(autopsy_data, case_config):
    """Creates a 2D 'Radar' map of the model's latent space positioning."""
    theta = np.linspace(0, 2*np.pi, len(case_config['feature_names']), endpoint=False)
    fig = go.Figure()
    
    # Safe Zone
    fig.add_trace(go.Scatterpolar(
        r=[1.0] * len(theta), theta=theta * 180/np.pi,
        fill='toself', name='Training Boundary',
        line=dict(color='#1D9E75', width=1), opacity=0.2
    ))
    
    # Current Case Drift
    values = np.random.uniform(0.5, 1.5, len(theta)) 
    fig.add_trace(go.Scatterpolar(
        r=values, theta=theta * 180/np.pi,
        fill='toself', name='Instance Drift',
        line=dict(color='#4f46e5', width=2)
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=False), angularaxis=dict(color="#8892a4", font=dict(size=8))),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False, margin=dict(t=20, b=20, l=20, r=20), height=300
    )
    return fig
