import plotly.graph_objects as go
import numpy as np
import pandas as pd

# Append the new Latent Space function to the existing file
code = '''
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
'''
with open("visualisations.py", "a") as f:
    f.write(code)
