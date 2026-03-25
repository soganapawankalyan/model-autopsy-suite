# Model Autopsy Suite

> *When a model fails, don't retrain and hope. Investigate.*

A forensic investigation console for machine learning failures. Load any trained classifier, select a wrong prediction, and run a full five-layer autopsy — SHAP feature attribution, nearest neighbour forensics, confidence decomposition, feature z-score profiling, and an AI-written structured failure report.

## Why this exists

Most ML debugging stops at "the model got it wrong." Model Autopsy Suite asks the next five questions: which features caused the failure, are similar training examples also failing, is the model overconfident or genuinely uncertain, where does this sample sit relative to the training distribution, and what should be fixed. The result is a structured incident report a team can actually act on.

## Three pre-loaded cases

| Case | Model | Known bias |
|---|---|---|
| Credit risk assessment | RandomForest | Undertrained on young applicants with short credit history |
| Medical diagnosis | GradientBoosting | Overfit on high-BMI presentations, misses atypical markers |
| Equipment failure | RandomForest | Misses interaction effects between moderate temperature and vibration |

## Five forensic layers

- **SHAP attribution** — which features drove the prediction and by how much
- **Nearest neighbour forensics** — are similar training examples also failing, or is this isolated
- **Confidence decomposition** — is the model overconfident, uncertain, or internally divided
- **Feature z-score profile** — where does this sample sit relative to the training distribution
- **AI failure report** — Ollama-written structured report with probable cause and corrective actions

## Stack

- Python 3.11 · scikit-learn · XGBoost · SHAP · Plotly · Streamlit · Ollama llama3.2:3b

## Setup
```bash
git clone https://github.com/soganapawankalyan/model-autopsy-suite.git
cd model-autopsy-suite
python -m venv venv && source venv/bin/activate
brew install libomp
pip install -r requirements.txt
ollama pull llama3.2:3b
python -m streamlit run app.py
```

## Interview talking points

- Built SHAP-based feature attribution from scratch using TreeExplainer — produces per-prediction waterfall charts, not average importance scores
- Implemented nearest neighbour forensics to distinguish isolated failures from systematic model blind spots — directly analogous to root cause analysis in production ML systems
- Designed a composite failure severity classifier combining confidence, isolation, and label consistency scores
- Identified and deliberately injected known biases into each model to produce interpretable, instructive failure cases for the autopsy
