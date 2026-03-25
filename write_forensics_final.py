code = '''import numpy as np
import pandas as pd
import shap
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

def run_shap_analysis(model, X, sample_idx):
    sample = X.iloc[[sample_idx]]
    try:
        # Using the generic Explainer for better compatibility with 2026 library versions
        # We use a small background set (100 samples) for speed
        explainer = shap.Explainer(model.predict, X.iloc[:100])
        shap_values = explainer(sample)
        
        sv = shap_values.values[0]
        bv = shap_values.base_values[0]
        
        features = X.columns.tolist()
        values = X.iloc[sample_idx].tolist()
        
        contribs = [{"feature": f, "value": round(float(v), 4), "shap": round(float(s), 4)}
                    for f, v, s in zip(features, values, sv)]
        contribs.sort(key=lambda x: abs(x["shap"]), reverse=True)
        
        return {"contribs": contribs, "base_value": round(float(bv), 4),
                "prediction_value": round(float(bv + sum(c["shap"] for c in contribs)), 4),
                "valid": True, "error": None}
    except Exception as e:
        return {"contribs": [], "base_value": 0, "prediction_value": 0,
                "valid": False, "error": str(e)}

# ... [Keep run_nearest_neighbour_forensics, run_confidence_decomposition, 
# classify_failure_severity, find_actual_failures, and run_full_autopsy 
# from the previous write_forensics2.py] ...

def run_nearest_neighbour_forensics(X, y, sample_idx, n_neighbours=8):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    sample_scaled = X_scaled[[sample_idx]]
    nbrs = NearestNeighbors(n_neighbors=n_neighbours + 1, metric="euclidean")
    nbrs.fit(X_scaled)
    distances, indices = nbrs.kneighbors(sample_scaled)
    neighbour_indices = indices[0][1:]
    neighbour_dists = distances[0][1:]
    neighbours = []
    for idx, dist in zip(neighbour_indices, neighbour_dists):
        neighbours.append({
            "index": int(idx), "distance": round(float(dist), 4),
            "label": int(y.iloc[idx]),
            "features": {c: round(float(X.iloc[idx][c]), 4) for c in X.columns},
        })
    same_label = sum(1 for n in neighbours if n["label"] == int(y.iloc[sample_idx]))
    label_consistency = same_label / len(neighbours)
    avg_distance = float(np.mean(neighbour_dists))
    isolation_score = min(1.0, avg_distance / 5.0)
    return {"neighbours": neighbours, "label_consistency": round(label_consistency, 3),
            "avg_distance": round(avg_distance, 4), "isolation_score": round(isolation_score, 4),
            "same_label_count": same_label, "sample_label": int(y.iloc[sample_idx])}

def run_confidence_decomposition(model, X, sample_idx):
    sample = X.iloc[[sample_idx]]
    proba = model.predict_proba(sample)[0]
    pred_class = int(np.argmax(proba))
    confidence = float(np.max(proba))
    uncertainty = 1.0 - confidence
    entropy = float(-np.sum(proba * np.log(proba + 1e-10)))
    max_entropy = float(np.log(len(proba)))
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0
    tree_std = disagreement = 0.0
    if hasattr(model, "estimators_"):
        estimators_flat = model.estimators_.flatten() if hasattr(model.estimators_, "flatten") else list(model.estimators_)
        tree_preds = np.array([est.predict_proba(sample.values)[0][pred_class] for est in estimators_flat])
        tree_std = float(np.std(tree_preds))
        disagreement = tree_std
    
    if confidence > 0.85 and norm_entropy > 0.4: flag = "overconfident"
    elif confidence < 0.6: flag = "uncertain"
    elif disagreement > 0.15: flag = "divided"
    else: flag = "consistent"
    
    return {"probabilities": {i: round(float(p), 4) for i, p in enumerate(proba)},
            "predicted_class": pred_class, "confidence": round(confidence, 4),
            "uncertainty": round(uncertainty, 4), "entropy": round(entropy, 4),
            "norm_entropy": round(norm_entropy, 4), "tree_std": round(tree_std, 4),
            "disagreement": round(disagreement, 4), "confidence_flag": flag}

def classify_failure_severity(model, X, y, sample_idx, nn_result, conf_result):
    pred = int(model.predict(X.iloc[[sample_idx]])[0])
    true = int(y.iloc[sample_idx])
    is_wrong = pred != true
    conf = conf_result["confidence"]
    iso = nn_result["isolation_score"]
    if is_wrong and conf > 0.75 and iso > 0.4:
        severity, score = "catastrophic", 0.9
        reason = "High-confidence wrong prediction in an isolated region"
    elif is_wrong and conf > 0.6:
        severity, score = "significant", 0.65
        reason = "Moderate-confidence wrong prediction"
    elif is_wrong:
        severity, score = "marginal", 0.35
        reason = "Low-confidence failure near decision boundary"
    else:
        severity, score = "correct", 0.0
        reason = "Prediction is correct"
    return {"severity": severity, "score": score, "reason": reason,
            "is_wrong": is_wrong, "predicted": pred, "true_label": true,
            "confidence": conf, "isolation": iso, "consistency": nn_result["label_consistency"]}

def find_actual_failures(case):
    preds = case["model"].predict(case["X"])
    return np.where(preds != case["y"].values)[0].tolist()

def run_full_autopsy(case, sample_idx):
    return {"case": case, "sample_idx": sample_idx, "sample": case["X"].iloc[sample_idx].to_dict(),
            "shap": run_shap_analysis(case["model"], case["X"], sample_idx),
            "neighbours": run_nearest_neighbour_forensics(case["X"], case["y"], sample_idx),
            "confidence": run_confidence_decomposition(case["model"], case["X"], sample_idx),
            "severity": classify_failure_severity(case["model"], case["X"], case["y"], sample_idx, 
                                               run_nearest_neighbour_forensics(case["X"], case["y"], sample_idx),
                                               run_confidence_decomposition(case["model"], case["X"], sample_idx))}
'''
with open("forensics.py", "w") as f:
    f.write(code)
