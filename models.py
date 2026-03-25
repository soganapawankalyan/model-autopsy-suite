import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)


def build_credit_risk_case() -> dict:
    """
    Credit risk classification.
    Deliberate bias: model undertrained on young applicants with short
    credit history — produces systematic false positives for this group.
    """
    n = 1500
    age              = np.random.randint(18, 70, n)
    income           = np.random.normal(55000, 20000, n).clip(15000, 150000)
    credit_score     = np.random.normal(680, 80, n).clip(300, 850)
    loan_amount      = np.random.normal(25000, 12000, n).clip(1000, 80000)
    employment_years = np.random.exponential(5, n).clip(0, 30)
    debt_ratio       = np.random.beta(2, 5, n)
    credit_history   = np.random.exponential(4, n).clip(0, 20)
    num_accounts     = np.random.poisson(4, n).clip(0, 15)

    risk = (
        (credit_score < 620).astype(int) * 3 +
        (debt_ratio > 0.45).astype(int) * 2 +
        (income < 30000).astype(int) * 2 +
        (loan_amount / income > 0.6).astype(int) * 2 +
        (employment_years < 1).astype(int) +
        (credit_history < 2).astype(int) +
        np.random.binomial(1, 0.1, n)
    )
    y = (risk >= 4).astype(int)

    df = pd.DataFrame({
        "age": age, "income": income.round(0),
        "credit_score": credit_score.round(0),
        "loan_amount": loan_amount.round(0),
        "employment_years": employment_years.round(1),
        "debt_ratio": debt_ratio.round(3),
        "credit_history_years": credit_history.round(1),
        "num_accounts": num_accounts,
    })

    young_mask = (age < 28) & (credit_history < 3)
    train_mask = ~young_mask | (np.random.random(n) > 0.75)
    X_train = df[train_mask]
    y_train = y[train_mask]

    model = RandomForestClassifier(n_estimators=100, max_depth=6,
                                   min_samples_leaf=5, random_state=42)
    model.fit(X_train, y_train)

    high_risk_indices = np.where(
        (y == 0) & young_mask & (credit_score > 650) & (income > 40000)
    )[0]
    interesting_failures = high_risk_indices[:8].tolist()

    return {
        "name":        "Credit risk assessment",
        "description": "Loan default risk classifier — RandomForest trained on applicant financial profile",
        "target":      "default_risk",
        "class_names": ["Low risk", "High risk"],
        "model":       model,
        "X":           df,
        "y":           pd.Series(y, name="default_risk"),
        "features":    df.columns.tolist(),
        "failure_indices": interesting_failures,
        "known_bias":  "Model undertrained on young applicants with short credit history — produces systematic false positives for age < 28 with credit_history < 3 years",
        "color":       "#E24B4A",
    }


def build_medical_diagnosis_case() -> dict:
    """
    Medical diagnosis classification.
    Deliberate bias: model overfit on high-BMI patients,
    misses diagnosis on normal-BMI patients with atypical presentation.
    """
    n = 1200
    age         = np.random.randint(25, 80, n)
    bmi         = np.random.normal(27, 6, n).clip(16, 50)
    glucose     = np.random.normal(105, 30, n).clip(60, 300)
    blood_press = np.random.normal(72, 12, n).clip(40, 110)
    insulin     = np.random.exponential(80, n).clip(0, 400)
    skin_thick  = np.random.normal(23, 8, n).clip(0, 60)
    pregnancies = np.random.poisson(2, n).clip(0, 12)
    dpf         = np.random.exponential(0.4, n).clip(0.08, 2.5)

    disease = (
        (glucose > 140).astype(int) * 3 +
        (bmi > 32).astype(int) * 2 +
        (age > 50).astype(int) * 2 +
        (blood_press > 85).astype(int) +
        (insulin > 150).astype(int) +
        (dpf > 0.8).astype(int) +
        np.random.binomial(1, 0.08, n)
    )
    y = (disease >= 4).astype(int)

    df = pd.DataFrame({
        "age": age, "bmi": bmi.round(1),
        "glucose": glucose.round(0),
        "blood_pressure": blood_press.round(0),
        "insulin": insulin.round(0),
        "skin_thickness": skin_thick.round(0),
        "pregnancies": pregnancies,
        "diabetes_pedigree": dpf.round(3),
    })

    model = GradientBoostingClassifier(n_estimators=80, max_depth=4,
                                       learning_rate=0.1, random_state=42)
    model.fit(df, y)

    failure_idx = np.where(
        (y == 1) & (bmi < 25) & (glucose > 145) & (age < 45)
    )[0]
    interesting_failures = failure_idx[:8].tolist()

    return {
        "name":        "Medical diagnosis",
        "description": "Diabetes risk classifier — GradientBoosting trained on patient clinical markers",
        "target":      "diabetes_positive",
        "class_names": ["Negative", "Positive"],
        "model":       model,
        "X":           df,
        "y":           pd.Series(y, name="diabetes_positive"),
        "features":    df.columns.tolist(),
        "failure_indices": interesting_failures,
        "known_bias":  "Model overfit on high-BMI presentations — misses diagnosis on normal-BMI patients with elevated glucose and atypical clinical markers",
        "color":       "#EF9F27",
    }


def build_equipment_failure_case() -> dict:
    """
    Equipment failure prediction.
    Deliberate bias: model ignores interaction effects between
    temperature and vibration — misses imminent failures when
    both are moderately elevated simultaneously.
    """
    n = 1800
    temperature  = np.random.normal(72, 12, n).clip(40, 110)
    vibration    = np.random.normal(2.1, 0.8, n).clip(0.5, 8)
    pressure     = np.random.normal(14.5, 2, n).clip(8, 25)
    runtime_hrs  = np.random.exponential(2000, n).clip(100, 8000)
    error_rate   = np.random.beta(1, 8, n) * 15
    rpm          = np.random.normal(3200, 200, n).clip(2400, 4000)
    maintenance  = np.random.exponential(180, n).clip(7, 730)
    load_pct     = np.random.beta(3, 2, n) * 100

    failure = (
        (temperature > 88).astype(int) * 3 +
        (vibration > 4).astype(int) * 3 +
        ((temperature > 78) & (vibration > 3)).astype(int) * 2 +
        (error_rate > 8).astype(int) * 2 +
        (runtime_hrs > 5000).astype(int) * 2 +
        (maintenance > 365).astype(int) +
        (pressure > 19).astype(int) +
        np.random.binomial(1, 0.05, n)
    )
    y = (failure >= 4).astype(int)

    df = pd.DataFrame({
        "temperature": temperature.round(1),
        "vibration":   vibration.round(2),
        "pressure":    pressure.round(1),
        "runtime_hrs": runtime_hrs.round(0),
        "error_rate":  error_rate.round(2),
        "rpm":         rpm.round(0),
        "days_since_maintenance": maintenance.round(0),
        "load_pct":    load_pct.round(1),
    })

    model = RandomForestClassifier(n_estimators=120, max_depth=5,
                                   min_samples_leaf=8, random_state=99)
    model.fit(df, y)

    failure_idx = np.where(
    (y == 1) &
    (temperature > 76) & (temperature < 92) &
    (vibration > 2.8) & (vibration < 5.5)
    )[0]
    interesting_failures = failure_idx[:8].tolist()

    return {
        "name":        "Equipment failure prediction",
        "description": "Machinery failure classifier — RandomForest trained on IoT sensor readings",
        "target":      "imminent_failure",
        "class_names": ["Normal", "Failure imminent"],
        "model":       model,
        "X":           df,
        "y":           pd.Series(y, name="imminent_failure"),
        "features":    df.columns.tolist(),
        "failure_indices": interesting_failures,
        "known_bias":  "Model misses interaction effects — fails to detect imminent failure when temperature and vibration are both moderately elevated simultaneously",
        "color":       "#7F77DD",
    }


ALL_CASES = {
    "Credit risk assessment":      build_credit_risk_case,
    "Medical diagnosis":           build_medical_diagnosis_case,
    "Equipment failure prediction": build_equipment_failure_case,
}


def get_case(name: str) -> dict:
    return ALL_CASES[name]()


if __name__ == "__main__":
    for name, fn in ALL_CASES.items():
        case = fn()
        model = case["model"]
        X, y  = case["X"], case["y"]
        preds = model.predict(X)
        acc   = (preds == y).mean()
        print(f"\n{case['name']}")
        print(f"  Samples: {len(X)} | Features: {len(case['features'])}")
        print(f"  Accuracy: {acc:.3f} | Class balance: {y.mean():.3f}")
        print(f"  Interesting failure cases: {len(case['failure_indices'])}")
        print(f"  Known bias: {case['known_bias'][:60]}...")
